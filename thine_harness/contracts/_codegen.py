"""Generate statically typed immutable DTO views from the frozen v1 schemas."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


PACKAGE_ROOT = Path(__file__).parent
PACK_ROOT = PACKAGE_ROOT / "local-hermes-thine" / "v1"


def _pascal(value: str) -> str:
    return "".join(
        part.upper() if part in {"p0", "p1", "p2"} else part.title()
        for part in value.split("_")
    )


class ViewGenerator:
    def __init__(self, document_name: str, document: dict[str, Any]):
        self.prefix = _pascal(document_name.removesuffix(".schema"))
        self.document = document
        self.protocols: dict[str, dict[str, Any]] = {}

    def type_for(self, schema: dict[str, Any], hint: str) -> str:
        if "$ref" in schema:
            reference = schema["$ref"]
            if not reference.startswith("#/$defs/"):
                return "FrozenJSONValue"
            name = reference.rsplit("/", 1)[-1]
            resolved = self.document["$defs"][name]
            siblings = {key: value for key, value in schema.items() if key != "$ref"}
            if siblings:
                return self.type_for(_overlay_schema(resolved, siblings), hint)
            return self.type_for(resolved, _pascal(name))
        if "const" in schema:
            return f"Literal[{schema['const']!r}]"
        if "enum" in schema:
            return "Literal[" + ", ".join(repr(value) for value in schema["enum"]) + "]"
        if "oneOf" in schema or "allOf" in schema:
            return " | ".join(
                self.type_for(option, f"{hint}Variant{index}")
                for index, option in enumerate(self.variants(schema), start=1)
            )

        schema_type = schema.get("type")
        if schema_type is None and "properties" in schema:
            schema_type = "object"
        if isinstance(schema_type, list):
            return " | ".join(
                self.type_for({**schema, "type": option}, hint)
                for option in schema_type
            )
        if schema_type == "null":
            return "None"
        if schema_type == "boolean":
            return "bool"
        if schema_type == "integer":
            return "int"
        if schema_type == "number":
            return "int | float"
        if schema_type == "string":
            return "str"
        if schema_type == "array":
            return (
                f"tuple[{self.type_for(schema.get('items', {}), hint + 'Item')}, ...]"
            )
        if schema_type == "object":
            properties = schema.get("properties", {})
            if properties:
                name = f"{self.prefix}{hint}View"
                self.protocols.setdefault(name, schema)
                return name
            additional = schema.get("additionalProperties", True)
            child = (
                self.type_for(additional, hint + "Value")
                if isinstance(additional, dict)
                else "FrozenJSONValue"
            )
            return f"Mapping[str, {child}]"
        return "FrozenJSONValue"

    def target_view(self, target_name: str, schema: dict[str, Any]) -> str:
        return self.type_for(schema, _pascal(target_name))

    def variants(self, schema: dict[str, Any]) -> list[dict[str, Any]]:
        if "oneOf" in schema:
            base = {key: value for key, value in schema.items() if key != "oneOf"}
            return [
                nested
                for option in schema["oneOf"]
                for nested in self.variants(_overlay_schema(base, option))
            ]
        if "allOf" in schema:
            combinations = [
                {key: value for key, value in schema.items() if key != "allOf"}
            ]
            for component in schema["allOf"]:
                combinations = [
                    _overlay_schema(current, refinement)
                    for current in combinations
                    for refinement in self.variants(component)
                ]
            return combinations
        return [schema]

    def optional_paths(
        self,
        schema: dict[str, Any],
        path: tuple[str, ...] = (),
    ) -> dict[tuple[str, ...], set[str]]:
        if "$ref" in schema:
            reference = schema["$ref"]
            if not reference.startswith("#/$defs/"):
                return {}
            resolved = self.document["$defs"][reference.rsplit("/", 1)[-1]]
            siblings = {key: value for key, value in schema.items() if key != "$ref"}
            return self.optional_paths(
                _overlay_schema(resolved, siblings) if siblings else resolved,
                path,
            )
        if "oneOf" in schema or "allOf" in schema:
            combined: dict[tuple[str, ...], set[str]] = {}
            for variant in self.variants(schema):
                _merge_optional_paths(combined, self.optional_paths(variant, path))
            return combined

        schema_type = schema.get("type")
        if schema_type is None and "properties" in schema:
            schema_type = "object"
        if schema_type == "array":
            return self.optional_paths(schema.get("items", {}), (*path, "*"))
        if schema_type != "object":
            return {}

        required = set(schema.get("required", []))
        result: dict[tuple[str, ...], set[str]] = {}
        for field, child in schema.get("properties", {}).items():
            if field not in required:
                result.setdefault(path, set()).add(field)
            _merge_optional_paths(
                result,
                self.optional_paths(child, (*path, field)),
            )
        return result

    def render(self) -> str:
        rendered: set[str] = set()
        blocks: list[str] = []
        while pending := sorted(set(self.protocols) - rendered):
            for name in pending:
                schema = self.protocols[name]
                required = set(schema.get("required", []))
                lines = [f"class {name}(Protocol):"]
                for field, child in schema.get("properties", {}).items():
                    child_type = self.type_for(
                        child, f"{name.removesuffix('View')}{_pascal(field)}"
                    )
                    if field not in required and "None" not in child_type.split(" | "):
                        child_type += " | None"
                    lines.extend([
                        "    @property",
                        f"    def {field}(self) -> {child_type}: ...",
                        "",
                    ])
                if len(lines) == 1:
                    lines.append("    ...")
                blocks.append("\n".join(lines).rstrip())
                rendered.add(name)
        return "\n\n\n".join(blocks)


def generate() -> None:
    manifest = json.loads((PACK_ROOT / "manifest.json").read_text(encoding="utf-8"))
    serialization = json.loads(
        (PACK_ROOT / "metadata" / "serialization-map.json").read_text(encoding="utf-8")
    )
    documents: dict[str, dict[str, Any]] = {}
    generators: dict[str, ViewGenerator] = {}
    target_views: dict[str, str] = {}
    target_optional_paths: dict[str, dict[tuple[str, ...], set[str]]] = {}

    for target_name, pointer in manifest["schema_targets"].items():
        relative, fragment = pointer.split("#", 1)
        document = documents.setdefault(
            relative,
            json.loads((PACK_ROOT / relative).read_text(encoding="utf-8")),
        )
        generator = generators.setdefault(
            relative, ViewGenerator(Path(relative).stem, document)
        )
        schema: Any = document
        for part in fragment.removeprefix("/").split("/"):
            schema = schema[part]
        target_views[target_name] = generator.target_view(target_name, schema)
        target_optional_paths[target_name] = generator.optional_paths(schema)

    view_blocks = [generator.render() for generator in generators.values()]
    view_source = '''"""Generated immutable structural views for local-Hermes contract v1.

Regenerate with ``python -m thine_harness.contracts._codegen`` after accepting
a new contract pack version. Do not edit this file by hand.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

from ._base import FrozenJSONValue


'''
    (PACKAGE_ROOT / "_views_generated.py").write_text(
        view_source + "\n\n\n".join(block for block in view_blocks if block) + "\n",
        encoding="utf-8",
    )

    by_module: dict[str, list[str]] = {}
    for target_name, metadata in serialization["types"].items():
        module_name = metadata["python"].rsplit(".", 1)[-1]
        by_module.setdefault(module_name, []).append(target_name)

    for module_name, targets in by_module.items():
        target_rows = [
            (target, _pascal(target), target_views[target])
            for target in sorted(targets)
        ]
        view_imports = {
            name
            for _, _, expression in target_rows
            for name in re.findall(r"\b[A-Z][A-Za-z0-9]+View\b", expression)
        }
        imports = ",\n    ".join(sorted(view_imports))
        body = [
            f'"""Generated typed {module_name} contract DTOs."""',
            "",
            "from ._base import ContractDTO, contract_type",
            "from ._views_generated import (",
            f"    {imports},",
            ")",
            "",
        ]
        for target, class_name, view_name in target_rows:
            optional_literal = _render_optional_paths(target_optional_paths[target])
            body.extend([
                "",
                f'@contract_type("{target}")',
                f"class {class_name}(ContractDTO[{view_name}]):",
                f'    """Immutable typed view of a validated {target} payload."""',
                "",
                "    __slots__ = ()",
                f"    _optional_fields = {optional_literal}",
            ])
        exports = ",\n    ".join(f'"{class_name}"' for _, class_name, _ in target_rows)
        body.extend(["", "", "__all__ = [", f"    {exports},", "]", ""])
        (PACKAGE_ROOT / f"{module_name}.py").write_text(
            "\n".join(body), encoding="utf-8"
        )


def _overlay_schema(base: dict[str, Any], refinement: dict[str, Any]) -> dict[str, Any]:
    """Overlay one discriminated branch without dropping its common object shape."""

    merged = {**base, **refinement}
    if "required" in base or "required" in refinement:
        merged["required"] = list(
            dict.fromkeys([*base.get("required", []), *refinement.get("required", [])])
        )
    base_properties = base.get("properties", {})
    refinement_properties = refinement.get("properties", {})
    if base_properties or refinement_properties:
        merged["properties"] = {
            key: _overlay_schema(base_properties.get(key, {}), child)
            if key in base_properties
            else child
            for key, child in refinement_properties.items()
        }
        merged["properties"] = {**base_properties, **merged["properties"]}
    return merged


def _merge_optional_paths(
    destination: dict[tuple[str, ...], set[str]],
    source: dict[tuple[str, ...], set[str]],
) -> None:
    for path, fields in source.items():
        destination.setdefault(path, set()).update(fields)


def _render_optional_paths(paths: dict[tuple[str, ...], set[str]]) -> str:
    if not paths:
        return "{}"
    entries = [
        f"{path!r}: frozenset({tuple(sorted(fields))!r})"
        for path, fields in sorted(paths.items())
    ]
    return "{" + ", ".join(entries) + "}"


if __name__ == "__main__":
    generate()
