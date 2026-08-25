"""Dependency-free validation primitives for the frozen schema subset."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


EXACT_FORBIDDEN_FIELDS = {"tokens", "raw_keystrokes", "raw_notification_payload"}
FORBIDDEN_SENSITIVE_PARTS = {
    "password",
    "passwords",
    "credential",
    "credentials",
    "access_token",
    "authorization",
    "clipboard",
    "hidden_reasoning",
    "system_prompt",
    "stack_trace",
    "raw_audio",
}


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    valid_fixture_count: int
    invalid_fixture_count: int


class SchemaValidator:
    """Validate the deliberately small JSON Schema 2020-12 subset in v1."""

    def __init__(self, document: dict[str, Any]):
        self.document = document

    def validate(self, instance: Any, schema: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        self._validate(instance, schema, "$", errors)
        return errors

    def _validate(
        self,
        instance: Any,
        schema: dict[str, Any],
        path: str,
        errors: list[str],
    ) -> None:
        if "$ref" in schema:
            self._validate(
                instance, self._resolve_pointer(schema["$ref"]), path, errors
            )
            schema = {key: value for key, value in schema.items() if key != "$ref"}

        if "const" in schema and not self._json_equal(instance, schema["const"]):
            errors.append(f"{path}: expected constant {schema['const']!r}")
            return
        if "enum" in schema and not any(
            self._json_equal(instance, option) for option in schema["enum"]
        ):
            errors.append(f"{path}: unknown enum value {instance!r}")
            return
        if isinstance(instance, float) and not math.isfinite(instance):
            errors.append(f"{path}: non-finite number is not valid JSON")
            return
        if _is_unsafe_integral_number(instance):
            errors.append(f"{path}: integer exceeds interoperable JSON safe range")
            return

        expected_type = schema.get("type")
        if expected_type is not None and not self._matches_type(
            instance, expected_type
        ):
            errors.append(f"{path}: expected type {expected_type!r}")
            return

        if isinstance(instance, dict):
            for key in schema.get("required", []):
                if key not in instance:
                    errors.append(f"{path}: missing required property {key!r}")
            properties = schema.get("properties", {})
            for key, value in instance.items():
                if key in properties:
                    self._validate(value, properties[key], f"{path}.{key}", errors)
                elif schema.get("additionalProperties", True) is False:
                    errors.append(f"{path}: undeclared additive property {key!r}")
                elif isinstance(schema.get("additionalProperties"), dict):
                    self._validate(
                        value,
                        schema["additionalProperties"],
                        f"{path}.{key}",
                        errors,
                    )
            if len(instance) < schema.get("minProperties", 0):
                errors.append(f"{path}: too few properties")
            if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
                errors.append(f"{path}: too many properties")

        if isinstance(instance, list):
            if len(instance) < schema.get("minItems", 0):
                errors.append(f"{path}: too few items")
            if "maxItems" in schema and len(instance) > schema["maxItems"]:
                errors.append(f"{path}: too many items")
            if schema.get("uniqueItems") and len({
                _json_fingerprint(item) for item in instance
            }) != len(instance):
                errors.append(f"{path}: duplicate items")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, value in enumerate(instance):
                    self._validate(value, item_schema, f"{path}[{index}]", errors)

        if isinstance(instance, str):
            if len(instance) < schema.get("minLength", 0):
                errors.append(f"{path}: string shorter than minimum")
            if "maxLength" in schema and len(instance) > schema["maxLength"]:
                errors.append(f"{path}: string longer than maximum")
            if "pattern" in schema and re.search(schema["pattern"], instance) is None:
                errors.append(f"{path}: string does not match required pattern")

        if isinstance(instance, (int, float)) and not isinstance(instance, bool):
            if "minimum" in schema and instance < schema["minimum"]:
                errors.append(f"{path}: number below minimum")
            if "maximum" in schema and instance > schema["maximum"]:
                errors.append(f"{path}: number above maximum")

        for keyword, required_matches in (("oneOf", 1), ("anyOf", None)):
            alternatives = schema.get(keyword)
            if alternatives:
                matches = sum(
                    not self.validate(instance, option) for option in alternatives
                )
                if (required_matches is not None and matches != required_matches) or (
                    required_matches is None and matches == 0
                ):
                    errors.append(f"{path}: does not satisfy {keyword}")

        for component in schema.get("allOf", []):
            self._validate(instance, component, path, errors)

    def _resolve_pointer(self, reference: str) -> dict[str, Any]:
        if not reference.startswith("#/"):
            raise ValueError(f"external schema reference is forbidden: {reference}")
        value: Any = self.document
        for raw_part in reference[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            value = value[part]
        if not isinstance(value, dict):
            raise ValueError(f"schema reference is not an object: {reference}")
        return value

    @staticmethod
    def _matches_type(instance: Any, expected: str | list[str]) -> bool:
        expected_types = [expected] if isinstance(expected, str) else expected
        predicates = {
            "null": lambda value: value is None,
            "boolean": lambda value: isinstance(value, bool),
            "integer": lambda value: (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and (not isinstance(value, float) or value.is_integer())
            ),
            "number": lambda value: (
                isinstance(value, (int, float)) and not isinstance(value, bool)
            ),
            "string": lambda value: isinstance(value, str),
            "array": lambda value: isinstance(value, list),
            "object": lambda value: isinstance(value, dict),
        }
        return any(predicates[name](instance) for name in expected_types)

    @classmethod
    def _json_equal(cls, left: Any, right: Any) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return isinstance(left, bool) and isinstance(right, bool) and left == right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return (
                (not isinstance(left, float) or math.isfinite(left))
                and (not isinstance(right, float) or math.isfinite(right))
                and left == right
            )
        if type(left) is not type(right):
            return False
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(
                cls._json_equal(a, b) for a, b in zip(left, right)
            )
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(
                cls._json_equal(left[key], right[key]) for key in left
            )
        return left == right


def reject_non_finite_json(value: str) -> Any:
    raise json.JSONDecodeError(
        f"non-finite number is not valid JSON: {value!r}", value, 0
    )


def object_without_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise json.JSONDecodeError(f"duplicate JSON object member: {key!r}", key, 0)
        value[key] = child
    return value


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=reject_non_finite_json,
            object_pairs_hook=object_without_duplicate_members,
        )
    reject_unsafe_integral_numbers(value)
    return value


def reject_unsafe_integral_numbers(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            reject_unsafe_integral_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_unsafe_integral_numbers(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: non-finite number is not valid JSON")
    elif _is_unsafe_integral_number(value):
        raise ValueError(f"{path}: integer exceeds interoperable JSON safe range")


def resolve_schema(
    pack_root: Path, target: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative_path, fragment = target.split("#", 1)
    path = (pack_root / relative_path).resolve()
    if not path.is_relative_to(pack_root.resolve()):
        raise ValueError(f"schema target escapes pack root: {target}")
    document = load_json(path)
    if not isinstance(document, dict):
        raise ValueError(f"schema document is not an object: {relative_path}")
    schema: Any = document
    for raw_part in fragment.removeprefix("/").split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        schema = schema[part]
    if not isinstance(schema, dict):
        raise ValueError(f"schema target is not an object: {target}")
    return document, schema


def semantic_errors(
    target: str,
    payload: dict[str, Any],
    interaction_actions: dict[str, list[str]],
) -> list[str]:
    errors: list[str] = []
    if target == "tick":
        expected: dict[str, tuple[set[str], str]] = {
            "p0_user_chat": ({"p0"}, "user_message"),
            "p1_transcript": ({"p1"}, "transcript_availability"),
            "p1_speaker": ({"p1"}, "speaker_mapping"),
            "p1_interaction": ({"p1"}, "interaction_window"),
            "p2_scheduled": ({"p2", "p1"}, "schedule"),
        }
        priorities, source_kind = expected.get(str(payload.get("kind")), (set(), ""))
        if payload.get("priority") not in priorities:
            errors.append("tick kind priority mismatch")
        if (
            payload.get("source_ref", {}).get("kind") != source_kind
            or payload.get("payload", {}).get("payload_kind") != source_kind
        ):
            errors.append("tick kind source mismatch")
        if payload.get("source_ref", {}).get("id") != payload.get("payload", {}).get(
            "reference_id"
        ):
            errors.append("tick source reference mismatch")

    if target == "home_state":
        component_ids = [node.get("component_id") for node in payload.get("nodes", [])]
        if len(component_ids) != len(set(component_ids)):
            errors.append("Home State contains duplicate singleton component IDs")

    if target == "interaction_event":
        kind = str(payload.get("kind"))
        action = payload.get("safe_payload", {}).get("action")
        if action not in interaction_actions.get(kind, []):
            errors.append("interaction action is not allowlisted for kind")

    if target == "hermes_control_request" and payload.get(
        "created_at_ms", 0
    ) >= payload.get("deadline_at_ms", 0):
        errors.append("HermesControlPort deadline must follow request creation")

    if target == "tool_result":
        if payload.get("started_at_ms", 0) >= payload.get("deadline_at_ms", 0):
            errors.append("tool deadline must follow tool start")
        if payload.get("finished_at_ms", 0) < payload.get("started_at_ms", 0):
            errors.append("tool finish precedes tool start")

    if target == "topic_lifecycle":
        asked = payload.get("last_asked_at_ms")
        eligible = payload.get("next_eligible_at_ms")
        if asked is not None and eligible is not None and eligible < asked:
            errors.append("topic next eligible time precedes last ask")

    if target in {"final_reply_outbox", "final_reply_receipt"}:
        expected_key = f"assistant-message:{payload.get('assistant_message_id')}"
        if payload.get("idempotency_key") != expected_key:
            label = "outbox" if target == "final_reply_outbox" else "receipt"
            errors.append(f"final reply {label} is not keyed by assistant_message_id")

    if target in {"p0_submission_outbox", "mobile_chat_outbox", "queue_receipt"}:
        if (
            payload.get("idempotency_key")
            != f"user-message:{payload.get('user_message_id')}"
        ):
            errors.append("chat submission is not keyed by user_message_id")

    if target == "preferences":
        keys = [entry.get("key") for entry in payload.get("narrow_preferences", [])]
        if len(keys) != len(set(keys)):
            errors.append("preferences contain duplicate keys")

    if target == "reset_command":
        snapshot = payload.get("explicit_preferences_to_delete", [])
        keys = [entry.get("key") for entry in snapshot if isinstance(entry, dict)]
        if len(keys) != len(set(keys)):
            errors.append("reset preference snapshot contains duplicate keys")
        if payload.get("scope") == "all_hermes_state" and payload.get(
            "confirmed_preferences_revision"
        ) != payload.get("current_preferences_revision"):
            errors.append("reset preference confirmation revision is stale")

    if target == "quarantine_record":
        immutable_range = payload.get("immutable_range") or {}
        if payload.get("source_kind") == "interaction" and immutable_range.get(
            "first_cursor", 0
        ) > immutable_range.get("last_cursor", 0):
            errors.append("quarantine interaction cursor range invalid")
        if payload.get("source_kind") == "transcript":
            first = immutable_range.get("first_sequence_number")
            last = immutable_range.get("last_sequence_number")
            if (first is None) != (last is None) or (
                first is not None and last is not None and first > last
            ):
                errors.append("quarantine transcript sequence range invalid")

    _inspect_extensions(payload, errors)
    for path, key in _walk_keys(payload):
        if key in EXACT_FORBIDDEN_FIELDS:
            errors.append(f"forbidden exact {key} field: {path}")
        sensitive_part = _sensitive_field_part(key)
        if sensitive_part:
            errors.append(
                f"fixture contains forbidden sensitive field: {sensitive_part} at {path}"
            )
    return errors


def _is_unsafe_integral_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or value.is_integer())
        and abs(value) > 9_007_199_254_740_991
    )


def _walk_keys(value: Any, path: str = "$") -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append((f"{path}.{key}", key))
            keys.extend(_walk_keys(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            keys.extend(_walk_keys(child, f"{path}[{index}]"))
    return keys


def _sensitive_field_part(key: str) -> str | None:
    normalized = _normalized_wire_key(key)
    return next(
        (
            part
            for part in FORBIDDEN_SENSITIVE_PARTS
            if f"_{part}_" in f"_{normalized}_"
        ),
        None,
    )


def _inspect_extensions(value: Any, errors: list[str], path: str = "$") -> None:
    forbidden = {
        "action",
        "actions",
        "command",
        "commands",
        "code",
        "navigation_intent",
        "navigation_intents",
        "tool_call",
        "tool_calls",
        "tool_name",
        "tool_names",
        "url",
        "urls",
        "wire_name",
        "wire_names",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "extensions" and isinstance(child, dict):
                for extension_path, extension_key in _walk_keys(
                    child, f"{path}.extensions"
                ):
                    normalized = _normalized_wire_key(extension_key)
                    if normalized in forbidden:
                        errors.append(
                            f"extensions must be inert and non-executable: {extension_path}"
                        )
            _inspect_extensions(child, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _inspect_extensions(child, errors, f"{path}[{index}]")


def _normalized_wire_key(key: str) -> str:
    with_acronym_boundaries = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
    with_word_boundaries = re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", with_acronym_boundaries
    )
    return re.sub(r"[^a-z0-9]+", "_", with_word_boundaries.lower()).strip("_")


def _json_fingerprint(value: Any) -> tuple[Any, ...]:
    """Return a hashable fingerprint with the same equality as JSON Schema."""

    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, int):
        return ("number", value, 1)
    if isinstance(value, float):
        numerator, denominator = value.as_integer_ratio()
        return ("number", numerator, denominator)
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, list):
        return ("array", *(_json_fingerprint(child) for child in value))
    if isinstance(value, dict):
        return (
            "object",
            *(sorted((key, _json_fingerprint(child)) for key, child in value.items())),
        )
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")
