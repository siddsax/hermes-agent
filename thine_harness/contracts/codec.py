"""Strict decoder and encoder for the vendored v1 contract pack."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ._base import ContractDTO, JSONValue, _DTO_BY_TARGET
from ._schema import (
    SchemaValidator,
    ValidationReport,
    load_json,
    object_without_duplicate_members,
    reject_non_finite_json,
    reject_unsafe_integral_numbers,
    resolve_schema,
    semantic_errors,
)


CONTRACT_PACK_ROOT = Path(__file__).parent / "local-hermes-thine" / "v1"
PROVENANCE_PATH = CONTRACT_PACK_ROOT.parent / "local-hermes-thine-v1.provenance.json"


class ContractDecodeError(ValueError):
    """The payload cannot be safely interpreted under contract v1."""


def load_manifest() -> dict[str, Any]:
    manifest = load_json(CONTRACT_PACK_ROOT / "manifest.json")
    if not isinstance(manifest, dict):
        raise ContractDecodeError("contract manifest must be an object")
    return manifest


def _interaction_actions() -> dict[str, list[str]]:
    document = load_json(CONTRACT_PACK_ROOT / "metadata" / "interaction-allowlist.json")
    if not isinstance(document, dict):
        return {}
    actions = document.get("allowlisted_kinds")
    return actions if isinstance(actions, dict) else {}


def _parse_wire(wire: str | bytes | bytearray) -> JSONValue:
    try:
        value = json.loads(
            wire,
            parse_constant=reject_non_finite_json,
            object_pairs_hook=object_without_duplicate_members,
        )
        reject_unsafe_integral_numbers(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractDecodeError(str(exc)) from exc
    return value


def validate_payload(type_id: str, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
    manifest = load_manifest()
    target = manifest.get("schema_targets", {}).get(type_id)
    if not isinstance(target, str):
        raise ContractDecodeError(f"unknown contract target {type_id!r}")
    try:
        document, schema = resolve_schema(CONTRACT_PACK_ROOT, target)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ContractDecodeError(f"contract schema unavailable for {type_id!r}: {exc}") from exc

    errors = SchemaValidator(document).validate(payload, schema)
    errors.extend(semantic_errors(type_id, payload, _interaction_actions()))
    if errors:
        raise ContractDecodeError("; ".join(errors))
    return payload


def decode_contract(type_id: str, wire: str | bytes | bytearray) -> ContractDTO:
    payload = _parse_wire(wire)
    if not isinstance(payload, dict):
        raise ContractDecodeError("contract payload must be a JSON object")
    validated = validate_payload(type_id, payload)
    dto_type = _DTO_BY_TARGET.get(type_id)
    if dto_type is None:
        raise ContractDecodeError(f"no Python DTO is registered for {type_id!r}")
    return dto_type._from_validated(validated)


def _snapshot_errors() -> list[str]:
    try:
        provenance = load_json(PROVENANCE_PATH)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"provenance: {exc}"]
    if not isinstance(provenance, dict) or not isinstance(provenance.get("file_sha256"), dict):
        return ["provenance: invalid file_sha256 map"]

    expected = provenance["file_sha256"]
    actual_paths = {
        str(path.relative_to(CONTRACT_PACK_ROOT))
        for path in CONTRACT_PACK_ROOT.rglob("*")
        if path.is_file()
    }
    errors: list[str] = []
    if actual_paths != set(expected):
        errors.append("provenance: vendored file set differs from accepted snapshot")
    for relative_path, expected_hash in expected.items():
        path = CONTRACT_PACK_ROOT / relative_path
        if not path.is_file():
            continue
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            errors.append(f"provenance: hash mismatch for {relative_path}")
    return errors


def validate_contract_pack() -> ValidationReport:
    manifest = load_manifest()
    errors = _snapshot_errors()
    valid_count = 0
    invalid_count = 0
    covered_targets: set[str] = set()
    for expectation, suite_paths in manifest["fixture_suites"].items():
        for relative_path in suite_paths:
            suite = load_json(CONTRACT_PACK_ROOT / relative_path)
            for case in suite["cases"]:
                try:
                    validate_payload(case["target"], case["payload"])
                    case_errors: list[str] = []
                except ContractDecodeError as exc:
                    case_errors = [str(exc)]
                if expectation == "valid":
                    valid_count += 1
                    covered_targets.add(case["target"])
                    errors.extend(
                        f"fixture {case['case_id']}: {message}" for message in case_errors
                    )
                else:
                    invalid_count += 1
                    if not case_errors:
                        errors.append(
                            f"fixture {case['case_id']}: invalid fixture unexpectedly passed"
                        )
                    elif case.get("expected_error") and case["expected_error"] not in case_errors[0]:
                        errors.append(
                            f"fixture {case['case_id']}: expected {case['expected_error']!r}; "
                            f"observed {case_errors!r}"
                        )
    for required_target in manifest["required_fixture_targets"]:
        if required_target not in covered_targets:
            errors.append(f"manifest: required fixture target not covered: {required_target}")
    return ValidationReport(
        errors=tuple(errors),
        valid_fixture_count=valid_count,
        invalid_fixture_count=invalid_count,
    )
