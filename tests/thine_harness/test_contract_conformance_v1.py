from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from thine_harness.contracts import (
    CONTRACT_PACK_ROOT,
    ContractDecodeError,
    decode_contract,
    load_manifest,
    validate_contract_pack,
)
from thine_harness.contracts.tool_metadata import PRODUCT_TOOL_NAMESPACES


def _fixture_cases(expectation: str):
    manifest = load_manifest()
    for relative_path in manifest["fixture_suites"][expectation]:
        suite = json.loads((CONTRACT_PACK_ROOT / relative_path).read_text(encoding="utf-8"))
        yield from suite["cases"]


def test_vendored_pack_is_the_complete_accepted_controller_snapshot():
    provenance = json.loads(
        (CONTRACT_PACK_ROOT.parent / "local-hermes-thine-v1.provenance.json").read_text(
            encoding="utf-8"
        )
    )

    assert provenance["controller_commit"] == (
        "2479efa6059ae2b0185cfdf575c53c74eb64ce59"
    )
    assert provenance["contract_version"] == {"major": 1, "minor": 0}
    assert provenance["file_sha256"]
    assert set(provenance["file_sha256"]) == {
        str(path.relative_to(CONTRACT_PACK_ROOT))
        for path in CONTRACT_PACK_ROOT.rglob("*")
        if path.is_file()
    }

    report = validate_contract_pack()
    assert report.errors == ()
    assert report.valid_fixture_count == 89
    assert report.invalid_fixture_count == 62


def test_every_manifest_target_decodes_to_its_assigned_typed_python_dto():
    manifest = load_manifest()
    serialization_map = json.loads(
        (CONTRACT_PACK_ROOT / "metadata" / "serialization-map.json").read_text(
            encoding="utf-8"
        )
    )
    cases = list(_fixture_cases("valid"))

    assert {case["target"] for case in cases} == set(manifest["required_fixture_targets"])
    for case in cases:
        dto = decode_contract(case["target"], json.dumps(case["payload"]))
        expected_module = serialization_map["types"][case["target"]]["python"]

        assert dto.type_id == case["target"]
        assert dto.to_dict() == case["payload"]
        assert json.loads(dto.to_json()) == case["payload"]
        assert dto.__class__.__module__ == expected_module
        assert getattr(importlib.import_module(expected_module), dto.__class__.__name__) is dto.__class__


def test_every_unchanged_negative_fixture_fails_closed_with_the_frozen_reason():
    for case in _fixture_cases("invalid"):
        with pytest.raises(ContractDecodeError) as error:
            decode_contract(case["target"], json.dumps(case["payload"]))

        assert case["expected_error"] in str(error.value)


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ('{"schema_version":{"major":1,"minor":0},"ordinal":1,"ordinal":1}', "duplicate"),
        ('{"schema_version":{"major":1,"minor":0},"ordinal":NaN}', "non-finite"),
        (
            '{"schema_version":{"major":1,"minor":0},"ordinal":9007199254740992}',
            "safe range",
        ),
    ],
)
def test_json_boundary_rejects_duplicate_keys_non_finite_and_unsafe_numbers(wire, expected):
    with pytest.raises(ContractDecodeError, match=expected):
        decode_contract("attempt", wire)


def test_same_major_future_minor_accepts_only_declared_inert_extensions():
    case = next(case for case in _fixture_cases("valid") if case["target"] == "attempt")
    payload = case["payload"] | {
        "schema_version": {"major": 1, "minor": 1},
        "extensions": {"vendor_note": "stored only"},
    }

    assert decode_contract("attempt", json.dumps(payload)).to_dict() == payload

    payload["extensions"] = {"command": "do_something"}
    with pytest.raises(ContractDecodeError, match="inert"):
        decode_contract("attempt", json.dumps(payload))


def test_product_tool_catalog_is_discovery_only_and_keeps_the_core_schema_small():
    assert {item.namespace for item in PRODUCT_TOOL_NAMESPACES} == {
        "transcripts",
        "speakers",
        "communications",
        "ui.state",
        "schedules",
        "working_memory",
        "topics",
        "permissions",
        "run",
    }
    assert all(20 <= len(item.description) <= 120 for item in PRODUCT_TOOL_NAMESPACES)
    assert all(item.eager_tool_schemas == () for item in PRODUCT_TOOL_NAMESPACES)
