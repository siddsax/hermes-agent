from __future__ import annotations

import json
from pathlib import Path

from thine_harness.envelope import RuntimeEnvelopeBudget
from thine_harness.runtime import RuntimeModelConfig


CONTRACTS = Path(__file__).parents[2] / "thine_harness" / "contracts"


def test_frozen_transcript_envelope_keeps_every_reserved_bucket_below_context():
    budget = RuntimeEnvelopeBudget.pinned()

    assert budget.measured_residual_tokens == 220_619
    assert budget.absolute_transcript_tokens == 200_000
    assert budget.unallocated_safety_tokens == 20_619
    assert budget.routine_batch_target_tokens == 8_000
    assert budget.total_reserved_tokens == 272_000


def test_model_and_envelope_fixtures_match_public_runtime_contract():
    model_fixture = json.loads(
        (CONTRACTS / "runtime-model-v1.json").read_text(encoding="utf-8")
    )
    envelope_fixture = json.loads(
        (CONTRACTS / "runtime-envelope-v1.json").read_text(encoding="utf-8")
    )

    assert model_fixture["runtime"] == RuntimeModelConfig.openai_gpt_5_6_sol_medium().__dict__
    assert envelope_fixture["budget"] == RuntimeEnvelopeBudget.pinned().as_dict()
    assert envelope_fixture["measurement"]["working_memory_hard_guard"] == (
        "unresolved_no_exact_openai_codex_gpt_5_6_sol_tokenizer_changed_writes_fail_closed"
    )
    assert envelope_fixture["measurement"]["working_memory_token_count_storage"] == (
        "exact_configured_model_tokens_only_nullable_when_unmeasured"
    )
    assert envelope_fixture["measurement"]["working_memory_auxiliary_utf8_byte_ceiling"] == 16_000
    assert envelope_fixture["measurement"][
        "working_memory_correction_target_tokens"
    ] == 14_000


def test_binding_module_ownership_has_no_overlapping_paths():
    ownership = json.loads(
        (CONTRACTS / "hermes-ownership-v1.json").read_text(encoding="utf-8")
    )
    owned_paths = [
        path
        for paths in ownership["module_owners"].values()
        for path in paths
    ]

    assert len(owned_paths) == len(set(owned_paths))
    assert ownership["module_owners"]["H3_run_coordinator"] == [
        "thine_harness/run_coordinator.py",
        "thine_harness/run_state.py",
    ]
    assert "thine_harness/runtime.py" in ownership["module_owners"][
        "H9_production_runtime_progress"
    ]
