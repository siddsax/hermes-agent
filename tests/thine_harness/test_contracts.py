from __future__ import annotations

import json
from pathlib import Path

from thine_harness.envelope import RuntimeEnvelopeBudget
from thine_harness.runtime import RuntimeModelConfig


CONTRACTS = Path(__file__).parents[2] / "thine_harness" / "contracts"


def test_frozen_transcript_envelope_keeps_every_reserved_bucket_below_context():
    budget = RuntimeEnvelopeBudget.pinned()

    assert budget.fixed_prefix_reserve_tokens == 4_096
    assert budget.measured_residual_tokens == 219_136
    assert budget.absolute_transcript_tokens == 200_000
    assert budget.unallocated_safety_tokens == 19_136
    assert (
        budget.measured_residual_tokens - budget.absolute_transcript_tokens
        == budget.unallocated_safety_tokens
    )
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


def test_binding_module_ownership_separates_production_modules_and_shared_seams():
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
    assert ownership["module_owners"]["H9_production_runtime_progress"] == [
        "thine_harness/progress_adapter.py"
    ]
    assert ownership["reference_spikes"]["THI3-41_runtime"]["path"] == (
        "thine_harness/runtime.py"
    )
    assert ownership["reference_spikes"]["THI3-41_runtime"][
        "temporarily_co_locates"
    ] == ["H3_coordinator_behavior", "H9_adapter_behavior"]
    assert ownership["shared_core_touchpoints"] == {
        "outbound_observation": [
            "agent/codex_runtime.py",
            "agent/outbound_request_scope.py",
        ],
        "isolation_and_safe_boundary_cancellation": [
            "agent/agent_init.py",
            "run_agent.py",
        ],
        "stop_hook_tool_denial": [
            "agent/tool_execution_scope.py",
            "model_tools.py",
            "tools/registry.py",
        ],
    }
