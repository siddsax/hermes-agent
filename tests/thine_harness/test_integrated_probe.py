from __future__ import annotations

import hashlib

import pytest

from thine_harness.integrated_probe import (
    OutboundTransportRecorder,
    _summarize_prefix_evidence,
    run_integrated_live_probe,
)
from thine_harness.working_memory import tool_schema_sha256
from agent.outbound_request_scope import notify_outbound_request


def test_outbound_recorder_captures_exact_wire_cache_key_and_tools_by_phase():
    class _Agent:
        def __init__(self) -> None:
            self.forwarded = []

        def _run_codex_stream(self, api_kwargs, client=None, on_first_delta=None):
            notify_outbound_request("codex_responses", api_kwargs)
            self.forwarded.append((api_kwargs, client, on_first_delta))
            return "provider-response"

    agent = _Agent()
    recorder = OutboundTransportRecorder()
    tools = [
        {
            "type": "function",
            "name": "tool_search",
            "description": "Search deferred helpers.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    with recorder.phase("primary"):
        result = agent._run_codex_stream(
            {
                "model": "gpt-5.6-sol",
                "prompt_cache_key": "pck-proof",
                "instructions": "Stable Harness policy",
                "tools": tools,
            },
            client="client",
        )

    assert result == "provider-response"
    assert agent.forwarded[0][0]["tools"] is tools
    assert recorder.records == [
        {
            "phase": "primary",
            "api_mode": "codex_responses",
            "model": "gpt-5.6-sol",
            "prompt_cache_key": "pck-proof",
            "tools": tools,
            "tool_schema_sha256": tool_schema_sha256(tools),
            "system_prompt_sha256": hashlib.sha256(
                b"Stable Harness policy"
            ).hexdigest(),
            "system_prompt_chars": 21,
            "fixed_prefix_estimated_tokens": 23,
        }
    ]


def test_integrated_probe_pins_deferred_catalog_listing_off_before_agent_construction(
    monkeypatch,
    tmp_path,
):
    class _ConstructionObserved(Exception):
        pass

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def factory(**_kwargs):
        from tools.tool_search import load_config

        config = load_config()
        assert config.enabled == "on"
        assert config.listing == "off"
        raise _ConstructionObserved

    with pytest.raises(_ConstructionObserved):
        run_integrated_live_probe(
            token_loader=lambda: {"access_token": "not-a-real-token"},
            agent_factory=factory,
        )


def test_prefix_proof_preserves_observations_without_freezing_an_exact_estimate():
    records = [
        {
            "fixed_prefix_estimated_tokens": estimate,
            "system_prompt_chars": chars,
            "system_prompt_sha256": "stable-system",
            "tool_schema_sha256": "stable-tools",
        }
        for estimate, chars in [(3_000, 9_000), (3_100, 9_100)]
    ]

    evidence = _summarize_prefix_evidence(records, reserve_tokens=4_096)

    assert evidence == {
        "fixed_prefix_reserve_tokens": 4_096,
        "fixed_prefix_estimated_tokens": [3_000, 3_100],
        "fixed_prefix_within_reserve": True,
        "system_prompt_chars": [9_000, 9_100],
        "system_prompt_sha256": ["stable-system", "stable-system"],
        "wire_tool_schema_sha256": ["stable-tools", "stable-tools"],
        "same_system_prompt_sha256": True,
        "same_tool_schema_sha256": True,
    }


def test_prefix_proof_rejects_over_reserve_or_within_run_hash_drift():
    evidence = _summarize_prefix_evidence(
        [
            {
                "fixed_prefix_estimated_tokens": 4_097,
                "system_prompt_chars": 9_160,
                "system_prompt_sha256": "system-primary",
                "tool_schema_sha256": "tools-primary",
            },
            {
                "fixed_prefix_estimated_tokens": 3_100,
                "system_prompt_chars": 9_100,
                "system_prompt_sha256": "system-stop",
                "tool_schema_sha256": "tools-stop",
            },
        ],
        reserve_tokens=4_096,
    )

    assert evidence["fixed_prefix_within_reserve"] is False
    assert evidence["same_system_prompt_sha256"] is False
    assert evidence["same_tool_schema_sha256"] is False
