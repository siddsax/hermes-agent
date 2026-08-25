from __future__ import annotations

from thine_harness.integrated_probe import OutboundTransportRecorder
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
        }
    ]
