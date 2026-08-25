"""Integrated live proof for deferred tools, streaming, cache, and Stop Hook."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import copy
import json
import logging
import os
import re
import tempfile
from typing import Any, Callable, Iterator

from httpx import HTTPError
from openai import APIError

from agent.outbound_request_scope import observe_outbound_requests
from thine_harness.probe import (
    CODEX_BASE_URL,
    CodexCredentialUnavailable,
    _aiagent_factory,
    _load_codex_cli_token,
)
from thine_harness.runtime import (
    HermesAIAgentSession,
    HermesInvocationRuntime,
    InvocationEvent,
    InvocationKind,
    InvocationRequest,
    RuntimeModelConfig,
    RuntimeSelectionError,
)
from thine_harness.working_memory import (
    CacheIdentity,
    HermesCachedStopHookContext,
    StopHookRunner,
    WorkingMemorySnapshot,
    tool_schema_sha256,
)


PROOF_HELPER_NAME = "thine_transcripts_probe_lookup"
PROOF_TOOLSET = "mcp-thine-proof"
PROOF_RESULT_MARKER = "THINE_DEFERRED_HELPER_OK"
PROOF_FINAL_MARKER = "THI3_41_INTEGRATED_OK"
logger = logging.getLogger(__name__)


class OutboundTransportRecorder:
    """Record exact kwargs observed immediately before provider SDK dispatch."""

    def __init__(self):
        self._phase: ContextVar[str] = ContextVar(
            f"thine_probe_phase_{id(self)}",
            default="unscoped",
        )
        self.records: list[dict[str, Any]] = []

    def _record(self, api_mode: str, kwargs: dict[str, Any]) -> None:
        # Hermes may move the already-wire-form ``tools`` array into
        # ``extra_body`` to bypass the OpenAI SDK's recursive TypedDict walk.
        # The SDK merges that object into the JSON body after transformation,
        # so this resolves the exact post-merge wire value.
        extra_body = kwargs.get("extra_body")
        wire_tools = kwargs.get("tools")
        if isinstance(extra_body, dict) and "tools" in extra_body:
            wire_tools = extra_body["tools"]
        tools = copy.deepcopy(list(wire_tools or []))
        self.records.append(
            {
                "phase": self._phase.get(),
                "api_mode": api_mode,
                "model": str(kwargs.get("model") or ""),
                "prompt_cache_key": str(kwargs.get("prompt_cache_key") or ""),
                "tools": tools,
                "tool_schema_sha256": tool_schema_sha256(tools),
            }
        )

    @contextmanager
    def phase(self, phase: str) -> Iterator[None]:
        token = self._phase.set(phase)
        try:
            with observe_outbound_requests(self._record):
                yield
        finally:
            self._phase.reset(token)


class _ProbeMemoryStore:
    def __init__(self) -> None:
        self.commits: list[dict[str, Any]] = []
        self.unchanged: list[dict[str, Any]] = []

    def commit(self, *, expected_version, markdown, token_count, run_id):
        record = {
            "expected_version": expected_version,
            "markdown": markdown,
            "token_count": token_count,
            "run_id": run_id,
        }
        self.commits.append(record)
        return expected_version + 1

    def mark_unchanged(self, *, expected_version, run_id):
        self.unchanged.append(
            {"expected_version": expected_version, "run_id": run_id}
        )


def _helper_schema() -> dict[str, Any]:
    return {
        "name": PROOF_HELPER_NAME,
        "description": (
            "Look up one canonical transcript segment for the THI3-41 proof."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "integer",
                    "description": "Canonical transcript sequence number.",
                }
            },
            "required": ["sequence"],
        },
    }


def _sum_usage_delta(total: dict[str, int], prior: dict[str, int]) -> dict[str, int]:
    return {
        key: max(int(total.get(key) or 0) - int(prior.get(key) or 0), 0)
        for key in {
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        }
    }


def _phase_records(
    records: list[dict[str, Any]], phase: str
) -> list[dict[str, Any]]:
    return [record for record in records if record["phase"] == phase]


def run_integrated_live_probe(
    *,
    token_loader: Callable[[], dict[str, Any] | None] = _load_codex_cli_token,
    agent_factory: Callable[..., Any] = _aiagent_factory,
) -> dict[str, Any]:
    """Run the complete proof on one real ``AIAgent`` instance."""
    from tools.registry import registry

    credentials = token_loader()
    access_token = str((credentials or {}).get("access_token") or "")
    if not access_token:
        raise CodexCredentialUnavailable(
            "Codex CLI credential is missing, expired, or unavailable"
        )

    handler_calls: list[dict[str, Any]] = []

    def helper_handler(args, **_kwargs):
        handler_calls.append(dict(args))
        return json.dumps(
            {
                "marker": PROOF_RESULT_MARKER,
                "sequence": args.get("sequence"),
            },
            separators=(",", ":"),
        )

    registry.register(
        name=PROOF_HELPER_NAME,
        toolset=PROOF_TOOLSET,
        schema=_helper_schema(),
        handler=helper_handler,
    )
    try:
        config = RuntimeModelConfig.openai_gpt_5_6_sol_medium()
        agent = agent_factory(
            base_url=CODEX_BASE_URL,
            api_key=access_token,
            provider=config.provider,
            requested_provider=config.provider,
            api_mode=config.api_mode,
            model=config.model,
            reasoning_config={"enabled": True, "effort": config.reasoning_effort},
            fallback_model=None,
            enabled_toolsets=[PROOF_TOOLSET],
            quiet_mode=True,
            max_iterations=8,
            max_tokens=256,
            session_id="thi3-41-integrated-live-proof",
            ephemeral_system_prompt=(
                "Follow the requested deferred-tool sequence exactly. Return only "
                "the requested final marker after the helper result. For Stop Hook "
                "requests, return only the requested JSON object and never call tools."
            ),
            skip_context_files=True,
            skip_memory=True,
            skip_background_review=True,
        )
        session = HermesAIAgentSession(agent=agent, expected=config)
        runtime = HermesInvocationRuntime(session=session, config=config)
        recorder = OutboundTransportRecorder()
        events: list[InvocationEvent] = []
        with recorder.phase("primary"):
            primary = runtime.invoke(
                InvocationRequest(
                    logical_run_id="thi3-41-integrated-live-proof",
                    kind=InvocationKind.USER_CHAT,
                    prompt=(
                        "You must demonstrate deferred helper use in this exact order: "
                        "(1) call tool_search for 'canonical transcript sequence'; "
                        f"(2) call tool_describe for '{PROOF_HELPER_NAME}'; "
                        f"(3) call tool_call with name '{PROOF_HELPER_NAME}' and "
                        "arguments {\"sequence\":41}. After observing the helper marker, "
                        f"return exactly {PROOF_FINAL_MARKER}."
                    ),
                ),
                emit=events.append,
            )

        primary_wire = _phase_records(recorder.records, "primary")
        if not primary_wire:
            raise RuntimeError("integrated probe captured no primary outbound request")
        first_wire = primary_wire[0]
        cache_identity = CacheIdentity.from_request(
            session_id=str(agent.session_id),
            prompt_cache_key=first_wire["prompt_cache_key"],
            tools=first_wire["tools"],
        )
        stop_context = HermesCachedStopHookContext(
            agent=agent,
            conversation_history=list(primary.context_messages),
            cache_identity=cache_identity,
        )
        memory_store = _ProbeMemoryStore()
        handler_count_before_hook = len(handler_calls)
        with recorder.phase("stop_hook"):
            stop_outcome = StopHookRunner().finalize(
                run_id="thi3-41-integrated-live-proof",
                current=WorkingMemorySnapshot(
                    version=1,
                    markdown=(
                        "The THI3-41 probe has no durable user preference to retain."
                    ),
                    token_count=None,
                ),
                context=stop_context,
                store=memory_store,
                interrupted=False,
            )
        session.validate_runtime()
        hook_wire = _phase_records(recorder.records, "stop_hook")
        if not hook_wire:
            raise RuntimeError("integrated probe captured no Stop Hook outbound request")

        tool_names = [
            str((message.get("tool_name") or message.get("name") or ""))
            for message in primary.context_messages
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        primary_keys = [record["prompt_cache_key"] for record in primary_wire]
        hook_keys = [record["prompt_cache_key"] for record in hook_wire]
        primary_hashes = [record["tool_schema_sha256"] for record in primary_wire]
        hook_hashes = [record["tool_schema_sha256"] for record in hook_wire]
        hook_total_usage = stop_context.last_usage
        evidence = {
            "status": "ok",
            "diagnostics": runtime.diagnostics().as_dict(),
            "same_agent_instance": True,
            "session_id": str(agent.session_id),
            "event_kinds": [event.kind.value for event in events],
            "progress_event_count": sum(
                event.kind.value == "progress" for event in events
            ),
            "final_marker": str(primary.final_output or "").strip(),
            "deferred_tool_result_names": tool_names,
            "helper_calls": list(handler_calls),
            "helper_calls_during_stop_hook": len(handler_calls)
            - handler_count_before_hook,
            "primary_usage": primary.usage,
            "stop_hook_cumulative_usage": hook_total_usage,
            "stop_hook_usage_delta": _sum_usage_delta(
                hook_total_usage, primary.usage
            ),
            "primary_prompt_cache_keys": primary_keys,
            "stop_hook_prompt_cache_keys": hook_keys,
            "primary_tool_schema_hashes": primary_hashes,
            "stop_hook_tool_schema_hashes": hook_hashes,
            "same_prompt_cache_key": len(set(primary_keys + hook_keys)) == 1,
            "same_wire_tool_array": all(
                record["tools"] == first_wire["tools"]
                for record in [*primary_wire, *hook_wire]
            ),
            "primary_wire_requests": primary_wire,
            "stop_hook_wire_requests": hook_wire,
            "stop_hook_outcome": stop_outcome.kind.value,
            "working_memory_commits": memory_store.commits,
            "working_memory_unchanged_markers": memory_store.unchanged,
            "tokenizer_status": StopHookRunner().tokenizer_status,
        }
        # Hermes intentionally unwraps the bridge's ``tool_call`` before
        # appending the canonical tool result, so history records the deferred
        # helper's real name. The wire arrays prove that helper was not eagerly
        # available and therefore could only have executed through the bridge.
        required_tool_results = {
            "tool_search",
            "tool_describe",
            PROOF_HELPER_NAME,
        }
        if not required_tool_results.issubset(set(tool_names)):
            raise RuntimeError(
                "model did not complete deferred search/describe/call sequence: "
                + repr(tool_names)
            )
        if handler_calls != [{"sequence": 41}]:
            raise RuntimeError(
                "deferred helper call was not exactly-once with sequence 41"
            )
        if evidence["final_marker"] != PROOF_FINAL_MARKER:
            raise RuntimeError("integrated probe final marker mismatch")
        if not evidence["same_prompt_cache_key"] or not evidence["same_wire_tool_array"]:
            raise RuntimeError("Stop Hook cache envelope drifted from primary")
        if evidence["helper_calls_during_stop_hook"] != 0:
            raise RuntimeError("Stop Hook executed a helper handler")
        return evidence
    finally:
        registry.deregister(PROOF_HELPER_NAME)


def _sanitized_error(exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    message = str(exc)[:800]
    message = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", message, flags=re.I)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", message)
    return {
        "status": "blocked",
        "error_type": type(exc).__name__,
        "http_status": status,
        "provider_code": getattr(exc, "code", None),
        "message": message,
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="thi3-41-integrated-") as proof_home:
        os.environ["HERMES_HOME"] = proof_home
        try:
            evidence = run_integrated_live_probe()
        except (
            APIError,
            CodexCredentialUnavailable,
            HTTPError,
            RuntimeSelectionError,
        ) as exc:
            evidence = _sanitized_error(exc)
        except Exception:
            logger.exception("Unexpected THI3-41 integrated live probe failure")
            raise
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 0 if evidence["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["OutboundTransportRecorder", "run_integrated_live_probe"]
