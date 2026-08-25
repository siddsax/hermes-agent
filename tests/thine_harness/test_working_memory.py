from __future__ import annotations

import pytest

from thine_harness.working_memory import (
    CacheIdentity,
    HermesCachedStopHookContext,
    StopHookContextChanged,
    StopHookOutcomeKind,
    StopHookRequest,
    StopHookRunner,
    WorkingMemoryLimitError,
    WorkingMemoryProposal,
    WorkingMemorySnapshot,
    tool_schema_sha256,
)
from agent.transports.codex import ResponsesApiTransport


class _MemoryStore:
    def __init__(self) -> None:
        self.commits = []
        self.unchanged = []

    def commit(self, *, expected_version, markdown, token_count, run_id):
        self.commits.append((expected_version, markdown, token_count, run_id))
        return expected_version + 1

    def mark_unchanged(self, *, expected_version, run_id):
        self.unchanged.append((expected_version, run_id))


class _UnchangedContext:
    identity = CacheIdentity(
        session_id="session-1",
        prompt_cache_key="cache-1",
        tool_schema_sha256="tools-1",
    )

    def __init__(self) -> None:
        self.requests: list[StopHookRequest] = []

    def continue_stop_hook(self, request):
        self.requests.append(request)
        return WorkingMemoryProposal.unchanged()


def test_stop_hook_marks_unchanged_without_writing_a_memory_version():
    store = _MemoryStore()
    context = _UnchangedContext()
    runner = StopHookRunner(token_counter=len)
    current = WorkingMemorySnapshot(version=7, markdown="Keep this", token_count=9)

    outcome = runner.finalize(
        run_id="run-1",
        current=current,
        context=context,
        store=store,
        interrupted=False,
    )

    assert outcome.kind is StopHookOutcomeKind.UNCHANGED
    assert outcome.cache_identity == context.identity
    assert len(context.requests) == 1
    assert context.requests[0].target_tokens == 16_000
    assert store.commits == []
    assert store.unchanged == [(7, "run-1")]


class _CachedAIAgent:
    session_id = "session-aiagent"

    def __init__(self) -> None:
        self.calls = []
        self._persist_disabled = False
        self.skip_memory = True
        self.skip_background_review = True
        self.tools = [{"type": "function", "name": "tool_search"}]

    def _build_system_prompt(self):
        raise AssertionError("Stop Hook must not rebuild the frozen system prompt")

    def _build_api_kwargs(self, messages):
        raise AssertionError("Stop Hook must consume the captured primary request identity")

    def run_conversation(self, prompt, **kwargs):
        assert self._persist_disabled is True
        assert self.skip_memory is True
        assert self.skip_background_review is True
        self.calls.append((prompt, kwargs))
        return {
            "final_response": '{"worth_remembering":false,"markdown":null}',
            "input_tokens": 1200,
            "output_tokens": 12,
            "cache_read_tokens": 1024,
            "cache_write_tokens": 0,
            "messages": [
                *kwargs["conversation_history"],
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": '{"worth_remembering":false,"markdown":null}',
                },
            ],
        }


def test_aiagent_stop_hook_continues_on_same_agent_history_and_cache_identity():
    agent = _CachedAIAgent()
    prior_history = [
        {"role": "user", "content": "Do the work"},
        {"role": "assistant", "content": "Done"},
    ]
    context = HermesCachedStopHookContext(
        agent=agent,
        conversation_history=prior_history,
        cache_identity=CacheIdentity.from_request(
            session_id="session-aiagent",
            prompt_cache_key="pck-aiagent",
            tools=agent.tools,
        ),
    )
    store = _MemoryStore()

    outcome = StopHookRunner(token_counter=len).finalize(
        run_id="run-aiagent",
        current=WorkingMemorySnapshot(4, "Remember this", 13),
        context=context,
        store=store,
        interrupted=False,
    )

    assert outcome.kind is StopHookOutcomeKind.UNCHANGED
    assert outcome.cache_identity.session_id == "session-aiagent"
    assert outcome.cache_identity.prompt_cache_key == "pck-aiagent"
    assert len(agent.calls) == 1
    assert agent._persist_disabled is False
    assert agent.calls[0][1]["conversation_history"] == prior_history
    assert "Remember this" in agent.calls[0][0]
    assert context.last_usage == {
        "input_tokens": 1200,
        "output_tokens": 12,
        "cache_read_tokens": 1024,
        "cache_write_tokens": 0,
    }


class _CompactingContext:
    identity = CacheIdentity("session-1", "cache-1", "tools-1")

    def __init__(self) -> None:
        self.requests: list[StopHookRequest] = []

    def continue_stop_hook(self, request):
        self.requests.append(request)
        if request.oversized_candidate is None:
            return WorkingMemoryProposal.changed("x" * 16_001)
        return WorkingMemoryProposal.changed("y" * 14_000)


def test_oversized_proposal_gets_one_agent_directed_same_context_compaction():
    store = _MemoryStore()
    context = _CompactingContext()
    runner = StopHookRunner(token_counter=len)

    outcome = runner.finalize(
        run_id="run-2",
        current=WorkingMemorySnapshot(3, "prior", 5),
        context=context,
        store=store,
        interrupted=False,
    )

    assert outcome.kind is StopHookOutcomeKind.COMMITTED
    assert outcome.memory_version == 4
    assert outcome.token_count == 14_000
    assert [request.target_tokens for request in context.requests] == [16_000, 14_000]
    assert context.requests[1].oversized_candidate == "x" * 16_001
    assert store.commits == [(3, "y" * 14_000, 14_000, "run-2")]


class _UnicodeCompactingContext:
    identity = CacheIdentity("session-unicode", "cache-unicode", "tools-unicode")

    def __init__(self) -> None:
        self.requests: list[StopHookRequest] = []

    def continue_stop_hook(self, request):
        self.requests.append(request)
        if request.oversized_candidate is None:
            # Hermes' rough estimator counts this as only 1,250 tokens, while
            # its UTF-8 representation is 20,000 bytes. The production guard
            # must conservatively reject it without an optional tokenizer.
            return WorkingMemoryProposal.changed("😀" * 5_000)
        return WorkingMemoryProposal.changed("bounded")


def test_default_working_memory_guard_is_hard_for_token_dense_unicode():
    store = _MemoryStore()
    context = _UnicodeCompactingContext()

    outcome = StopHookRunner().finalize(
        run_id="run-unicode",
        current=WorkingMemorySnapshot(1, "prior", 5),
        context=context,
        store=store,
        interrupted=False,
    )

    assert len(context.requests) == 2
    assert context.requests[1].target_tokens == 14_000
    assert outcome.token_count == len("bounded".encode("utf-8"))


def test_corrected_working_memory_must_meet_the_14k_correction_target():
    class _StillTooLarge(_CompactingContext):
        def continue_stop_hook(self, request):
            self.requests.append(request)
            if request.oversized_candidate is None:
                return WorkingMemoryProposal.changed("x" * 16_001)
            return WorkingMemoryProposal.changed("y" * 14_001)

    with pytest.raises(WorkingMemoryLimitError, match="target is 14000"):
        StopHookRunner(token_counter=len).finalize(
            run_id="run-correction-target",
            current=WorkingMemorySnapshot(3, "prior", 5),
            context=_StillTooLarge(),
            store=_MemoryStore(),
            interrupted=False,
        )


class _MustNotRunContext:
    identity = CacheIdentity("session-bg", "cache-bg", "tools-bg")

    def continue_stop_hook(self, request):
        raise AssertionError("interrupted background work must skip the Stop Hook")


def test_interrupted_background_invocation_skips_stop_hook_and_memory_writes():
    store = _MemoryStore()
    outcome = StopHookRunner(token_counter=len).finalize(
        run_id="run-background",
        current=WorkingMemorySnapshot(11, "prior", 5),
        context=_MustNotRunContext(),
        store=store,
        interrupted=True,
    )

    assert outcome.kind is StopHookOutcomeKind.SKIPPED_INTERRUPTED
    assert outcome.memory_version == 11
    assert store.commits == []
    assert store.unchanged == []


def test_stop_hook_continuation_keeps_responses_cache_key_and_tool_array():
    transport = ResponsesApiTransport()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "tool_search",
                "description": "Search deferred helpers.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    common = {
        "model": "gpt-5.6-sol",
        "tools": tools,
        "instructions": "Stable Harness policy",
        "session_id": "thine-run-1",
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "is_codex_backend": True,
        "reasoning_config": {"enabled": True, "effort": "medium"},
    }
    primary = transport.build_kwargs(
        messages=[{"role": "user", "content": "Process this tick"}],
        **common,
    )
    stop_hook = transport.build_kwargs(
        messages=[
            {"role": "user", "content": "Process this tick"},
            {"role": "assistant", "content": "Done"},
            {"role": "user", "content": "Finalize Working Memory"},
        ],
        **common,
    )

    assert primary["model"] == stop_hook["model"] == "gpt-5.6-sol"
    assert primary["reasoning"] == stop_hook["reasoning"] == {
        "effort": "medium",
        "summary": "auto",
    }
    assert primary["prompt_cache_key"] == stop_hook["prompt_cache_key"]
    assert primary["tools"] == stop_hook["tools"]
    assert tool_schema_sha256(primary["tools"]) == tool_schema_sha256(
        stop_hook["tools"]
    )


class _DriftingContext(_UnchangedContext):
    @property
    def identity(self):
        suffix = len(self.requests)
        return CacheIdentity("session-1", "cache-1", f"tools-{suffix}")


def test_stop_hook_fails_closed_if_cached_context_changes():
    store = _MemoryStore()

    with pytest.raises(StopHookContextChanged, match="cache identity changed"):
        StopHookRunner(token_counter=len).finalize(
            run_id="run-drift",
            current=WorkingMemorySnapshot(2, "prior", 5),
            context=_DriftingContext(),
            store=store,
            interrupted=False,
        )

    assert store.commits == []
    assert store.unchanged == []
