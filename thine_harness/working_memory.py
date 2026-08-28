"""Bounded Working Memory and same-context Stop Hook seam."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Protocol

MAX_WORKING_MEMORY_TOKENS = 16_000
COMPACTION_TARGET_TOKENS = 14_000
MAX_WORKING_MEMORY_UTF8_BYTES = 16_000
COMPACTION_TARGET_UTF8_BYTES = 14_000
CONFIGURED_MODEL_TOKENIZER_LIMITATION = (
    "unresolved: no exact tokenizer is available for openai-codex/gpt-5.6-sol"
)


def tool_schema_sha256(tools: list[dict]) -> str:
    """Content identity for the exact stable tool array sent to the model."""
    canonical = json.dumps(
        tools,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class CacheIdentity:
    session_id: str
    prompt_cache_key: str
    tool_schema_sha256: str

    @classmethod
    def from_request(
        cls,
        *,
        session_id: str,
        prompt_cache_key: str,
        tools: list[dict],
    ) -> "CacheIdentity":
        """Capture identity from the already-frozen primary request envelope."""
        return cls(
            session_id=session_id,
            prompt_cache_key=prompt_cache_key,
            tool_schema_sha256=tool_schema_sha256(tools),
        )


def utf8_byte_size(text: str) -> int:
    """Return an auxiliary byte ceiling measurement, never a token count."""
    return len(str(text).encode("utf-8", errors="surrogatepass"))


@dataclass(frozen=True)
class WorkingMemorySnapshot:
    version: int
    markdown: str
    token_count: int | None


@dataclass(frozen=True)
class WorkingMemoryProposal:
    markdown: str | None
    worth_remembering: bool

    @classmethod
    def unchanged(cls) -> "WorkingMemoryProposal":
        return cls(markdown=None, worth_remembering=False)

    @classmethod
    def changed(cls, markdown: str) -> "WorkingMemoryProposal":
        return cls(markdown=markdown, worth_remembering=True)


@dataclass(frozen=True)
class StopHookRequest:
    current_markdown: str
    target_tokens: int
    oversized_candidate: str | None = None


class StopHookOutcomeKind(str, Enum):
    COMMITTED = "committed"
    UNCHANGED = "unchanged"
    SKIPPED_INTERRUPTED = "skipped_interrupted"


@dataclass(frozen=True)
class StopHookOutcome:
    kind: StopHookOutcomeKind
    cache_identity: CacheIdentity
    memory_version: int
    token_count: int | None


class CachedStopHookContextPort(Protocol):
    @property
    def identity(self) -> CacheIdentity: ...

    def continue_stop_hook(self, request: StopHookRequest) -> WorkingMemoryProposal: ...

    def count_candidate_tokens(self, candidate: str) -> int: ...


class WorkingMemoryStorePort(Protocol):
    def commit(
        self,
        *,
        expected_version: int,
        markdown: str,
        token_count: int,
        run_id: str,
    ) -> int: ...

    def mark_unchanged(self, *, expected_version: int, run_id: str) -> None: ...


class StopHookContextChanged(RuntimeError):
    """The Stop Hook attempted to leave the cached invocation context."""


class WorkingMemoryLimitError(RuntimeError):
    """Agent-directed compaction failed to produce a bounded document."""


class WorkingMemoryTokenizerUnavailable(RuntimeError):
    """A changed document cannot be measured in configured-model tokens."""


class StopHookResponseError(RuntimeError):
    """The same-context Stop Hook returned an invalid structured decision."""


class HermesCachedStopHookContext:
    """Continue a completed turn on the same ``AIAgent`` and message history.

    The Stop Hook is appended as user-side content.  Model, system prompt,
    session ID, and tool array stay owned by the existing agent, so Hermes'
    content-addressed Responses cache prefix remains unchanged.
    """

    def __init__(
        self,
        *,
        agent: Any,
        conversation_history: list[dict[str, Any]],
        cache_identity: CacheIdentity,
    ):
        self._agent = agent
        self._conversation_history = list(conversation_history)
        self._cache_identity = cache_identity
        self._agent_tool_fingerprint = tool_schema_sha256(
            list(getattr(agent, "tools", None) or [])
        )
        self._agent_runtime_fingerprint = self._runtime_fingerprint(agent)
        if str(getattr(agent, "session_id", "") or "") != cache_identity.session_id:
            raise StopHookContextChanged(
                "captured cache identity does not match the existing AIAgent session"
            )
        if not bool(getattr(agent, "skip_memory", False)) or not bool(
            getattr(agent, "skip_background_review", False)
        ):
            raise StopHookContextChanged(
                "same-context Stop Hook requires Hermes memory and background review disabled"
            )
        self._last_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
        }

    @property
    def last_usage(self) -> dict[str, int]:
        return dict(self._last_usage)

    @property
    def identity(self) -> CacheIdentity:
        current_tool_fingerprint = tool_schema_sha256(
            list(getattr(self._agent, "tools", None) or [])
        )
        tool_fingerprint = self._cache_identity.tool_schema_sha256
        if current_tool_fingerprint != self._agent_tool_fingerprint:
            tool_fingerprint = "agent-tool-array-changed"
        prompt_cache_key = self._cache_identity.prompt_cache_key
        if self._runtime_fingerprint(self._agent) != self._agent_runtime_fingerprint:
            prompt_cache_key = "agent-runtime-selection-changed"
        return CacheIdentity(
            session_id=str(getattr(self._agent, "session_id", "") or ""),
            prompt_cache_key=prompt_cache_key,
            tool_schema_sha256=tool_fingerprint,
        )

    @staticmethod
    def _runtime_fingerprint(agent: Any) -> tuple[str, str, str, str]:
        from thine_harness.runtime import runtime_selection_fingerprint

        return runtime_selection_fingerprint(agent)

    def continue_stop_hook(self, request: StopHookRequest) -> WorkingMemoryProposal:
        payload = {
            "current_markdown": request.current_markdown,
            "target_tokens": request.target_tokens,
            "oversized_candidate": request.oversized_candidate,
        }
        prompt = (
            "Stop Hook: update Working Memory only if something worth remembering "
            "changed during the completed invocation. Keep only recent agent actions, "
            "commitments, unresolved threads, expected follow-ups, and at most the "
            "last ten Tick summaries so the next invocation avoids repetition. Do "
            "not copy transcript/source facts into memory. When a communication or "
            "prompt changed a durable Topic, retain its exact topic_key and latest "
            "action/receipt until it is no longer needed for immediate repetition "
            "avoidance. Durable explicit preferences and user corrections are "
            "authoritative outside Working Memory; never contradict or duplicate them. "
            "Compact resolved routine history before the target. Return one JSON object with "
            "exact keys worth_remembering (boolean) and markdown (string or null). "
            "Do not call tools. Input:"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        prior_persist_disabled = bool(
            getattr(self._agent, "_persist_disabled", False)
        )
        self._agent._persist_disabled = True
        try:
            from agent.tool_execution_scope import deny_tool_execution

            with deny_tool_execution("Stop Hook is memory-only"):
                raw = self._agent.run_conversation(
                    prompt,
                    conversation_history=list(self._conversation_history),
                )
        finally:
            self._agent._persist_disabled = prior_persist_disabled
        self._last_usage = {
            key: int(raw.get(key) or 0)
            for key in self._last_usage
        }
        messages = raw.get("messages")
        if isinstance(messages, list):
            self._conversation_history = list(messages)
        try:
            decision = json.loads(str(raw.get("final_response") or ""))
        except (TypeError, ValueError) as exc:
            raise StopHookResponseError("Stop Hook did not return valid JSON") from exc
        if not isinstance(decision, dict) or set(decision) != {
            "worth_remembering",
            "markdown",
        }:
            raise StopHookResponseError(
                "Stop Hook response requires exactly worth_remembering and markdown"
            )
        if not isinstance(decision["worth_remembering"], bool):
            raise StopHookResponseError(
                "Stop Hook response requires boolean worth_remembering"
            )
        markdown = decision["markdown"]
        if markdown is not None and not isinstance(markdown, str):
            raise StopHookResponseError(
                "Stop Hook response markdown must be a string or null"
            )
        if decision["worth_remembering"]:
            if markdown is None:
                raise StopHookResponseError(
                    "Stop Hook changed decision requires markdown"
                )
            return WorkingMemoryProposal.changed(markdown)
        if markdown is not None:
            raise StopHookResponseError(
                "Stop Hook unchanged decision requires null markdown"
            )
        return WorkingMemoryProposal.unchanged()

    def count_candidate_tokens(self, candidate: str) -> int:
        """Ask the configured provider to tokenize one exact visible candidate.

        Codex Responses reports cumulative ``output_tokens`` and the exact hidden
        ``reasoning_tokens`` subset.  A candidate is measurable only when it is
        the sole visible output and round-trips byte-for-byte; otherwise this
        fails closed instead of treating a local estimate as model tokens.
        """
        before = dict(self._last_usage)
        history_before = len(self._conversation_history)
        prompt = (
            "Configured-model token measurement. Return the text between the "
            "candidate delimiters byte-for-byte as your entire visible response. "
            "Do not add delimiters, Markdown fences, commentary, or tool calls.\n"
            "<candidate>"
            + candidate
            + "</candidate>"
        )
        prior_persist_disabled = bool(
            getattr(self._agent, "_persist_disabled", False)
        )
        self._agent._persist_disabled = True
        try:
            from agent.tool_execution_scope import deny_tool_execution

            with deny_tool_execution("Working Memory token measurement is read-only"):
                raw = self._agent.run_conversation(
                    prompt,
                    conversation_history=list(self._conversation_history),
                )
        finally:
            self._agent._persist_disabled = prior_persist_disabled
        measured = {
            key: int(raw.get(key) or 0)
            for key in self._last_usage
        }
        messages = raw.get("messages")
        if isinstance(messages, list):
            self._conversation_history = list(messages)
        self._last_usage = measured
        new_messages = (
            list(messages[history_before:]) if isinstance(messages, list) else []
        )
        if str(raw.get("final_response") or "") != candidate:
            raise WorkingMemoryTokenizerUnavailable(
                "configured model did not reproduce the Working Memory candidate "
                "exactly for authoritative token measurement"
            )
        if raw.get("response_transformed") is True:
            raise WorkingMemoryTokenizerUnavailable(
                "configured model response was transformed after token accounting"
            )
        if raw.get("api_calls") not in {None, 1} or any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in new_messages
        ):
            raise WorkingMemoryTokenizerUnavailable(
                "configured-model token measurement was not one tool-free response"
            )
        output_delta = measured["output_tokens"] - before["output_tokens"]
        reasoning_delta = measured["reasoning_tokens"] - before["reasoning_tokens"]
        visible_tokens = output_delta - reasoning_delta
        if output_delta < 0 or reasoning_delta < 0 or visible_tokens < 0:
            raise WorkingMemoryTokenizerUnavailable(
                "configured model returned inconsistent cumulative token usage"
            )
        return visible_tokens


class StopHookRunner:
    def __init__(
        self,
        *,
        token_counter: Callable[[str], int] | None = None,
    ):
        self._token_counter = token_counter

    @property
    def tokenizer_status(self) -> str:
        if self._token_counter is None:
            return CONFIGURED_MODEL_TOKENIZER_LIMITATION
        return "configured-model tokenizer supplied"

    def finalize(
        self,
        *,
        run_id: str,
        current: WorkingMemorySnapshot,
        context: CachedStopHookContextPort,
        store: WorkingMemoryStorePort,
        interrupted: bool,
    ) -> StopHookOutcome:
        identity = context.identity
        if interrupted:
            return StopHookOutcome(
                StopHookOutcomeKind.SKIPPED_INTERRUPTED,
                identity,
                current.version,
                current.token_count,
            )

        proposal = context.continue_stop_hook(
            StopHookRequest(
                current_markdown=current.markdown,
                target_tokens=MAX_WORKING_MEMORY_TOKENS,
            )
        )
        self._require_same_context(context, identity)

        if (
            not proposal.worth_remembering
            or proposal.markdown is None
            or proposal.markdown == current.markdown
        ):
            store.mark_unchanged(expected_version=current.version, run_id=run_id)
            return StopHookOutcome(
                StopHookOutcomeKind.UNCHANGED,
                identity,
                current.version,
                current.token_count,
            )

        markdown = proposal.markdown
        byte_count = utf8_byte_size(markdown)
        token_count = (
            None
            if byte_count > MAX_WORKING_MEMORY_UTF8_BYTES
            else self._measure_tokens(markdown, context, identity)
        )
        if byte_count > MAX_WORKING_MEMORY_UTF8_BYTES or (
            token_count is not None and token_count > MAX_WORKING_MEMORY_TOKENS
        ):
            oversized_candidate = markdown
            proposal = context.continue_stop_hook(
                StopHookRequest(
                    current_markdown=current.markdown,
                    target_tokens=COMPACTION_TARGET_TOKENS,
                    oversized_candidate=oversized_candidate,
                )
            )
            self._require_same_context(context, identity)
            if not proposal.worth_remembering or proposal.markdown is None:
                raise WorkingMemoryLimitError(
                    "working memory correction did not return a compacted proposal"
                )
            markdown = proposal.markdown
            byte_count = utf8_byte_size(markdown)
            token_count = self._measure_tokens(markdown, context, identity)
            if byte_count > COMPACTION_TARGET_UTF8_BYTES:
                raise WorkingMemoryLimitError(
                    f"compacted working memory is {byte_count} UTF-8 bytes; "
                    f"auxiliary byte target is {COMPACTION_TARGET_UTF8_BYTES}"
                )
            if token_count is not None and token_count > COMPACTION_TARGET_TOKENS:
                raise WorkingMemoryLimitError(
                    f"compacted working memory is {token_count} tokens; "
                    f"correction target is {COMPACTION_TARGET_TOKENS}"
                )
            if markdown == current.markdown:
                store.mark_unchanged(expected_version=current.version, run_id=run_id)
                return StopHookOutcome(
                    StopHookOutcomeKind.UNCHANGED,
                    identity,
                    current.version,
                    current.token_count,
                )
        if token_count is None:
            raise WorkingMemoryTokenizerUnavailable(
                "configured-model tokenizer is unavailable; refusing to store "
                "UTF-8 bytes as token_count"
            )
        version = store.commit(
            expected_version=current.version,
            markdown=markdown,
            token_count=token_count,
            run_id=run_id,
        )
        return StopHookOutcome(
            StopHookOutcomeKind.COMMITTED,
            identity,
            version,
            token_count,
        )

    def _measure_tokens(
        self,
        markdown: str,
        context: CachedStopHookContextPort,
        expected_identity: CacheIdentity,
    ) -> int:
        if self._token_counter is not None:
            return self._token_counter(markdown)
        counter = getattr(context, "count_candidate_tokens", None)
        if not callable(counter):
            raise WorkingMemoryTokenizerUnavailable(
                "configured-model tokenizer is unavailable; refusing to store "
                "UTF-8 bytes as token_count"
            )
        count = counter(markdown)
        self._require_same_context(context, expected_identity)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise WorkingMemoryTokenizerUnavailable(
                "configured-model token measurement returned an invalid count"
            )
        return count

    @staticmethod
    def _require_same_context(
        context: CachedStopHookContextPort,
        expected: CacheIdentity,
    ) -> None:
        if context.identity != expected:
            raise StopHookContextChanged(
                "Stop Hook cache identity changed during same-context continuation"
            )


__all__ = [
    "COMPACTION_TARGET_UTF8_BYTES",
    "COMPACTION_TARGET_TOKENS",
    "CONFIGURED_MODEL_TOKENIZER_LIMITATION",
    "MAX_WORKING_MEMORY_UTF8_BYTES",
    "MAX_WORKING_MEMORY_TOKENS",
    "CacheIdentity",
    "CachedStopHookContextPort",
    "HermesCachedStopHookContext",
    "StopHookContextChanged",
    "StopHookOutcome",
    "StopHookOutcomeKind",
    "StopHookRequest",
    "StopHookResponseError",
    "StopHookRunner",
    "WorkingMemoryLimitError",
    "WorkingMemoryProposal",
    "WorkingMemorySnapshot",
    "WorkingMemoryStorePort",
    "WorkingMemoryTokenizerUnavailable",
    "tool_schema_sha256",
    "utf8_byte_size",
]
