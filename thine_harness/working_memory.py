"""Bounded Working Memory and same-context Stop Hook seam."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Protocol

MAX_WORKING_MEMORY_TOKENS = 16_000
COMPACTION_TARGET_TOKENS = 14_000


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


def conservative_model_token_bound(text: str) -> int:
    """Return a dependency-free hard upper bound for byte-level BPE tokens.

    GPT-family token vocabularies encode non-empty byte sequences, so a UTF-8
    byte count cannot undercount model tokens. It is deliberately conservative
    when the configured model tokenizer is unavailable in the core runtime.
    """
    return len(str(text).encode("utf-8", errors="surrogatepass"))


@dataclass(frozen=True)
class WorkingMemorySnapshot:
    version: int
    markdown: str
    token_count: int


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
    token_count: int


class CachedStopHookContextPort(Protocol):
    @property
    def identity(self) -> CacheIdentity: ...

    def continue_stop_hook(self, request: StopHookRequest) -> WorkingMemoryProposal: ...


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
        return CacheIdentity(
            session_id=str(getattr(self._agent, "session_id", "") or ""),
            prompt_cache_key=self._cache_identity.prompt_cache_key,
            tool_schema_sha256=tool_fingerprint,
        )

    def continue_stop_hook(self, request: StopHookRequest) -> WorkingMemoryProposal:
        payload = {
            "current_markdown": request.current_markdown,
            "target_tokens": request.target_tokens,
            "oversized_candidate": request.oversized_candidate,
        }
        prompt = (
            "Stop Hook: update Working Memory only if something worth remembering "
            "changed during the completed invocation. Return one JSON object with "
            "exact keys worth_remembering (boolean) and markdown (string or null). "
            "Do not call tools. Input:"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        prior_persist_disabled = bool(
            getattr(self._agent, "_persist_disabled", False)
        )
        self._agent._persist_disabled = True
        try:
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
        if not isinstance(decision, dict) or not isinstance(
            decision.get("worth_remembering"), bool
        ):
            raise StopHookResponseError(
                "Stop Hook response requires boolean worth_remembering"
            )
        markdown = decision.get("markdown")
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
        return WorkingMemoryProposal.unchanged()


class StopHookRunner:
    def __init__(
        self,
        *,
        token_counter: Callable[[str], int] = conservative_model_token_bound,
    ):
        self._token_counter = token_counter

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
        token_count = self._token_counter(markdown)
        if token_count > MAX_WORKING_MEMORY_TOKENS:
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
            token_count = self._token_counter(markdown)
            if token_count > COMPACTION_TARGET_TOKENS:
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
    "COMPACTION_TARGET_TOKENS",
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
    "conservative_model_token_bound",
    "tool_schema_sha256",
]
