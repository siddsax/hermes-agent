"""Public invocation lifecycle seam for the local Thine Harness.

This module deliberately wraps Hermes' existing agent runtime.  It does not
create a second agent loop or a plugin surface; callers provide one session
port and receive a small, transport-neutral lifecycle contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import threading
from typing import Any, Callable, Protocol


SAFE_BOUNDARY_RESUME_PROMPT = (
    "Resume this Logical Run from the durable safe-boundary history. The original "
    "input is already present; do not repeat completed tool effects. Re-plan and "
    "perform only unfinished work."
)


class InvocationKind(str, Enum):
    USER_CHAT = "p0_user_chat"
    BACKGROUND = "p1_background"


class InvocationEventKind(str, Enum):
    ACCEPTED = "accepted"
    STARTED = "started"
    PROGRESS = "progress"
    FINAL = "final"
    INTERRUPTED = "interrupted"


class RuntimeSelectionError(RuntimeError):
    """The live Hermes session drifted from the required fail-closed model."""


@dataclass(frozen=True)
class InvocationEvent:
    kind: InvocationEventKind
    phase: str
    text: str
    ephemeral: bool = True

    @classmethod
    def progress(cls, phase: str, text: str) -> "InvocationEvent":
        return cls(InvocationEventKind.PROGRESS, phase, text)


@dataclass(frozen=True)
class RuntimeModelConfig:
    provider: str
    model: str
    api_mode: str
    reasoning_effort: str
    context_window_tokens: int

    @classmethod
    def openai_gpt_5_6_sol_medium(cls) -> "RuntimeModelConfig":
        return cls(
            provider="openai-codex",
            model="gpt-5.6-sol",
            api_mode="codex_responses",
            reasoning_effort="medium",
            context_window_tokens=272_000,
        )


@dataclass(frozen=True)
class RuntimeDiagnostics:
    provider: str
    model: str
    api_mode: str
    reasoning_effort: str
    context_window_tokens: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_mode": self.api_mode,
            "reasoning_effort": self.reasoning_effort,
            "context_window_tokens": self.context_window_tokens,
        }


@dataclass(frozen=True)
class InvocationRequest:
    logical_run_id: str
    kind: InvocationKind
    prompt: str
    resume_token: str | None = None
    context_messages: list[dict[str, Any]] = field(default_factory=list)
    original_input: str | None = None
    completed_tool_results: list[dict[str, Any]] = field(default_factory=list)
    successful_action_receipts: list[dict[str, Any]] = field(default_factory=list)
    partial_visible_assistant_output: str = ""


@dataclass(frozen=True)
class AgentTurnResult:
    final_output: str | None = None
    context_messages: list[dict[str, Any]] = field(default_factory=list)
    interrupted: bool = False
    resume_token: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    remaining_work: str | None = None
    completed_tool_results: list[dict[str, Any]] = field(default_factory=list)
    successful_action_receipts: list[dict[str, Any]] = field(default_factory=list)
    partial_visible_assistant_output: str = ""


@dataclass(frozen=True)
class BackgroundCheckpoint:
    """Durable safe-boundary input for resuming one Logical Run."""

    resume_token: str
    logical_run_id: str
    input_prompt: str
    remaining_work: str
    context_messages: list[dict[str, Any]]
    completed_tool_results: list[dict[str, Any]]
    successful_action_receipts: list[dict[str, Any]]
    partial_visible_assistant_output: str
    updated_at: str


class BackgroundCheckpointStorePort(Protocol):
    def save(self, checkpoint: BackgroundCheckpoint) -> None: ...

    def load(self, resume_token: str) -> BackgroundCheckpoint: ...


class InvocationControl:
    """Thread-safe cancellation signal shared with one provider invocation."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._safe_boundary_deferred = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None
        self._callbacks: list[Callable[[str], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def wait_cancelled(self, timeout: float | None = None) -> bool:
        return self._cancelled.wait(timeout)

    def wait_safe_boundary_deferred(self, timeout: float | None = None) -> bool:
        """Wait until cancellation is observably queued behind an active tool."""
        return self._safe_boundary_deferred.wait(timeout)

    def cancel(self, reason: str) -> None:
        callbacks: list[Callable[[str], None]]
        with self._lock:
            if self._cancelled.is_set():
                return
            self._reason = reason
            self._cancelled.set()
            callbacks = list(self._callbacks)
        for callback in callbacks:
            callback(reason)

    def bind_cancel(self, callback: Callable[[str], None]) -> Callable[[], None]:
        reason: str | None = None
        with self._lock:
            if self._cancelled.is_set():
                reason = self._reason
            else:
                self._callbacks.append(callback)
        if reason is not None:
            callback(reason)

        def unbind() -> None:
            with self._lock:
                try:
                    self._callbacks.remove(callback)
                except ValueError:
                    pass

        return unbind


class AgentSessionPort(Protocol):
    def invoke(
        self,
        request: InvocationRequest,
        *,
        emit: Callable[[InvocationEvent], None],
        control: InvocationControl,
    ) -> AgentTurnResult: ...


class HermesAIAgentSession:
    """Adapter for one existing ``AIAgent`` instance.

    Construction validates the already-resolved runtime rather than mutating
    it.  A provider-side rejection therefore stays visible to the caller;
    this adapter never substitutes a provider, model, protocol, or reasoning
    effort.
    """

    _USAGE_KEYS = (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    )

    def __init__(self, *, agent: Any, expected: RuntimeModelConfig):
        self._agent = agent
        self._expected = expected
        actual = {
            "provider": str(getattr(agent, "provider", "") or ""),
            "model": str(getattr(agent, "model", "") or ""),
            "api_mode": str(getattr(agent, "api_mode", "") or ""),
            "reasoning_effort": self._reasoning_effort(agent),
            "context_window_tokens": int(
                getattr(getattr(agent, "context_compressor", None), "context_length", 0)
                or 0
            ),
        }
        wanted = RuntimeDiagnostics(**expected.__dict__).as_dict()
        drift = [
            f"{key}: expected {wanted[key]!r}, got {actual[key]!r}"
            for key in wanted
            if actual[key] != wanted[key]
        ]
        if not bool(getattr(agent, "skip_memory", False)):
            drift.append("Hermes memory plugin must be disabled for Harness turns")
        if not bool(getattr(agent, "skip_background_review", False)):
            drift.append("Hermes background review must be disabled for Harness turns")
        if drift:
            raise RuntimeSelectionError("runtime selection drift: " + "; ".join(drift))

    @staticmethod
    def _reasoning_effort(agent: Any) -> str:
        config = getattr(agent, "reasoning_config", None)
        if not isinstance(config, dict) or config.get("enabled") is False:
            return ""
        return str(config.get("effort") or "")

    def invoke(
        self,
        request: InvocationRequest,
        *,
        emit: Callable[[InvocationEvent], None],
        control: InvocationControl,
    ) -> AgentTurnResult:
        invocation_done = threading.Event()
        cancellation_workers: list[threading.Thread] = []

        def interrupt_at_safe_boundary(reason: str) -> None:
            request_at_boundary = getattr(
                self._agent,
                "request_interrupt_at_tool_safe_boundary",
                None,
            )
            if callable(request_at_boundary):
                if bool(request_at_boundary(reason)):
                    control._safe_boundary_deferred.set()
                return

            def deliver() -> None:
                # Hermes sets _executing_tools around the complete executor,
                # including canonical result append and persistence. Never use
                # AIAgent.interrupt() while that fence is active: it explicitly
                # aborts tool and child-agent workers.
                while bool(getattr(self._agent, "_executing_tools", False)):
                    control._safe_boundary_deferred.set()
                    if invocation_done.wait(0.01):
                        return
                if not invocation_done.is_set():
                    self._agent.interrupt(reason)

            worker = threading.Thread(
                target=deliver,
                name="thine-safe-boundary-cancel",
                daemon=True,
            )
            cancellation_workers.append(worker)
            worker.start()

        unbind_cancel = control.bind_cancel(interrupt_at_safe_boundary)

        def on_delta(delta: Any) -> None:
            text = str(delta or "")
            if text:
                visible_deltas.append(text)
                emit(InvocationEvent.progress("assistant_delta", text))

        visible_deltas: list[str] = []
        try:
            raw = self._agent.run_conversation(
                request.prompt,
                conversation_history=request.context_messages or None,
                stream_callback=on_delta,
            )
        finally:
            invocation_done.set()
            unbind_cancel()
            for worker in cancellation_workers:
                worker.join(timeout=0.1)
        usage = {
            key: int(raw.get(key) or 0)
            for key in self._USAGE_KEYS
        }
        messages = list(raw.get("messages") or [])
        completed_tool_results = [
            {
                "tool_call_id": str(message.get("tool_call_id") or ""),
                "name": str(
                    message.get("tool_name") or message.get("name") or ""
                ),
                "content": message.get("content"),
                "effect_disposition": message.get("effect_disposition"),
            }
            for message in messages
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        successful_action_receipts = [
            {
                "tool_call_id": result["tool_call_id"],
                "name": result["name"],
                "status": "applied",
            }
            for result in completed_tool_results
            if result["effect_disposition"] == "applied"
        ]
        return AgentTurnResult(
            final_output=raw.get("final_response"),
            context_messages=messages,
            interrupted=bool(raw.get("interrupted")),
            resume_token=request.resume_token if raw.get("interrupted") else None,
            usage=usage,
            remaining_work=(
                SAFE_BOUNDARY_RESUME_PROMPT if raw.get("interrupted") else None
            ),
            completed_tool_results=completed_tool_results,
            successful_action_receipts=successful_action_receipts,
            partial_visible_assistant_output="".join(visible_deltas),
        )


class HermesInvocationRuntime:
    """Project one Hermes session onto the Thine invocation port."""

    def __init__(
        self,
        *,
        session: AgentSessionPort,
        config: RuntimeModelConfig,
        checkpoint_store: BackgroundCheckpointStorePort | None = None,
        clock: Callable[[], str] | None = None,
    ):
        self._session = session
        self._config = config
        self._checkpoint_store = checkpoint_store
        self._clock = clock or (
            lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

    def diagnostics(self) -> RuntimeDiagnostics:
        return RuntimeDiagnostics(**self._config.__dict__)

    def invoke(
        self,
        request: InvocationRequest,
        *,
        emit: Callable[[InvocationEvent], None],
        control: InvocationControl | None = None,
    ) -> AgentTurnResult:
        if request.kind is InvocationKind.BACKGROUND and not request.resume_token:
            raise ValueError(
                "background invocation requires a durable resume_token before admission"
            )
        if request.kind is InvocationKind.BACKGROUND:
            if self._checkpoint_store is None:
                raise ValueError(
                    "background invocation requires a durable checkpoint_store"
                )
            self._checkpoint_store.save(
                BackgroundCheckpoint(
                    resume_token=request.resume_token or "",
                    logical_run_id=request.logical_run_id,
                    input_prompt=request.original_input or request.prompt,
                    remaining_work=request.prompt,
                    context_messages=list(request.context_messages),
                    completed_tool_results=list(request.completed_tool_results),
                    successful_action_receipts=list(
                        request.successful_action_receipts
                    ),
                    partial_visible_assistant_output=(
                        request.partial_visible_assistant_output
                    ),
                    updated_at=self._clock(),
                )
            )
        invocation_control = control or InvocationControl()
        emit(InvocationEvent(InvocationEventKind.ACCEPTED, "queue", "Accepted"))
        emit(InvocationEvent(InvocationEventKind.STARTED, "runtime", "Started"))

        def emit_progress(event: InvocationEvent) -> None:
            if event.kind is not InvocationEventKind.PROGRESS:
                raise ValueError("session ports may emit progress events only")
            if request.kind is InvocationKind.USER_CHAT:
                emit(event)

        result = self._session.invoke(
            request,
            emit=emit_progress,
            control=invocation_control,
        )
        if result.interrupted:
            if request.kind is InvocationKind.BACKGROUND:
                assert self._checkpoint_store is not None
                self._checkpoint_store.save(
                    BackgroundCheckpoint(
                        resume_token=request.resume_token or "",
                        logical_run_id=request.logical_run_id,
                        input_prompt=request.original_input or request.prompt,
                        remaining_work=result.remaining_work or request.prompt,
                        context_messages=list(
                            result.context_messages or request.context_messages
                        ),
                        completed_tool_results=self._merge_checkpoint_records(
                            request.completed_tool_results,
                            result.completed_tool_results,
                        ),
                        successful_action_receipts=self._merge_checkpoint_records(
                            request.successful_action_receipts,
                            result.successful_action_receipts,
                        ),
                        partial_visible_assistant_output=(
                            request.partial_visible_assistant_output
                            + result.partial_visible_assistant_output
                        ),
                        updated_at=self._clock(),
                    )
                )
            emit(
                InvocationEvent(
                    InvocationEventKind.INTERRUPTED,
                    "runtime",
                    invocation_control.reason or "Interrupted",
                )
            )
            return result

        if result.final_output is None:
            raise RuntimeError("completed invocation did not produce a final output")
        emit(
            InvocationEvent(
                InvocationEventKind.FINAL,
                "final",
                result.final_output,
                ephemeral=False,
            )
        )
        return result

    def resume(
        self,
        resume_token: str,
        *,
        emit: Callable[[InvocationEvent], None],
        control: InvocationControl | None = None,
    ) -> AgentTurnResult:
        """Start a new invocation of the same Logical Run at its checkpoint."""
        if self._checkpoint_store is None:
            raise ValueError("resume requires a durable checkpoint_store")
        checkpoint = self._checkpoint_store.load(resume_token)
        return self.invoke(
            InvocationRequest(
                logical_run_id=checkpoint.logical_run_id,
                kind=InvocationKind.BACKGROUND,
                prompt=checkpoint.remaining_work,
                resume_token=checkpoint.resume_token,
                context_messages=list(checkpoint.context_messages),
                original_input=checkpoint.input_prompt,
                completed_tool_results=list(checkpoint.completed_tool_results),
                successful_action_receipts=list(
                    checkpoint.successful_action_receipts
                ),
                partial_visible_assistant_output=(
                    checkpoint.partial_visible_assistant_output
                ),
            ),
            emit=emit,
            control=control,
        )

    @staticmethod
    def _merge_checkpoint_records(
        previous: list[dict[str, Any]],
        current: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in [*previous, *current]:
            marker = repr(sorted(record.items(), key=lambda item: item[0]))
            if marker not in seen:
                seen.add(marker)
                merged.append(dict(record))
        return merged


__all__ = [
    "AgentSessionPort",
    "AgentTurnResult",
    "BackgroundCheckpoint",
    "BackgroundCheckpointStorePort",
    "HermesAIAgentSession",
    "HermesInvocationRuntime",
    "InvocationControl",
    "InvocationEvent",
    "InvocationEventKind",
    "InvocationKind",
    "InvocationRequest",
    "RuntimeDiagnostics",
    "RuntimeModelConfig",
    "RuntimeSelectionError",
    "SAFE_BOUNDARY_RESUME_PROMPT",
]
