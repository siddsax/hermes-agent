"""Single-flight durable Tick coordinator with a deterministic fake seam."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
import threading
import time
from typing import Any, Callable, cast, Literal, Mapping, Protocol

from .contracts.runtime import Tick
from .contracts.tool_metadata import PRODUCT_TOOL_NAMESPACES
from .run_state import (
    CheckpointRecord,
    DurableRunState,
    StateDiagnostics,
    ToolReceiptRecord,
    diagnostics_as_dict,
)
from .runtime import RuntimeModelConfig


_FINGERPRINT = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class FakeFeatureCommand:
    """Typed command at the fake feature boundary; it contains no DB handle."""

    action_id: str
    intent_fingerprint: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id is required")
        if _FINGERPRINT.fullmatch(self.intent_fingerprint) is None:
            raise ValueError("intent_fingerprint must be 64 lowercase hex characters")


@dataclass(frozen=True)
class FakeFeatureAcknowledgement:
    provider_reference: str
    result: Mapping[str, Any]


class FakeFeaturePort(Protocol):
    def apply(self, command: FakeFeatureCommand) -> FakeFeatureAcknowledgement: ...


class InvocationControl:
    """Safe-boundary preemption signal for the currently active fake invocation."""

    def __init__(self) -> None:
        self._preempted = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None

    @property
    def preemption_requested(self) -> bool:
        return self._preempted.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def request_preemption(self, reason: str) -> None:
        with self._lock:
            if self._preempted.is_set():
                return
            self._reason = reason
            self._preempted.set()

    def wait_for_preemption(self, timeout: float | None = None) -> bool:
        return self._preempted.wait(timeout)


@dataclass(frozen=True)
class InvocationContext:
    tick: Tick
    attempt_id: str
    attempt_ordinal: int
    checkpoint: CheckpointRecord | None
    acknowledged_receipts: tuple[ToolReceiptRecord, ...]
    prepared_input: Any | None = None


OutcomeStatus = Literal[
    "completed",
    "cancelled",
    "checkpointed",
    "preempted",
    "yielded",
    "continuation",
    "fault",
]


@dataclass(frozen=True)
class InvocationOutcome:
    status: OutcomeStatus
    remaining_work: str = ""
    failure_code: str | None = None
    decision_outcome: Literal["no_action"] | None = None
    finalization_context: Any | None = None

    @classmethod
    def completed(cls) -> "InvocationOutcome":
        return cls("completed")

    @classmethod
    def no_action(cls, *, finalization_context: Any | None = None) -> "InvocationOutcome":
        return cls(
            "completed",
            decision_outcome="no_action",
            finalization_context=finalization_context,
        )

    @classmethod
    def checkpointed(
        cls,
        *,
        remaining_work: str,
    ) -> "InvocationOutcome":
        return cls("checkpointed", remaining_work)

    @classmethod
    def cancelled(
        cls,
        *,
        remaining_work: str,
    ) -> "InvocationOutcome":
        return cls("cancelled", remaining_work)

    @classmethod
    def preempted(
        cls,
        *,
        remaining_work: str,
    ) -> "InvocationOutcome":
        return cls("preempted", remaining_work)

    @classmethod
    def yielded(
        cls,
        *,
        remaining_work: str,
    ) -> "InvocationOutcome":
        return cls("yielded", remaining_work)

    @classmethod
    def continuation(
        cls,
        *,
        remaining_work: str,
    ) -> "InvocationOutcome":
        return cls("continuation", remaining_work)

    @classmethod
    def fault(cls, failure_code: str) -> "InvocationOutcome":
        if not failure_code:
            raise ValueError("failure_code is required")
        return cls("fault", failure_code=failure_code)


class FakeInvocationPort(Protocol):
    def invoke(
        self,
        context: InvocationContext,
        *,
        tools: "DurableFeatureTools",
        control: InvocationControl,
    ) -> InvocationOutcome: ...


@dataclass(frozen=True)
class ActiveRunLease:
    """Coordinator-owned acquisition passed only to trusted deep modules."""

    user_id: str
    logical_run_id: str
    owner: str
    attempt_id: str
    attempt_ordinal: int
    lease_token: str


class RunInputPort(Protocol):
    """Prepare kind-specific input only after the queue lease is durable."""

    def prepare(
        self,
        context: InvocationContext,
        *,
        lease: ActiveRunLease,
    ) -> Any | None: ...


@dataclass(frozen=True)
class RunFinalizationResult:
    tick_id: str
    logical_run_id: str
    attempt_ordinal: int
    status: Literal["completed", "awaiting_audio_ack"]


class RunFinalizerPort(Protocol):
    """Commit kind-specific finalization and resume acknowledgement suffixes."""

    def resume_pending(self, user_id: str) -> RunFinalizationResult | None: ...

    def finalize(
        self,
        context: InvocationContext,
        outcome: InvocationOutcome,
        *,
        lease: ActiveRunLease,
    ) -> RunFinalizationResult: ...


@dataclass(frozen=True)
class HarnessRuntimeConfiguration:
    provider: str
    model: str
    api_mode: str
    reasoning_effort: str
    context_window_tokens: int
    tool_search_enabled: bool
    tool_search_listing: bool
    tool_namespaces: tuple[str, ...]

    @classmethod
    def from_model_config(
        cls,
        config: RuntimeModelConfig,
        *,
        tool_search_enabled: bool = True,
        tool_search_listing: bool = False,
    ) -> "HarnessRuntimeConfiguration":
        return cls(
            provider=config.provider,
            model=config.model,
            api_mode=config.api_mode,
            reasoning_effort=config.reasoning_effort,
            context_window_tokens=config.context_window_tokens,
            tool_search_enabled=tool_search_enabled,
            tool_search_listing=tool_search_listing,
            tool_namespaces=tuple(item.namespace for item in PRODUCT_TOOL_NAMESPACES),
        )


@dataclass(frozen=True)
class HarnessDiagnostics:
    state: StateDiagnostics
    runtime: HarnessRuntimeConfiguration

    @property
    def queue(self):
        return self.state.queue

    @property
    def leases(self):
        return self.state.leases

    @property
    def attempts(self):
        return self.state.attempts

    @property
    def checkpoints(self):
        return self.state.checkpoints

    @property
    def receipts(self):
        return self.state.receipts

    @property
    def quarantines(self):
        return self.state.quarantines

    def as_dict(self) -> dict[str, Any]:
        result = cast(dict[str, Any], diagnostics_as_dict(self.state))
        runtime = asdict(self.runtime)
        runtime["tool_namespaces"] = list(self.runtime.tool_namespaces)
        result["runtime"] = runtime
        return result


@dataclass(frozen=True)
class RunResult:
    tick_id: str
    logical_run_id: str
    attempt_ordinal: int
    status: str
    checkpoint_id: str | None = None


class DurableFeatureTools:
    """Execute a typed fake feature command once and persist its acknowledgement."""

    def __init__(
        self,
        *,
        state: DurableRunState,
        feature_port: FakeFeaturePort,
        user_id: str,
        logical_run_id: str,
        lease_owner: str,
        attempt_id: str,
        lease_token: str,
        clock_ms: Callable[[], int],
    ) -> None:
        self._state = state
        self._feature_port = feature_port
        self._user_id = user_id
        self._logical_run_id = logical_run_id
        self._lease_owner = lease_owner
        self._attempt_id = attempt_id
        self._lease_token = lease_token
        self._clock_ms = clock_ms

    def execute_once(self, command: FakeFeatureCommand) -> ToolReceiptRecord:
        self._state.assert_active_lease(
            user_id=self._user_id,
            logical_run_id=self._logical_run_id,
            owner=self._lease_owner,
            attempt_id=self._attempt_id,
            lease_token=self._lease_token,
            now_ms=self._clock_ms(),
        )
        existing = self._state.get_receipt(
            user_id=self._user_id,
            logical_run_id=self._logical_run_id,
            action_id=command.action_id,
        )
        if existing is not None:
            if existing.intent_fingerprint != command.intent_fingerprint:
                from .run_state import ReceiptConflict

                raise ReceiptConflict(
                    "action identity was reused with a different intent"
                )
            return existing
        acknowledgement = self._feature_port.apply(command)
        return self._state.record_or_get_receipt(
            user_id=self._user_id,
            logical_run_id=self._logical_run_id,
            action_id=command.action_id,
            intent_fingerprint=command.intent_fingerprint,
            owner=self._lease_owner,
            attempt_id=self._attempt_id,
            lease_token=self._lease_token,
            provider_reference=acknowledgement.provider_reference,
            result=dict(acknowledgement.result),
            acknowledged_at_ms=self._clock_ms(),
        )


class RunCoordinator:
    """Select and execute one durable Tick at a time."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        runtime: FakeInvocationPort,
        feature_port: FakeFeaturePort,
        runtime_configuration: HarnessRuntimeConfiguration | None = None,
        input_port: RunInputPort | None = None,
        finalizer: RunFinalizerPort | None = None,
        clock_ms: Callable[[], int] | None = None,
        lease_owner: str = "local-harness",
    ) -> None:
        self._state = state
        self._runtime = runtime
        self._feature_port = feature_port
        self._input_port = input_port
        self._finalizer = finalizer
        self._runtime_configuration = runtime_configuration or (
            HarnessRuntimeConfiguration.from_model_config(
                RuntimeModelConfig.openai_gpt_5_6_sol_medium()
            )
        )
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._lease_owner = lease_owner
        self._invocation_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active: tuple[str, str, InvocationControl] | None = None

    def enqueue(self, tick: Tick) -> str:
        tick_id = self._state.enqueue(tick, now_ms=self._clock_ms())
        payload = tick.payload
        if payload.kind == "p0_user_chat":
            with self._active_lock:
                active = self._active
            if active is not None:
                active_user, active_kind, control = active
                if active_user == payload.user_id and active_kind != "p0_user_chat":
                    control.request_preemption("p0_user_tick")
        return tick_id

    def run_next(self, user_id: str) -> RunResult | None:
        if not self._invocation_lock.acquire(blocking=False):
            return None
        try:
            if self._finalizer is not None:
                resumed = self._finalizer.resume_pending(user_id)
                if resumed is not None:
                    return RunResult(
                        tick_id=resumed.tick_id,
                        logical_run_id=resumed.logical_run_id,
                        attempt_ordinal=resumed.attempt_ordinal,
                        status=resumed.status,
                    )
            leased = self._state.lease_next(
                user_id,
                owner=self._lease_owner,
                now_ms=self._clock_ms(),
            )
            if leased is None:
                return None
            payload = leased.tick.payload
            control = InvocationControl()
            with self._active_lock:
                self._active = (user_id, str(payload.kind), control)
            if payload.kind != "p0_user_chat" and self._state.has_queued_p0(user_id):
                control.request_preemption("p0_user_tick")
            renewal_stop = threading.Event()
            renewal_interval = max(
                min(self._state.lease_duration_ms / 3_000, 5.0), 0.01
            )

            def renew_live_lease() -> None:
                while not renewal_stop.wait(renewal_interval):
                    try:
                        renewed = self._state.renew_lease(
                            user_id=user_id,
                            logical_run_id=str(payload.logical_run_id),
                            owner=self._lease_owner,
                            attempt_id=leased.attempt_id,
                            lease_token=leased.lease_token,
                            now_ms=self._clock_ms(),
                        )
                    except Exception:
                        control.request_preemption("lease_renewal_error")
                        return
                    if not renewed:
                        control.request_preemption("lease_lost")
                        return

            renewal_thread = threading.Thread(
                target=renew_live_lease,
                name=f"thine-lease:{payload.logical_run_id}",
                daemon=True,
            )
            renewal_thread.start()
            context = InvocationContext(
                tick=leased.tick,
                attempt_id=leased.attempt_id,
                attempt_ordinal=leased.attempt_ordinal,
                checkpoint=leased.checkpoint,
                acknowledged_receipts=leased.acknowledged_receipts,
            )
            active_lease = ActiveRunLease(
                user_id=user_id,
                logical_run_id=str(payload.logical_run_id),
                owner=self._lease_owner,
                attempt_id=leased.attempt_id,
                attempt_ordinal=leased.attempt_ordinal,
                lease_token=leased.lease_token,
            )
            tools = DurableFeatureTools(
                state=self._state,
                feature_port=self._feature_port,
                user_id=user_id,
                logical_run_id=str(payload.logical_run_id),
                lease_owner=self._lease_owner,
                attempt_id=leased.attempt_id,
                lease_token=leased.lease_token,
                clock_ms=self._clock_ms,
            )
            try:
                if self._input_port is not None:
                    context = replace(
                        context,
                        prepared_input=self._input_port.prepare(
                            context,
                            lease=active_lease,
                        ),
                    )
                outcome = self._runtime.invoke(context, tools=tools, control=control)
            except Exception as exc:
                outcome = InvocationOutcome.fault(
                    f"runtime_exception:{type(exc).__name__}"
                )
            finally:
                renewal_stop.set()
                renewal_thread.join(timeout=renewal_interval + 0.1)
                with self._active_lock:
                    self._active = None
            if outcome.status == "completed":
                if self._finalizer is not None:
                    try:
                        finalized = self._finalizer.finalize(
                            context,
                            outcome,
                            lease=active_lease,
                        )
                    except Exception as exc:
                        status = self._state.record_fault(
                            user_id=user_id,
                            logical_run_id=str(payload.logical_run_id),
                            owner=self._lease_owner,
                            attempt_id=leased.attempt_id,
                            lease_token=leased.lease_token,
                            failure_code=(
                                f"finalization_exception:{type(exc).__name__}"
                            ),
                            now_ms=self._clock_ms(),
                        )
                        return RunResult(
                            tick_id=str(payload.tick_id),
                            logical_run_id=str(payload.logical_run_id),
                            attempt_ordinal=leased.attempt_ordinal,
                            status=status,
                        )
                    return RunResult(
                        tick_id=finalized.tick_id,
                        logical_run_id=finalized.logical_run_id,
                        attempt_ordinal=finalized.attempt_ordinal,
                        status=finalized.status,
                    )
                self._state.complete(
                    user_id=user_id,
                    logical_run_id=str(payload.logical_run_id),
                    owner=self._lease_owner,
                    attempt_id=leased.attempt_id,
                    lease_token=leased.lease_token,
                    now_ms=self._clock_ms(),
                )
                return RunResult(
                    tick_id=str(payload.tick_id),
                    logical_run_id=str(payload.logical_run_id),
                    attempt_ordinal=leased.attempt_ordinal,
                    status="completed",
                )
            if outcome.status in {
                "cancelled",
                "checkpointed",
                "preempted",
                "yielded",
                "continuation",
            }:
                persisted_receipts = tuple(
                    receipt.receipt_id
                    for receipt in self._state.receipts_for_run(
                        user_id=user_id,
                        logical_run_id=str(payload.logical_run_id),
                    )
                )
                checkpoint = self._state.save_checkpoint_and_requeue(
                    user_id=user_id,
                    logical_run_id=str(payload.logical_run_id),
                    owner=self._lease_owner,
                    attempt_id=leased.attempt_id,
                    lease_token=leased.lease_token,
                    cause=outcome.status,
                    remaining_work=outcome.remaining_work,
                    completed_receipt_ids=persisted_receipts,
                    now_ms=self._clock_ms(),
                )
                return RunResult(
                    tick_id=str(payload.tick_id),
                    logical_run_id=str(payload.logical_run_id),
                    attempt_ordinal=leased.attempt_ordinal,
                    status="checkpointed",
                    checkpoint_id=checkpoint.checkpoint_id,
                )
            if outcome.status == "fault":
                status = self._state.record_fault(
                    user_id=user_id,
                    logical_run_id=str(payload.logical_run_id),
                    owner=self._lease_owner,
                    attempt_id=leased.attempt_id,
                    lease_token=leased.lease_token,
                    failure_code=outcome.failure_code or "runtime_fault",
                    now_ms=self._clock_ms(),
                )
                return RunResult(
                    tick_id=str(payload.tick_id),
                    logical_run_id=str(payload.logical_run_id),
                    attempt_ordinal=leased.attempt_ordinal,
                    status=status,
                )
            raise AssertionError(f"unhandled outcome {outcome.status!r}")
        finally:
            self._invocation_lock.release()

    def diagnostics(self, user_id: str) -> HarnessDiagnostics:
        return HarnessDiagnostics(
            state=self._state.diagnostics(user_id),
            runtime=self._runtime_configuration,
        )


__all__ = [
    "ActiveRunLease",
    "DurableFeatureTools",
    "FakeFeatureAcknowledgement",
    "FakeFeatureCommand",
    "FakeFeaturePort",
    "FakeInvocationPort",
    "HarnessDiagnostics",
    "HarnessRuntimeConfiguration",
    "InvocationContext",
    "InvocationControl",
    "InvocationOutcome",
    "RunFinalizationResult",
    "RunFinalizerPort",
    "RunInputPort",
    "RunCoordinator",
    "RunResult",
]
