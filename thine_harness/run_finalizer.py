"""Kind-specific finalization for transcript no-action Logical Runs."""

from __future__ import annotations

import time
from typing import Callable

from .contracts.ports import ClaimId, MemoryVersion, RunId, TranscriptPort
from .input_pump import PreparedTranscriptInput
from .run_coordinator import (
    ActiveRunLease,
    InvocationContext,
    InvocationOutcome,
    RunFinalizationResult,
)
from .run_state import DurableRunState, PendingTranscriptAck


class TranscriptNoActionFinalizer:
    """Atomically record unchanged memory, then run the ack-only suffix."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        transcript_port: TranscriptPort,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._state = state
        self._transcript_port = transcript_port
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def resume_pending(self, user_id: str) -> RunFinalizationResult | None:
        pending = self._state.next_pending_transcript_ack(user_id)
        if pending is None:
            return None
        return self._ack_pending(pending)

    def finalize(
        self,
        context: InvocationContext,
        outcome: InvocationOutcome,
        *,
        lease: ActiveRunLease,
    ) -> RunFinalizationResult:
        payload = context.tick.payload
        if payload.kind != "p1_transcript":
            raise ValueError("transcript finalizer cannot finalize another Tick kind")
        if outcome.decision_outcome != "no_action":
            raise ValueError("THI3-49 finalization requires explicit no_action")
        if not isinstance(context.prepared_input, PreparedTranscriptInput):
            raise ValueError("transcript finalization requires the durable claim")
        pending = self._state.finalize_transcript_no_action(
            user_id=lease.user_id,
            logical_run_id=lease.logical_run_id,
            owner=lease.owner,
            attempt_id=lease.attempt_id,
            lease_token=lease.lease_token,
            now_ms=self._clock_ms(),
        )
        return self._ack_pending(pending)

    def _ack_pending(self, pending: PendingTranscriptAck) -> RunFinalizationResult:
        try:
            acknowledgement = self._transcript_port.acknowledge(
                ClaimId(pending.claim_id),
                RunId(pending.logical_run_id),
                MemoryVersion(str(pending.memory_version)),
            )
        except Exception:
            # The backend may have committed cleanup while its response was lost.
            # Keep the run non-runnable and retry only this suffix on the next wake.
            return RunFinalizationResult(
                tick_id=pending.tick_id,
                logical_run_id=pending.logical_run_id,
                attempt_ordinal=pending.attempt_ordinal,
                status="awaiting_audio_ack",
            )
        self._state.complete_transcript_ack(
            pending=pending,
            acknowledgement=acknowledgement,
        )
        return RunFinalizationResult(
            tick_id=pending.tick_id,
            logical_run_id=pending.logical_run_id,
            attempt_ordinal=pending.attempt_ordinal,
            status="completed",
        )


__all__ = ["TranscriptNoActionFinalizer"]

