"""Credential-safe controlled proof of the real transcript/Working Memory path."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

from .contracts.transcripts import (
    TranscriptAck,
    TranscriptCanonicalLookup,
    TranscriptClaim,
    TranscriptClaimLookup,
)
from .input_pump import TranscriptClaimNotFound, TranscriptInputPump
from .integrated_probe import _pin_tool_search_listing_off, _sanitized_error
from .run_coordinator import RunCoordinator
from .run_state import DurableRunState
from .transcript_agent import TranscriptAgentFinalizer, build_real_transcript_runtime


class _ControlledTranscriptPort:
    def __init__(self) -> None:
        self.claims: dict[str, TranscriptClaim] = {}
        self.ack_count = 0

    def lookup_claim(self, claim_request_id):
        request_id = str(claim_request_id)
        if request_id not in self.claims:
            raise TranscriptClaimNotFound(request_id)
        return TranscriptClaimLookup.from_dict(self.claims[request_id].to_dict())

    def claim(self, request):
        payload = request.payload
        claim = TranscriptClaim.from_dict({
            "schema_version": {"major": 1, "minor": 0},
            "claim_id": f"controlled:{payload.claim_request_id}",
            "claim_request_id": payload.claim_request_id,
            "lease_owner": payload.lease_owner,
            "claimed_at_ms": payload.now_ms,
            "lease_expires_at_ms": payload.now_ms + 600_000,
            "claim_state": "claimed",
            "acknowledged_at_ms": None,
            "released_at_ms": None,
            "release_reason": None,
            "window_token_count": 12,
            "window_source_duration_ms": 2_000,
            "input_continuation_cursor": None,
            "next_continuation_cursor": None,
            "window_complete": True,
            "entries": [{
                "aggregation_buffer_id": 1,
                "buffer_status": "hermes_pending",
                "created_at_ms": payload.now_ms - 2_000,
                "updated_at_ms": payload.now_ms - 1_000,
                "source_start_ms": payload.now_ms - 2_000,
                "source_end_ms": payload.now_ms,
                "sequence_number": 1,
                "provenance": "legacy_buffer:1",
                "transcript": (
                    "This is a controlled local proof transcript. No user-visible "
                    "action is requested."
                ),
                "segments": [{
                    "segment_id": "controlled-segment-1",
                    "type": "speech",
                    "start_ms": 0,
                    "end_ms": 2_000,
                    "occurred_at_ms": payload.now_ms - 2_000,
                    "text": (
                        "This is a controlled local proof transcript. No user-visible "
                        "action is requested."
                    ),
                    "speaker": {
                        "source_speaker_id": "SPEAKER_UNKNOWN_1",
                        "canonical_speaker_id": None,
                        "canonical_display_name": None,
                        "canonical_is_user": None,
                        "attribution": "unknown",
                    },
                    "audio_ref": "controlled://audio/1",
                    "chunk_ref": None,
                }],
            }],
            "extensions": {},
        })
        self.claims[str(payload.claim_request_id)] = claim
        return claim

    def acknowledge(self, claim_id, run_id, memory_version):
        self.ack_count += 1
        return TranscriptAck.from_dict({
            "schema_version": {"major": 1, "minor": 0},
            "ack_id": f"controlled-ack:{claim_id}",
            "claim_id": str(claim_id),
            "run_id": str(run_id),
            "memory_version": str(memory_version),
            "acknowledged_at_ms": int(time.time() * 1000),
            "deleted_buffer_entry_ids": [1],
            "durable_receipt_written": True,
            "canonical_transcript_retained": True,
            "extensions": {},
        })

    def canonical_lookup(self, sequence_number):
        return TranscriptCanonicalLookup.from_dict({
            "schema_version": {"major": 1, "minor": 0},
            "sequence_number": int(sequence_number),
            "transcript": "Controlled canonical transcript remains retained.",
            "segments": [],
            "extensions": {},
        })

    def renew(self, request):
        raise AssertionError(f"controlled proof does not renew: {request!r}")

    def reclaim(self, request):
        raise AssertionError(f"controlled proof does not reclaim: {request!r}")

    def release(self, claim_id, reason):
        raise AssertionError(
            f"controlled proof does not release {claim_id!r}: {reason!r}"
        )


class _NoEffects:
    def apply(self, command):
        raise AssertionError(f"controlled proof cannot execute effects: {command!r}")


def run_controlled_probe(*, firebase_uid: str) -> dict[str, object]:
    _pin_tool_search_listing_off()
    with tempfile.TemporaryDirectory(prefix="thi3-50-transcript-") as directory:
        state = DurableRunState(Path(directory) / "state.sqlite3")
        transcript = _ControlledTranscriptPort()
        now_ms = int(time.time() * 1000)
        pump = TranscriptInputPump(state, transcript_port=transcript)
        runtime = build_real_transcript_runtime(
            state, firebase_uid=firebase_uid
        )
        coordinator = RunCoordinator(
            state,
            runtime=runtime,
            feature_port=_NoEffects(),
            input_port=pump,
            finalizer=TranscriptAgentFinalizer(state, transcript_port=transcript),
        )
        pump.enqueue_availability(
            user_id=firebase_uid,
            source_hint="controlled-live-proof",
            occurred_at_ms=now_ms - 2_000,
            received_at_ms=now_ms,
        )
        result = coordinator.run_next(firebase_uid)
        if result is None:
            raise RuntimeError("controlled transcript Tick was not admitted")
        if result.status != "completed":
            attempts = state.diagnostics(firebase_uid).attempts
            return {
                "status": result.status,
                "attempt_ordinal": result.attempt_ordinal,
                "failure_code": (
                    attempts[-1].failure_code if attempts else "missing_attempt"
                ),
            }
        inspection = state.inspect_agent_run(
            user_id=firebase_uid, logical_run_id=result.logical_run_id
        )
        memory = state.working_memory_snapshot(firebase_uid)
        return {
            "status": "ok" if result.status == "completed" else result.status,
            "model": inspection.model,
            "provider": inspection.provider,
            "reasoning_effort": inspection.reasoning_effort,
            "decision_outcome": inspection.decision_outcome,
            "tool_discoveries": list(inspection.tool_discoveries),
            "stop_hook_outcome": inspection.stop_hook_outcome,
            "memory_version": memory.version,
            "memory_token_count": memory.token_count,
            "memory_chars": len(memory.markdown),
            "ack_count": transcript.ack_count,
            "usage": inspection.usage,
        }


def main() -> int:
    firebase_uid = os.environ.get("THINE_FIREBASE_UID", "").strip()
    if not firebase_uid:
        print(json.dumps({"status": "blocked", "message": "THINE_FIREBASE_UID is required"}))
        return 2
    try:
        evidence = run_controlled_probe(firebase_uid=firebase_uid)
    except Exception as exc:
        evidence = _sanitized_error(exc)
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 0 if evidence.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_controlled_probe"]
