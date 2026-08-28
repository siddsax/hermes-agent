from __future__ import annotations

from pathlib import Path
from typing import Callable

import httpx

from thine_harness.contracts.transcripts import (
    TranscriptAck,
    TranscriptCanonicalLookup,
    TranscriptClaim,
    TranscriptClaimLookup,
)
from thine_harness.input_pump import (
    BackendTranscriptClient,
    FakeTranscriptNoActionRuntime,
    TranscriptClaimNotFound,
    TranscriptInputPump,
)
from thine_harness.run_coordinator import RunCoordinator
from thine_harness.run_finalizer import TranscriptNoActionFinalizer
from thine_harness.run_state import DurableRunState


def _claimed_transcript(
    *,
    claim_request_id: str,
    lease_owner: str,
    claim_id: str | None = None,
) -> TranscriptClaim:
    return TranscriptClaim.from_dict(
        {
            "schema_version": {"major": 1, "minor": 0},
            "claim_id": claim_id or f"audio-claim:{claim_request_id}",
            "claim_request_id": claim_request_id,
            "lease_owner": lease_owner,
            "claimed_at_ms": 100,
            "lease_expires_at_ms": 600_100,
            "claim_state": "claimed",
            "acknowledged_at_ms": None,
            "released_at_ms": None,
            "release_reason": None,
            "window_token_count": 4,
            "window_source_duration_ms": 2_000,
            "input_continuation_cursor": None,
            "next_continuation_cursor": None,
            "window_complete": True,
            "entries": [
                {
                    "aggregation_buffer_id": 41,
                    "buffer_status": "hermes_pending",
                    "created_at_ms": 10,
                    "updated_at_ms": 20,
                    "source_start_ms": 1_000,
                    "source_end_ms": 3_000,
                    "sequence_number": 901,
                    "provenance": "transcription_sequence:901",
                    "transcript": "Unknown speaker material is still eligible.",
                    "segments": [
                        {
                            "segment_id": "segment-901",
                            "type": "speech",
                            "start_ms": 0,
                            "end_ms": 2_000,
                            "occurred_at_ms": 1_000,
                            "text": "Unknown speaker material is still eligible.",
                            "speaker": {
                                "source_speaker_id": "SPEAKER_UNKNOWN_1",
                                "canonical_speaker_id": None,
                                "canonical_display_name": None,
                                "canonical_is_user": None,
                                "attribution": "unknown",
                            },
                            "audio_ref": "gs://redacted/audio-901",
                            "chunk_ref": None,
                        }
                    ],
                }
            ],
            "extensions": {},
        }
    )


def _acknowledgement(
    *, claim_id: str, run_id: str, memory_version: str
) -> TranscriptAck:
    return TranscriptAck.from_dict(
        {
            "schema_version": {"major": 1, "minor": 0},
            "ack_id": f"ack:{claim_id}",
            "claim_id": claim_id,
            "run_id": run_id,
            "memory_version": memory_version,
            "acknowledged_at_ms": 500,
            "deleted_buffer_entry_ids": [41],
            "durable_receipt_written": True,
            "canonical_transcript_retained": True,
            "extensions": {},
        }
    )


class _TranscriptPort:
    def __init__(
        self,
        *,
        on_claim: Callable[[], None] | None = None,
        lose_first_claim_response: bool = False,
        lose_first_ack_response: bool = False,
    ) -> None:
        self.on_claim = on_claim
        self.lose_first_claim_response = lose_first_claim_response
        self.lose_first_ack_response = lose_first_ack_response
        self.claim_requests = []
        self.lookup_requests: list[str] = []
        self.ack_requests: list[tuple[str, str, str]] = []
        self.claim_by_request: dict[str, TranscriptClaim] = {}

    def claim(self, request):
        self.claim_requests.append(request)
        if self.on_claim is not None:
            self.on_claim()
        request_id = request.payload.claim_request_id
        claim = self.claim_by_request.setdefault(
            request_id,
            _claimed_transcript(
                claim_request_id=request_id,
                lease_owner=request.payload.lease_owner,
            ),
        )
        if self.lose_first_claim_response and len(self.claim_requests) == 1:
            raise TimeoutError("claim response was lost after backend commit")
        return claim

    def lookup_claim(self, claim_request_id):
        request_id = str(claim_request_id)
        self.lookup_requests.append(request_id)
        try:
            return TranscriptClaimLookup.from_dict(
                self.claim_by_request[request_id].to_dict()
            )
        except KeyError as exc:
            raise TranscriptClaimNotFound(request_id) from exc

    def acknowledge(self, claim_id, run_id, memory_version):
        request = (str(claim_id), str(run_id), str(memory_version))
        self.ack_requests.append(request)
        acknowledgement = _acknowledgement(
            claim_id=request[0], run_id=request[1], memory_version=request[2]
        )
        if self.lose_first_ack_response and len(self.ack_requests) == 1:
            raise TimeoutError("ack response was lost after backend cleanup")
        return acknowledgement

    def renew(self, request):  # pragma: no cover - not used by this vertical slice
        raise AssertionError("renew is not expected in the no-action happy path")

    def reclaim(self, request):  # pragma: no cover - not used by this vertical slice
        raise AssertionError("reclaim is not expected in the no-action happy path")

    def release(self, claim_id, reason):  # pragma: no cover
        raise AssertionError("release is not expected in the no-action happy path")

    def canonical_lookup(self, sequence_number):
        return TranscriptCanonicalLookup.from_dict(
            {
                "schema_version": {"major": 1, "minor": 0},
                "sequence_number": int(sequence_number),
                "transcript": "Canonical transcript remains.",
                "segments": [],
                "extensions": {},
            }
        )


class _NoFeatureEffects:
    def __init__(self) -> None:
        self.calls = []

    def apply(self, command):
        self.calls.append(command)
        raise AssertionError("a no-action transcript run cannot execute visible effects")


def _coordinator(
    database: Path,
    *,
    transcript_port: _TranscriptPort,
    runtime: FakeTranscriptNoActionRuntime,
    feature_port: _NoFeatureEffects,
    clock: Callable[[], int],
):
    state = DurableRunState(database)
    pump = TranscriptInputPump(state, transcript_port=transcript_port, clock_ms=clock)
    finalizer = TranscriptNoActionFinalizer(
        state, transcript_port=transcript_port, clock_ms=clock
    )
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=feature_port,
        input_port=pump,
        finalizer=finalizer,
        clock_ms=clock,
    )
    return state, pump, coordinator


def test_transcript_is_claimed_only_after_lease_then_no_action_is_finalized_and_acked(
    tmp_path: Path,
):
    now = 100
    database = tmp_path / "state.sqlite3"
    runtime = FakeTranscriptNoActionRuntime()
    effects = _NoFeatureEffects()
    transcript = _TranscriptPort()
    state, pump, coordinator = _coordinator(
        database,
        transcript_port=transcript,
        runtime=runtime,
        feature_port=effects,
        clock=lambda: now,
    )
    transcript.on_claim = lambda: (
        len(state.diagnostics("daily-user").leases) == 1
        or (_ for _ in ()).throw(AssertionError("claim happened before lease"))
    )

    tick_id = pump.enqueue_availability(
        user_id="daily-user",
        source_hint="aggregation-buffer-ready:901",
        occurred_at_ms=90,
        received_at_ms=95,
    )

    assert transcript.claim_requests == []
    assert coordinator.diagnostics("daily-user").queue[0].tick_id == tick_id

    result = coordinator.run_next("daily-user")

    assert result is not None
    assert result.status == "completed"
    assert len(transcript.claim_requests) == 1
    assert len(runtime.invocations) == 1
    prepared = runtime.invocations[0].prepared_input
    assert prepared.claim.payload.entries[0].segments[0].speaker.attribution == "unknown"
    assert transcript.ack_requests == [
        (
            prepared.claim.payload.claim_id,
            result.logical_run_id,
            "0",
        )
    ]
    assert effects.calls == []

    record = state.transcript_run_record(
        user_id="daily-user", logical_run_id=result.logical_run_id
    )
    assert record.queue_state == "completed"
    assert record.decision_outcome == "no_action"
    assert record.visible_action_intent_count == 0
    assert record.working_memory_outcome == "unchanged"
    assert record.memory_version == 0
    assert record.finalization_phase == "completed"
    assert record.input_receipt_id == f"input-receipt:{record.ack_id}"
    assert record.run_receipt_id == f"run-receipt:{result.logical_run_id}"
    assert record.canonical_transcript_retained is True
    assert coordinator.run_next("daily-user") is None


def test_lost_claim_response_reuses_request_and_does_not_duplicate_inference(
    tmp_path: Path,
):
    transcript = _TranscriptPort(lose_first_claim_response=True)
    runtime = FakeTranscriptNoActionRuntime()
    state, pump, coordinator = _coordinator(
        tmp_path / "state.sqlite3",
        transcript_port=transcript,
        runtime=runtime,
        feature_port=_NoFeatureEffects(),
        clock=lambda: 100,
    )
    pump.enqueue_availability(
        user_id="daily-user",
        source_hint="aggregation-buffer-ready:901",
        occurred_at_ms=90,
        received_at_ms=95,
    )

    first = coordinator.run_next("daily-user")
    second = coordinator.run_next("daily-user")

    assert first is not None and first.status == "failed_retryable"
    assert second is not None and second.status == "completed"
    assert len(transcript.claim_requests) == 1
    assert len(runtime.invocations) == 1
    assert transcript.lookup_requests[0] == transcript.lookup_requests[1]
    assert state.transcript_run_record(
        user_id="daily-user", logical_run_id=second.logical_run_id
    ).attempts_total == 2


def test_ack_only_recovery_never_reinvokes_fake_decision_or_stop_hook(
    tmp_path: Path,
):
    database = tmp_path / "state.sqlite3"
    transcript = _TranscriptPort(lose_first_ack_response=True)
    runtime = FakeTranscriptNoActionRuntime()
    effects = _NoFeatureEffects()
    state, pump, coordinator = _coordinator(
        database,
        transcript_port=transcript,
        runtime=runtime,
        feature_port=effects,
        clock=lambda: 100,
    )
    tick_id = pump.enqueue_availability(
        user_id="daily-user",
        source_hint="aggregation-buffer-ready:901",
        occurred_at_ms=90,
        received_at_ms=95,
    )

    pending = coordinator.run_next("daily-user")

    assert pending is not None and pending.status == "awaiting_audio_ack"
    assert len(runtime.invocations) == 1
    interim = state.transcript_run_record(
        user_id="daily-user", logical_run_id=pending.logical_run_id
    )
    assert interim.finalization_phase == "awaiting_audio_ack"
    assert interim.working_memory_outcome == "unchanged"
    assert interim.input_receipt_id is None

    _, replay_pump, restarted = _coordinator(
        database,
        transcript_port=transcript,
        runtime=runtime,
        feature_port=effects,
        clock=lambda: 200,
    )
    completed = restarted.run_next("daily-user")

    assert completed is not None and completed.status == "completed"
    assert completed.tick_id == tick_id
    assert len(runtime.invocations) == 1
    assert len(transcript.claim_requests) == 1
    assert len(transcript.ack_requests) == 2
    assert replay_pump.enqueue_availability(
        user_id="daily-user",
        source_hint="aggregation-buffer-ready:901",
        occurred_at_ms=90,
        received_at_ms=95,
    ) == tick_id
    assert restarted.run_next("daily-user") is None
    assert effects.calls == []


def test_backend_transcript_client_uses_only_explicit_loopback_helpers():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = __import__("json").loads(request.content)
        if request.url.path == "/v1/transcripts/claims":
            return httpx.Response(
                200,
                json=_claimed_transcript(
                    claim_request_id=body["claim_request_id"],
                    lease_owner=body["lease_owner"],
                ).to_dict(),
            )
        if request.url.path == "/v1/transcripts/claims/ack":
            return httpx.Response(
                200,
                json=_acknowledgement(
                    claim_id=body["claim_id"],
                    run_id=body["run_id"],
                    memory_version=body["memory_version"],
                ).to_dict(),
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = BackendTranscriptClient(
        origin="http://127.0.0.1:8790",
        credential="private-token",
        firebase_uid="daily-user",
        clock_ms=lambda: 123,
        transport=httpx.MockTransport(handler),
    )
    request = TranscriptInputPump.claim_request(
        user_id="daily-user",
        logical_run_id="run-1",
        claim_request_id="claim-request-1",
        now_ms=100,
    )

    try:
        claim = client.claim(request)
        ack = client.acknowledge(claim.payload.claim_id, "run-1", "0")
    finally:
        client.close()

    assert claim.payload.claim_request_id == "claim-request-1"
    assert ack.payload.canonical_transcript_retained is True
    assert [request.url.path for request in requests] == [
        "/v1/transcripts/claims",
        "/v1/transcripts/claims/ack",
    ]
    assert all(request.headers["authorization"] == "Bearer private-token" for request in requests)
    assert all(request.headers["x-thine-firebase-uid"] == "daily-user" for request in requests)
