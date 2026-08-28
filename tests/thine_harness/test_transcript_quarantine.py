from __future__ import annotations

from pathlib import Path

import httpx

from thine_harness.contracts.recovery import InputGap
from thine_harness.contracts.transcripts import (
    TranscriptAck,
    TranscriptCanonicalLookup,
    TranscriptClaim,
    TranscriptClaimLookup,
)
from thine_harness.input_pump import (
    BackendTranscriptClient,
    FakeTranscriptNoActionRuntime,
    PreparedTranscriptInput,
    TranscriptClaimNotFound,
    TranscriptInputPump,
    TranscriptQuarantineRequest,
    TranscriptQuarantineResult,
    TranscriptRetryRequest,
    TranscriptRetryResult,
)
from thine_harness.run_coordinator import InvocationOutcome, RunCoordinator
from thine_harness.run_finalizer import TranscriptNoActionFinalizer
from thine_harness.run_state import DurableRunState


def _claim(
    *, request_id: str, run_id: str, buffer_id: int, sequence: int
) -> TranscriptClaim:
    return TranscriptClaim.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "claim_id": f"audio-claim:{request_id}",
        "claim_request_id": request_id,
        "lease_owner": run_id,
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
                "aggregation_buffer_id": buffer_id,
                "buffer_status": "hermes_pending",
                "created_at_ms": 10,
                "updated_at_ms": 20,
                "source_start_ms": 1_000,
                "source_end_ms": 3_000,
                "sequence_number": sequence,
                "provenance": f"transcription_sequence:{sequence}",
                "transcript": "Unknown speaker material remains eligible.",
                "segments": [
                    {
                        "segment_id": f"segment-{sequence}",
                        "type": "speech",
                        "start_ms": 0,
                        "end_ms": 2_000,
                        "occurred_at_ms": 1_000,
                        "text": "Unknown speaker material remains eligible.",
                        "speaker": {
                            "source_speaker_id": "SPEAKER_UNKNOWN_1",
                            "canonical_speaker_id": None,
                            "canonical_display_name": None,
                            "canonical_is_user": None,
                            "attribution": "unknown",
                        },
                        "audio_ref": f"gs://redacted/audio-{sequence}",
                        "chunk_ref": None,
                    }
                ],
            }
        ],
        "extensions": {},
    })


def _ack(*, claim_id: str, run_id: str, memory_version: str) -> TranscriptAck:
    return TranscriptAck.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "ack_id": f"ack:{claim_id}:{run_id}",
        "claim_id": claim_id,
        "run_id": run_id,
        "memory_version": memory_version,
        "acknowledged_at_ms": 800,
        "deleted_buffer_entry_ids": [41],
        "durable_receipt_written": True,
        "canonical_transcript_retained": True,
        "extensions": {},
    })


class _TranscriptRecoveryPort:
    def __init__(self) -> None:
        self.claims: dict[str, TranscriptClaim] = {}
        self.next_buffer_id = 41
        self.next_sequence = 901
        self.quarantine_requests: list[TranscriptQuarantineRequest] = []
        self.retry_requests: list[TranscriptRetryRequest] = []
        self.ack_requests: list[tuple[str, str, str]] = []
        self.quarantine_sources: dict[str, str] = {}
        self.lose_first_quarantine_response = False
        self.lose_first_retry_response = False

    def claim(self, request):
        request_id = str(request.payload.claim_request_id)
        claim = self.claims.setdefault(
            request_id,
            _claim(
                request_id=request_id,
                run_id=str(request.payload.lease_owner),
                buffer_id=self.next_buffer_id,
                sequence=self.next_sequence,
            ),
        )
        self.next_buffer_id += 1
        self.next_sequence += 1
        return claim

    def lookup_claim(self, claim_request_id):
        try:
            return TranscriptClaimLookup.from_dict(
                self.claims[str(claim_request_id)].to_dict()
            )
        except KeyError as exc:
            raise TranscriptClaimNotFound(str(claim_request_id)) from exc

    def quarantine(self, request: TranscriptQuarantineRequest):
        self.quarantine_requests.append(request)
        self.quarantine_sources[request.quarantine_id] = request.claim_id
        gap = InputGap.from_dict({
            "schema_version": {"major": 1, "minor": 0},
            "gap_id": f"gap:{request.quarantine_id}",
            "source_kind": "transcript",
            "source_identity": request.claim_id,
            "quarantine_id": request.quarantine_id,
            "normal_cursor_advanced": True,
            "reason": "attempts_exhausted",
            "recorded_at_ms": request.quarantined_at_ms,
            "extensions": {},
        })
        result = TranscriptQuarantineResult(
            status="quarantined",
            quarantine_id=request.quarantine_id,
            claim_id=request.claim_id,
            logical_run_id=request.logical_run_id,
            source_identity=request.claim_id,
            aggregation_buffer_ids=(41,),
            sequence_numbers=(901,),
            provenance=("transcription_sequence:901",),
            adoption_kinds=(None,),
            failure_code=request.failure_code,
            fault_attempts_total=3,
            quarantined_at_ms=request.quarantined_at_ms,
            normal_cursor_advanced=True,
            input_retained=True,
            canonical_transcript_retained=True,
            input_gap=gap,
        )
        if self.lose_first_quarantine_response and len(self.quarantine_requests) == 1:
            raise TimeoutError("quarantine response lost after backend commit")
        return result

    def retry_quarantine(self, request: TranscriptRetryRequest):
        self.retry_requests.append(request)
        claim = self.claims.setdefault(
            request.retry_request_id,
            _claim(
                request_id=request.retry_request_id,
                run_id=request.retry_run_id,
                buffer_id=41,
                sequence=901,
            ),
        )
        result = TranscriptRetryResult(
            quarantine_id=request.quarantine_id,
            retry_run_id=request.retry_run_id,
            retry_request_id=request.retry_request_id,
            requested_at_ms=request.requested_at_ms,
            original_claim_id=self.quarantine_sources[request.quarantine_id],
            original_source_identity=self.quarantine_sources[request.quarantine_id],
            original_provenance=("transcription_sequence:901",),
            normal_cursor_rewound=False,
            quarantine_retained=True,
            claim=claim,
        )
        if self.lose_first_retry_response and len(self.retry_requests) == 1:
            raise TimeoutError("retry response lost after backend reservation")
        return result

    def acknowledge(self, claim_id, run_id, memory_version):
        request = (str(claim_id), str(run_id), str(memory_version))
        self.ack_requests.append(request)
        return _ack(claim_id=request[0], run_id=request[1], memory_version=request[2])

    def renew(self, request):  # pragma: no cover - not part of this slice
        raise AssertionError

    def reclaim(self, request):  # pragma: no cover - not part of this slice
        raise AssertionError

    def release(self, claim_id, reason):  # pragma: no cover - not part of this slice
        raise AssertionError

    def canonical_lookup(self, sequence_number):
        return TranscriptCanonicalLookup.from_dict({
            "schema_version": {"major": 1, "minor": 0},
            "sequence_number": int(sequence_number),
            "transcript": "Canonical transcript remains.",
            "segments": [],
            "extensions": {},
        })


class _NoEffects:
    def apply(
        self, command
    ):  # pragma: no cover - a failure/no-action run has no effect
        raise AssertionError(command)


class _FaultFirstClaim:
    def __init__(self) -> None:
        self.invocations = []

    def invoke(self, context, *, tools, control):
        del tools, control
        self.invocations.append(context)
        sequence = context.prepared_input.claim.payload.entries[0].sequence_number
        if int(sequence) == 901:
            return InvocationOutcome.fault("provider_timeout")
        return InvocationOutcome.no_action()


class _StopHookFaultFinalizer(TranscriptNoActionFinalizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_hook_attempts = 0

    def finalize(self, context, outcome, *, lease):
        del context, outcome, lease
        self.stop_hook_attempts += 1
        raise RuntimeError("stop hook failed")


def _coordinator(
    database: Path,
    *,
    transcript: _TranscriptRecoveryPort,
    runtime,
    now: int,
):
    state = DurableRunState(database)
    pump = TranscriptInputPump(state, transcript_port=transcript, clock_ms=lambda: now)
    finalizer = TranscriptNoActionFinalizer(
        state, transcript_port=transcript, clock_ms=lambda: now
    )
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=_NoEffects(),
        input_port=pump,
        finalizer=finalizer,
        clock_ms=lambda: now,
    )
    return state, pump, coordinator


def test_third_real_fault_syncs_quarantine_and_delivers_gap_to_later_transcript(
    tmp_path: Path,
):
    database = tmp_path / "state.sqlite3"
    transcript = _TranscriptRecoveryPort()
    runtime = _FaultFirstClaim()
    state, pump, coordinator = _coordinator(
        database, transcript=transcript, runtime=runtime, now=100
    )
    pump.enqueue_availability(
        user_id="daily-user",
        source_hint="aggregation-buffer-ready:901",
        occurred_at_ms=90,
        received_at_ms=95,
    )

    assert coordinator.run_next("daily-user").status == "failed_retryable"
    assert coordinator.run_next("daily-user").status == "failed_retryable"
    third = coordinator.run_next("daily-user")

    assert third is not None and third.status == "quarantined"
    assert len(runtime.invocations) == 3
    assert [item.attempt_ordinal for item in runtime.invocations] == [1, 2, 3]
    assert len(transcript.quarantine_requests) == 1
    quarantine = state.inspect_transcript_quarantine(
        user_id="daily-user",
        quarantine_id=f"{third.logical_run_id}:quarantine",
    )
    assert quarantine.sync_state == "synchronized"
    assert quarantine.record.payload.immutable_range.aggregation_buffer_ids == (41,)
    assert quarantine.record.payload.immutable_range.first_sequence_number == 901

    pump.enqueue_availability(
        user_id="daily-user",
        source_hint="aggregation-buffer-ready:902",
        occurred_at_ms=101,
        received_at_ms=102,
    )
    later = coordinator.run_next("daily-user")

    assert later is not None and later.status == "completed"
    later_context = runtime.invocations[-1]
    assert (
        later_context.prepared_input.claim.payload
        .entries[0]
        .segments[0]
        .speaker.attribution
        == "unknown"
    )
    assert [
        gap.payload.quarantine_id for gap in later_context.prepared_input.input_gaps
    ] == [quarantine.quarantine_id]


def test_lost_quarantine_response_recovers_suffix_without_fourth_inference(
    tmp_path: Path,
):
    database = tmp_path / "state.sqlite3"
    transcript = _TranscriptRecoveryPort()
    transcript.lose_first_quarantine_response = True
    runtime = _FaultFirstClaim()
    _, pump, coordinator = _coordinator(
        database, transcript=transcript, runtime=runtime, now=100
    )
    pump.enqueue_availability(
        user_id="daily-user",
        source_hint="aggregation-buffer-ready:901",
        occurred_at_ms=90,
        received_at_ms=95,
    )

    assert coordinator.run_next("daily-user").status == "failed_retryable"
    assert coordinator.run_next("daily-user").status == "failed_retryable"
    assert coordinator.run_next("daily-user").status == "quarantine_pending"

    _, _, restarted = _coordinator(
        database, transcript=transcript, runtime=runtime, now=200
    )
    recovered = restarted.run_next("daily-user")

    assert recovered is not None and recovered.status == "quarantined"
    assert len(runtime.invocations) == 3
    assert len(transcript.quarantine_requests) == 2


def test_explicit_retry_is_fresh_work_and_preserves_original_quarantine(
    tmp_path: Path,
):
    database = tmp_path / "state.sqlite3"
    transcript = _TranscriptRecoveryPort()
    failing = _FaultFirstClaim()
    state, pump, coordinator = _coordinator(
        database, transcript=transcript, runtime=failing, now=100
    )
    pump.enqueue_availability(
        user_id="daily-user",
        source_hint="aggregation-buffer-ready:901",
        occurred_at_ms=90,
        received_at_ms=95,
    )
    coordinator.run_next("daily-user")
    coordinator.run_next("daily-user")
    quarantined = coordinator.run_next("daily-user")
    assert quarantined is not None and quarantined.status == "quarantined"
    quarantine_id = f"{quarantined.logical_run_id}:quarantine"

    retry_run_id = "run:explicit-retry-901"
    explicit_retry = pump.enqueue_explicit_retry(
        user_id="daily-user",
        quarantine_id=quarantine_id,
        retry_run_id=retry_run_id,
        created_at_ms=300,
    )
    no_action = FakeTranscriptNoActionRuntime()
    retry_finalizer = TranscriptNoActionFinalizer(
        state, transcript_port=transcript, clock_ms=lambda: 300
    )
    retry_coordinator = RunCoordinator(
        state,
        runtime=no_action,
        feature_port=_NoEffects(),
        input_port=pump,
        finalizer=retry_finalizer,
        clock_ms=lambda: 300,
    )
    result = retry_coordinator.run_next("daily-user")

    assert result is not None and result.status == "completed"
    assert result.logical_run_id == retry_run_id
    assert explicit_retry.payload.retry_run_id == retry_run_id
    assert explicit_retry.payload.rewinds_normal_cursor is False
    assert transcript.retry_requests[0].quarantine_id == quarantine_id
    assert transcript.retry_requests[0].lease_duration_ms == 600_000
    prepared_retry = no_action.invocations[0].prepared_input
    assert isinstance(prepared_retry, PreparedTranscriptInput)
    assert prepared_retry.explicit_retry is not None
    assert prepared_retry.explicit_retry.payload.quarantine_id == quarantine_id
    preserved = state.inspect_transcript_quarantine(
        user_id="daily-user", quarantine_id=quarantine_id
    )
    assert preserved.sync_state == "synchronized"
    assert preserved.retry_run_ids == (retry_run_id,)
    assert (
        state.transcript_run_record(
            user_id="daily-user", logical_run_id=retry_run_id
        ).queue_state
        == "completed"
    )


def test_stop_hook_faults_count_but_quarantine_delivery_does_not(tmp_path: Path):
    database = tmp_path / "state.sqlite3"
    transcript = _TranscriptRecoveryPort()
    state = DurableRunState(database)
    pump = TranscriptInputPump(state, transcript_port=transcript, clock_ms=lambda: 100)
    finalizer = _StopHookFaultFinalizer(
        state, transcript_port=transcript, clock_ms=lambda: 100
    )
    coordinator = RunCoordinator(
        state,
        runtime=FakeTranscriptNoActionRuntime(),
        feature_port=_NoEffects(),
        input_port=pump,
        finalizer=finalizer,
        clock_ms=lambda: 100,
    )
    pump.enqueue_availability(
        user_id="daily-user",
        source_hint="aggregation-buffer-ready:901",
        occurred_at_ms=90,
        received_at_ms=95,
    )

    first = coordinator.run_next("daily-user")
    second = coordinator.run_next("daily-user")
    third = coordinator.run_next("daily-user")
    assert first is not None and first.status == "failed_retryable"
    assert second is not None and second.status == "failed_retryable"
    assert third is not None and third.status == "quarantined"

    assert finalizer.stop_hook_attempts == 3
    assert len(transcript.quarantine_requests) == 1
    assert [
        (item.ordinal, item.status) for item in state.diagnostics("daily-user").attempts
    ] == [
        (1, "failed_fault"),
        (2, "failed_fault"),
        (3, "failed_fault"),
    ]


def test_lost_explicit_retry_response_reuses_new_run_without_counting_fault(
    tmp_path: Path,
):
    database = tmp_path / "state.sqlite3"
    transcript = _TranscriptRecoveryPort()
    failing = _FaultFirstClaim()
    state, pump, coordinator = _coordinator(
        database, transcript=transcript, runtime=failing, now=100
    )
    pump.enqueue_availability(
        user_id="daily-user",
        source_hint="aggregation-buffer-ready:901",
        occurred_at_ms=90,
        received_at_ms=95,
    )
    coordinator.run_next("daily-user")
    coordinator.run_next("daily-user")
    quarantined = coordinator.run_next("daily-user")
    quarantine_id = f"{quarantined.logical_run_id}:quarantine"
    retry_run_id = "run:lost-retry-response"
    pump.enqueue_explicit_retry(
        user_id="daily-user",
        quarantine_id=quarantine_id,
        retry_run_id=retry_run_id,
        created_at_ms=300,
    )
    transcript.lose_first_retry_response = True
    retry = RunCoordinator(
        state,
        runtime=FakeTranscriptNoActionRuntime(),
        feature_port=_NoEffects(),
        input_port=pump,
        finalizer=TranscriptNoActionFinalizer(
            state, transcript_port=transcript, clock_ms=lambda: 300
        ),
        clock_ms=lambda: 300,
    )

    first = retry.run_next("daily-user")
    second = retry.run_next("daily-user")

    assert first is not None and first.status == "input_retry_pending"
    assert second is not None and second.status == "completed"
    assert len(transcript.retry_requests) == 2
    assert transcript.retry_requests[0] == transcript.retry_requests[1]
    attempts = [
        item
        for item in state.diagnostics("daily-user").attempts
        if item.logical_run_id == retry_run_id
    ]
    assert [(item.ordinal, item.status) for item in attempts] == [(1, "succeeded")]


def test_backend_client_uses_exact_quarantine_retry_and_inspect_helpers():
    requests: list[httpx.Request] = []
    quarantine_payload: dict | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal quarantine_payload
        requests.append(request)
        body = __import__("json").loads(request.content)
        if request.url.path == "/v1/transcripts/claims/quarantine":
            quarantine_payload = {
                "status": "quarantined",
                "quarantine_id": body["quarantine_id"],
                "claim_id": body["claim_id"],
                "logical_run_id": body["logical_run_id"],
                "source_identity": body["claim_id"],
                "aggregation_buffer_ids": [41],
                "sequence_numbers": [901],
                "provenance": ["transcription_sequence:901"],
                "adoption_kinds": ["startup_existing_buffer"],
                "failure_code": body["failure_code"],
                "fault_attempts_total": 3,
                "quarantined_at_ms": body["quarantined_at_ms"],
                "normal_cursor_advanced": True,
                "input_retained": True,
                "canonical_transcript_retained": True,
                "input_gap": {
                    "schema_version": {"major": 1, "minor": 0},
                    "gap_id": "gap-1",
                    "source_kind": "transcript",
                    "source_identity": body["claim_id"],
                    "quarantine_id": body["quarantine_id"],
                    "normal_cursor_advanced": True,
                    "reason": "attempts_exhausted",
                    "recorded_at_ms": body["quarantined_at_ms"],
                    "extensions": {},
                },
            }
            return httpx.Response(200, json=quarantine_payload)
        if request.url.path == "/v1/transcripts/quarantines/retry":
            return httpx.Response(
                200,
                json={
                    "quarantine_id": body["quarantine_id"],
                    "retry_run_id": body["retry_run_id"],
                    "retry_request_id": body["retry_request_id"],
                    "requested_at_ms": body["requested_at_ms"],
                    "original_claim_id": "audio-claim:original",
                    "original_source_identity": "audio-claim:original",
                    "original_provenance": ["transcription_sequence:901"],
                    "normal_cursor_rewound": False,
                    "quarantine_retained": True,
                    "claim": _claim(
                        request_id=body["retry_request_id"],
                        run_id=body["retry_run_id"],
                        buffer_id=41,
                        sequence=901,
                    ).to_dict(),
                },
            )
        if request.url.path == "/v1/transcripts/quarantines/inspect":
            assert quarantine_payload is not None
            return httpx.Response(
                200,
                json={
                    "quarantine": quarantine_payload,
                    "source_rows_present": True,
                    "retries": [
                        {
                            "retry_request_id": "retry-request-1",
                            "retry_run_id": "retry-run-1",
                            "claim_id": "retry-claim-1",
                            "requested_at_ms": 200,
                            "claim_state": "claimed",
                        }
                    ],
                },
            )
        raise AssertionError(request.url.path)

    client = BackendTranscriptClient(
        origin="http://127.0.0.1:8790",
        credential="private-token",
        firebase_uid="daily-user",
        transport=httpx.MockTransport(handler),
    )
    quarantine_request = TranscriptQuarantineRequest(
        claim_id="audio-claim:original",
        logical_run_id="original-run",
        quarantine_id="quarantine-1",
        failure_code="provider_timeout",
        fault_attempts_total=3,
        quarantined_at_ms=100,
    )
    retry_request = TranscriptRetryRequest(
        quarantine_id="quarantine-1",
        retry_run_id="retry-run-1",
        retry_request_id="retry-request-1",
        requested_at_ms=200,
        lease_duration_ms=600_000,
    )
    try:
        quarantined = client.quarantine(quarantine_request)
        retried = client.retry_quarantine(retry_request)
        inspected = client.inspect_quarantine("quarantine-1")
    finally:
        client.close()

    assert quarantined.adoption_kinds == ("startup_existing_buffer",)
    assert retried.quarantine_retained is True
    assert inspected.quarantine.quarantine_id == "quarantine-1"
    assert inspected.retries[0].retry_run_id == "retry-run-1"
    assert [request.url.path for request in requests] == [
        "/v1/transcripts/claims/quarantine",
        "/v1/transcripts/quarantines/retry",
        "/v1/transcripts/quarantines/inspect",
    ]
    assert all(
        request.headers["x-thine-firebase-uid"] == "daily-user" for request in requests
    )
