from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx

from thine_harness.contracts.interactions import InteractionBatch
from thine_harness.interactions import (
    FakeInteractionNoActionRuntime,
    BackendInteractionClient,
    InteractionAvailability,
    InteractionClaimNotFound,
    InteractionClaimRequest,
    InteractionInputPump,
    InteractionQuarantineResult,
    InteractionRetryResult,
    InteractionRunFinalizer,
    latest_half_hour_boundary_ms,
    next_half_hour_boundary_ms,
)
from thine_harness.run_coordinator import (
    FakeFeatureAcknowledgement,
    InvocationOutcome,
    RunCoordinator,
)
from thine_harness.run_state import DurableRunState, SCHEMA_VERSION


def _batch(
    batch_id: str,
    *,
    first_cursor: int,
    last_cursor: int,
    window_end_ms: int,
    user_id: str = "daily-user",
) -> InteractionBatch:
    events = []
    for cursor in range(first_cursor, last_cursor + 1):
        events.append({
            "schema_version": {"major": 1, "minor": 0},
            "event_id": f"event-{cursor}",
            "user_id": user_id,
            "device_id": "iphone-local",
            "occurred_at_ms": window_end_ms - 1_000,
            "received_at_ms": window_end_ms - 500,
            "screen_id": "route.home",
            "surface_id": "home.card.people",
            "kind": "home_activation",
            "object_ref": "node-people",
            "outcome": "succeeded",
            "home_revision": 42,
            "home_node_id": "node-people",
            "primary_correlation": None,
            "safe_payload": {
                "action": "component_tap",
                "closed_value": None,
            },
            "extensions": {},
        })
    return InteractionBatch.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "batch_id": batch_id,
        "user_id": user_id,
        "first_cursor": first_cursor,
        "last_cursor": last_cursor,
        "window_start_ms": window_end_ms - 1_800_000,
        "window_end_ms": window_end_ms,
        "events": events,
        "delivery_attempt": 1,
        "extensions": {},
    })


class _Source:
    def __init__(self) -> None:
        self.pending: list[InteractionBatch] = []
        self.claims: dict[str, InteractionBatch] = {}
        self.claim_requests = []
        self.lookup_requests: list[str] = []
        self.consume_requests = []
        self.quarantine_requests = []
        self.retry_requests = []
        self.quarantined: dict[str, InteractionBatch] = {}
        self.lose_first_claim_response = False
        self.lose_first_consume_response = False

    def availability(self, *, boundary_end_ms: int):
        return InteractionAvailability(
            available=any(
                int(batch.payload.window_end_ms) <= boundary_end_ms
                for batch in self.pending
            ),
            next_cursor=(
                int(self.pending[0].payload.first_cursor) if self.pending else None
            ),
        )

    def claim(self, request):
        self.claim_requests.append(request)
        batch = self.claims.get(request.claim_request_id)
        if batch is None:
            batch = next(
                item
                for item in self.pending
                if int(item.payload.window_end_ms) <= request.boundary_end_ms
            )
            self.claims[request.claim_request_id] = batch
        if self.lose_first_claim_response and len(self.claim_requests) == 1:
            raise TimeoutError("backend committed claim but response was lost")
        return batch

    def lookup_claim(self, claim_request_id):
        self.lookup_requests.append(claim_request_id)
        try:
            return self.claims[claim_request_id]
        except KeyError as exc:
            raise InteractionClaimNotFound(claim_request_id) from exc

    def consume(self, receipt):
        self.consume_requests.append(receipt)
        batch_id = str(receipt.payload.batch_id)
        self.pending = [
            item for item in self.pending if str(item.payload.batch_id) != batch_id
        ]
        if self.lose_first_consume_response and len(self.consume_requests) == 1:
            raise TimeoutError("backend consumed cursor but response was lost")

    def quarantine(self, request):
        self.quarantine_requests.append(request)
        batch = next(
            item
            for item in self.pending
            if str(item.payload.batch_id) == request.batch_id
        )
        self.quarantined[request.quarantine_id] = batch
        self.pending = [item for item in self.pending if item is not batch]
        return InteractionQuarantineResult(
            quarantine_id=request.quarantine_id,
            logical_run_id=request.logical_run_id,
            batch_id=request.batch_id,
            first_cursor=request.first_cursor,
            last_cursor=request.last_cursor,
            normal_cursor_advanced=True,
            input_retained=True,
        )

    def retry(self, request):
        self.retry_requests.append(request)
        return InteractionRetryResult(
            quarantine_id=request.quarantine_id,
            retry_run_id=request.retry_run_id,
            retry_request_id=request.retry_request_id,
            batch=self.quarantined[request.quarantine_id],
            normal_cursor_rewound=False,
            quarantine_retained=True,
        )


class _NoFeatureEffects:
    def apply(self, command):
        raise AssertionError(f"interaction no-action used visible effect {command}")


class _CompleteRuntime:
    def __init__(self) -> None:
        self.order: list[str] = []

    def invoke(self, context, *, tools, control):
        self.order.append(str(context.tick.payload.kind))
        return InvocationOutcome.completed()


def _coordinator(
    database: Path,
    *,
    source: _Source,
    runtime: FakeInteractionNoActionRuntime,
    clock,
):
    state = DurableRunState(database)
    pump = InteractionInputPump(state, source=source, clock_ms=clock)
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=_NoFeatureEffects(),
        input_port=pump,
        finalizer=InteractionRunFinalizer(state, source=source, clock_ms=clock),
        clock_ms=clock,
    )
    return state, pump, coordinator


def test_boundary_is_local_wall_clock_and_empty_windows_do_not_enqueue(tmp_path: Path):
    now = int(datetime(2026, 8, 25, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
    # 05:00 UTC is 10:30 Asia/Kolkata.
    assert latest_half_hour_boundary_ms(now + 29_999, "Asia/Kolkata") == now
    kathmandu_now = int(
        datetime(2026, 8, 25, 4, 14, tzinfo=timezone.utc).timestamp() * 1000
    )
    # Kathmandu's +05:45 offset means the next local :00 is 04:15 UTC.
    assert next_half_hour_boundary_ms(kathmandu_now, "Asia/Kathmandu") == (
        kathmandu_now + 60_000
    )
    state = DurableRunState(tmp_path / "state.sqlite3")
    source = _Source()
    pump = InteractionInputPump(state, source=source, clock_ms=lambda: now)

    assert pump.scan_due(user_id="daily-user") is None
    assert state.diagnostics("daily-user").queue == ()
    with state._connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_backend_client_uses_closed_claim_wrapper_and_private_headers():
    boundary = 1_787_644_800_000
    batch = _batch("batch-http", first_cursor=7, last_cursor=7, window_end_ms=boundary)
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path == "/v1/interactions/availability":
            return httpx.Response(200, json={"available": True, "next_cursor": 7})
        if request.url.path == "/v1/interactions/claims":
            return httpx.Response(
                200,
                json={
                    "claim": {
                        "claim_id": "interaction-claim-1",
                        "claim_request_id": "claim-request-1",
                        "logical_run_id": "run-1",
                        "claim_kind": "normal",
                        "retry_of_quarantine_id": None,
                        "batch": batch.to_dict(),
                    }
                },
            )
        raise AssertionError(request.url.path)

    client = BackendInteractionClient(
        origin="http://127.0.0.1:8790",
        credential="private-token",
        firebase_uid="daily-user",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert client.availability(boundary_end_ms=boundary).next_cursor == 7
        claimed = client.claim(
            InteractionClaimRequest(
                claim_request_id="claim-request-1",
                logical_run_id="run-1",
                boundary_start_ms=boundary - 1_800_000,
                boundary_end_ms=boundary,
            )
        )
    finally:
        client.close()

    assert claimed.to_json() == batch.to_json()
    assert all(
        item.headers["x-thine-firebase-uid"] == "daily-user" for item in observed
    )
    assert all(
        item.headers["authorization"] == "Bearer private-token" for item in observed
    )


def test_restart_catches_up_once_through_latest_boundary_and_replay_is_idempotent(
    tmp_path: Path,
):
    boundary = int(datetime(2026, 8, 25, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
    now = boundary + 2 * 1_800_000 + 123
    source = _Source()
    source.pending.append(
        _batch("batch-catch-up", first_cursor=1, last_cursor=2, window_end_ms=boundary)
    )
    database = tmp_path / "state.sqlite3"
    state = DurableRunState(database)
    first = InteractionInputPump(state, source=source, clock_ms=lambda: now)

    tick_id = first.scan_due(user_id="daily-user")
    restarted = InteractionInputPump(
        DurableRunState(database), source=source, clock_ms=lambda: now
    )

    assert tick_id is not None
    assert restarted.scan_due(user_id="daily-user") is None
    queue = state.diagnostics("daily-user").queue
    assert len(queue) == 1
    assert queue[0].kind == "p1_interaction"


def test_claim_transport_retry_reuses_range_and_success_consumes_cursor_once(
    tmp_path: Path,
):
    boundary = 1_787_644_800_000
    source = _Source()
    source.pending.append(
        _batch("batch-1", first_cursor=1, last_cursor=1, window_end_ms=boundary)
    )
    source.lose_first_claim_response = True
    runtime = FakeInteractionNoActionRuntime()
    state, pump, coordinator = _coordinator(
        tmp_path / "state.sqlite3",
        source=source,
        runtime=runtime,
        clock=lambda: boundary,
    )
    pump.scan_due(user_id="daily-user")

    first = coordinator.run_next("daily-user")
    second = coordinator.run_next("daily-user")

    assert first is not None and first.status == "input_retry_pending"
    assert second is not None and second.status == "completed"
    assert len(source.claim_requests) == 1
    assert len(runtime.invocations) == 1
    assert len(source.consume_requests) == 1
    attempts = [
        item
        for item in state.diagnostics("daily-user").attempts
        if item.logical_run_id == second.logical_run_id
    ]
    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"


def test_lost_consume_response_retries_ack_suffix_without_inference(tmp_path: Path):
    boundary = 1_787_644_800_000
    source = _Source()
    source.pending.append(
        _batch("batch-1", first_cursor=1, last_cursor=1, window_end_ms=boundary)
    )
    source.lose_first_consume_response = True
    runtime = FakeInteractionNoActionRuntime()
    database = tmp_path / "state.sqlite3"
    _, pump, coordinator = _coordinator(
        database, source=source, runtime=runtime, clock=lambda: boundary
    )
    pump.scan_due(user_id="daily-user")

    pending = coordinator.run_next("daily-user")
    _, _, restarted = _coordinator(
        database, source=source, runtime=runtime, clock=lambda: boundary + 1
    )
    completed = restarted.run_next("daily-user")

    assert pending is not None and pending.status == "awaiting_interaction_ack"
    assert completed is not None and completed.status == "completed"
    assert len(runtime.invocations) == 1
    assert len(source.consume_requests) == 2


def test_third_real_fault_quarantines_range_and_later_boundary_continues(
    tmp_path: Path,
):
    boundary = 1_787_644_800_000
    source = _Source()
    source.pending.extend([
        _batch("poison", first_cursor=1, last_cursor=2, window_end_ms=boundary),
        _batch(
            "later",
            first_cursor=3,
            last_cursor=3,
            window_end_ms=boundary + 1_800_000,
        ),
    ])
    now = [boundary]
    runtime = FakeInteractionNoActionRuntime([
        InvocationOutcome.fault("provider_fault"),
        InvocationOutcome.fault("provider_fault"),
        InvocationOutcome.fault("provider_fault"),
        InvocationOutcome.no_action(),
    ])
    state, pump, coordinator = _coordinator(
        tmp_path / "state.sqlite3",
        source=source,
        runtime=runtime,
        clock=lambda: now[0],
    )
    pump.scan_due(user_id="daily-user")

    assert coordinator.run_next("daily-user").status == "failed_retryable"  # type: ignore[union-attr]
    assert coordinator.run_next("daily-user").status == "failed_retryable"  # type: ignore[union-attr]
    assert coordinator.run_next("daily-user").status == "quarantined"  # type: ignore[union-attr]
    assert len(source.quarantine_requests) == 1
    assert source.quarantine_requests[0].fault_attempts_total == 3
    assert len(runtime.invocations) == 3

    now[0] = boundary + 1_800_000
    assert pump.scan_due(user_id="daily-user") is not None
    completed = coordinator.run_next("daily-user")

    assert completed is not None and completed.status == "completed"
    assert len(runtime.invocations) == 4
    later_input = runtime.invocations[3].prepared_input
    assert later_input.input_gaps[0].payload.source_kind == "interaction"
    assert later_input.input_gaps[0].payload.normal_cursor_advanced is True
    assert [
        item.failure_code for item in state.diagnostics("daily-user").attempts[:3]
    ] == [
        "provider_fault",
        "provider_fault",
        "provider_fault",
    ]

    retry_run_id = "interaction-explicit-retry-1"
    pump.enqueue_explicit_retry(
        user_id="daily-user",
        quarantine_id=source.quarantine_requests[0].quarantine_id,
        retry_run_id=retry_run_id,
        created_at_ms=now[0] + 1,
    )
    retried = coordinator.run_next("daily-user")

    assert retried is not None and retried.logical_run_id == retry_run_id
    assert retried.status == "completed"
    assert len(source.retry_requests) == 1
    inspection = pump.inspect(user_id="daily-user")
    assert inspection["quarantines"][0]["sync_state"] == "synchronized"  # type: ignore[index]
