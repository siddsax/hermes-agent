from __future__ import annotations

import json
from pathlib import Path

import httpx

from thine_harness.action_dispatcher import (
    ActionDispatcher,
    COMMUNICATION_ALLOWANCE_WINDOW_MS,
)
from thine_harness.communications import (
    BackendCommunicationClient,
    COMMUNICATION_SEND_TOOL_NAME,
    COMMUNICATION_SEND_TOOL_SCHEMA,
    COMMUNICATION_STATUS_TOOL_NAME,
    COMMUNICATION_STATUS_TOOL_SCHEMA,
    CommunicationToolBinding,
)
from thine_harness.contracts.notifications import (
    NotificationIntent,
    NotificationOutcome,
    NotificationPermission,
)
from thine_harness.contracts.runtime import Tick
from thine_harness.deferred_tools import DeferredNamespaceCatalog
from thine_harness.interactions import BackgroundRuntimeRouter
from thine_harness.run_coordinator import (
    InvocationContext,
    InvocationControl,
    InvocationOutcome,
)
from thine_harness.run_state import DurableRunState


_VERSION = {"major": 1, "minor": 0}


class _Clock:
    def __init__(self, value: int = 1_000_000) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


class _Backend:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.deliveries: list[NotificationIntent] = []
        self.failures_remaining = 0
        self.outcome = "accepted"
        self.permission_state = "authorized"

    def permission(self) -> NotificationPermission:
        return NotificationPermission.from_dict({
            "schema_version": _VERSION,
            "user_preference": "enabled",
            "os_permission": self.permission_state,
            "last_permission_ask_at_ms": None,
            "last_permission_ask_topic_id": None,
            "observed_at_ms": self.clock(),
            "extensions": {},
        })

    def deliver(self, intent: NotificationIntent) -> NotificationOutcome:
        self.deliveries.append(intent)
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise httpx.ReadTimeout("lost response")
        return NotificationOutcome.from_dict({
            "schema_version": _VERSION,
            "action_id": intent.payload.action_id,
            "communication_kind": "background_message",
            "outcome": self.outcome,
            "provider_correlation_id": (
                "push-provider-1" if self.outcome == "accepted" else None
            ),
            "persisted_message_id": intent.payload.persisted_message_id,
            "permission_state": self.permission_state,
            "allowance_consumed": True,
            "completed_at_ms": self.clock(),
            "extensions": {},
        })


def _tick(tick_id: str, *, queued_at_ms: int) -> Tick:
    return Tick.from_dict({
        "schema_version": _VERSION,
        "tick_id": tick_id,
        "user_id": "daily-user",
        "logical_run_id": f"run:{tick_id}",
        "kind": "p1_transcript",
        "priority": "p1",
        "occurred_at_ms": queued_at_ms,
        "received_at_ms": queued_at_ms,
        "queued_at_ms": queued_at_ms,
        "source_ref": {"kind": "transcript_availability", "id": tick_id},
        "causation_id": None,
        "correlation_id": f"correlation:{tick_id}",
        "attempt_ordinal": 1,
        "lease": None,
        "communication_allowance_snapshot": None,
        "payload": {
            "payload_kind": "transcript_availability",
            "reference_id": tick_id,
        },
        "extensions": {},
    })


def _active_context(
    state: DurableRunState, clock: _Clock, tick_id: str
) -> tuple[InvocationContext, object]:
    tick = _tick(tick_id, queued_at_ms=clock())
    state.enqueue(tick, now_ms=clock())
    lease = state.lease_next("daily-user", owner="test-harness", now_ms=clock())
    assert lease is not None
    state.mark_inference_started(
        user_id="daily-user",
        logical_run_id=f"run:{tick_id}",
        owner="test-harness",
        attempt_id=lease.attempt_id,
        lease_token=lease.lease_token,
        now_ms=clock(),
    )
    return (
        InvocationContext(
            tick=lease.tick,
            attempt_id=lease.attempt_id,
            attempt_ordinal=lease.attempt_ordinal,
            checkpoint=lease.checkpoint,
            acknowledged_receipts=lease.acknowledged_receipts,
        ),
        lease,
    )


def _binding(
    database: Path, clock: _Clock, backend: _Backend
) -> tuple[DurableRunState, ActionDispatcher, CommunicationToolBinding]:
    state = DurableRunState(database)
    dispatcher = ActionDispatcher(state, clock_ms=clock)
    return (
        state,
        dispatcher,
        CommunicationToolBinding(
            dispatcher=dispatcher,
            backend=backend,
            clock_ms=clock,
        ),
    )


def test_proactive_message_persists_once_and_push_body_is_exact(tmp_path: Path):
    clock = _Clock()
    backend = _Backend(clock)
    state, dispatcher, binding = _binding(tmp_path / "state.sqlite3", clock, backend)
    context, _lease = _active_context(state, clock, "tick-message-1")

    with binding.activate(context):
        first = json.loads(binding.send({"message": "Please tag the new speaker."}))
        replay = json.loads(binding.send({"message": "Please tag the new speaker."}))
        conflict = json.loads(binding.send({"message": "A different message."}))

    assert first == replay
    assert first["message_persisted"] is True
    assert first["push_status"] == "accepted"
    assert first["allowance"]["status"] == "consumed"
    assert conflict == {"error_code": "action_intent_conflict", "ok": False}
    assert len(backend.deliveries) == 1
    intent = backend.deliveries[0]
    assert intent.payload.body == "Please tag the new speaker."
    assert intent.payload.navigation_template == "route.chat"
    assert intent.payload.push_required_for_background_message is True
    assert intent.payload.persisted_message_id == first["assistant_message_id"]
    record = dispatcher.record(first["action_id"])
    assert record.state == "succeeded"
    assert record.receipt is not None
    assert (
        len(
            state.receipts_for_run(
                user_id="daily-user", logical_run_id="run:tick-message-1"
            )
        )
        == 1
    )


def test_missing_push_permission_does_not_misreport_message_failure(tmp_path: Path):
    clock = _Clock()
    backend = _Backend(clock)
    backend.outcome = "skipped_permission_disabled"
    backend.permission_state = "denied"
    state, _dispatcher, binding = _binding(tmp_path / "state.sqlite3", clock, backend)
    context, _lease = _active_context(state, clock, "tick-message-permission")

    with binding.activate(context):
        result = json.loads(binding.send({"message": "Your update is ready."}))

    assert result["status"] == "succeeded"
    assert result["message_persisted"] is True
    assert result["push_status"] == "skipped_permission_disabled"
    assert result["notification_permission"] == "denied"
    assert result["allowance"]["remaining"] == 0


def test_terminal_push_failure_still_reports_persisted_message_success(tmp_path: Path):
    clock = _Clock()
    backend = _Backend(clock)
    backend.outcome = "failed_terminal"
    state, _dispatcher, binding = _binding(tmp_path / "state.sqlite3", clock, backend)
    context, _lease = _active_context(state, clock, "tick-message-push-failed")

    with binding.activate(context):
        result = json.loads(binding.send({"message": "The message still exists."}))

    assert result["status"] == "succeeded"
    assert result["message_persisted"] is True
    assert result["push_status"] == "failed_terminal"
    assert result["allowance"]["status"] == "consumed"


def test_ambiguous_transport_reconciles_after_restart_without_model_replay(
    tmp_path: Path,
):
    database = tmp_path / "state.sqlite3"
    clock = _Clock()
    backend = _Backend(clock)
    backend.failures_remaining = 1
    state, dispatcher, binding = _binding(database, clock, backend)
    context, _lease = _active_context(state, clock, "tick-message-recovery")

    with binding.activate(context):
        pending = json.loads(binding.send({"message": "One durable update."}))

    assert pending["status"] == "queued_for_retry"
    assert pending["message_persistence_state"] == ("unknown_pending_reconciliation")
    assert pending["allowance"]["status"] == "reserved"
    action_id = pending["action_id"]
    persisted_intent = backend.deliveries[0].to_json()

    restarted_state = DurableRunState(database)
    restarted_dispatcher = ActionDispatcher(restarted_state, clock_ms=clock)
    restarted_binding = CommunicationToolBinding(
        dispatcher=restarted_dispatcher,
        backend=backend,
        clock_ms=clock,
    )
    assert restarted_binding.reconcile_due("daily-user") == (action_id,)

    assert len(backend.deliveries) == 2
    assert backend.deliveries[1].to_json() == persisted_intent
    recovered = restarted_dispatcher.record(action_id)
    assert recovered.state == "succeeded"
    assert recovered.outcome is not None
    assert (
        recovered.outcome.payload.persisted_message_id
        == pending["assistant_message_id"]
    )


def test_allowance_blocks_another_tick_until_fixed_window_expires(tmp_path: Path):
    clock = _Clock()
    backend = _Backend(clock)
    state, _dispatcher, binding = _binding(tmp_path / "state.sqlite3", clock, backend)
    first_context, first_lease = _active_context(state, clock, "tick-first")
    with binding.activate(first_context):
        first = json.loads(binding.send({"message": "First message."}))
    state.complete(
        user_id="daily-user",
        logical_run_id="run:tick-first",
        owner="test-harness",
        attempt_id=first_lease.attempt_id,
        lease_token=first_lease.lease_token,
        now_ms=clock(),
    )

    clock.value += 1
    second_context, second_lease = _active_context(state, clock, "tick-second")
    with binding.activate(second_context):
        blocked = json.loads(binding.send({"message": "Too soon."}))
    assert blocked["error_code"] == "communication_allowance_unavailable"
    assert blocked["allowance"]["next_eligible_at_ms"] == (
        first["allowance"]["generated_at_ms"] + COMMUNICATION_ALLOWANCE_WINDOW_MS
    )
    state.complete(
        user_id="daily-user",
        logical_run_id="run:tick-second",
        owner="test-harness",
        attempt_id=second_lease.attempt_id,
        lease_token=second_lease.lease_token,
        now_ms=clock(),
    )

    clock.value += COMMUNICATION_ALLOWANCE_WINDOW_MS
    third_context, _third_lease = _active_context(state, clock, "tick-third")
    with binding.activate(third_context):
        allowed = json.loads(binding.send({"message": "Allowed now."}))
    assert allowed["message_persisted"] is True
    assert len(backend.deliveries) == 2


def test_status_is_deferred_and_exposes_permission_and_allowance(tmp_path: Path):
    clock = _Clock()
    backend = _Backend(clock)
    state, _dispatcher, binding = _binding(tmp_path / "state.sqlite3", clock, backend)
    assert json.loads(binding.send({"message": "Not from user chat."})) == {
        "error_code": "no_active_background_tick",
        "ok": False,
    }
    assert backend.deliveries == []
    context, _lease = _active_context(state, clock, "tick-status")
    with binding.activate(context):
        status = json.loads(binding.status({}))

    assert status["ok"] is True
    assert status["allowance"]["remaining"] == 1
    assert status["notification_permission"]["os_permission"] == "authorized"
    assert "automatically requests one push" in status["proactive_message_delivery"]

    from tools.registry import registry

    binding.register(registry_instance=registry)
    try:
        definitions = [
            {"type": "function", "function": COMMUNICATION_STATUS_TOOL_SCHEMA},
            {"type": "function", "function": COMMUNICATION_SEND_TOOL_SCHEMA},
        ]
        catalog = DeferredNamespaceCatalog(definitions, context_length=272_000)
        eager = {tool["function"]["name"] for tool in catalog.model_tool_definitions()}
        assert COMMUNICATION_STATUS_TOOL_NAME not in eager
        assert COMMUNICATION_SEND_TOOL_NAME not in eager
        matches = catalog.search("send a proactive message")
        assert any(
            match["name"] == COMMUNICATION_SEND_TOOL_NAME
            and match["namespace"] == "communications"
            for match in matches
        )
    finally:
        registry.deregister(COMMUNICATION_STATUS_TOOL_NAME)
        registry.deregister(COMMUNICATION_SEND_TOOL_NAME)


def test_backend_client_posts_only_frozen_intent_and_replays_exactly():
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "schema_version": _VERSION,
                    "user_preference": "enabled",
                    "os_permission": "authorized",
                    "last_permission_ask_at_ms": None,
                    "last_permission_ask_topic_id": None,
                    "observed_at_ms": 1,
                    "extensions": {},
                },
            )
        intent = NotificationIntent.from_json(request.content)
        return httpx.Response(
            200,
            json={
                "schema_version": _VERSION,
                "action_id": intent.payload.action_id,
                "communication_kind": "background_message",
                "outcome": "accepted",
                "provider_correlation_id": "push-1",
                "persisted_message_id": intent.payload.persisted_message_id,
                "permission_state": "authorized",
                "allowance_consumed": True,
                "completed_at_ms": 2,
                "extensions": {},
            },
        )

    intent = NotificationIntent.from_dict({
        "schema_version": _VERSION,
        "action_id": "action-1",
        "kind": "background_message_push",
        "title": "Thine",
        "body": "Exact message body.",
        "persisted_message_id": "message-1",
        "navigation_template": "route.chat",
        "push_required_for_background_message": True,
        "created_at_ms": 1,
        "extensions": {},
    })
    client = BackendCommunicationClient(
        origin="http://127.0.0.1:8790",
        credential="private-token",
        user_id="daily-user",
        transport=httpx.MockTransport(handle),
    )
    try:
        assert client.permission().payload.os_permission == "authorized"
        first = client.deliver(intent)
        replay = client.deliver(intent)
    finally:
        client.close()

    assert first.to_json() == replay.to_json()
    assert [request.url.path for request in requests] == [
        "/_local-hermes/private/v1/communications/permission",
        "/_local-hermes/private/v1/communications/background-message",
        "/_local-hermes/private/v1/communications/background-message",
    ]
    posted = json.loads(requests[1].content)
    assert posted == intent.to_dict()
    assert requests[1].headers["x-thine-firebase-uid"] == "daily-user"


def test_background_router_activates_tools_only_for_the_routed_tick(tmp_path: Path):
    clock = _Clock()
    backend = _Backend(clock)
    state, _dispatcher, binding = _binding(tmp_path / "state.sqlite3", clock, backend)
    context, _lease = _active_context(state, clock, "tick-router")

    class _Runtime:
        def invoke(self, routed_context, *, tools, control):
            del tools, control
            assert routed_context is context
            status = json.loads(binding.status({}))
            assert status["ok"] is True
            return InvocationOutcome.no_action()

    router = BackgroundRuntimeRouter(
        {"p1_transcript": _Runtime()},
        context_bindings=(binding,),
    )

    outcome = router.invoke(
        context,
        tools=object(),
        control=InvocationControl(),
    )

    assert outcome.decision_outcome == "no_action"
    assert json.loads(binding.status({})) == {
        "error_code": "no_active_background_tick",
        "ok": False,
    }
