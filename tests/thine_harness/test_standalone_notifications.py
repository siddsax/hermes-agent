from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import httpx

from thine_harness.action_dispatcher import ActionDispatcher
from thine_harness.communications import (
    BackendCommunicationClient,
    CommunicationToolBinding,
)
from thine_harness.contracts.notifications import (
    NotificationIntent,
    NotificationOutcome,
    NotificationPermission,
)
from thine_harness.contracts.runtime import Tick
from thine_harness.deferred_tools import DeferredNamespaceCatalog
from thine_harness.run_coordinator import InvocationContext
from thine_harness.run_state import DurableRunState, LeasedRun, SCHEMA_VERSION
from thine_harness.standalone_notifications import (
    STANDALONE_NOTIFICATION_SEND_TOOL_NAME,
    STANDALONE_NOTIFICATION_SEND_TOOL_SCHEMA,
    STANDALONE_NOTIFICATION_STATUS_TOOL_NAME,
    STANDALONE_NOTIFICATION_STATUS_TOOL_SCHEMA,
    StandaloneNotificationToolBinding,
)


_VERSION = {"major": 1, "minor": 0}


class _Clock:
    def __init__(self, value: int = 1_000_000) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


class _Backend:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.user_preference = "enabled"
        self.os_permission = "authorized"
        self.standalone_deliveries: list[NotificationIntent] = []
        self.background_deliveries: list[NotificationIntent] = []
        self.receipts: dict[str, NotificationOutcome] = {}
        self.lose_next_response = False

    def permission(self) -> NotificationPermission:
        return NotificationPermission.from_dict({
            "schema_version": _VERSION,
            "user_preference": self.user_preference,
            "os_permission": self.os_permission,
            "last_permission_ask_at_ms": 900_000,
            "last_permission_ask_topic_id": "topic-enable-notifications",
            "observed_at_ms": self.clock(),
            "extensions": {},
        })

    def deliver_standalone(self, intent: NotificationIntent) -> NotificationOutcome:
        self.standalone_deliveries.append(intent)
        if self.user_preference == "disabled":
            outcome = "capability_disabled"
        elif self.os_permission not in {"authorized", "provisional"}:
            outcome = "permission_required"
        else:
            outcome = "accepted"
        receipt = NotificationOutcome.from_dict({
            "schema_version": _VERSION,
            "action_id": intent.payload.action_id,
            "communication_kind": "standalone_notification",
            "outcome": outcome,
            "provider_correlation_id": (
                "push-standalone-1" if outcome == "accepted" else None
            ),
            "persisted_message_id": None,
            "permission_state": self.os_permission,
            "allowance_consumed": outcome == "accepted",
            "completed_at_ms": self.clock(),
            "extensions": {},
        })
        self.receipts[str(intent.payload.action_id)] = receipt
        if self.lose_next_response:
            self.lose_next_response = False
            raise httpx.ReadTimeout("lost response after durable backend outcome")
        return receipt

    def standalone_receipt(self, action_id: str) -> NotificationOutcome | None:
        return self.receipts.get(action_id)

    def deliver(self, intent: NotificationIntent) -> NotificationOutcome:
        self.background_deliveries.append(intent)
        return NotificationOutcome.from_dict({
            "schema_version": _VERSION,
            "action_id": intent.payload.action_id,
            "communication_kind": "background_message",
            "outcome": "skipped_permission_disabled",
            "provider_correlation_id": None,
            "persisted_message_id": intent.payload.persisted_message_id,
            "permission_state": self.os_permission,
            "allowance_consumed": True,
            "completed_at_ms": self.clock(),
            "extensions": {},
        })


def _tick(tick_id: str, clock: _Clock) -> Tick:
    return Tick.from_dict({
        "schema_version": _VERSION,
        "tick_id": tick_id,
        "user_id": "daily-user",
        "logical_run_id": f"run:{tick_id}",
        "kind": "p1_transcript",
        "priority": "p1",
        "occurred_at_ms": clock(),
        "received_at_ms": clock(),
        "queued_at_ms": clock(),
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


def _active(
    state: DurableRunState, clock: _Clock, tick_id: str
) -> tuple[InvocationContext, LeasedRun]:
    state.enqueue(_tick(tick_id, clock), now_ms=clock())
    lease = state.lease_next("daily-user", owner="test", now_ms=clock())
    assert lease is not None
    state.mark_inference_started(
        user_id="daily-user",
        logical_run_id=f"run:{tick_id}",
        owner="test",
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


def _bindings(database: Path, clock: _Clock, backend: _Backend):
    state = DurableRunState(database)
    dispatcher = ActionDispatcher(state, clock_ms=clock)
    return (
        state,
        dispatcher,
        StandaloneNotificationToolBinding(
            dispatcher=dispatcher, backend=backend, clock_ms=clock
        ),
        CommunicationToolBinding(
            dispatcher=dispatcher, backend=backend, clock_ms=clock
        ),
    )


def test_accepted_notification_is_distinct_and_consumes_shared_allowance(
    tmp_path: Path,
):
    clock = _Clock()
    backend = _Backend(clock)
    state, dispatcher, notification, message = _bindings(
        tmp_path / "state.sqlite3", clock, backend
    )
    context, _lease = _active(state, clock, "tick-accepted")

    with notification.activate(context), message.activate(context):
        first = json.loads(
            notification.send({
                "title": "Speaker ready",
                "body": "Tap to review the speaker.",
                "navigation_template": "route.speakers-list",
            })
        )
        replay = json.loads(
            notification.send({
                "title": "Speaker ready",
                "body": "Tap to review the speaker.",
                "navigation_template": "route.speakers-list",
            })
        )
        blocked_message = json.loads(
            message.send({"message": "The allowance is already consumed."})
        )

    assert first == replay
    assert first["status"] == "succeeded"
    assert first["allowance_consumed"] is True
    assert first["allowance"]["status"] == "consumed"
    assert blocked_message["error_code"] == "communication_allowance_unavailable"
    assert len(backend.standalone_deliveries) == 1
    intent = backend.standalone_deliveries[0].payload
    assert intent.kind == "standalone_notification"
    assert intent.persisted_message_id is None
    assert intent.navigation_template == "route.speakers-list"
    record = dispatcher.record(first["action_id"])
    assert record.action_kind == "standalone_notification"
    assert record.effect_ordinal == 2
    assert record.assistant_message_id is None


def test_permission_negative_is_durable_does_not_consume_and_prevents_nagging(
    tmp_path: Path,
):
    clock = _Clock()
    backend = _Backend(clock)
    backend.os_permission = "not_determined"
    state, dispatcher, notification, message = _bindings(
        tmp_path / "state.sqlite3", clock, backend
    )
    context, lease = _active(state, clock, "tick-permission")

    with notification.activate(context), message.activate(context):
        denied = json.loads(
            notification.send({
                "title": "Ready",
                "body": "Your update is ready.",
            })
        )
        replay = json.loads(
            notification.send({
                "title": "Ready",
                "body": "Your update is ready.",
            })
        )
        fallback_message = json.loads(
            message.send({"message": "Enable notifications if you want alerts."})
        )

    assert denied == replay
    assert denied["error_code"] == "permission_required"
    assert denied["allowance_consumed"] is False
    assert len(backend.standalone_deliveries) == 1
    assert fallback_message["message_persisted"] is True
    assert dispatcher.record(denied["action_id"]).state == "terminal_negative"
    state.complete(
        user_id="daily-user",
        logical_run_id="run:tick-permission",
        owner="test",
        attempt_id=lease.attempt_id,
        lease_token=lease.lease_token,
        now_ms=clock(),
    )

    clock.value += 1
    context_2, _lease_2 = _active(state, clock, "tick-permission-repeat")
    with notification.activate(context_2):
        status = json.loads(notification.status({}))["standalone_notification"]
        blocked = json.loads(
            notification.send({
                "title": "Ready again",
                "body": "This must not nag.",
            })
        )

    assert status["last_request"]["outcome"] == "permission_required"
    assert status["last_permission_request_at_ms"] == 900_000
    assert status["permission_request_route"] == {
        "activation": "user_tap_only",
        "agent_may_force_navigation": False,
        "chat_markdown_link": "thine:///permissions/other",
    }
    assert blocked["error_code"] == "permission_required"
    assert (
        blocked["durable_repeat_guard"][
            "do_not_repeat_until_policy_or_permission_changes"
        ]
        is True
    )
    assert len(backend.standalone_deliveries) == 1


def test_ambiguous_delivery_uses_receipt_lookup_after_restart(tmp_path: Path):
    database = tmp_path / "state.sqlite3"
    clock = _Clock()
    backend = _Backend(clock)
    backend.lose_next_response = True
    state, _dispatcher, notification, _message = _bindings(database, clock, backend)
    context, _lease = _active(state, clock, "tick-reconcile")
    with notification.activate(context):
        pending = json.loads(
            notification.send({
                "title": "Durable",
                "body": "This delivery has a lost response.",
            })
        )
    assert pending["status"] == "queued_for_retry"

    restarted_state = DurableRunState(database)
    restarted_dispatcher = ActionDispatcher(restarted_state, clock_ms=clock)
    restarted = StandaloneNotificationToolBinding(
        dispatcher=restarted_dispatcher, backend=backend, clock_ms=clock
    )
    assert restarted.reconcile_due("daily-user") == (pending["action_id"],)
    assert len(backend.standalone_deliveries) == 1
    assert restarted_dispatcher.record(pending["action_id"]).state == "succeeded"


def test_user_disabled_is_terminal_until_policy_changes(tmp_path: Path):
    clock = _Clock()
    backend = _Backend(clock)
    backend.user_preference = "disabled"
    state, _dispatcher, notification, _message = _bindings(
        tmp_path / "state.sqlite3", clock, backend
    )
    context, lease = _active(state, clock, "tick-disabled")
    with notification.activate(context):
        disabled = json.loads(
            notification.send({"title": "Disabled", "body": "Not delivered."})
        )
    assert disabled["error_code"] == "capability_disabled"
    assert disabled["allowance_consumed"] is False
    assert disabled["allowance"]["status"] == "available"
    state.complete(
        user_id="daily-user",
        logical_run_id="run:tick-disabled",
        owner="test",
        attempt_id=lease.attempt_id,
        lease_token=lease.lease_token,
        now_ms=clock(),
    )

    backend.user_preference = "enabled"
    clock.value += 1
    context_2, _lease_2 = _active(state, clock, "tick-enabled")
    with notification.activate(context_2):
        enabled = json.loads(
            notification.send({"title": "Enabled", "body": "Delivered now."})
        )
    assert enabled["status"] == "succeeded"
    assert len(backend.standalone_deliveries) == 2


def test_notification_tools_are_deferred_in_communications_namespace(tmp_path: Path):
    clock = _Clock()
    backend = _Backend(clock)
    _state, _dispatcher, notification, _message = _bindings(
        tmp_path / "state.sqlite3", clock, backend
    )
    from tools.registry import registry

    notification.register(registry_instance=registry)
    try:
        definitions = [
            {
                "type": "function",
                "function": STANDALONE_NOTIFICATION_STATUS_TOOL_SCHEMA,
            },
            {"type": "function", "function": STANDALONE_NOTIFICATION_SEND_TOOL_SCHEMA},
        ]
        catalog = DeferredNamespaceCatalog(definitions, context_length=272_000)
        eager = {tool["function"]["name"] for tool in catalog.model_tool_definitions()}
        assert STANDALONE_NOTIFICATION_STATUS_TOOL_NAME not in eager
        assert STANDALONE_NOTIFICATION_SEND_TOOL_NAME not in eager
        matches = catalog.search("send a standalone notification")
        assert any(
            match["name"] == STANDALONE_NOTIFICATION_SEND_TOOL_NAME
            and match["namespace"] == "communications"
            for match in matches
        )
    finally:
        registry.deregister(STANDALONE_NOTIFICATION_STATUS_TOOL_NAME)
        registry.deregister(STANDALONE_NOTIFICATION_SEND_TOOL_NAME)


def test_backend_client_uses_frozen_standalone_routes_and_receipt_lookup():
    requests: list[httpx.Request] = []
    outcomes: dict[str, dict[str, object]] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            intent = NotificationIntent.from_json(request.content)
            outcome: dict[str, object] = {
                "schema_version": _VERSION,
                "action_id": intent.payload.action_id,
                "communication_kind": "standalone_notification",
                "outcome": "accepted",
                "provider_correlation_id": "push-1",
                "persisted_message_id": None,
                "permission_state": "authorized",
                "allowance_consumed": True,
                "completed_at_ms": 2,
                "extensions": {},
            }
            outcomes[str(intent.payload.action_id)] = outcome
            return httpx.Response(200, json=outcome)
        action_id = request.url.path.rsplit("/", 1)[-1]
        outcome = outcomes.get(action_id)
        return httpx.Response(404 if outcome is None else 200, json=outcome or {})

    intent = NotificationIntent.from_dict({
        "schema_version": _VERSION,
        "action_id": "action-standalone-1",
        "kind": "standalone_notification",
        "title": "Title",
        "body": "Body",
        "persisted_message_id": None,
        "navigation_template": "route.profile",
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
        assert client.standalone_receipt("missing") is None
        delivered = client.deliver_standalone(intent)
        replay = client.standalone_receipt("action-standalone-1")
    finally:
        client.close()

    assert replay is not None
    assert delivered.to_json() == replay.to_json()
    assert [request.url.path for request in requests] == [
        "/v1/communications/standalone-notification/missing",
        "/v1/communications/standalone-notification",
        "/v1/communications/standalone-notification/action-standalone-1",
    ]
    assert json.loads(requests[1].content) == intent.to_dict()
    for request in requests:
        assert request.headers["authorization"] == "Bearer private-token"
        assert request.headers["x-thine-firebase-uid"] == "daily-user"
        assert request.headers["x-request-id"]
    assert len({request.headers["x-request-id"] for request in requests}) == 3


def test_schema_upgrade_preserves_existing_proactive_action(tmp_path: Path):
    database = tmp_path / "v7.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE queue_items (
            logical_run_id TEXT PRIMARY KEY
        );
        CREATE TABLE checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            logical_run_id TEXT NOT NULL,
            cause TEXT NOT NULL,
            remaining_work TEXT NOT NULL,
            completed_receipt_ids_json TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
        );
        CREATE TABLE communication_actions (
            action_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            logical_run_id TEXT NOT NULL,
            tick_id TEXT NOT NULL,
            effect_ordinal INTEGER NOT NULL CHECK (effect_ordinal = 1),
            intent_fingerprint TEXT NOT NULL,
            assistant_message_id TEXT NOT NULL,
            message_text TEXT NOT NULL,
            action_intent_json TEXT NOT NULL,
            notification_intent_json TEXT NOT NULL,
            state TEXT NOT NULL,
            outcome_json TEXT,
            receipt_json TEXT,
            last_error_code TEXT,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            UNIQUE(user_id, logical_run_id, effect_ordinal),
            UNIQUE(user_id, assistant_message_id),
            FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
        );
        CREATE INDEX communication_actions_due
            ON communication_actions(user_id, state, updated_at_ms);
        CREATE TABLE communication_allowance_ledger (
            action_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            state TEXT NOT NULL,
            reserved_at_ms INTEGER NOT NULL,
            consumed_at_ms INTEGER,
            released_at_ms INTEGER,
            FOREIGN KEY(action_id) REFERENCES communication_actions(action_id)
        );
        CREATE INDEX communication_allowance_by_user
            ON communication_allowance_ledger(user_id, state, reserved_at_ms DESC);
        CREATE TABLE communication_permission_observations (
            user_id TEXT PRIMARY KEY,
            permission_json TEXT NOT NULL,
            observed_at_ms INTEGER NOT NULL
        );
        INSERT INTO queue_items VALUES ('run-old');
        INSERT INTO communication_actions VALUES (
            'tick-old:effect:1', 'daily-user', 'run-old', 'tick-old', 1,
            'fingerprint', 'message-old', 'Old message', '{}', '{}',
            'succeeded', NULL, NULL, NULL, 1, 2
        );
        INSERT INTO communication_allowance_ledger VALUES (
            'tick-old:effect:1', 'daily-user', 'consumed', 1, 2, NULL
        );
        PRAGMA user_version = 7;
        """
    )
    connection.close()

    DurableRunState(database)

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT * FROM communication_actions WHERE action_id = ?",
        ("tick-old:effect:1",),
    ).fetchone()
    assert row is not None
    assert row["action_kind"] == "background_message"
    assert row["title"] == "Thine"
    assert row["navigation_template"] == "route.chat"
    assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    connection.close()
