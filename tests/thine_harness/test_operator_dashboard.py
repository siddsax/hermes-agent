from __future__ import annotations

from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import threading

import pytest
from fastapi.testclient import TestClient

from thine_harness.action_dispatcher import ActionDispatcher
from thine_harness.communications import (
    BackendCommunicationClient,
    PushRegistrationStatus,
)
from thine_harness.contracts.notifications import (
    NotificationIntent,
    NotificationPermission,
)
from thine_harness.contracts.runtime import Tick
from thine_harness.home_state import HomeStateProjector
from thine_harness.maintenance import AuthoritativeStateReader, RetentionResetService
from thine_harness.operator_dashboard import (
    OperatorDashboard,
    OperatorDashboardConfigurationError,
    OperatorDashboardControl,
    OperatorDashboardReadService,
    PushRegistrationReadPort,
    load_operator_dashboard_config,
)
from thine_harness.operator_dashboard_server import create_operator_dashboard_app
from thine_harness.private_server import build_product_p0_controller
from thine_harness.private_topology import (
    load_backend_private_config,
    load_private_service_config,
)
from thine_harness.run_state import DurableRunState
from thine_harness.schedules import OneShotScheduleService
from thine_harness.topics_preferences import TopicPreferenceService


USER = "daily-user"
NOW = 1_900_000_000_000


def _dashboard(
    tmp_path: Path,
    *,
    harness_stopped: bool = False,
    run_diagnostics: Callable[[str], object] | None = None,
    live_run: Callable[[str], dict[str, object] | None] | None = None,
    communications: PushRegistrationReadPort | None = None,
    state: DurableRunState | None = None,
    home: HomeStateProjector | None = None,
    retry_quarantine: Callable[[str, str], str] | None = None,
    retry_action: Callable[[str], dict[str, object]] | None = None,
    wake_harness: Callable[[], None] | None = None,
) -> OperatorDashboard:
    tmp_path.mkdir(parents=True, exist_ok=True)
    state = state or DurableRunState(tmp_path / "run-state.sqlite3")
    home = home or HomeStateProjector(
        tmp_path / "home-state.sqlite3", clock_ms=lambda: NOW
    )
    reader = AuthoritativeStateReader(state, home=home, clock_ms=lambda: NOW)
    maintenance = RetentionResetService(state, home=home, clock_ms=lambda: NOW)
    schedules = OneShotScheduleService(state, clock_ms=lambda: NOW)
    return OperatorDashboard(
        reads=OperatorDashboardReadService(
            reader,
            state=state,
            actions=ActionDispatcher(state, clock_ms=lambda: NOW),
            topics=TopicPreferenceService(state, clock_ms=lambda: NOW),
            schedules=schedules,
            maintenance=maintenance,
            communications=communications,
            run_diagnostics=run_diagnostics,
            live_run=live_run,
            clock_ms=lambda: NOW,
        ),
        controls=OperatorDashboardControl(
            user_id=USER,
            home=home,
            schedules=schedules,
            maintenance=maintenance,
            retry_quarantine=retry_quarantine,
            retry_action=retry_action,
            wake_harness=wake_harness,
            harness_stopped=lambda: harness_stopped,
        ),
        user_id=USER,
    )


def _tick(index: int) -> Tick:
    tick_id = f"diagnostic-{index}"
    return Tick.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "tick_id": tick_id,
        "user_id": USER,
        "logical_run_id": f"run:{tick_id}",
        "kind": "p1_transcript",
        "priority": "p1",
        "occurred_at_ms": index + 1,
        "received_at_ms": index + 1,
        "queued_at_ms": index + 1,
        "source_ref": {"kind": "transcript_availability", "id": tick_id},
        "causation_id": None,
        "correlation_id": tick_id,
        "attempt_ordinal": 1,
        "lease": None,
        "communication_allowance_snapshot": None,
        "payload": {
            "payload_kind": "transcript_availability",
            "reference_id": tick_id,
        },
        "extensions": {},
    })


def _operator_client(dashboard: OperatorDashboard) -> tuple[TestClient, str]:
    client = TestClient(
        create_operator_dashboard_app(dashboard), client=("127.0.0.1", 50000)
    )
    body = client.get("/").text
    match = re.search(r"const TOKEN=(\"[^\"]+\")", body)
    assert match is not None
    return client, str(json.loads(match.group(1)))


def _confirmed_http_control(
    client: TestClient, token: str, command: dict[str, object]
) -> dict[str, object]:
    headers = {"X-Operator-Token": token}
    preview = client.post("/api/controls/preview", headers=headers, json=command)
    assert preview.status_code == 200
    preview_body = preview.json()
    assert preview_body["requires_confirmation"] is True
    executed = client.post(
        "/api/controls/execute",
        headers=headers,
        json=preview_body["execute_payload"],
    )
    assert executed.status_code == 200
    return dict(executed.json())


def _seed_schedule(state: DurableRunState, schedule_id: str, *, due_at_ms: int) -> None:
    with state._transaction() as connection:
        connection.execute(
            """
            INSERT INTO one_shot_schedules (
                schedule_id, user_id, due_at_ms, timezone_name, due_time_input,
                created_at_ms, updated_at_ms, status, reason, creator_tick_id,
                originating_run_id, originating_action_id, intent_fingerprint
            ) VALUES (?, ?, ?, 'Asia/Kolkata', '2030-03-17T12:00:00+05:30',
                      ?, ?, 'active', 'Original reason', 'tick-schedule',
                      'run-schedule', ?, ?)
            """,
            (
                schedule_id,
                USER,
                due_at_ms,
                NOW,
                NOW,
                f"action:{schedule_id}",
                f"intent:{schedule_id}",
            ),
        )


def test_config_accepts_only_a_loopback_literal() -> None:
    defaults = load_operator_dashboard_config({})
    assert defaults.enabled is False
    assert defaults.port == 8792
    config = load_operator_dashboard_config({
        "thine_harness": {
            "operator_dashboard": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 8792,
            }
        }
    })
    assert config.enabled is True
    assert config.host == "127.0.0.1"
    assert config.port == 8792

    with pytest.raises(OperatorDashboardConfigurationError):
        load_operator_dashboard_config({
            "thine_harness": {
                "operator_dashboard": {
                    "enabled": True,
                    "host": "0.0.0.0",
                    "port": 8792,
                }
            }
        })


def test_snapshot_has_bounded_owner_sourced_panels_and_explicit_gaps(
    tmp_path: Path,
) -> None:
    snapshot = _dashboard(tmp_path).snapshot()

    assert snapshot["authoritative"] is False
    assert snapshot["binding"] == "mac_loopback_only"
    assert snapshot["limit"] == 50
    expected = {
        "queue",
        "current_run",
        "transcripts",
        "working_memory",
        "home",
        "interactions",
        "speakers",
        "communications",
        "schedules",
        "topics_preferences",
        "retention_reset",
        "debug_timeline",
    }
    assert expected.issubset(snapshot["panels"])
    for panel in snapshot["panels"].values():
        assert panel["source"]
        assert panel["generated_at_ms"] == NOW
        assert panel["status"] in {"ok", "partial", "unavailable", "error"}
        assert "error" in panel
        assert panel["freshness"]["snapshot_generated_at_ms"] == NOW
        assert panel["freshness"]["read_at_ms"] == NOW
        assert "owner_observed_at_ms" in panel["freshness"]
        assert "components" in panel["freshness"]
    assert (
        snapshot["panels"]["home"]["data"]["last_mobile_ack"]["status"] == "unavailable"
    )
    assert (
        snapshot["panels"]["transcripts"]["data"]["canonical_transcripts"]["status"]
        == "unavailable"
    )
    assert "hermes_retained_mapping_inputs" in snapshot["panels"]["speakers"]["data"]
    assert snapshot["panels"]["working_memory"]["data"]["restore_available"] is False
    communications = snapshot["panels"]["communications"]
    assert communications["status"] == "partial"
    assert communications["data"]["push_registration"] == {
        "status": "unavailable",
        "owner": "thine.backend.push_transport",
        "reason": "standalone dashboard has no backend communication client",
    }


def test_product_attached_dashboard_reads_redacted_push_registration_status(
    tmp_path: Path,
) -> None:
    class _Communications:
        def permission(self) -> NotificationPermission:
            return NotificationPermission.from_dict({
                "schema_version": {"major": 1, "minor": 0},
                "user_preference": "enabled",
                "os_permission": "authorized",
                "last_permission_ask_at_ms": None,
                "last_permission_ask_topic_id": None,
                "observed_at_ms": NOW - 75,
                "extensions": {},
            })

        def push_registration_status(self) -> PushRegistrationStatus:
            return PushRegistrationStatus.from_dict({
                "has_registration": True,
                "registration_count": 2,
                "last_observed_at_ms": NOW - 50,
            })

    panels = _dashboard(tmp_path, communications=_Communications()).snapshot()["panels"]

    assert panels["communications"]["status"] == "ok"
    assert panels["communications"]["data"]["permission"] == {
        "schema_version": {"major": 1, "minor": 0},
        "user_preference": "enabled",
        "os_permission": "authorized",
        "last_permission_ask_at_ms": None,
        "last_permission_ask_topic_id": None,
        "observed_at_ms": NOW - 75,
        "extensions": {},
    }
    assert panels["communications"]["data"]["push_registration"] == {
        "has_registration": True,
        "registration_count": 2,
        "last_observed_at_ms": NOW - 50,
    }
    freshness = panels["communications"]["freshness"]
    assert freshness["read_at_ms"] == NOW
    assert freshness["owner_observed_at_ms"] == NOW - 50
    assert freshness["components"]["permission"]["owner_observed_at_ms"] == NOW - 75
    assert (
        freshness["components"]["push_registration"]["owner_observed_at_ms"] == NOW - 50
    )


def test_push_registration_failure_is_communications_panel_scoped_and_redacted(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingCommunications:
        def permission(self) -> NotificationPermission:
            return NotificationPermission.from_dict({
                "schema_version": {"major": 1, "minor": 0},
                "user_preference": "enabled",
                "os_permission": "authorized",
                "last_permission_ask_at_ms": None,
                "last_permission_ask_topic_id": None,
                "observed_at_ms": NOW - 75,
                "extensions": {},
            })

        def push_registration_status(self) -> PushRegistrationStatus:
            raise RuntimeError("private-device-token-must-not-leak")

    with caplog.at_level(logging.ERROR, logger="thine_harness.operator_dashboard"):
        panels = _dashboard(
            tmp_path, communications=_FailingCommunications()
        ).snapshot()["panels"]

    communications = panels["communications"]
    assert communications["status"] == "partial"
    assert communications["error"] == "one_or_more_owner_values_unavailable"
    assert communications["data"]["permission"]["os_permission"] == ("authorized")
    assert communications["data"]["push_registration"] == {
        "status": "error",
        "owner": "thine.backend.push_transport",
        "error": "owner_read_failed:RuntimeError",
    }
    assert "actions" in communications["data"]
    assert communications["freshness"]["owner_observed_at_ms"] == NOW - 75
    assert panels["schedules"]["status"] == "ok"
    assert "private-device-token-must-not-leak" not in str(communications)


def test_product_controller_projects_registration_from_real_loopback_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, str | None]] = []
    standalone_outcomes: dict[str, dict[str, object]] = {}

    class _BackendHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._capture("GET")
            if self.path == "/v1/communications/push-registration":
                self._respond(
                    {
                        "has_registration": True,
                        "registration_count": 3,
                        "last_observed_at_ms": NOW - 25,
                    },
                    status=200,
                )
                return
            if self.path == "/v1/communications/permission":
                self._respond({
                    "schema_version": {"major": 1, "minor": 0},
                    "user_preference": "enabled",
                    "os_permission": "authorized",
                    "last_permission_ask_at_ms": None,
                    "last_permission_ask_topic_id": None,
                    "observed_at_ms": NOW,
                    "extensions": {},
                })
                return
            receipt_prefix = "/v1/communications/standalone-notification/"
            if self.path.startswith(receipt_prefix):
                action_id = self.path.removeprefix(receipt_prefix)
                outcome = standalone_outcomes.get(action_id)
                self._respond(outcome or {}, status=200 if outcome is not None else 404)
                return
            self._respond({"error": "route_not_found"}, status=404)

        def do_POST(self) -> None:
            self._capture("POST")
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length)) if length else None
            if self.path == "/v1/communications/background-message":
                assert isinstance(payload, dict)
                self._respond({
                    "schema_version": {"major": 1, "minor": 0},
                    "action_id": payload["action_id"],
                    "communication_kind": "background_message",
                    "outcome": "accepted",
                    "provider_correlation_id": "push-background",
                    "persisted_message_id": payload["persisted_message_id"],
                    "permission_state": "authorized",
                    "allowance_consumed": True,
                    "completed_at_ms": NOW,
                    "extensions": {},
                })
                return
            if self.path == "/v1/communications/standalone-notification":
                assert isinstance(payload, dict)
                action_id = str(payload["action_id"])
                outcome: dict[str, object] = {
                    "schema_version": {"major": 1, "minor": 0},
                    "action_id": action_id,
                    "communication_kind": "standalone_notification",
                    "outcome": "accepted",
                    "provider_correlation_id": "push-standalone",
                    "persisted_message_id": None,
                    "permission_state": "authorized",
                    "allowance_consumed": True,
                    "completed_at_ms": NOW,
                    "extensions": {},
                }
                standalone_outcomes[action_id] = outcome
                self._respond(outcome)
                return
            # The product controller's background scanner asks the same local
            # backend whether a speaker event is waiting. Empty is valid.
            if self.path == "/v1/speaker-mappings/next":
                self._respond(None)
                return
            self._respond({"error": "route_not_found"}, status=404)

        def _capture(self, method: str) -> None:
            requests.append({
                "method": method,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "user_id": self.headers.get("X-Thine-Firebase-UID"),
                "request_id": self.headers.get("X-Request-ID"),
            })

        def _respond(self, payload: object, *, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            del args

    backend = ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config = {
        "thine_harness": {
            "private_service": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 8789,
                "firebase_uid": USER,
                "request_timeout_seconds": 2,
                "credential": {"env": "HERMES_CONTROL_TOKEN", "file": ""},
            },
            "private_backend": {
                "origin": f"http://127.0.0.1:{backend.server_address[1]}",
                "request_timeout_seconds": 2,
                "credential": {"env": "BACKEND_PRIVATE_TOKEN", "file": ""},
            },
        }
    }
    private_config = load_private_service_config(
        config,
        environ={"HERMES_CONTROL_TOKEN": "private-control-token"},
    )
    backend_config = load_backend_private_config(
        config,
        environ={"BACKEND_PRIVATE_TOKEN": "backend-private-token"},
    )
    controller = build_product_p0_controller(
        private_config=private_config,
        backend_config=backend_config,
        database_path=hermes_home / "thine-harness" / "run-state.sqlite3",
        runtime_factory=lambda: object(),  # Not loaded without an admitted Tick.
    )
    client = BackendCommunicationClient(
        origin=backend_config.origin,
        credential=backend_config.credential,
        user_id=backend_config.firebase_uid,
        timeout_seconds=backend_config.request_timeout_seconds,
    )
    background_intent = NotificationIntent.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "action_id": "background-action",
        "kind": "background_message_push",
        "title": "Thine",
        "body": "Background body",
        "persisted_message_id": "message-1",
        "navigation_template": "route.chat",
        "push_required_for_background_message": True,
        "created_at_ms": NOW,
        "extensions": {},
    })
    standalone_intent = NotificationIntent.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "action_id": "standalone-action",
        "kind": "standalone_notification",
        "title": "Thine",
        "body": "Standalone body",
        "persisted_message_id": None,
        "navigation_template": "route.profile",
        "push_required_for_background_message": True,
        "created_at_ms": NOW,
        "extensions": {},
    })
    try:
        dashboard = controller.operator_dashboard
        assert isinstance(dashboard, OperatorDashboard)
        response = TestClient(
            create_operator_dashboard_app(dashboard),
            client=("127.0.0.1", 50000),
        ).get("/api/snapshot")
        permission = client.permission()
        background = client.deliver(background_intent)
        missing_receipt = client.standalone_receipt("missing")
        standalone = client.deliver_standalone(standalone_intent)
        receipt = client.standalone_receipt("standalone-action")
    finally:
        client.close()
        controller.close()
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=2)

    assert permission.payload.os_permission == "authorized"
    assert background.payload.action_id == "background-action"
    assert missing_receipt is None
    assert receipt is not None
    assert standalone.to_json() == receipt.to_json()

    assert response.status_code == 200
    communications = response.json()["panels"]["communications"]
    assert communications["status"] == "ok"
    assert communications["data"]["push_registration"] == {
        "has_registration": True,
        "registration_count": 3,
        "last_observed_at_ms": NOW - 25,
    }
    communication_requests = [
        request
        for request in requests
        if str(request["path"]).startswith("/v1/communications/")
    ]
    assert [request["path"] for request in communication_requests] == [
        "/v1/communications/permission",
        "/v1/communications/push-registration",
        "/v1/communications/permission",
        "/v1/communications/background-message",
        "/v1/communications/standalone-notification/missing",
        "/v1/communications/standalone-notification",
        "/v1/communications/standalone-notification/standalone-action",
    ]
    assert [request["method"] for request in communication_requests] == [
        "GET",
        "GET",
        "GET",
        "POST",
        "GET",
        "POST",
        "GET",
    ]
    for request in communication_requests:
        assert request["authorization"] == "Bearer backend-private-token"
        assert request["user_id"] == USER
        assert request["request_id"]
    assert len({request["request_id"] for request in communication_requests}) == 7


def test_runtime_config_comes_from_coordinator_diagnostics(tmp_path: Path) -> None:
    def diagnostics(_user_id: str) -> dict[str, object]:
        return {
            "queue": [],
            "leases": [],
            "attempts": [],
            "checkpoints": [],
            "receipts": [
                {"logical_run_id": "run-live"},
                {"logical_run_id": "run-other"},
                {"logical_run_id": "run-live"},
            ],
            "quarantines": [],
            "runtime": {
                "provider": "test-provider",
                "model": "non-default-model",
                "api_mode": "test-mode",
                "reasoning_effort": "high",
                "context_window_tokens": 1234,
                "tool_search_enabled": False,
                "tool_search_listing": True,
                "tool_namespaces": ["test-tools"],
            },
        }

    panel = _dashboard(
        tmp_path,
        run_diagnostics=diagnostics,
        live_run=lambda _user_id: {
            "logical_run_id": "run-live",
            "interruption_request": {"requested": False, "reason": None},
        },
    ).snapshot()["panels"]["current_run"]
    assert panel["data"]["runtime"]["model"] == "non-default-model"
    assert panel["data"]["runtime"]["reasoning_effort"] == "high"
    assert panel["data"]["active"]["completed_tool_receipts"] == 0
    assert panel["data"]["active"]["interruption_request"]["requested"] is False


def test_dashboard_queue_reads_newest_bounded_rows_and_targets_active_receipts(
    tmp_path: Path,
) -> None:
    state = DurableRunState(tmp_path / "state.sqlite3")
    home = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: NOW)
    for index in range(55):
        state.enqueue(_tick(index), now_ms=index + 1)
    with state._transaction() as connection:
        for index in range(55):
            logical_run_id = f"run:diagnostic-{index}"
            connection.execute(
                """
                INSERT INTO checkpoints (
                    checkpoint_id, user_id, logical_run_id, cause,
                    remaining_work, completed_receipt_ids_json,
                    updated_at_ms, original_input, context_messages_json,
                    completed_tool_results_json,
                    successful_action_receipts_json,
                    partial_visible_assistant_output
                ) VALUES (?, ?, ?, 'continuation', 'resume', '[]', ?, '',
                          '[]', '[]', '[]', '')
                """,
                (f"checkpoint-{index}", USER, logical_run_id, index + 1),
            )
            connection.execute(
                """
                INSERT INTO tool_receipts (
                    receipt_id, user_id, logical_run_id, action_id,
                    intent_fingerprint, provider_reference, result_json,
                    acknowledged_at_ms
                ) VALUES (?, ?, ?, ?, ?, 'provider', '{}', ?)
                """,
                (
                    f"receipt-{index}",
                    USER,
                    logical_run_id,
                    f"action-{index}",
                    f"intent-{index}",
                    index + 1,
                ),
            )

    def coordinator_diagnostics(_user_id: str) -> dict[str, object]:
        return {
            "queue": [],
            "leases": [],
            "attempts": [],
            "checkpoints": [],
            "receipts": [],
            "quarantines": [],
            "runtime": {"model": "gpt-5.6-sol"},
        }

    panels = _dashboard(
        tmp_path,
        state=state,
        home=home,
        run_diagnostics=coordinator_diagnostics,
        live_run=lambda _user_id: {"logical_run_id": "run:diagnostic-0"},
    ).snapshot()["panels"]

    assert [
        item["checkpoint_id"] for item in panels["queue"]["data"]["checkpoints"]
    ] == [f"checkpoint-{index}" for index in range(54, 4, -1)]
    assert len(panels["queue"]["data"]["tool_receipts"]) == 50
    assert panels["current_run"]["data"]["active"]["completed_tool_receipts"] == 1


def test_owner_failure_is_isolated_and_freshness_is_truthful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_transcripts(
        _self: DurableRunState, _user_id: str, *, limit: int = 50
    ) -> tuple[dict[str, object], ...]:
        del limit
        raise RuntimeError("private detail must not leak")

    monkeypatch.setattr(DurableRunState, "recent_transcript_runs", fail_transcripts)
    with caplog.at_level(logging.ERROR, logger="thine_harness.operator_dashboard"):
        panels = _dashboard(tmp_path).snapshot()["panels"]

    assert panels["transcripts"]["status"] == "error"
    assert panels["transcripts"]["error"] == "owner_read_failed:RuntimeError"
    assert panels["transcripts"]["freshness"]["status"] == "unknown"
    assert panels["transcripts"]["freshness"]["read_at_ms"] == NOW
    assert panels["transcripts"]["freshness"]["owner_observed_at_ms"] is None
    assert panels["transcripts"]["freshness"]["observed_at_ms"] is None
    assert panels["working_memory"]["status"] == "ok"
    assert panels["working_memory"]["freshness"]["status"] == "read"
    assert panels["working_memory"]["freshness"]["read_at_ms"] == NOW
    assert panels["working_memory"]["freshness"]["owner_observed_at_ms"] is None
    failures = [
        record
        for record in caplog.records
        if record.getMessage() == "operator dashboard owner read failed"
    ]
    assert len(failures) == 1
    assert failures[0].dashboard_source == "hermes.run_state.recent_transcript_runs"
    assert failures[0].exc_info is not None


def test_reset_requires_preview_exact_confirmation_and_harness_stop(
    tmp_path: Path,
) -> None:
    running_dashboard = _dashboard(tmp_path)
    dashboard = _dashboard(tmp_path / "stopped", harness_stopped=True)
    preview = dashboard.preview_control({
        "action": "reset",
        "scope": "working_memory_topics",
    })
    assert preview["requires_confirmation"] is True
    assert preview["execute_payload"]["confirmation"]

    running_preview = running_dashboard.preview_control({
        "action": "reset",
        "scope": "working_memory_topics",
    })
    refused = running_dashboard.execute_control(running_preview["execute_payload"])
    assert refused["status"] == "refused_live_work"

    completed = dashboard.execute_control(preview["execute_payload"])
    assert completed["status"] == "completed"


def test_http_controls_edit_cancel_and_run_now_schedules_through_owner(
    tmp_path: Path,
) -> None:
    state = DurableRunState(tmp_path / "state.sqlite3")
    home = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: NOW)
    _seed_schedule(state, "schedule-edit", due_at_ms=NOW + 100_000)
    _seed_schedule(state, "schedule-cancel", due_at_ms=NOW + 200_000)
    _seed_schedule(state, "schedule-now", due_at_ms=NOW + 300_000)
    wake_count = 0

    def wake() -> None:
        nonlocal wake_count
        wake_count += 1

    client, token = _operator_client(
        _dashboard(
            tmp_path,
            state=state,
            home=home,
            wake_harness=wake,
        )
    )

    edited = _confirmed_http_control(
        client,
        token,
        {
            "action": "schedule_edit",
            "schedule_id": "schedule-edit",
            "due_at": "2031-04-05T09:45:00+05:30",
            "timezone": "Asia/Kolkata",
            "reason": "Inspect the local daily-driver state.",
        },
    )
    cancelled = _confirmed_http_control(
        client,
        token,
        {"action": "schedule_cancel", "schedule_id": "schedule-cancel"},
    )
    run_now = _confirmed_http_control(
        client,
        token,
        {"action": "schedule_run_now", "schedule_id": "schedule-now"},
    )

    assert edited["schedule"]["schedule"]["reason"] == (
        "Inspect the local daily-driver state."
    )
    assert edited["schedule"]["schedule"]["status"] == "active"
    assert cancelled["schedule"]["schedule"]["status"] == "cancelled"
    assert run_now["schedule"]["schedule"]["status"] == "enqueued"
    assert wake_count == 3


def test_http_controls_retry_quarantine_and_action_through_attached_owners(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []
    wake_count = 0

    def retry_quarantine(source_kind: str, quarantine_id: str) -> str:
        calls.append(("quarantine", source_kind, quarantine_id))
        return "run:explicit-retry"

    def retry_action(action_id: str) -> dict[str, object]:
        calls.append(("action", action_id))
        return {"action_id": action_id, "state": "completed"}

    def wake() -> None:
        nonlocal wake_count
        wake_count += 1

    client, token = _operator_client(
        _dashboard(
            tmp_path,
            retry_quarantine=retry_quarantine,
            retry_action=retry_action,
            wake_harness=wake,
        )
    )

    quarantine = _confirmed_http_control(
        client,
        token,
        {
            "action": "retry_quarantined",
            "source_kind": "interaction",
            "quarantine_id": "quarantine-1",
        },
    )
    action = _confirmed_http_control(
        client,
        token,
        {"action": "retry_action", "action_id": "communication-1"},
    )

    assert quarantine == {
        "status": "completed",
        "retry_run_id": "run:explicit-retry",
    }
    assert action == {
        "status": "completed",
        "result": {"action_id": "communication-1", "state": "completed"},
    }
    assert calls == [
        ("quarantine", "interaction", "quarantine-1"),
        ("action", "communication-1"),
    ]
    assert wake_count == 1


def test_http_controls_replace_and_reactivate_home_revisions_through_owner(
    tmp_path: Path,
) -> None:
    state = DurableRunState(tmp_path / "state.sqlite3")
    home = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: NOW)
    assert home.current(USER).payload.revision == 1
    client, token = _operator_client(_dashboard(tmp_path, state=state, home=home))

    replaced = _confirmed_http_control(
        client,
        token,
        {
            "action": "home_replace",
            "reason": "Show the current daily-driver focus.",
            "nodes": [
                {
                    "node_id": "greeting",
                    "component_id": "home.hero.greeting",
                    "visible": True,
                    "content": {
                        "title": "Hello, Siddhartha",
                        "body": "Here is what matters now.",
                        "count": None,
                    },
                }
            ],
        },
    )
    reactivated = _confirmed_http_control(
        client,
        token,
        {
            "action": "home_activate",
            "source_revision": 1,
            "reason": "Return to the earlier Home composition.",
        },
    )

    assert replaced["revision"]["revision"] == 2
    assert reactivated["activation"]["new_revision"] == 3
    assert reactivated["activation"]["source_revision"] == 1
    assert home.current(USER).payload.revision == 3
    assert home.current(USER).payload.nodes == ()


def test_http_reset_preview_confirmation_and_execution_use_maintenance_owner(
    tmp_path: Path,
) -> None:
    dashboard = _dashboard(tmp_path, harness_stopped=True)
    client, token = _operator_client(dashboard)
    headers = {"X-Operator-Token": token}
    preview_response = client.post(
        "/api/controls/preview",
        headers=headers,
        json={"action": "reset", "scope": "working_memory_topics"},
    )
    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["summary"]["scope"] == "working_memory_topics"
    execute_payload = dict(preview["execute_payload"])

    refused = client.post(
        "/api/controls/execute",
        headers=headers,
        json={**execute_payload, "confirmation": "not-the-confirmation"},
    )
    assert refused.status_code == 200
    assert refused.json()["status"] == "confirmation_required"

    completed = client.post(
        "/api/controls/execute", headers=headers, json=execute_payload
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    plans = dashboard.snapshot()["panels"]["retention_reset"]["data"]["reset_plans"]
    assert plans[0]["reset_id"] == execute_payload["reset_id"]
    assert plans[0]["status"] == "completed"


def test_loopback_app_rejects_proxy_headers_and_serves_ui(tmp_path: Path) -> None:
    app = create_operator_dashboard_app(_dashboard(tmp_path))
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.get("/")
    assert response.status_code == 200
    assert "Local Thine Operator" in response.text
    assert "owner observation unavailable" in response.text
    assert "source freshness" in response.text
    assert client.get("/api/snapshot").status_code == 200

    proxied = client.get("/api/snapshot", headers={"X-Forwarded-For": "127.0.0.1"})
    assert proxied.status_code == 403
    assert proxied.json()["error"] == "proxied_operator_access_forbidden"


def test_ui_exposes_no_memory_restore_or_global_autonomy_switches(
    tmp_path: Path,
) -> None:
    app = create_operator_dashboard_app(_dashboard(tmp_path))
    body = TestClient(app, client=("127.0.0.1", 50000)).get("/").text.lower()
    assert "memory restore" not in body
    assert "disable background inference" not in body
    assert "disable home mutation" not in body
    assert "disable schedule creation" not in body
