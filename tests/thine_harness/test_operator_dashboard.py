from __future__ import annotations

from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading

import pytest
from fastapi.testclient import TestClient

from thine_harness.action_dispatcher import ActionDispatcher
from thine_harness.communications import PushRegistrationStatus
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
from thine_harness.private_topology import BackendPrivateConfig, PrivateServiceConfig
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
) -> OperatorDashboard:
    tmp_path.mkdir(parents=True, exist_ok=True)
    state = DurableRunState(tmp_path / "run-state.sqlite3")
    home = HomeStateProjector(tmp_path / "home-state.sqlite3", clock_ms=lambda: NOW)
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
            harness_stopped=lambda: harness_stopped,
        ),
        user_id=USER,
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
        def push_registration_status(self) -> PushRegistrationStatus:
            return PushRegistrationStatus.from_dict({
                "has_registration": True,
                "registration_count": 2,
                "last_observed_at_ms": NOW - 50,
            })

    panels = _dashboard(tmp_path, communications=_Communications()).snapshot()["panels"]

    assert panels["communications"]["status"] == "ok"
    assert panels["communications"]["data"]["push_registration"] == {
        "has_registration": True,
        "registration_count": 2,
        "last_observed_at_ms": NOW - 50,
    }


def test_push_registration_failure_is_communications_panel_scoped_and_redacted(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingCommunications:
        def push_registration_status(self) -> PushRegistrationStatus:
            raise RuntimeError("private-device-token-must-not-leak")

    with caplog.at_level(logging.ERROR, logger="thine_harness.operator_dashboard"):
        panels = _dashboard(
            tmp_path, communications=_FailingCommunications()
        ).snapshot()["panels"]

    communications = panels["communications"]
    assert communications["status"] == "error"
    assert communications["error"] == "owner_read_failed:RuntimeError"
    assert communications["data"] == {}
    assert communications["freshness"]["status"] == "unknown"
    assert panels["schedules"]["status"] == "ok"
    assert "private-device-token-must-not-leak" not in str(communications)


def test_product_controller_projects_registration_from_real_loopback_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, str | None]] = []

    class _BackendHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append({
                "method": "GET",
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "user_id": self.headers.get("X-Thine-Firebase-UID"),
                "request_id": self.headers.get("X-Request-ID"),
            })
            self._respond({
                "has_registration": True,
                "registration_count": 3,
                "last_observed_at_ms": NOW - 25,
            })

        def do_POST(self) -> None:
            # The product controller's background scanner asks the same local
            # backend whether a speaker event is waiting. Empty is valid.
            self._respond(None)

        def _respond(self, payload: object) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
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
    controller = build_product_p0_controller(
        private_config=PrivateServiceConfig(
            enabled=True,
            host="127.0.0.1",
            port=8789,
            firebase_uid=USER,
            request_timeout_seconds=2,
            credential="private-control-token",
        ),
        backend_config=BackendPrivateConfig(
            origin=f"http://127.0.0.1:{backend.server_address[1]}",
            firebase_uid=USER,
            request_timeout_seconds=2,
            credential="backend-private-token",
        ),
        database_path=hermes_home / "thine-harness" / "run-state.sqlite3",
        runtime_factory=lambda: object(),  # Not loaded without an admitted Tick.
    )
    try:
        dashboard = controller.operator_dashboard
        assert isinstance(dashboard, OperatorDashboard)
        response = TestClient(
            create_operator_dashboard_app(dashboard),
            client=("127.0.0.1", 50000),
        ).get("/api/snapshot")
    finally:
        controller.close()
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=2)

    assert response.status_code == 200
    communications = response.json()["panels"]["communications"]
    assert communications["status"] == "ok"
    assert communications["data"]["push_registration"] == {
        "has_registration": True,
        "registration_count": 3,
        "last_observed_at_ms": NOW - 25,
    }
    registration_requests = [
        request
        for request in requests
        if request["path"]
        == "/_local-hermes/private/v1/communications/push-registration"
    ]
    assert registration_requests == [
        {
            "method": "GET",
            "path": "/_local-hermes/private/v1/communications/push-registration",
            "authorization": "Bearer backend-private-token",
            "user_id": USER,
            "request_id": registration_requests[0]["request_id"],
        }
    ]
    assert registration_requests[0]["request_id"]


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
    assert panel["data"]["active"]["completed_tool_receipts"] == 2
    assert panel["data"]["active"]["interruption_request"]["requested"] is False


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
    assert panels["transcripts"]["freshness"]["observed_at_ms"] is None
    assert panels["working_memory"]["status"] == "ok"
    assert panels["working_memory"]["freshness"]["observed_at_ms"] == NOW
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


def test_loopback_app_rejects_proxy_headers_and_serves_ui(tmp_path: Path) -> None:
    app = create_operator_dashboard_app(_dashboard(tmp_path))
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.get("/")
    assert response.status_code == 200
    assert "Local Thine Operator" in response.text
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
