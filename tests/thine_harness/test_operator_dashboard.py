from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from thine_harness.home_state import HomeStateProjector
from thine_harness.maintenance import AuthoritativeStateReader, RetentionResetService
from thine_harness.operator_dashboard import (
    OperatorDashboard,
    OperatorDashboardConfigurationError,
    OperatorDashboardControl,
    OperatorDashboardReadService,
    load_operator_dashboard_config,
)
from thine_harness.operator_dashboard_server import create_operator_dashboard_app
from thine_harness.run_state import DurableRunState
from thine_harness.schedules import OneShotScheduleService


USER = "daily-user"
NOW = 1_900_000_000_000


def _dashboard(tmp_path: Path, *, harness_stopped: bool = False) -> OperatorDashboard:
    tmp_path.mkdir(parents=True, exist_ok=True)
    state = DurableRunState(tmp_path / "run-state.sqlite3")
    home = HomeStateProjector(tmp_path / "home-state.sqlite3", clock_ms=lambda: NOW)
    reader = AuthoritativeStateReader(state, home=home, clock_ms=lambda: NOW)
    maintenance = RetentionResetService(state, home=home, clock_ms=lambda: NOW)
    return OperatorDashboard(
        reads=OperatorDashboardReadService(
            reader,
            maintenance=maintenance,
            clock_ms=lambda: NOW,
        ),
        controls=OperatorDashboardControl(
            user_id=USER,
            home=home,
            schedules=OneShotScheduleService(state, clock_ms=lambda: NOW),
            maintenance=maintenance,
            harness_stopped=lambda: harness_stopped,
        ),
        user_id=USER,
    )


def test_config_accepts_only_a_loopback_literal() -> None:
    assert load_operator_dashboard_config({}).enabled is False
    config = load_operator_dashboard_config(
        {
            "thine_harness": {
                "operator_dashboard": {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": 8791,
                }
            }
        }
    )
    assert config.enabled is True
    assert config.host == "127.0.0.1"
    assert config.port == 8791

    with pytest.raises(OperatorDashboardConfigurationError):
        load_operator_dashboard_config(
            {
                "thine_harness": {
                    "operator_dashboard": {
                        "enabled": True,
                        "host": "0.0.0.0",
                        "port": 8791,
                    }
                }
            }
        )


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
    assert snapshot["panels"]["home"]["data"]["last_mobile_ack"]["status"] == "unavailable"
    assert snapshot["panels"]["transcripts"]["data"]["canonical_transcripts"]["status"] == "unavailable"
    assert snapshot["panels"]["working_memory"]["data"]["restore_available"] is False


def test_reset_requires_preview_exact_confirmation_and_harness_stop(
    tmp_path: Path,
) -> None:
    running_dashboard = _dashboard(tmp_path)
    dashboard = _dashboard(tmp_path / "stopped", harness_stopped=True)
    preview = dashboard.preview_control(
        {"action": "reset", "scope": "working_memory_topics"}
    )
    assert preview["requires_confirmation"] is True
    assert preview["execute_payload"]["confirmation"]

    running_preview = running_dashboard.preview_control(
        {"action": "reset", "scope": "working_memory_topics"}
    )
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
