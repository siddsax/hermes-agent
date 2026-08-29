from __future__ import annotations

import asyncio
import http.client
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from agent.secret_scope import (
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from fastapi.testclient import TestClient
from hermes_cli.config_defaults import DEFAULT_CONFIG

from thine_harness.private_server import build_private_service_server
from thine_harness.private_service import create_private_service_app
from thine_harness.private_topology import (
    PrivateServiceConfigurationError,
    load_backend_private_config,
    load_private_service_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _enabled_config(**overrides: object) -> dict[str, object]:
    private_service: dict[str, object] = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 8789,
        "firebase_uid": "firebase-user-1",
        "request_timeout_seconds": 5.0,
        "credential": {"env": "HERMES_CONTROL_TOKEN", "file": ""},
    }
    private_service.update(overrides)
    return {"thine_harness": {"private_service": private_service}}


def test_enabled_private_service_loads_behavior_from_config_and_secret_from_env() -> (
    None
):
    config = load_private_service_config(
        _enabled_config(),
        environ={"HERMES_CONTROL_TOKEN": "private-test-token"},
    )

    assert config.host == "127.0.0.1"
    assert config.port == 8789
    assert config.firebase_uid == "firebase-user-1"
    assert config.request_timeout_seconds == 5.0
    assert config.credential == "private-test-token"
    assert "private-test-token" not in repr(config)


def test_backend_private_callback_config_defaults_to_the_fixed_loopback_bridge() -> (
    None
):
    config = load_backend_private_config(
        _enabled_config(),
        environ={"BACKEND_PRIVATE_TOKEN": "backend-private-token"},
    )

    assert config.origin == "http://127.0.0.1:8790"
    assert config.firebase_uid == "firebase-user-1"
    assert config.request_timeout_seconds == 5.0
    assert config.credential == "backend-private-token"
    assert "backend-private-token" not in repr(config)


def test_default_config_keeps_private_service_disabled_without_a_secret() -> None:
    config = load_private_service_config(DEFAULT_CONFIG, environ={})

    assert config.enabled is False
    assert config.host == "127.0.0.1"
    assert config.port == 8789
    assert config.credential == ""


def test_private_credential_can_be_loaded_from_a_file(tmp_path) -> None:
    credential_file = tmp_path / "control-token"
    credential_file.write_text("file-secret\n", encoding="utf-8")
    config = _enabled_config(credential={"env": "", "file": str(credential_file)})

    loaded = load_private_service_config(config, environ={})

    assert loaded.credential == "file-secret"
    assert "file-secret" not in repr(loaded)


def test_production_loader_reads_profile_config_and_scoped_secret(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes_home = tmp_path / "daily-driver-hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """\
thine_harness:
  private_service:
    enabled: true
    host: 127.0.0.1
    port: 8789
    firebase_uid: firebase-profile-user
    request_timeout_seconds: 5
    credential:
      env: HERMES_CONTROL_TOKEN
      file: ""
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_CONTROL_TOKEN", "profile-private-token")

    config = load_private_service_config()

    assert config.enabled is True
    assert config.firebase_uid == "firebase-profile-user"
    assert config.credential == "profile-private-token"


def test_private_service_uses_active_profile_secret_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_CONTROL_TOKEN", "wrong-process-token")
    scope_token = set_secret_scope({"HERMES_CONTROL_TOKEN": "scoped-private-token"})
    set_multiplex_active(True)
    try:
        config = load_private_service_config(_enabled_config())
    finally:
        reset_secret_scope(scope_token)
        set_multiplex_active(False)

    assert config.credential == "scoped-private-token"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com"])
def test_private_service_rejects_non_loopback_bind(host: str) -> None:
    with pytest.raises(PrivateServiceConfigurationError, match="loopback"):
        load_private_service_config(
            _enabled_config(host=host),
            environ={"HERMES_CONTROL_TOKEN": "private-test-token"},
        )


def _client() -> TestClient:
    config = load_private_service_config(
        _enabled_config(),
        environ={"HERMES_CONTROL_TOKEN": "private-test-token"},
    )
    return TestClient(
        create_private_service_app(
            config,
            process_instance_id="hermes-private-test-instance",
            started_at_ms=1_787_644_800_000,
        )
    )


def _headers(**overrides: str) -> dict[str, str]:
    headers = {
        "Authorization": "Bearer private-test-token",
        "X-Thine-Firebase-UID": "firebase-user-1",
        "X-Request-ID": "request-1",
    }
    headers.update(overrides)
    return headers


def test_authenticated_health_is_stable_for_reconnect_detection() -> None:
    with _client() as client:
        first = client.get("/health", headers=_headers())
        second = client.get(
            "/health", headers=_headers(**{"X-Request-ID": "request-2"})
        )

    assert first.status_code == 200
    assert first.json() == {
        "schema_version": {"major": 1, "minor": 0},
        "service": "hermes_control",
        "status": "ready",
        "process_instance_id": "hermes-private-test-instance",
        "started_at_ms": 1_787_644_800_000,
        "request_id": "request-1",
    }
    assert second.json()["process_instance_id"] == first.json()["process_instance_id"]
    assert second.json()["request_id"] == "request-2"


@pytest.mark.parametrize(
    ("headers", "status_code", "error_code"),
    [
        ({"Authorization": "Bearer wrong-token"}, 401, "unauthorized"),
        ({"X-Thine-Firebase-UID": "another-user"}, 403, "uid_mismatch"),
        ({"X-Request-ID": ""}, 400, "invalid_request_id"),
    ],
)
def test_private_service_fails_closed_at_auth_and_uid_boundary(
    headers: dict[str, str], status_code: int, error_code: str
) -> None:
    with _client() as client:
        response = client.get("/health", headers=_headers(**headers))

    assert response.status_code == status_code
    assert response.json()["error"] == error_code
    assert "private-test-token" not in response.text


@pytest.mark.parametrize(
    ("header_name", "header_value", "status_code", "error_code"),
    [
        ("Authorization", "Bearer töken", 401, "unauthorized"),
        ("X-Thine-Firebase-UID", "usér", 403, "uid_mismatch"),
    ],
)
def test_non_ascii_authentication_headers_are_rejected_without_a_500(
    header_name: str, header_value: str, status_code: int, error_code: str
) -> None:
    headers = [
        (name.encode("ascii"), value.encode("utf-8"))
        for name, value in _headers(**{header_name: header_value}).items()
    ]
    with _client() as client:
        response = client.get("/health", headers=headers)

    assert response.status_code == status_code
    assert response.json()["error"] == error_code


def test_private_service_enforces_an_active_request_deadline() -> None:
    config = load_private_service_config(
        _enabled_config(request_timeout_seconds=0.1),
        environ={"HERMES_CONTROL_TOKEN": "private-test-token"},
    )
    app = create_private_service_app(config)

    @app.get("/test-only/never-finishes")
    async def never_finishes() -> None:
        await asyncio.Event().wait()

    with TestClient(app) as client:
        response = client.get(
            "/test-only/never-finishes",
            headers=_headers(),
        )

    assert response.status_code == 504
    assert response.json() == {
        "error": "request_timed_out",
        "request_id": "request-1",
    }


def test_control_route_is_reserved_but_exposes_no_generic_rpc() -> None:
    with _client() as client:
        reserved = client.post("/v1/control", headers=_headers(), json={})
        arbitrary = client.post("/v1/tools/execute", headers=_headers(), json={})

    assert reserved.status_code == 501
    assert reserved.json() == {
        "error": "control_not_implemented",
        "request_id": "request-1",
    }
    assert arbitrary.status_code == 404


def test_private_server_launch_settings_stay_loopback_and_finite() -> None:
    config = load_private_service_config(
        _enabled_config(request_timeout_seconds=7.0),
        environ={"HERMES_CONTROL_TOKEN": "private-test-token"},
    )

    server = build_private_service_server(config)

    assert server.config.host == "127.0.0.1"
    assert server.config.port == 8789
    assert server.config.proxy_headers is False
    assert server.config.forwarded_allow_ips == ""
    assert server.config.timeout_keep_alive == 7
    assert server.config.timeout_graceful_shutdown == 7
    assert server.config.server_header is False


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _network_health_request(
    port: int,
    *,
    firebase_uid: str,
) -> tuple[int, str]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
    try:
        connection.request(
            "GET",
            "/health",
            headers={
                "Authorization": "Bearer private-e2e-token",
                "X-Thine-Firebase-UID": firebase_uid,
                "X-Request-ID": "private-e2e-request",
            },
        )
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        connection.close()


def _network_operator_request(port: int, *, forwarded: bool = False) -> tuple[int, str]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
    headers = {"X-Forwarded-For": "127.0.0.1"} if forwarded else {}
    try:
        connection.request("GET", "/api/snapshot", headers=headers)
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        connection.close()


@pytest.mark.macos_only
def test_private_server_module_serves_authenticated_loopback_requests(
    tmp_path: Path,
) -> None:
    port = _unused_loopback_port()
    dashboard_port = _unused_loopback_port()
    hermes_home = tmp_path / "daily-driver-hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"""\
thine_harness:
  private_service:
    enabled: true
    host: 127.0.0.1
    port: {port}
    firebase_uid: firebase-e2e-user
    request_timeout_seconds: 2
    credential:
      env: HERMES_CONTROL_TOKEN
      file: ""
  operator_dashboard:
    enabled: true
    host: 127.0.0.1
    port: {dashboard_port}
""",
        encoding="utf-8",
    )
    (hermes_home / ".env").write_text(
        "HERMES_CONTROL_TOKEN=private-e2e-token\n"
        "BACKEND_PRIVATE_TOKEN=backend-e2e-token\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        HERMES_HOME=str(hermes_home),
        PYTHONUNBUFFERED="1",
    )
    environment.pop("HERMES_CONTROL_TOKEN", None)
    process = subprocess.Popen(
        [sys.executable, "-m", "thine_harness.private_server"],
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = ""
    try:
        deadline = time.monotonic() + 10
        while True:
            if process.poll() is not None:
                output, _ = process.communicate()
                pytest.fail("private server exited before becoming ready:\n" + output)
            try:
                status, body = _network_health_request(
                    port,
                    firebase_uid="firebase-e2e-user",
                )
            except (OSError, http.client.HTTPException):
                if time.monotonic() >= deadline:
                    pytest.fail("private server did not become ready within 10 seconds")
                time.sleep(0.05)
                continue
            assert status == 200
            assert '"status":"ready"' in body
            break

        deadline = time.monotonic() + 10
        while True:
            try:
                dashboard_status, dashboard_body = _network_operator_request(
                    dashboard_port
                )
            except (OSError, http.client.HTTPException):
                if time.monotonic() >= deadline:
                    pytest.fail(
                        "operator dashboard did not become ready within 10 seconds"
                    )
                time.sleep(0.05)
                continue
            assert dashboard_status == 200
            assert '"binding":"mac_loopback_only"' in dashboard_body
            assert '"current_run"' in dashboard_body
            break

        proxy_status, proxy_body = _network_operator_request(
            dashboard_port, forwarded=True
        )
        assert proxy_status == 403
        assert '"error":"proxied_operator_access_forbidden"' in proxy_body

        rejected_status, rejected_body = _network_health_request(
            port,
            firebase_uid="another-user",
        )
        assert rejected_status == 403
        assert '"error":"uid_mismatch"' in rejected_body

        process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=10)
        assert process.returncode == 0
        assert "Traceback" not in output
        assert not (hermes_home / "thine-harness" / "harness-active.pid").exists()
        for stopped_port in (port, dashboard_port):
            with pytest.raises(OSError):
                _network_operator_request(stopped_port)
    finally:
        if process.poll() is None:
            process.kill()
            trailing_output, _ = process.communicate(timeout=5)
            output += trailing_output


@pytest.mark.macos_only
def test_invalid_operator_config_cleans_controller_and_owned_marker(
    tmp_path: Path,
) -> None:
    hermes_home = tmp_path / "invalid-operator-hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """\
thine_harness:
  private_service:
    enabled: true
    host: 127.0.0.1
    port: 8789
    firebase_uid: firebase-e2e-user
    credential:
      env: HERMES_CONTROL_TOKEN
      file: ""
  operator_dashboard:
    enabled: true
    host: 0.0.0.0
    port: 8792
""",
        encoding="utf-8",
    )
    (hermes_home / ".env").write_text(
        "HERMES_CONTROL_TOKEN=private-e2e-token\n"
        "BACKEND_PRIVATE_TOKEN=backend-e2e-token\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(HERMES_HOME=str(hermes_home), PYTHONUNBUFFERED="1")
    environment.pop("HERMES_CONTROL_TOKEN", None)

    result = subprocess.run(
        [sys.executable, "-m", "thine_harness.private_server"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 2
    assert "operator dashboard host must be loopback-only" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (hermes_home / "thine-harness" / "harness-active.pid").exists()


@pytest.mark.macos_only
def test_operator_startup_failure_cleans_controller_and_owned_marker(
    tmp_path: Path,
) -> None:
    private_port = _unused_loopback_port()
    hermes_home = tmp_path / "occupied-operator-hermes"
    hermes_home.mkdir()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied_listener:
        occupied_listener.bind(("127.0.0.1", 0))
        occupied_listener.listen()
        dashboard_port = int(occupied_listener.getsockname()[1])
        (hermes_home / "config.yaml").write_text(
            f"""\
thine_harness:
  private_service:
    enabled: true
    host: 127.0.0.1
    port: {private_port}
    firebase_uid: firebase-e2e-user
    credential:
      env: HERMES_CONTROL_TOKEN
      file: ""
  operator_dashboard:
    enabled: true
    host: 127.0.0.1
    port: {dashboard_port}
""",
            encoding="utf-8",
        )
        (hermes_home / ".env").write_text(
            "HERMES_CONTROL_TOKEN=private-e2e-token\n"
            "BACKEND_PRIVATE_TOKEN=backend-e2e-token\n",
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment.update(HERMES_HOME=str(hermes_home), PYTHONUNBUFFERED="1")
        environment.pop("HERMES_CONTROL_TOKEN", None)

        result = subprocess.run(
            [sys.executable, "-m", "thine_harness.private_server"],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    assert result.returncode == 2
    assert "operator dashboard listener did not start" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (hermes_home / "thine-harness" / "harness-active.pid").exists()
