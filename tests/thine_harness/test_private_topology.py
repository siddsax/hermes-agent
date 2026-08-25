from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from hermes_cli.config_defaults import DEFAULT_CONFIG

from thine_harness.private_server import build_private_service_server
from thine_harness.private_service import create_private_service_app
from thine_harness.private_topology import (
    PrivateServiceConfigurationError,
    load_private_service_config,
)


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


def test_enabled_private_service_loads_behavior_from_config_and_secret_from_env() -> None:
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


def test_default_config_keeps_private_service_disabled_without_a_secret() -> None:
    config = load_private_service_config(DEFAULT_CONFIG, environ={})

    assert config.enabled is False
    assert config.host == "127.0.0.1"
    assert config.port == 8789
    assert config.credential == ""


def test_private_credential_can_be_loaded_from_a_file(tmp_path) -> None:
    credential_file = tmp_path / "control-token"
    credential_file.write_text("file-secret\n", encoding="utf-8")
    config = _enabled_config(
        credential={"env": "", "file": str(credential_file)}
    )

    loaded = load_private_service_config(config, environ={})

    assert loaded.credential == "file-secret"
    assert "file-secret" not in repr(loaded)


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
        second = client.get("/health", headers=_headers(**{"X-Request-ID": "request-2"}))

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
