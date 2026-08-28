"""Fail-closed configuration for the local Thine/Hermes private service."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from urllib.parse import urlparse


class PrivateServiceConfigurationError(ValueError):
    """Raised when the private service cannot start safely."""


@dataclass(frozen=True)
class PrivateServiceConfig:
    """Validated launch configuration with its resolved private credential."""

    enabled: bool
    host: str
    port: int
    firebase_uid: str
    request_timeout_seconds: float
    credential: str = field(repr=False)


@dataclass(frozen=True)
class BackendPrivateConfig:
    """Explicit loopback callback target and its separate backend credential."""

    origin: str
    firebase_uid: str
    request_timeout_seconds: float
    credential: str = field(repr=False)


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PrivateServiceConfigurationError(f"{path} must be a mapping")
    return cast(Mapping[str, object], value)


def _loopback_literal(value: object) -> str:
    host = str(value or "").strip()
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise PrivateServiceConfigurationError(
            "thine_harness.private_service.host must be a loopback IP literal"
        ) from exc
    if not address.is_loopback:
        raise PrivateServiceConfigurationError(
            "thine_harness.private_service.host must be loopback-only"
        )
    return host


def _resolve_credential(
    credential_config: Mapping[str, object],
    *,
    environ: Mapping[str, str] | None,
) -> str:
    env_name = str(credential_config.get("env") or "").strip()
    file_name = str(credential_config.get("file") or "").strip()
    if bool(env_name) == bool(file_name):
        raise PrivateServiceConfigurationError(
            "configure exactly one private credential source: env or file"
        )

    if env_name:
        if environ is None:
            from agent.secret_scope import get_secret

            raw_value = get_secret(env_name)
        else:
            raw_value = environ.get(env_name)
        value = str(raw_value or "").strip()
        source = f"environment variable {env_name}"
    else:
        try:
            value = Path(file_name).expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise PrivateServiceConfigurationError(
                "private credential file could not be read"
            ) from exc
        source = "private credential file"

    if not value:
        raise PrivateServiceConfigurationError(f"{source} is empty")
    return value


def load_private_service_config(
    config: Mapping[str, object] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> PrivateServiceConfig:
    """Load and validate the loopback service from Hermes ``config.yaml``.

    Non-secret behavior is read from ``thine_harness.private_service``. The
    bearer value itself is resolved only from a named environment variable or
    a private file, so it never needs to enter ``config.yaml``.
    """

    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    harness = _mapping(config.get("thine_harness", {}), path="thine_harness")
    service = _mapping(
        harness.get("private_service", {}),
        path="thine_harness.private_service",
    )
    enabled = service.get("enabled", False)
    if not isinstance(enabled, bool):
        raise PrivateServiceConfigurationError(
            "thine_harness.private_service.enabled must be a boolean"
        )

    host = _loopback_literal(service.get("host", "127.0.0.1"))
    port = service.get("port", 8789)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise PrivateServiceConfigurationError(
            "thine_harness.private_service.port must be between 1 and 65535"
        )
    timeout = service.get("request_timeout_seconds", 5.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise PrivateServiceConfigurationError(
            "thine_harness.private_service.request_timeout_seconds must be numeric"
        )
    timeout = float(timeout)
    if not 0.1 <= timeout <= 120.0:
        raise PrivateServiceConfigurationError(
            "thine_harness.private_service.request_timeout_seconds must be between 0.1 and 120"
        )

    firebase_uid = str(service.get("firebase_uid") or "").strip()
    if enabled and (not firebase_uid or len(firebase_uid) > 128):
        raise PrivateServiceConfigurationError(
            "enabled private service requires one Firebase UID of at most 128 characters"
        )

    credential = ""
    if enabled:
        credential = _resolve_credential(
            _mapping(service.get("credential", {}), path="credential"),
            environ=environ,
        )

    return PrivateServiceConfig(
        enabled=enabled,
        host=host,
        port=port,
        firebase_uid=firebase_uid,
        request_timeout_seconds=timeout,
        credential=credential,
    )


def load_backend_private_config(
    config: Mapping[str, object] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> BackendPrivateConfig:
    """Load the fixed backend-private resource/callback adapter configuration."""
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    harness = _mapping(config.get("thine_harness", {}), path="thine_harness")
    service = _mapping(
        harness.get("private_service", {}),
        path="thine_harness.private_service",
    )
    backend = _mapping(
        harness.get("private_backend", {}),
        path="thine_harness.private_backend",
    )
    firebase_uid = str(service.get("firebase_uid") or "").strip()
    if not firebase_uid or len(firebase_uid) > 128:
        raise PrivateServiceConfigurationError(
            "backend private callbacks require the configured Firebase UID"
        )
    origin = str(backend.get("origin") or "http://127.0.0.1:8790").strip()
    parsed = urlparse(origin)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
    except ValueError as exc:
        raise PrivateServiceConfigurationError(
            "thine_harness.private_backend.origin must use a loopback IP literal"
        ) from exc
    if (
        parsed.scheme != "http"
        or not address.is_loopback
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise PrivateServiceConfigurationError(
            "thine_harness.private_backend.origin must be loopback-only HTTP"
        )
    timeout = backend.get(
        "request_timeout_seconds",
        service.get("request_timeout_seconds", 5.0),
    )
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise PrivateServiceConfigurationError(
            "thine_harness.private_backend.request_timeout_seconds must be numeric"
        )
    timeout = float(timeout)
    if not 0.1 <= timeout <= 120.0:
        raise PrivateServiceConfigurationError(
            "thine_harness.private_backend.request_timeout_seconds must be between 0.1 and 120"
        )
    credential_config = backend.get(
        "credential",
        {"env": "BACKEND_PRIVATE_TOKEN", "file": ""},
    )
    credential = _resolve_credential(
        _mapping(credential_config, path="thine_harness.private_backend.credential"),
        environ=environ,
    )
    return BackendPrivateConfig(
        origin=origin.rstrip("/"),
        firebase_uid=firebase_uid,
        request_timeout_seconds=timeout,
        credential=credential,
    )


__all__ = [
    "BackendPrivateConfig",
    "PrivateServiceConfig",
    "PrivateServiceConfigurationError",
    "load_private_service_config",
    "load_backend_private_config",
]
