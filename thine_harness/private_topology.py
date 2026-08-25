"""Fail-closed configuration for the local Thine/Hermes private service."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast


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


__all__ = [
    "PrivateServiceConfig",
    "PrivateServiceConfigurationError",
    "load_private_service_config",
]
