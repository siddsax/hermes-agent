"""Launch the authenticated Hermes control boundary on Mac loopback."""

from __future__ import annotations

import math
import sys

import uvicorn

from .private_service import create_private_service_app
from .private_topology import (
    PrivateServiceConfig,
    PrivateServiceConfigurationError,
    load_private_service_config,
)


def build_private_service_server(config: PrivateServiceConfig) -> uvicorn.Server:
    """Create a bounded Uvicorn server from validated topology config."""

    timeout_seconds = max(1, math.ceil(config.request_timeout_seconds))
    app = create_private_service_app(config)
    uvicorn_config = uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        access_log=True,
        proxy_headers=False,
        forwarded_allow_ips="",
        server_header=False,
        timeout_keep_alive=timeout_seconds,
        timeout_graceful_shutdown=timeout_seconds,
    )
    return uvicorn.Server(uvicorn_config)


def main() -> int:
    """Validate configuration, then serve until interrupted."""

    try:
        config = load_private_service_config()
        if not config.enabled:
            raise PrivateServiceConfigurationError(
                "thine_harness.private_service.enabled is false"
            )
        build_private_service_server(config).run()
    except KeyboardInterrupt:
        return 0
    except PrivateServiceConfigurationError as exc:
        print(f"Hermes private service configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_private_service_server", "main"]
