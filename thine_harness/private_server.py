"""Launch the authenticated Hermes control boundary on Mac loopback."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import uvicorn

from .private_service import create_private_service_app
from .home_state import HomeStateProjector
from .input_pump import BackendTranscriptClient, TranscriptInputPump
from .private_topology import (
    BackendPrivateConfig,
    PrivateServiceConfig,
    PrivateServiceConfigurationError,
    load_backend_private_config,
    load_private_service_config,
)
from .p0_chat import (
    BackendPrivateChatClient,
    P0ChatController,
    P0ChatStore,
    build_p0_runtime,
)
from .runtime import HermesInvocationRuntime
from .transcript_agent import TranscriptAgentFinalizer, build_real_transcript_runtime


def build_private_service_server(
    config: PrivateServiceConfig,
    *,
    p0_control: P0ChatController | None = None,
    home_state: HomeStateProjector | None = None,
) -> uvicorn.Server:
    """Create a bounded Uvicorn server from validated topology config."""

    timeout_seconds = max(1, math.ceil(config.request_timeout_seconds))
    app = create_private_service_app(
        config,
        p0_control=p0_control,
        home_state=home_state,
    )
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


def build_product_p0_controller(
    *,
    private_config: PrivateServiceConfig,
    backend_config: BackendPrivateConfig,
    database_path: Path,
    runtime_factory: Callable[[], HermesInvocationRuntime] | None = None,
) -> P0ChatController:
    """Construct production adapters while deferring model login until first work."""
    backend = BackendPrivateChatClient(
        origin=backend_config.origin,
        credential=backend_config.credential,
        firebase_uid=backend_config.firebase_uid,
        timeout_seconds=backend_config.request_timeout_seconds,
    )
    store = P0ChatStore(database_path)
    transcript = BackendTranscriptClient(
        origin=backend_config.origin,
        credential=backend_config.credential,
        firebase_uid=backend_config.firebase_uid,
        timeout_seconds=backend_config.request_timeout_seconds,
    )
    transcript_runtime = build_real_transcript_runtime(
        store.run_state,
        firebase_uid=private_config.firebase_uid,
    )
    return P0ChatController(
        store=store,
        backend=backend,
        runtime_factory=runtime_factory
        or (lambda: build_p0_runtime(firebase_uid=private_config.firebase_uid)),
        background_runtime=transcript_runtime,
        background_input=TranscriptInputPump(
            store.run_state,
            transcript_port=transcript,
        ),
        background_finalizer=TranscriptAgentFinalizer(
            store.run_state,
            transcript_port=transcript,
        ),
        extra_closables=(transcript,),
    )


def main() -> int:
    """Validate configuration, then serve until interrupted."""

    try:
        from hermes_cli.env_loader import load_hermes_dotenv
        from hermes_constants import get_hermes_home

        load_hermes_dotenv(hermes_home=get_hermes_home())
        config = load_private_service_config()
        if not config.enabled:
            raise PrivateServiceConfigurationError(
                "thine_harness.private_service.enabled is false"
            )
        backend_config = load_backend_private_config()
        controller = build_product_p0_controller(
            private_config=config,
            backend_config=backend_config,
            database_path=get_hermes_home() / "thine-harness" / "run-state.sqlite3",
        )
        home_state = HomeStateProjector(
            get_hermes_home() / "thine-harness" / "home-state.sqlite3"
        )
        try:
            build_private_service_server(
                config,
                p0_control=controller,
                home_state=home_state,
            ).run()
        finally:
            controller.close()
    except KeyboardInterrupt:
        return 0
    except PrivateServiceConfigurationError as exc:
        print(f"Hermes private service configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_private_service_server", "build_product_p0_controller", "main"]
