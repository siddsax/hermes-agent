"""Launch the authenticated Hermes control boundary on Mac loopback."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import uvicorn

from .action_dispatcher import ActionDispatcher
from .communications import BackendCommunicationClient, CommunicationToolBinding
from .private_service import create_private_service_app
from .home_state import HomeStateProjector
from .input_pump import BackendTranscriptClient, TranscriptInputPump
from .interactions import (
    BackendInteractionClient,
    BackgroundFinalizerRouter,
    BackgroundInputRouter,
    BackgroundRuntimeRouter,
    HalfHourInteractionDriver,
    InteractionBatchToolBinding,
    InteractionInputPump,
    InteractionRunFinalizer,
    RealInteractionAgentRuntime,
)
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
from .run_coordinator import RunCoordinator
from .runtime import HermesInvocationRuntime
from .speaker_mappings import (
    BackendSpeakerMappingClient,
    RealSpeakerMappingAgentRuntime,
    SpeakerMappingFinalizer,
    SpeakerMappingInputPump,
    SpeakerMappingInspectionToolBinding,
    SpeakerMappingToolBinding,
)
from .standalone_notifications import StandaloneNotificationToolBinding
from .transcript_agent import TranscriptAgentFinalizer, build_real_transcript_runtime
from .topics_preferences import TopicPreferenceService, TopicPreferenceToolBinding


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
    interactions = BackendInteractionClient(
        origin=backend_config.origin,
        credential=backend_config.credential,
        firebase_uid=backend_config.firebase_uid,
        timeout_seconds=backend_config.request_timeout_seconds,
    )
    speakers = BackendSpeakerMappingClient(
        origin=backend_config.origin,
        credential=backend_config.credential,
        firebase_uid=backend_config.firebase_uid,
        timeout_seconds=backend_config.request_timeout_seconds,
    )
    communications = BackendCommunicationClient(
        origin=backend_config.origin,
        credential=backend_config.credential,
        user_id=backend_config.firebase_uid,
        timeout_seconds=backend_config.request_timeout_seconds,
    )
    interaction_binding = InteractionBatchToolBinding()
    speaker_binding = SpeakerMappingToolBinding()
    communication_binding = CommunicationToolBinding(
        dispatcher=ActionDispatcher(store.run_state),
        backend=communications,
    )
    topic_service = TopicPreferenceService(store.run_state)
    topic_binding = TopicPreferenceToolBinding(service=topic_service)
    notification_binding = StandaloneNotificationToolBinding(
        dispatcher=ActionDispatcher(store.run_state),
        backend=communications,
        preference_lookup=lambda user_id: topic_service.preference_value(
            user_id, "notifications_enabled"
        ),
    )

    def communication_context(user_id: str) -> dict[str, object]:
        return {
            **communication_binding.prompt_context(user_id),
            **notification_binding.prompt_context(user_id),
            **topic_binding.prompt_context(user_id),
        }

    transcript_runtime = build_real_transcript_runtime(
        store.run_state,
        firebase_uid=private_config.firebase_uid,
        additional_tool_bindings=(
            interaction_binding,
            speaker_binding,
            communication_binding,
            notification_binding,
            topic_binding,
            SpeakerMappingInspectionToolBinding(
                state=store.run_state,
                user_id=private_config.firebase_uid,
            ),
        ),
        communication_context=communication_context,
    )
    transcript_input = TranscriptInputPump(
        store.run_state,
        transcript_port=transcript,
    )
    speaker_input = SpeakerMappingInputPump(
        store.run_state,
        speaker_port=speakers,
    )
    interaction_input = InteractionInputPump(
        store.run_state,
        source=interactions,
        timezone_name="Asia/Kolkata",
    )

    def scan_background(user_id: str, coordinator: RunCoordinator) -> object:
        communication_binding.reconcile_due(user_id)
        notification_binding.reconcile_due(user_id)
        return speaker_input.enqueue_next(user_id, coordinator=coordinator)

    controller = P0ChatController(
        store=store,
        backend=backend,
        runtime_factory=runtime_factory
        or (lambda: build_p0_runtime(firebase_uid=private_config.firebase_uid)),
        background_runtime=BackgroundRuntimeRouter(
            {
                "p1_transcript": transcript_runtime,
                "p1_interaction": RealInteractionAgentRuntime(
                    store.run_state,
                    agent=transcript_runtime.agent,
                    binding=interaction_binding,
                    communication_context=communication_context,
                ),
                "p1_speaker": RealSpeakerMappingAgentRuntime(
                    store.run_state,
                    agent=transcript_runtime.agent,
                    binding=speaker_binding,
                    communication_context=communication_context,
                ),
            },
            context_bindings=(
                communication_binding,
                notification_binding,
                topic_binding,
            ),
        ),
        background_input=BackgroundInputRouter({
            "p1_transcript": transcript_input,
            "p1_interaction": interaction_input,
            "p1_speaker": speaker_input,
        }),
        background_finalizer=BackgroundFinalizerRouter({
            "p1_transcript": TranscriptAgentFinalizer(
                store.run_state,
                transcript_port=transcript,
            ),
            "p1_interaction": InteractionRunFinalizer(
                store.run_state,
                source=interactions,
            ),
            "p1_speaker": SpeakerMappingFinalizer(
                store.run_state,
                speaker_port=speakers,
            ),
        }),
        background_scan=scan_background,
        p0_context_bindings=(topic_binding,),
        policy_context=topic_binding.prompt_context,
        extra_closables=(transcript, interactions, speakers, communications),
    )
    controller.add_closable(
        HalfHourInteractionDriver(
            pump=interaction_input,
            user_id=private_config.firebase_uid,
            wake_coordinator=controller.wake_background,
        )
    )
    return controller


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
