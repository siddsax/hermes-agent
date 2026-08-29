"""Launch the authenticated Hermes control boundary on Mac loopback."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

import uvicorn

from .action_dispatcher import ActionDispatcher
from .communications import BackendCommunicationClient, CommunicationToolBinding
from .private_service import create_private_service_app
from .home_state import HomeStateProjector, register_home_state_tools
from .input_pump import (
    BackendTranscriptClient,
    TenMinuteTranscriptDriver,
    TranscriptInputPump,
)
from .maintenance import (
    AuthoritativeReadToolBinding,
    AuthoritativeStateReader,
    RetentionResetService,
)
from .operator_dashboard import (
    OperatorDashboard,
    OperatorDashboardControl,
    OperatorDashboardConfigurationError,
    OperatorDashboardReadService,
    load_operator_dashboard_config,
)
from .operator_dashboard_server import create_operator_dashboard_app
from .run_state import DurableRunState
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
from .schedules import (
    OneShotScheduleDriver,
    OneShotScheduleService,
    RealScheduleAgentRuntime,
    ScheduleInputPort,
    ScheduleRunFinalizer,
    ScheduleToolBinding,
)
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


def _harness_marker(hermes_home: Path) -> Path:
    return hermes_home / "thine-harness" / "harness-active.pid"


def _harness_is_stopped(hermes_home: Path) -> bool:
    marker = _harness_marker(hermes_home)
    try:
        pid = int(marker.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


class OperatorDashboardStartupError(RuntimeError):
    """The loopback dashboard listener did not become ready."""


def _start_operator_dashboard_thread(
    server: uvicorn.Server, *, timeout_seconds: float = 5.0
) -> threading.Thread:
    """Start Uvicorn and surface background bind/startup failures synchronously."""
    finished = threading.Event()
    failures: list[BaseException] = []

    def run() -> None:
        try:
            server.run()
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(
        target=run,
        name="thine-operator-dashboard",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server.started and thread.is_alive():
            return thread
        if finished.wait(0.01):
            break
    server.should_exit = True
    if thread.ident is not None:
        thread.join(timeout=1)
    error = OperatorDashboardStartupError("operator dashboard listener did not start")
    if failures:
        raise error from failures[0]
    raise error


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


def build_operator_dashboard_server(
    dashboard: OperatorDashboard,
    *,
    host: str,
    port: int,
) -> uvicorn.Server:
    """Build the separate, non-tunneled loopback operator listener."""
    app = create_operator_dashboard_app(dashboard)
    return uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            access_log=False,
            proxy_headers=False,
            forwarded_allow_ips="",
            server_header=False,
        )
    )


def operator_dashboard_main() -> int:
    """Run the local operator page against Hermes-owned durable state."""
    try:
        from hermes_cli.env_loader import load_hermes_dotenv
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
        load_hermes_dotenv(hermes_home=hermes_home)
        config = load_operator_dashboard_config()
        if not config.enabled:
            raise PrivateServiceConfigurationError(
                "thine_harness.operator_dashboard.enabled is false"
            )
        private_config = load_private_service_config()
        state = DurableRunState(hermes_home / "thine-harness" / "run-state.sqlite3")
        home = HomeStateProjector(hermes_home / "thine-harness" / "home-state.sqlite3")
        schedules = OneShotScheduleService(state)
        topics = TopicPreferenceService(state)
        actions = ActionDispatcher(state)
        maintenance = RetentionResetService(state, home=home)
        dashboard = OperatorDashboard(
            reads=OperatorDashboardReadService(
                AuthoritativeStateReader(state, home=home, schedules=schedules),
                state=state,
                actions=actions,
                topics=topics,
                schedules=schedules,
                maintenance=maintenance,
            ),
            controls=OperatorDashboardControl(
                user_id=private_config.firebase_uid,
                home=home,
                schedules=schedules,
                maintenance=maintenance,
                harness_stopped=lambda: _harness_is_stopped(hermes_home),
            ),
            user_id=private_config.firebase_uid,
        )
        build_operator_dashboard_server(
            dashboard, host=config.host, port=config.port
        ).run()
    except KeyboardInterrupt:
        return 0
    except (
        PrivateServiceConfigurationError,
        OperatorDashboardConfigurationError,
        OperatorDashboardStartupError,
    ) as exc:
        print(f"Hermes operator dashboard configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


def build_product_p0_controller(
    *,
    private_config: PrivateServiceConfig,
    backend_config: BackendPrivateConfig,
    database_path: Path,
    runtime_factory: Callable[[], HermesInvocationRuntime] | None = None,
    home_state: HomeStateProjector | None = None,
) -> P0ChatController:
    """Construct production adapters while deferring model login until first work."""
    home_projector = home_state or HomeStateProjector(
        database_path.parent / "home-state.sqlite3"
    )
    register_home_state_tools(
        home_projector,
        user_id=private_config.firebase_uid,
    )
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
    action_dispatcher = ActionDispatcher(store.run_state)
    communication_binding = CommunicationToolBinding(
        dispatcher=action_dispatcher,
        backend=communications,
    )
    topic_service = TopicPreferenceService(store.run_state)
    topic_binding = TopicPreferenceToolBinding(service=topic_service)
    notification_binding = StandaloneNotificationToolBinding(
        dispatcher=action_dispatcher,
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

    schedules = OneShotScheduleService(store.run_state)
    schedule_binding = ScheduleToolBinding(
        state=store.run_state,
        service=schedules,
        user_id=private_config.firebase_uid,
    )
    authoritative_reader = AuthoritativeStateReader(
        store.run_state,
        home=home_projector,
        topics=topic_service,
        schedules=schedules,
    )
    authoritative_binding = AuthoritativeReadToolBinding(
        authoritative_reader,
        user_id=private_config.firebase_uid,
    )
    transcript_runtime = build_real_transcript_runtime(
        store.run_state,
        firebase_uid=private_config.firebase_uid,
        additional_tool_bindings=(
            interaction_binding,
            speaker_binding,
            communication_binding,
            notification_binding,
            topic_binding,
            schedule_binding,
            authoritative_binding,
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
        schedules.fire_due_once(user_id)
        schedules.promote_oldest_overdue(user_id)
        speaker_tick = speaker_input.enqueue_next(user_id, coordinator=coordinator)
        return speaker_tick

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
                "p2_scheduled": RealScheduleAgentRuntime(
                    store.run_state,
                    agent=transcript_runtime.agent,
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
            "p2_scheduled": ScheduleInputPort(schedules),
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
            "p2_scheduled": ScheduleRunFinalizer(store.run_state),
        }),
        background_scan=scan_background,
        p0_context_bindings=(topic_binding,),
        policy_context=topic_binding.prompt_context,
        extra_closables=(transcript, interactions, speakers, communications),
    )
    controller.add_closable(
        TenMinuteTranscriptDriver(
            pump=transcript_input,
            user_id=private_config.firebase_uid,
            wake_coordinator=controller.wake_background,
            timezone_name="Asia/Kolkata",
        )
    )
    controller.add_closable(
        HalfHourInteractionDriver(
            pump=interaction_input,
            user_id=private_config.firebase_uid,
            wake_coordinator=controller.wake_background,
        )
    )
    controller.add_closable(
        OneShotScheduleDriver(
            service=schedules,
            user_id=private_config.firebase_uid,
            wake_coordinator=controller.wake_background,
        )
    )
    maintenance = RetentionResetService(store.run_state, home=home_projector)

    def retry_quarantine(source_kind: str, quarantine_id: str) -> str:
        retry_run_id = f"operator-retry:{uuid.uuid4()}"
        now_ms = time.time_ns() // 1_000_000
        if source_kind == "transcript":
            transcript_input.enqueue_explicit_retry(
                user_id=private_config.firebase_uid,
                quarantine_id=quarantine_id,
                retry_run_id=retry_run_id,
                created_at_ms=now_ms,
            )
        elif source_kind == "interaction":
            interaction_input.enqueue_explicit_retry(
                user_id=private_config.firebase_uid,
                quarantine_id=quarantine_id,
                retry_run_id=retry_run_id,
                created_at_ms=now_ms,
            )
        elif source_kind == "speaker":
            retry_run_id = speaker_input.enqueue_explicit_retry(
                user_id=private_config.firebase_uid,
                quarantine_id=quarantine_id,
                coordinator=controller.coordinator,
            )
        else:
            raise ValueError("source_kind must be transcript, interaction, or speaker")
        return retry_run_id

    def retry_action(action_id: str) -> dict[str, object]:
        record = action_dispatcher.record(action_id)
        reconciled = (
            communication_binding.reconcile_one(private_config.firebase_uid, action_id)
            if record.action_kind == "background_message"
            else notification_binding.reconcile_one(
                private_config.firebase_uid, action_id
            )
        )
        return {"action_id": reconciled.action_id, "state": reconciled.state}

    controller.attach_operator_dashboard(
        OperatorDashboard(
            reads=OperatorDashboardReadService(
                authoritative_reader,
                state=store.run_state,
                actions=action_dispatcher,
                topics=topic_service,
                schedules=schedules,
                maintenance=maintenance,
                communications=communications,
                run_diagnostics=controller.coordinator.diagnostics,
                live_run=controller.coordinator.active_snapshot,
            ),
            controls=OperatorDashboardControl(
                user_id=private_config.firebase_uid,
                home=home_projector,
                schedules=schedules,
                maintenance=maintenance,
                retry_quarantine=retry_quarantine,
                retry_action=retry_action,
                wake_harness=controller.wake_background,
            ),
            user_id=private_config.firebase_uid,
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
        home_state = HomeStateProjector(
            get_hermes_home() / "thine-harness" / "home-state.sqlite3"
        )
        controller = build_product_p0_controller(
            private_config=config,
            backend_config=backend_config,
            database_path=get_hermes_home() / "thine-harness" / "run-state.sqlite3",
            home_state=home_state,
        )
        marker = _harness_marker(get_hermes_home())
        operator_server: uvicorn.Server | None = None
        operator_thread: threading.Thread | None = None
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(os.getpid()), encoding="utf-8")
            operator_config = load_operator_dashboard_config()
            if operator_config.enabled:
                dashboard = controller.operator_dashboard
                if not isinstance(dashboard, OperatorDashboard):
                    raise OperatorDashboardStartupError(
                        "product controller did not attach operator dashboard"
                    )
                operator_server = build_operator_dashboard_server(
                    dashboard,
                    host=operator_config.host,
                    port=operator_config.port,
                )
                operator_thread = _start_operator_dashboard_thread(operator_server)
            build_private_service_server(
                config,
                p0_control=controller,
                home_state=home_state,
            ).run()
        finally:
            try:
                if operator_server is not None:
                    operator_server.should_exit = True
                if operator_thread is not None and operator_thread.ident is not None:
                    operator_thread.join(timeout=5)
            finally:
                try:
                    controller.close()
                finally:
                    try:
                        if marker.read_text(encoding="utf-8").strip() == str(
                            os.getpid()
                        ):
                            marker.unlink()
                    except FileNotFoundError:
                        pass
    except KeyboardInterrupt:
        return 0
    except (
        PrivateServiceConfigurationError,
        OperatorDashboardConfigurationError,
        OperatorDashboardStartupError,
    ) as exc:
        print(f"Hermes private service configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_operator_dashboard_server",
    "build_private_service_server",
    "build_product_p0_controller",
    "main",
    "operator_dashboard_main",
]
