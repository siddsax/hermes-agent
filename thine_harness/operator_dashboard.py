"""Mac-loopback Operator Dashboard read projection and safe commands.

The dashboard is deliberately a projection over owner helpers.  It stores no
state of its own, never opens Thine backend/mobile storage, and returns an
explicit unavailable value when an owning process has no read seam.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import ipaddress
import logging
import time
from typing import Any, Protocol, cast
import uuid

from .action_dispatcher import ActionDispatcher
from .communications import BackendSpeakerMappingState, PushRegistrationStatus
from .contracts.notifications import NotificationPermission
from .home_state import HomeStateProjector
from .interactions import inspect_interaction_inputs
from .maintenance import (
    AuthoritativeStateReader,
    HISTORY_LIMIT,
    ResetScope,
    RetentionResetService,
)
from .schedules import OneShotScheduleService
from .run_state import DurableRunState
from .topics_preferences import TopicPreferenceService


logger = logging.getLogger(__name__)


class OperatorDashboardConfigurationError(ValueError):
    """The operator surface cannot be launched with the requested topology."""


@dataclass(frozen=True)
class OperatorDashboardConfig:
    enabled: bool
    host: str
    port: int


class QuarantineRetryPort(Protocol):
    def __call__(self, source_kind: str, quarantine_id: str) -> str: ...


class ActionRetryPort(Protocol):
    def __call__(self, action_id: str) -> Mapping[str, object]: ...


class PushRegistrationReadPort(Protocol):
    def permission(self) -> NotificationPermission: ...

    def push_registration_status(self) -> PushRegistrationStatus: ...


class BackendSpeakerReadPort(Protocol):
    def speaker_mapping_state(
        self, *, inspected_at_ms: int
    ) -> BackendSpeakerMappingState: ...


def load_operator_dashboard_config(
    config: Mapping[str, object] | None = None,
) -> OperatorDashboardConfig:
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    harness = config.get("thine_harness", {})
    if not isinstance(harness, Mapping):
        raise OperatorDashboardConfigurationError("thine_harness must be a mapping")
    harness_config = cast(Mapping[str, object], harness)
    raw = harness_config.get("operator_dashboard", {})
    if not isinstance(raw, Mapping):
        raise OperatorDashboardConfigurationError(
            "thine_harness.operator_dashboard must be a mapping"
        )
    dashboard_config = cast(Mapping[str, object], raw)
    enabled = dashboard_config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise OperatorDashboardConfigurationError(
            "thine_harness.operator_dashboard.enabled must be a boolean"
        )
    host = str(dashboard_config.get("host", "127.0.0.1")).strip()
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise OperatorDashboardConfigurationError(
            "operator dashboard host must be a loopback IP literal"
        ) from exc
    if not address.is_loopback:
        raise OperatorDashboardConfigurationError(
            "operator dashboard host must be loopback-only"
        )
    port = dashboard_config.get("port", 8792)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise OperatorDashboardConfigurationError(
            "operator dashboard port must be between 1 and 65535"
        )
    return OperatorDashboardConfig(enabled=enabled, host=host, port=port)


def unavailable(owner: str, reason: str) -> dict[str, object]:
    return {"status": "unavailable", "owner": owner, "reason": reason}


class OperatorDashboardReadService:
    """Build one bounded projection from explicit Hermes owner helpers."""

    def __init__(
        self,
        reader: AuthoritativeStateReader,
        *,
        state: DurableRunState,
        actions: ActionDispatcher,
        topics: TopicPreferenceService,
        schedules: OneShotScheduleService,
        maintenance: RetentionResetService,
        communications: PushRegistrationReadPort | None = None,
        speaker_state: BackendSpeakerReadPort | None = None,
        run_diagnostics: Callable[[str], object] | None = None,
        live_run: Callable[[str], Mapping[str, object] | None] | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._reader = reader
        self._state = state
        self._actions = actions
        self._topics = topics
        self._schedules = schedules
        self._maintenance = maintenance
        self._communications = communications
        self._speaker_state = speaker_state
        self._run_diagnostics = run_diagnostics
        self._live_run = live_run
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def snapshot(
        self, user_id: str, *, limit: int = HISTORY_LIMIT
    ) -> dict[str, object]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 50
        ):
            raise ValueError("limit_must_be_between_1_and_50")
        now_ms = int(self._clock_ms())
        panels = {
            "queue": self._capture(
                now_ms,
                "hermes.run_state.operator_diagnostics;hermes.maintenance.quarantines",
                lambda: self._queue_data(user_id, limit),
            ),
            "current_run": self._capture(
                now_ms,
                (
                    "hermes.run_coordinator.diagnostics+active_snapshot"
                    if self._run_diagnostics is not None
                    else "hermes.run_coordinator.unavailable"
                ),
                lambda: self._current_run_data(user_id),
                partial_error=(
                    "standalone dashboard is not attached to the live coordinator"
                    if self._run_diagnostics is None
                    else None
                ),
            ),
            "transcripts": self._capture(
                now_ms,
                "hermes.run_state.recent_transcript_runs",
                lambda: {
                    "claims": list(
                        self._state.recent_transcript_runs(user_id, limit=limit)
                    ),
                    "canonical_transcripts": unavailable(
                        "thine.dataplane",
                        "backend owner has no bounded dashboard read helper in this process",
                    ),
                },
                partial_error="canonical transcript content is backend-owned and unavailable here",
            ),
            "working_memory": self._capture(
                now_ms,
                "hermes.maintenance.working_memory_current+history;hermes.run_state.recent_finalizations",
                lambda: {
                    "current": self._reader.working_memory_current(user_id),
                    "versions": self._reader.working_memory_history(
                        user_id, limit=limit
                    ),
                    "restore_available": False,
                    "finalizations": list(
                        self._state.recent_finalizations(user_id, limit=limit)
                    ),
                },
            ),
            "home": self._capture(
                now_ms,
                "hermes.home_state_projector.current+history",
                lambda: {
                    "current": self._reader.home_current(user_id),
                    "history": self._reader.home_history(user_id),
                    "last_mobile_ack": unavailable(
                        "thine.mobile_experience",
                        "mobile acknowledgement is owned outside Hermes and has no helper yet",
                    ),
                },
                partial_error="last mobile acknowledgement has no Hermes owner helper",
            ),
            "interactions": self._capture(
                now_ms,
                "hermes.interactions.inspect_interaction_inputs",
                lambda: {
                    **inspect_interaction_inputs(
                        self._state, user_id=user_id, limit=limit
                    ),
                    "retention_days": 7,
                },
            ),
            "speakers": self._speakers_panel(now_ms, user_id, limit),
            "communications": self._communications_panel(now_ms, user_id, limit),
            "schedules": self._capture(
                now_ms,
                "hermes.one_shot_schedule_service.list",
                lambda: {
                    "items": [
                        item.to_tool_dict() for item in self._schedules.list(user_id)
                    ]
                },
            ),
            "topics_preferences": self._capture(
                now_ms,
                "hermes.topic_preference_service.inspect",
                lambda: self._topics.inspect(user_id),
            ),
            "retention_reset": self._capture(
                now_ms,
                "hermes.retention_reset_service.operator_snapshot",
                lambda: {
                    **self._maintenance.operator_snapshot(user_id, limit=limit),
                    "reset_scopes": [
                        "working_memory_topics",
                        "queues_schedules_receipts",
                        "home_state",
                        "all_hermes_state",
                    ],
                    "scoped_resets_preserve_explicit_preferences": True,
                    "full_reset_enumerates_explicit_preferences": True,
                },
            ),
            "debug_timeline": self._capture(
                now_ms,
                "hermes.maintenance.debug_invocations;hermes.run_state.operator_diagnostics",
                lambda: {
                    "redacted": True,
                    "invocations": self._reader.debug_invocations(user_id, limit=limit),
                    "tool_receipts": self._state.operator_diagnostics(
                        user_id, limit=limit
                    )["receipts"],
                },
            ),
        }
        return {
            "schema_version": {"major": 1, "minor": 0},
            "snapshot_id": f"operator:{uuid.uuid4()}",
            "user_id": user_id,
            "generated_at_ms": now_ms,
            "binding": "mac_loopback_only",
            "authoritative": False,
            "limit": limit,
            "panels": panels,
        }

    def _communications_panel(
        self, now_ms: int, user_id: str, limit: int
    ) -> dict[str, object]:
        source = (
            "hermes.action_dispatcher.operator_snapshot;"
            "hermes.topic_preference_service.inspect;"
            "thine.backend.push_transport.permission+push_registration"
        )
        try:
            local = self._actions.operator_snapshot(user_id, limit=limit)
            last_permission_request = self._topics.inspect(
                user_id, "enable_notifications"
            )
        except Exception:
            logger.error(
                "operator dashboard owner read failed",
                extra={"dashboard_source": source},
                exc_info=True,
            )
            return self._panel(
                now_ms,
                source,
                {},
                observed_at_ms=None,
                status="error",
                error="owner_read_failed",
            )

        backend_permission: object
        push_registration: object
        value_failed = False
        if self._communications is None:
            backend_permission = unavailable(
                "thine.backend.push_transport",
                "standalone dashboard has no backend communication client",
            )
            push_registration = unavailable(
                "thine.backend.push_transport",
                "standalone dashboard has no backend communication client",
            )
            value_failed = True
        else:
            communications = self._communications
            assert communications is not None
            backend_permission, permission_failed = self._communication_value(
                "permission",
                lambda: communications.permission().to_dict(),
            )
            push_registration, registration_failed = self._communication_value(
                "push_registration",
                lambda: communications.push_registration_status().to_dict(),
            )
            value_failed = permission_failed or registration_failed
        recorded_permission = local.pop("permission", None)
        data = {
            **local,
            "recorded_permission": recorded_permission,
            "permission": backend_permission,
            "last_permission_request": last_permission_request,
            "push_registration": push_registration,
        }
        local_actions_observed_at_ms = self._latest_record_timestamp(
            data.get("actions"), "updated_at_ms"
        )
        recorded_permission_observed_at_ms = self._mapping_timestamp(
            recorded_permission, "observed_at_ms"
        )
        permission_observed_at_ms = self._mapping_timestamp(
            backend_permission, "observed_at_ms"
        )
        push_observed_at_ms = self._mapping_timestamp(
            push_registration, "last_observed_at_ms"
        )
        permission_request_observed_at_ms = self._mapping_timestamp(
            last_permission_request, "updated_at_ms"
        )
        observed_candidates = [
            value
            for value in (
                local_actions_observed_at_ms,
                recorded_permission_observed_at_ms,
                permission_observed_at_ms,
                push_observed_at_ms,
                permission_request_observed_at_ms,
            )
            if value is not None
        ]
        return self._panel(
            now_ms,
            source,
            data,
            observed_at_ms=max(observed_candidates) if observed_candidates else None,
            status="partial" if value_failed else "ok",
            error=("one_or_more_owner_values_unavailable" if value_failed else None),
            components={
                "actions": self._freshness_value(local_actions_observed_at_ms, now_ms),
                "recorded_permission": self._freshness_value(
                    recorded_permission_observed_at_ms, now_ms
                ),
                "permission": self._freshness_value(permission_observed_at_ms, now_ms),
                "last_permission_request": self._freshness_value(
                    permission_request_observed_at_ms, now_ms
                ),
                "push_registration": self._freshness_value(push_observed_at_ms, now_ms),
            },
        )

    def _speakers_panel(
        self, now_ms: int, user_id: str, limit: int
    ) -> dict[str, object]:
        source = (
            "hermes.run_state.speaker_cursor+recent_speaker_mappings;"
            "thine.backend.maintenance.inspect.speaker_mappings"
        )
        try:
            cursor = self._state.speaker_cursor(user_id)
            retained_inputs = list(
                self._state.recent_speaker_mappings(user_id, limit=limit)
            )
        except Exception as exc:
            logger.error(
                "operator dashboard owner read failed",
                extra={"dashboard_source": source},
                exc_info=True,
            )
            return self._panel(
                now_ms,
                source,
                {},
                observed_at_ms=None,
                status="error",
                error=f"owner_read_failed:{type(exc).__name__}",
            )

        backend_failed = False
        if self._speaker_state is None:
            canonical: object = unavailable(
                "thine.dataplane.speaker_mappings",
                "standalone dashboard has no backend maintenance client",
            )
            backend_failed = True
        else:
            try:
                canonical = self._speaker_state.speaker_mapping_state(
                    inspected_at_ms=now_ms
                ).to_dict()
            except Exception as exc:
                logger.error(
                    "operator dashboard speaker owner read failed",
                    extra={"failure_type": type(exc).__name__},
                )
                canonical = {
                    "status": "error",
                    "owner": "thine.dataplane.speaker_mappings",
                    "error": f"owner_read_failed:{type(exc).__name__}",
                }
                backend_failed = True

        retained_observed_at_ms = self._latest_record_timestamp(
            retained_inputs, "updated_at_ms"
        )
        canonical_observed_at_ms = self._speaker_mapping_observed_at_ms(canonical)
        observed_candidates = [
            value
            for value in (retained_observed_at_ms, canonical_observed_at_ms)
            if value is not None
        ]
        data = {
            "cursor": cursor,
            "hermes_retained_mapping_inputs": retained_inputs,
            "canonical_mappings": canonical,
        }
        return self._panel(
            now_ms,
            source,
            data,
            observed_at_ms=max(observed_candidates) if observed_candidates else None,
            status="partial" if backend_failed else "ok",
            error="canonical_speaker_state_unavailable" if backend_failed else None,
            components={
                "hermes_retained_mapping_inputs": self._freshness_value(
                    retained_observed_at_ms, now_ms
                ),
                "canonical_mappings": self._freshness_value(
                    canonical_observed_at_ms, now_ms
                ),
            },
        )

    @staticmethod
    def _speaker_mapping_observed_at_ms(value: object) -> int | None:
        if not isinstance(value, Mapping):
            return None
        typed = cast(Mapping[object, object], value)
        recent = typed.get("recent_mappings")
        quarantines = typed.get("quarantines")
        candidates = [
            OperatorDashboardReadService._latest_record_timestamp(
                recent, "changed_at_ms"
            ),
            OperatorDashboardReadService._latest_record_timestamp(
                quarantines, "recorded_at_ms"
            ),
        ]
        present = [item for item in candidates if item is not None]
        return max(present) if present else None

    @staticmethod
    def _latest_record_timestamp(value: object, key: str) -> int | None:
        if not isinstance(value, (list, tuple)):
            return None
        candidates: list[int] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            typed_item = cast(Mapping[str, object], item)
            candidate = typed_item.get(key)
            if (
                isinstance(candidate, int)
                and not isinstance(candidate, bool)
                and candidate >= 0
            ):
                candidates.append(candidate)
        return max(candidates) if candidates else None

    @staticmethod
    def _freshness_value(
        observed_at_ms: int | None, read_at_ms: int
    ) -> dict[str, object]:
        return {
            "status": "observed" if observed_at_ms is not None else "read",
            "owner_observed_at_ms": observed_at_ms,
            "age_ms": (
                None if observed_at_ms is None else max(0, read_at_ms - observed_at_ms)
            ),
        }

    @staticmethod
    def _mapping_timestamp(value: object, key: str) -> int | None:
        if not isinstance(value, Mapping):
            return None
        typed_value = cast(Mapping[str, object], value)
        candidate = typed_value.get(key)
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and candidate >= 0
        ):
            return candidate
        return None

    @staticmethod
    def _communication_value(
        name: str, loader: Callable[[], object]
    ) -> tuple[object, bool]:
        try:
            return loader(), False
        except Exception as exc:
            logger.error(
                "operator dashboard communication owner read failed",
                extra={"dashboard_value": name, "failure_type": type(exc).__name__},
            )
            return (
                {
                    "status": "error",
                    "owner": "thine.backend.push_transport",
                    "error": f"owner_read_failed:{type(exc).__name__}",
                },
                True,
            )

    def _queue_data(self, user_id: str, limit: int) -> dict[str, object]:
        diagnostics = self._state.operator_diagnostics(user_id, limit=limit)
        return {
            "owner_observed_at_ms": diagnostics["owner_observed_at_ms"],
            "items": diagnostics["queue"],
            "leases": diagnostics["leases"],
            "attempts": diagnostics["attempts"],
            "checkpoints": diagnostics["checkpoints"],
            "tool_receipts": diagnostics["receipts"],
            "quarantines": self._reader.quarantines(user_id, limit=limit),
        }

    def _current_run_data(self, user_id: str) -> dict[str, object]:
        if self._run_diagnostics is None:
            return {
                "active": unavailable(
                    "hermes.run_coordinator",
                    "standalone dashboard is not attached to the live coordinator",
                ),
                "runtime": unavailable(
                    "hermes.run_coordinator.diagnostics",
                    "standalone dashboard is not attached to the live coordinator",
                ),
            }
        live = self._live_run(user_id) if self._live_run is not None else None
        logical_run_id = None if live is None else live.get("logical_run_id")
        diagnostics = self._state.operator_diagnostics(
            user_id,
            limit=HISTORY_LIMIT,
            active_logical_run_id=(
                logical_run_id if isinstance(logical_run_id, str) else None
            ),
        )
        runtime = self._runtime_diagnostics(user_id)
        if live is not None:
            live = dict(live)
            live["completed_tool_receipts"] = diagnostics["active_run_receipt_count"]
            live["token_estimate"] = unavailable(
                "hermes.agent_runtime",
                "provider usage is authoritative only after the current segment returns",
            )
        return {"active": live, "runtime": runtime}

    def _runtime_diagnostics(self, user_id: str) -> object:
        if self._run_diagnostics is None:
            return unavailable(
                "hermes.run_coordinator.diagnostics",
                "standalone dashboard is not attached to the live coordinator",
            )
        value = self._run_diagnostics(user_id)
        as_dict = getattr(value, "as_dict", None)
        value = as_dict() if callable(as_dict) else value
        if not isinstance(value, Mapping):
            raise TypeError("run diagnostics helper returned an invalid value")
        typed_value = cast(Mapping[str, object], value)
        runtime = typed_value.get("runtime")
        if not isinstance(runtime, Mapping):
            raise TypeError("run diagnostics helper returned invalid runtime state")
        return dict(cast(Mapping[str, object], runtime))

    @staticmethod
    def _capture(
        now_ms: int,
        source: str,
        loader: Callable[[], object],
        *,
        partial_error: str | None = None,
    ) -> dict[str, object]:
        try:
            data = loader()
        except Exception as exc:
            logger.error(
                "operator dashboard owner read failed",
                extra={"dashboard_source": source},
                exc_info=True,
            )
            return OperatorDashboardReadService._panel(
                now_ms,
                source,
                {},
                observed_at_ms=None,
                status="error",
                error=f"owner_read_failed:{type(exc).__name__}",
            )
        return OperatorDashboardReadService._panel(
            now_ms,
            source,
            data,
            observed_at_ms=OperatorDashboardReadService._latest_owner_timestamp(data),
            status="partial" if partial_error else "ok",
            error=partial_error,
            components=OperatorDashboardReadService._component_freshness(data, now_ms),
        )

    @staticmethod
    def _panel(
        now_ms: int,
        source: str,
        data: object,
        *,
        observed_at_ms: int | None,
        status: str,
        error: str | None,
        components: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        freshness_status = (
            "unknown"
            if status == "error"
            else "observed"
            if observed_at_ms is not None
            else "read"
        )
        return {
            "source": source,
            "generated_at_ms": now_ms,
            "freshness": {
                "status": freshness_status,
                "read_at_ms": now_ms,
                "owner_observed_at_ms": observed_at_ms,
                "observed_at_ms": observed_at_ms,
                "snapshot_generated_at_ms": now_ms,
                "age_ms": (
                    None if observed_at_ms is None else max(0, now_ms - observed_at_ms)
                ),
                "components": dict(components or {}),
            },
            "status": status,
            "error": error,
            "data": data,
        }

    @staticmethod
    def _latest_owner_timestamp(value: object) -> int | None:
        if isinstance(value, Mapping):
            return OperatorDashboardReadService._mapping_timestamp(
                value, "owner_observed_at_ms"
            )
        return None

    @staticmethod
    def _component_freshness(data: object, read_at_ms: int) -> dict[str, object]:
        if not isinstance(data, Mapping):
            return {}
        components: dict[str, object] = {}
        for name, value in data.items():
            if not isinstance(value, (Mapping, list, tuple)):
                continue
            observed_at_ms = OperatorDashboardReadService._latest_owner_timestamp(value)
            components[str(name)] = {
                "status": "observed" if observed_at_ms is not None else "read",
                "owner_observed_at_ms": observed_at_ms,
                "age_ms": (
                    None
                    if observed_at_ms is None
                    else max(0, read_at_ms - observed_at_ms)
                ),
            }
        return components


class OperatorDashboardControl:
    """Confirmation-gated facade over existing mutation owner Interfaces."""

    def __init__(
        self,
        *,
        user_id: str,
        home: HomeStateProjector,
        schedules: OneShotScheduleService,
        maintenance: RetentionResetService,
        retry_quarantine: QuarantineRetryPort | None = None,
        retry_action: ActionRetryPort | None = None,
        wake_harness: Callable[[], None] | None = None,
        harness_stopped: Callable[[], bool] | None = None,
    ) -> None:
        self._user_id = user_id
        self._home = home
        self._schedules = schedules
        self._maintenance = maintenance
        self._retry_quarantine = retry_quarantine
        self._retry_action = retry_action
        self._wake_harness = wake_harness or (lambda: None)
        self._harness_stopped = harness_stopped or (lambda: False)

    def preview(self, command: Mapping[str, object]) -> dict[str, object]:
        action = command.get("action")
        if action == "reset":
            scope = cast(ResetScope, command.get("scope"))
            plan = self._maintenance.plan_reset(self._user_id, scope)
            return {
                "action": "reset",
                "requires_confirmation": True,
                "summary": plan.to_dict(),
                "execute_payload": {
                    "action": "reset_execute",
                    "reset_id": plan.reset_id,
                    "confirmation": plan.confirmation,
                },
            }
        if action in {"schedule_run_now", "schedule_cancel", "schedule_edit"}:
            schedule_id = self._required_string(command, "schedule_id")
            record = self._schedules.inspect(self._user_id, schedule_id)
            confirmation = f"CONFIRM {action} {schedule_id}"
            return {
                "action": action,
                "requires_confirmation": True,
                "summary": record.to_tool_dict(),
                "execute_payload": {
                    "action": action,
                    "schedule_id": schedule_id,
                    "due_at": command.get("due_at"),
                    "timezone": command.get("timezone"),
                    "reason": command.get("reason"),
                    "confirmation": confirmation,
                },
            }
        if action == "home_activate":
            source_revision = self._required_int(command, "source_revision")
            current = self._home.current(self._user_id)
            confirmation = f"CONFIRM home_activate {source_revision}"
            return {
                "action": action,
                "requires_confirmation": True,
                "summary": {
                    "current_revision": current.payload.revision,
                    "source_revision": source_revision,
                },
                "execute_payload": {
                    "action": action,
                    "expected_revision": current.payload.revision,
                    "source_revision": source_revision,
                    "reason": str(command.get("reason") or "Operator activation"),
                    "confirmation": confirmation,
                },
            }
        if action == "home_replace":
            nodes = command.get("nodes")
            if not isinstance(nodes, list):
                raise ValueError("nodes must be an array")
            current = self._home.current(self._user_id)
            confirmation = f"CONFIRM home_replace {current.payload.revision}"
            return {
                "action": action,
                "requires_confirmation": True,
                "summary": {
                    "expected_revision": current.payload.revision,
                    "nodes": nodes,
                },
                "execute_payload": {
                    "action": action,
                    "expected_revision": current.payload.revision,
                    "nodes": nodes,
                    "reason": str(command.get("reason") or "Operator replacement"),
                    "confirmation": confirmation,
                },
            }
        if action == "retry_quarantined":
            if self._retry_quarantine is None:
                raise ValueError("quarantine retry owner helper is unavailable")
            kind = self._required_string(command, "source_kind")
            quarantine_id = self._required_string(command, "quarantine_id")
            confirmation = f"CONFIRM retry_quarantined {kind} {quarantine_id}"
            return {
                "action": action,
                "requires_confirmation": True,
                "summary": {"source_kind": kind, "quarantine_id": quarantine_id},
                "execute_payload": {
                    "action": action,
                    "source_kind": kind,
                    "quarantine_id": quarantine_id,
                    "confirmation": confirmation,
                },
            }
        if action == "retry_action":
            if self._retry_action is None:
                raise ValueError("action retry owner helper is unavailable")
            action_id = self._required_string(command, "action_id")
            confirmation = f"CONFIRM retry_action {action_id}"
            return {
                "action": action,
                "requires_confirmation": True,
                "summary": {"action_id": action_id},
                "execute_payload": {
                    "action": action,
                    "action_id": action_id,
                    "confirmation": confirmation,
                },
            }
        raise ValueError("unsupported operator action")

    def execute(self, command: Mapping[str, object]) -> dict[str, object]:
        action = command.get("action")
        if action == "reset_execute":
            result = self._maintenance.execute_reset(
                reset_id=self._required_string(command, "reset_id"),
                confirmation=self._required_string(command, "confirmation"),
                harness_stopped=self._harness_stopped(),
            )
            return cast(dict[str, object], result.to_dict())
        if action in {"schedule_run_now", "schedule_cancel", "schedule_edit"}:
            schedule_id = self._required_string(command, "schedule_id")
            self._confirm(command, f"CONFIRM {action} {schedule_id}")
            action_id = f"operator:{action}:{schedule_id}:{uuid.uuid4()}"
            if action == "schedule_run_now":
                record = self._schedules.run_now(
                    user_id=self._user_id,
                    schedule_id=schedule_id,
                    action_id=action_id,
                )
            elif action == "schedule_cancel":
                record = self._schedules.cancel(
                    user_id=self._user_id,
                    schedule_id=schedule_id,
                    action_id=action_id,
                )
            else:
                record = self._schedules.edit(
                    user_id=self._user_id,
                    schedule_id=schedule_id,
                    action_id=action_id,
                    due_at=cast(str | None, command.get("due_at")),
                    timezone_name=cast(str | None, command.get("timezone")),
                    reason=cast(str | None, command.get("reason")),
                )
            self._wake_harness()
            return {"status": "completed", "schedule": record.to_tool_dict()}
        if action == "home_activate":
            source_revision = self._required_int(command, "source_revision")
            self._confirm(command, f"CONFIRM home_activate {source_revision}")
            result = self._home.reactivate_revision(
                user_id=self._user_id,
                expected_revision=self._required_int(command, "expected_revision"),
                source_revision=source_revision,
                reason=self._required_string(command, "reason"),
                originating_run_id=f"operator:{uuid.uuid4()}",
                source_tick_id=f"operator:{uuid.uuid4()}",
                author="local_operator",
                action_id=f"operator:home_activate:{uuid.uuid4()}",
            )
            return {"status": "completed", "activation": result.to_dict()}
        if action == "home_replace":
            expected = self._required_int(command, "expected_revision")
            self._confirm(command, f"CONFIRM home_replace {expected}")
            nodes = command.get("nodes")
            if not isinstance(nodes, list):
                raise ValueError("nodes must be an array")
            result = self._home.replace_current(
                user_id=self._user_id,
                expected_revision=expected,
                nodes=nodes,
                reason=self._required_string(command, "reason"),
                originating_run_id=f"operator:{uuid.uuid4()}",
                source_tick_id=f"operator:{uuid.uuid4()}",
                author="local_operator",
                action_id=f"operator:home_replace:{uuid.uuid4()}",
            )
            return {"status": "completed", "revision": result.to_dict()}
        if action == "retry_quarantined":
            if self._retry_quarantine is None:
                raise ValueError("quarantine retry owner helper is unavailable")
            kind = self._required_string(command, "source_kind")
            quarantine_id = self._required_string(command, "quarantine_id")
            self._confirm(command, f"CONFIRM retry_quarantined {kind} {quarantine_id}")
            run_id = self._retry_quarantine(kind, quarantine_id)
            self._wake_harness()
            return {"status": "completed", "retry_run_id": run_id}
        if action == "retry_action":
            if self._retry_action is None:
                raise ValueError("action retry owner helper is unavailable")
            action_id = self._required_string(command, "action_id")
            self._confirm(command, f"CONFIRM retry_action {action_id}")
            return {
                "status": "completed",
                "result": dict(self._retry_action(action_id)),
            }
        raise ValueError("unsupported operator action")

    @staticmethod
    def _required_string(command: Mapping[str, object], name: str) -> str:
        value = command.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} is required")
        return value

    @staticmethod
    def _required_int(command: Mapping[str, object], name: str) -> int:
        value = command.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @classmethod
    def _confirm(cls, command: Mapping[str, object], expected: str) -> None:
        if cls._required_string(command, "confirmation") != expected:
            raise ValueError("exact confirmation is required")


class OperatorDashboard:
    """Small Interface shared by HTTP, tests, and future local launchers."""

    def __init__(
        self,
        *,
        reads: OperatorDashboardReadService,
        controls: OperatorDashboardControl,
        user_id: str,
    ) -> None:
        self._reads = reads
        self._controls = controls
        self._user_id = user_id

    def snapshot(self, *, limit: int = HISTORY_LIMIT) -> dict[str, object]:
        return self._reads.snapshot(self._user_id, limit=limit)

    def preview_control(self, command: Mapping[str, object]) -> dict[str, object]:
        return self._controls.preview(command)

    def execute_control(self, command: Mapping[str, object]) -> dict[str, object]:
        return self._controls.execute(command)


__all__ = [
    "ActionRetryPort",
    "OperatorDashboard",
    "OperatorDashboardConfig",
    "OperatorDashboardConfigurationError",
    "OperatorDashboardControl",
    "OperatorDashboardReadService",
    "PushRegistrationReadPort",
    "QuarantineRetryPort",
    "load_operator_dashboard_config",
    "unavailable",
]
