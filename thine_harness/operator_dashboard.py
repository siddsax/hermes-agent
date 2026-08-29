"""Mac-loopback Operator Dashboard read projection and safe commands.

The dashboard is deliberately a projection over owner helpers.  It stores no
state of its own, never opens Thine backend/mobile storage, and returns an
explicit unavailable value when an owning process has no read seam.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import ipaddress
import time
from typing import Any, Protocol, cast
import uuid

from .home_state import HomeStateProjector
from .maintenance import (
    AuthoritativeStateReader,
    HISTORY_LIMIT,
    ResetScope,
    RetentionResetService,
)
from .schedules import OneShotScheduleService


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
    port = dashboard_config.get("port", 8791)
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
        maintenance: RetentionResetService,
        live_run: Callable[[str], Mapping[str, object] | None] | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._reader = reader
        self._maintenance = maintenance
        self._live_run = live_run
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def snapshot(self, user_id: str, *, limit: int = HISTORY_LIMIT) -> dict[str, object]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError("limit_must_be_between_1_and_50")
        now_ms = int(self._clock_ms())
        state = self._reader.snapshot(user_id)
        details = self._reader.operator_details(user_id, limit=limit)
        history = self._reader.working_memory_history(user_id, limit=limit)
        debug = self._reader.debug_invocations(user_id, limit=limit)
        quarantines = self._reader.quarantines(user_id)
        home_history = self._reader.home_history(user_id)
        queue_state = cast(dict[str, object], state["queue_state"])
        leases = cast(list[dict[str, object]], queue_state["leases"])
        attempts = cast(list[dict[str, object]], queue_state["attempts"])[-limit:]
        checkpoints = cast(list[dict[str, object]], queue_state["checkpoints"])[-limit:]
        receipts = cast(list[dict[str, object]], queue_state["receipts"])[-limit:]
        live = self._live_run(user_id) if self._live_run is not None else None
        if live is not None:
            live = dict(live)
            logical_run_id = live.get("logical_run_id")
            live["completed_tool_receipts"] = sum(
                1 for item in receipts if item["logical_run_id"] == logical_run_id
            )
            live["token_estimate"] = unavailable(
                "hermes.agent_runtime",
                "provider usage is authoritative only after the current segment returns",
            )
        if live is None and leases:
            lease = leases[0]
            logical_run_id = str(lease["logical_run_id"])
            running_attempts = [
                item for item in attempts if item["logical_run_id"] == logical_run_id
            ]
            started_value = (
                running_attempts[-1].get("started_at_ms") if running_attempts else None
            )
            started_at = started_value if isinstance(started_value, int) else now_ms
            live = {
                "logical_run_id": logical_run_id,
                "state": lease["state"],
                "elapsed_ms": max(0, now_ms - started_at),
                "completed_tool_receipts": sum(
                    1 for item in receipts if item["logical_run_id"] == logical_run_id
                ),
                "token_estimate": unavailable(
                    "hermes.agent_runtime",
                    "provider usage is authoritative only after the current segment returns",
                ),
                "interruption_request": unavailable(
                    "hermes.run_coordinator",
                    "this reader is not attached to the live coordinator instance",
                ),
            }

        panels = {
            "queue": self._panel(
                now_ms,
                "hermes.run_state.diagnostics",
                {
                    "items": cast(list[object], queue_state["queue"])[-limit:],
                    "leases": leases[-limit:],
                    "attempts": attempts,
                    "checkpoints": checkpoints,
                    "tool_receipts": receipts,
                    "quarantines": quarantines,
                },
            ),
            "current_run": self._panel(
                now_ms,
                "hermes.run_coordinator.active_snapshot",
                {
                    "active": live,
                    "runtime": {
                        "provider": "openai-codex",
                        "model": "gpt-5.6-sol",
                        "api_mode": "codex_responses",
                        "reasoning_effort": "medium",
                        "tool_search_enabled": True,
                        "tool_search_listing": False,
                    },
                },
                status=(
                    "partial"
                    if live is not None
                    else "ok" if not leases else "unavailable"
                ),
                error=(
                    "live provider token usage is unavailable until segment completion"
                    if live is not None
                    else None
                    if not leases
                    else "live coordinator telemetry is not attached"
                ),
            ),
            "transcripts": self._panel(
                now_ms,
                "hermes.transcript_claim_repository",
                {
                    "claims": details["transcript_claims"],
                    "canonical_transcripts": unavailable(
                        "thine.dataplane",
                        "backend owner has no bounded dashboard read helper in this process",
                    ),
                },
                status="partial",
                error="canonical transcript content is backend-owned and unavailable here",
            ),
            "working_memory": self._panel(
                now_ms,
                "hermes.working_memory_custodian",
                {
                    "current": state["working_memory"],
                    "versions": history,
                    "restore_available": False,
                    "finalizations": details["run_finalizations"],
                },
            ),
            "home": self._panel(
                now_ms,
                "hermes.home_state_projector",
                {
                    "current": state["home"],
                    "history": home_history,
                    "last_mobile_ack": unavailable(
                        "thine.mobile_experience",
                        "mobile acknowledgement is owned outside Hermes and has no helper yet",
                    ),
                },
                status="partial",
                error="last mobile acknowledgement has no Hermes owner helper",
            ),
            "interactions": self._panel(
                now_ms,
                "hermes.interaction_input_repository",
                {
                    "clock": state["interaction_clock"],
                    "claims": details["interaction_claims"],
                    "retention_days": 7,
                    "quarantines": cast(dict[str, object], quarantines)["interaction"],
                },
            ),
            "speakers": self._panel(
                now_ms,
                "hermes.speaker_mapping_input_repository",
                {
                    "cursor": details["speaker_cursor"],
                    "inputs": details["speaker_inputs"],
                    "quarantines": cast(dict[str, object], quarantines)["speaker"],
                    "canonical_mappings": unavailable(
                        "thine.dataplane",
                        "backend owner has no bounded dashboard read helper in this process",
                    ),
                },
                status="partial",
                error="canonical speaker mappings are backend-owned and unavailable here",
            ),
            "communications": self._panel(
                now_ms,
                "hermes.action_dispatcher",
                {
                    "actions": details["communications"],
                    "allowance": details["communication_allowance"],
                    "permission": state["notification_permission"],
                    "last_permission_request": details[
                        "last_notification_permission_request"
                    ],
                },
            ),
            "schedules": self._panel(
                now_ms,
                "hermes.one_shot_schedule_service",
                {"items": state["schedules"]},
            ),
            "topics_preferences": self._panel(
                now_ms,
                "hermes.topic_preference_service",
                cast(dict[str, object], state["topics_preferences"]),
            ),
            "retention_reset": self._panel(
                now_ms,
                "hermes.retention_reset_service",
                {
                    "policy": self._maintenance.retention_policy(),
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
            "debug_timeline": self._panel(
                now_ms,
                "hermes.agent_run_inspection_repository",
                {
                    "redacted": True,
                    "invocations": debug,
                    "tool_receipts": receipts,
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

    @staticmethod
    def _panel(
        now_ms: int,
        source: str,
        data: object,
        *,
        status: str = "ok",
        error: str | None = None,
    ) -> dict[str, object]:
        return {
            "source": source,
            "generated_at_ms": now_ms,
            "freshness": {
                "status": "snapshot_time_only",
                "snapshot_generated_at_ms": now_ms,
            },
            "status": status,
            "error": error,
            "data": data,
        }


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
                "summary": {"expected_revision": current.payload.revision, "nodes": nodes},
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
            self._confirm(
                command, f"CONFIRM retry_quarantined {kind} {quarantine_id}"
            )
            run_id = self._retry_quarantine(kind, quarantine_id)
            self._wake_harness()
            return {"status": "completed", "retry_run_id": run_id}
        if action == "retry_action":
            if self._retry_action is None:
                raise ValueError("action retry owner helper is unavailable")
            action_id = self._required_string(command, "action_id")
            self._confirm(command, f"CONFIRM retry_action {action_id}")
            return {"status": "completed", "result": dict(self._retry_action(action_id))}
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
    "QuarantineRetryPort",
    "load_operator_dashboard_config",
    "unavailable",
]
