"""Standalone notification tools with durable policy and receipt inspection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import threading
import time
from typing import Any, Callable, Iterator, Mapping

from .action_dispatcher import (
    ActionDispatcher,
    CommunicationActionRecord,
    CommunicationAllowanceUnavailable,
)
from .communications import COMMUNICATION_TOOLSET, StandaloneNotificationPort
from .contracts.notifications import NotificationPermission
from .run_coordinator import InvocationContext
from .run_state import ReceiptConflict


STANDALONE_NOTIFICATION_STATUS_TOOL_NAME = (
    "thine_communications_standalone_notification_status"
)
STANDALONE_NOTIFICATION_SEND_TOOL_NAME = (
    "thine_communications_standalone_notification_send"
)

_NAVIGATION_TEMPLATES = (
    "route.chat",
    "route.connectors",
    "route.speakers-list",
    "route.notification-settings",
    "route.permission-other",
    "route.live-transcript",
    "route.profile",
)

STANDALONE_NOTIFICATION_STATUS_TOOL_SCHEMA = {
    "name": STANDALONE_NOTIFICATION_STATUS_TOOL_NAME,
    "description": (
        "Inspect standalone-notification eligibility, user policy, OS permission, "
        "the last permission request, and recent durable outcomes. This is read-only "
        "with respect to user-visible communication."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

STANDALONE_NOTIFICATION_SEND_TOOL_SCHEMA = {
    "name": STANDALONE_NOTIFICATION_SEND_TOOL_NAME,
    "description": (
        "Request one notification-only communication. It creates no chat message. "
        "The optional closed destination is resolved only after the user taps the "
        "operating-system notification; this tool cannot open or change a screen. "
        "Permission-required and capability-disabled outcomes are durable terminal "
        "negatives and do not consume the shared Communication Allowance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 120},
            "body": {"type": "string", "minLength": 1, "maxLength": 500},
            "navigation_template": {
                "type": ["string", "null"],
                "enum": [*_NAVIGATION_TEMPLATES, None],
            },
        },
        "required": ["title", "body"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class _ActiveBackgroundTick:
    user_id: str
    logical_run_id: str
    tick_id: str
    attempt_id: str


class StandaloneNotificationToolBinding:
    """Bind notification-only helpers to the active background Tick."""

    def __init__(
        self,
        *,
        dispatcher: ActionDispatcher,
        backend: StandaloneNotificationPort,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._dispatcher = dispatcher
        self._backend = backend
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._lock = threading.Lock()
        self._active: _ActiveBackgroundTick | None = None

    @contextmanager
    def activate(self, context: InvocationContext) -> Iterator[None]:
        payload = context.tick.payload
        if str(payload.kind) == "p0_user_chat":
            raise ValueError("standalone notification tools are background-only")
        active = _ActiveBackgroundTick(
            user_id=str(payload.user_id),
            logical_run_id=str(payload.logical_run_id),
            tick_id=str(payload.tick_id),
            attempt_id=context.attempt_id,
        )
        with self._lock:
            if self._active is not None:
                raise RuntimeError(
                    "another background Tick already owns standalone notifications"
                )
            self._active = active
        try:
            yield
        finally:
            with self._lock:
                self._active = None

    def register(self, *, registry_instance: Any | None = None) -> None:
        from tools.registry import registry

        active_registry = registry_instance or registry
        scope = active_registry.current_scope_key()
        active_registry.register(
            name=STANDALONE_NOTIFICATION_STATUS_TOOL_NAME,
            toolset=COMMUNICATION_TOOLSET,
            schema=STANDALONE_NOTIFICATION_STATUS_TOOL_SCHEMA,
            handler=self.status,
            scope=scope,
        )
        active_registry.register(
            name=STANDALONE_NOTIFICATION_SEND_TOOL_NAME,
            toolset=COMMUNICATION_TOOLSET,
            schema=STANDALONE_NOTIFICATION_SEND_TOOL_SCHEMA,
            handler=self.send,
            scope=scope,
        )

    def prompt_context(self, user_id: str) -> dict[str, object]:
        permission, permission_stale = self._permission(user_id)
        recent = self._recent_receipts(user_id)
        last_request = recent[0] if recent else None
        repeat_guard = self._repeat_guard(permission, recent)
        return {
            "standalone_notification": {
                "allowance": self._dispatcher.allowance_snapshot(user_id).to_dict(),
                "policy": permission.to_dict(),
                "permission_stale": permission_stale,
                "last_permission_request_at_ms": (
                    permission.payload.last_permission_ask_at_ms
                ),
                "last_permission_request_topic_id": (
                    permission.payload.last_permission_ask_topic_id
                ),
                "last_request": last_request,
                "recent_receipts": recent,
                "durable_repeat_guard": repeat_guard,
                "permission_request_route": {
                    "chat_markdown_link": "thine:///permissions/other",
                    "activation": "user_tap_only",
                    "agent_may_force_navigation": False,
                },
            }
        }

    def status(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if args:
            return self._json({"ok": False, "error_code": "unexpected_arguments"})
        active = self._active_tick()
        if active is None:
            return self._json({"ok": False, "error_code": "no_active_background_tick"})
        return self._json({"ok": True, **self.prompt_context(active.user_id)})

    def send(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if not self._valid_args(args):
            return self._json({"ok": False, "error_code": "invalid_notification"})
        active = self._active_tick()
        if active is None:
            return self._json({"ok": False, "error_code": "no_active_background_tick"})

        title = str(args["title"])
        body = str(args["body"])
        navigation = args.get("navigation_template")
        navigation_template = None if navigation is None else str(navigation)
        action_id = f"{active.tick_id}:effect:2"

        try:
            existing = self._dispatcher.record(action_id)
        except KeyError:
            existing = None
        if existing is None:
            permission, _stale = self._permission(active.user_id)
            repeat_guard = self._repeat_guard(
                permission, self._recent_receipts(active.user_id)
            )
            if repeat_guard is not None:
                return self._json({
                    "ok": False,
                    "status": "terminal_negative",
                    "error_code": repeat_guard["outcome"],
                    "durable_repeat_guard": repeat_guard,
                    "allowance": self._dispatcher.allowance_snapshot(
                        active.user_id
                    ).to_dict(),
                })

        try:
            record = self._dispatcher.reserve_standalone_notification(
                user_id=active.user_id,
                logical_run_id=active.logical_run_id,
                tick_id=active.tick_id,
                attempt_id=active.attempt_id,
                title=title,
                body=body,
                navigation_template=navigation_template,
            )
        except CommunicationAllowanceUnavailable:
            return self._json({
                "ok": False,
                "error_code": "communication_allowance_unavailable",
                "allowance": self._dispatcher.allowance_snapshot(
                    active.user_id
                ).to_dict(),
            })
        except ReceiptConflict:
            return self._json({"ok": False, "error_code": "action_intent_conflict"})
        except ValueError as exc:
            return self._json({"ok": False, "error_code": str(exc)})

        if record.state not in {"succeeded", "terminal_negative"}:
            record = self._deliver(record)
        return self._tool_result(record)

    def reconcile_due(self, user_id: str) -> tuple[str, ...]:
        """Reconcile transport only, without starting model reasoning."""
        completed: list[str] = []
        for record in self._dispatcher.pending_actions(
            user_id, action_kind="standalone_notification"
        ):
            reconciled = self._deliver(record)
            if reconciled.state in {"succeeded", "terminal_negative"}:
                completed.append(reconciled.action_id)
        return tuple(completed)

    def _deliver(self, record: CommunicationActionRecord) -> CommunicationActionRecord:
        if record.state in {"succeeded", "terminal_negative"}:
            return record
        if record.state != "reserved":
            try:
                receipt = self._backend.standalone_receipt(record.action_id)
            except Exception:
                receipt = None
            if receipt is not None:
                return self._dispatcher.complete(
                    action_id=record.action_id, outcome=receipt
                )
        self._dispatcher.mark_executing(record.action_id)
        try:
            outcome = self._backend.deliver_standalone(record.notification_intent)
        except Exception as exc:
            return self._dispatcher.mark_retry_pending(
                record.action_id,
                error_code=f"transport:{type(exc).__name__}",
            )
        return self._dispatcher.complete(action_id=record.action_id, outcome=outcome)

    def _tool_result(self, record: CommunicationActionRecord) -> str:
        allowance = self._dispatcher.allowance_snapshot(record.user_id).to_dict()
        if record.state not in {"succeeded", "terminal_negative"}:
            return self._json({
                "ok": True,
                "status": "queued_for_retry",
                "delivery_state": "unknown_pending_reconciliation",
                "action_id": record.action_id,
                "allowance": allowance,
            })
        assert record.outcome is not None and record.receipt is not None
        outcome = record.outcome.payload
        if record.state == "terminal_negative":
            return self._json({
                "ok": False,
                "status": "terminal_negative",
                "error_code": outcome.outcome,
                "action_id": record.action_id,
                "notification_permission": outcome.permission_state,
                "allowance_consumed": False,
                "allowance": allowance,
                "durable_repeat_guard": {
                    "action_id": record.action_id,
                    "outcome": outcome.outcome,
                    "do_not_repeat_until_policy_or_permission_changes": (
                        outcome.outcome
                        in {"permission_required", "capability_disabled"}
                    ),
                },
                "receipt": record.receipt.to_dict(),
            })
        return self._json({
            "ok": True,
            "status": "succeeded",
            "action_id": record.action_id,
            "provider_correlation_id": outcome.provider_correlation_id,
            "notification_permission": outcome.permission_state,
            "allowance_consumed": True,
            "allowance": allowance,
            "receipt": record.receipt.to_dict(),
        })

    def _permission(self, user_id: str) -> tuple[NotificationPermission, bool]:
        try:
            permission = self._backend.permission()
        except Exception:
            cached = self._dispatcher.latest_permission(user_id)
            if cached is not None:
                return cached, True
            return (
                NotificationPermission.from_dict({
                    "schema_version": {"major": 1, "minor": 0},
                    "user_preference": "enabled",
                    "os_permission": "unknown",
                    "last_permission_ask_at_ms": None,
                    "last_permission_ask_topic_id": None,
                    "observed_at_ms": self._clock_ms(),
                    "extensions": {},
                }),
                True,
            )
        self._dispatcher.record_permission_for_user(user_id, permission)
        return permission, False

    def _recent_receipts(self, user_id: str) -> list[dict[str, object]]:
        recent: list[dict[str, object]] = []
        for record in self._dispatcher.recent_actions(
            user_id, action_kind="standalone_notification", limit=10
        ):
            outcome = None if record.outcome is None else record.outcome.payload
            recent.append({
                "action_id": record.action_id,
                "requested_at_ms": record.created_at_ms,
                "state": record.state,
                "outcome": None if outcome is None else outcome.outcome,
                "permission_state": (
                    None if outcome is None else outcome.permission_state
                ),
                "allowance_consumed": (
                    None if outcome is None else outcome.allowance_consumed
                ),
                "navigation_template": record.navigation_template,
                "receipt_id": (
                    None
                    if record.receipt is None
                    else record.receipt.payload.receipt_id
                ),
            })
        return recent

    @staticmethod
    def _repeat_guard(
        permission: NotificationPermission,
        recent: list[dict[str, object]],
    ) -> dict[str, object] | None:
        policy = permission.payload
        for receipt in recent:
            outcome = receipt.get("outcome")
            if (
                outcome == "capability_disabled"
                and policy.user_preference == "disabled"
            ) or (
                outcome == "permission_required"
                and policy.os_permission in {"denied", "not_determined", "unknown"}
            ):
                return {
                    "action_id": receipt["action_id"],
                    "outcome": outcome,
                    "do_not_repeat_until_policy_or_permission_changes": True,
                }
        return None

    @staticmethod
    def _valid_args(args: Mapping[str, object]) -> bool:
        if not {"title", "body"}.issubset(args):
            return False
        if set(args) - {"title", "body", "navigation_template"}:
            return False
        if not isinstance(args.get("title"), str) or not isinstance(
            args.get("body"), str
        ):
            return False
        navigation = args.get("navigation_template")
        return navigation is None or navigation in _NAVIGATION_TEMPLATES

    def _active_tick(self) -> _ActiveBackgroundTick | None:
        with self._lock:
            return self._active

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


__all__ = [
    "STANDALONE_NOTIFICATION_SEND_TOOL_NAME",
    "STANDALONE_NOTIFICATION_SEND_TOOL_SCHEMA",
    "STANDALONE_NOTIFICATION_STATUS_TOOL_NAME",
    "STANDALONE_NOTIFICATION_STATUS_TOOL_SCHEMA",
    "StandaloneNotificationToolBinding",
]
