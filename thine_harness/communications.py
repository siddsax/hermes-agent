"""Proactive-message tools over the existing Thine chat and push transport."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import ipaddress
import json
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Protocol, cast
from urllib.parse import quote, urlparse
import uuid

import httpx

from .action_dispatcher import (
    ActionDispatcher,
    CommunicationActionRecord,
    CommunicationAllowanceUnavailable,
)
from .contracts import JSONValue
from .contracts.notifications import (
    NotificationIntent,
    NotificationOutcome,
    NotificationPermission,
)
from .run_coordinator import InvocationContext
from .run_state import ReceiptConflict


COMMUNICATION_TOOLSET = "local-thine-transcripts"
COMMUNICATION_STATUS_TOOL_NAME = "thine_communications_status"
COMMUNICATION_SEND_TOOL_NAME = "thine_communications_send"
_VERSION = {"major": 1, "minor": 0}

COMMUNICATION_STATUS_TOOL_SCHEMA = {
    "name": COMMUNICATION_STATUS_TOOL_NAME,
    "description": (
        "Inspect the current shared unsolicited Communication Allowance and the "
        "latest notification permission observation. This does not communicate."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

COMMUNICATION_SEND_TOOL_SCHEMA = {
    "name": COMMUNICATION_SEND_TOOL_NAME,
    "description": (
        "Persist one proactive assistant message for the current background Tick. "
        "A push using the exact same message is automatic; there is no separate "
        "notification choice. The shared allowance permits at most one successful "
        "unsolicited communication per 30 minutes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
            }
        },
        "required": ["message"],
        "additionalProperties": False,
    },
}


class BackgroundMessagePort(Protocol):
    def permission(self) -> NotificationPermission: ...

    def deliver(self, intent: NotificationIntent) -> NotificationOutcome: ...


class StandaloneNotificationPort(Protocol):
    def permission(self) -> NotificationPermission: ...

    def deliver_standalone(self, intent: NotificationIntent) -> NotificationOutcome: ...

    def standalone_receipt(self, action_id: str) -> NotificationOutcome | None: ...


@dataclass(frozen=True)
class PushRegistrationStatus:
    """Redacted backend-owned summary of the current user's push registrations."""

    has_registration: bool
    registration_count: int
    last_observed_at_ms: int | None

    @classmethod
    def from_dict(cls, payload: object) -> PushRegistrationStatus:
        if not isinstance(payload, Mapping):
            raise ValueError("push registration status must be an object")
        expected = {
            "has_registration",
            "registration_count",
            "last_observed_at_ms",
        }
        if set(payload) != expected:
            raise ValueError("push registration status has an invalid shape")

        has_registration = payload["has_registration"]
        registration_count = payload["registration_count"]
        last_observed_at_ms = payload["last_observed_at_ms"]
        if not isinstance(has_registration, bool):
            raise ValueError("has_registration must be a boolean")
        if (
            isinstance(registration_count, bool)
            or not isinstance(registration_count, int)
            or registration_count < 0
        ):
            raise ValueError("registration_count must be a nonnegative integer")
        if last_observed_at_ms is not None and (
            isinstance(last_observed_at_ms, bool)
            or not isinstance(last_observed_at_ms, int)
            or last_observed_at_ms < 0
        ):
            raise ValueError(
                "last_observed_at_ms must be a nonnegative integer or null"
            )
        return cls(
            has_registration=has_registration,
            registration_count=registration_count,
            last_observed_at_ms=last_observed_at_ms,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "has_registration": self.has_registration,
            "registration_count": self.registration_count,
            "last_observed_at_ms": self.last_observed_at_ms,
        }


class BackendCommunicationClient:
    """Authenticated client for the backend's closed communication helpers."""

    def __init__(
        self,
        *,
        origin: str,
        credential: str,
        user_id: str,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlparse(origin)
        try:
            address = ipaddress.ip_address(parsed.hostname or "")
        except ValueError as exc:
            raise ValueError(
                "backend communication origin must use a loopback IP literal"
            ) from exc
        if (
            parsed.scheme != "http"
            or not address.is_loopback
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("backend communication origin must be loopback-only HTTP")
        if not credential or not user_id:
            raise ValueError("backend credential and user ID are required")
        self._credential = credential
        self._user_id = user_id
        self._client = httpx.Client(
            base_url=origin.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def permission(self) -> NotificationPermission:
        response = self._client.get(
            "/_local-hermes/private/v1/communications/permission",
            headers=self._headers(),
        )
        response.raise_for_status()
        return NotificationPermission.from_dict(self._json_object(response))

    def push_registration_status(self) -> PushRegistrationStatus:
        response = self._client.get(
            "/_local-hermes/private/v1/communications/push-registration",
            headers=self._headers(),
        )
        response.raise_for_status()
        return PushRegistrationStatus.from_dict(self._json_object(response))

    def deliver(self, intent: NotificationIntent) -> NotificationOutcome:
        response = self._client.post(
            "/_local-hermes/private/v1/communications/background-message",
            headers=self._headers(),
            json=intent.to_dict(),
        )
        response.raise_for_status()
        outcome = NotificationOutcome.from_dict(self._json_object(response))
        if (
            outcome.payload.action_id != intent.payload.action_id
            or outcome.payload.communication_kind != "background_message"
            or outcome.payload.persisted_message_id
            != intent.payload.persisted_message_id
            or outcome.payload.allowance_consumed is not True
        ):
            raise ValueError("backend returned a mismatched background-message outcome")
        return outcome

    def deliver_standalone(self, intent: NotificationIntent) -> NotificationOutcome:
        response = self._client.post(
            "/_local-hermes/private/v1/communications/standalone-notification",
            headers=self._headers(),
            json=intent.to_dict(),
        )
        response.raise_for_status()
        outcome = NotificationOutcome.from_dict(self._json_object(response))
        self._validate_standalone_outcome(intent, outcome)
        return outcome

    def standalone_receipt(self, action_id: str) -> NotificationOutcome | None:
        response = self._client.get(
            "/_local-hermes/private/v1/communications/standalone-notification/"
            + quote(action_id, safe=""),
            headers=self._headers(),
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        outcome = NotificationOutcome.from_dict(self._json_object(response))
        if (
            outcome.payload.action_id != action_id
            or outcome.payload.communication_kind != "standalone_notification"
            or outcome.payload.persisted_message_id is not None
        ):
            raise ValueError("backend returned a mismatched standalone receipt")
        return outcome

    @staticmethod
    def _validate_standalone_outcome(
        intent: NotificationIntent, outcome: NotificationOutcome
    ) -> None:
        payload = outcome.payload
        if (
            payload.action_id != intent.payload.action_id
            or payload.communication_kind != "standalone_notification"
            or payload.persisted_message_id is not None
            or payload.allowance_consumed != (payload.outcome == "accepted")
        ):
            raise ValueError("backend returned a mismatched standalone outcome")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._credential}",
            "Content-Type": "application/json",
            "X-Thine-Firebase-UID": self._user_id,
            "X-Request-ID": str(uuid.uuid4()),
        }

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, JSONValue]:
        payload: object = response.json()
        if not isinstance(payload, dict):
            raise ValueError("backend communication response must be an object")
        return cast(dict[str, JSONValue], payload)


@dataclass(frozen=True)
class _ActiveBackgroundTick:
    user_id: str
    logical_run_id: str
    tick_id: str
    attempt_id: str


class CommunicationToolBinding:
    """Bind deferred communication helpers to one active background Tick."""

    def __init__(
        self,
        *,
        dispatcher: ActionDispatcher,
        backend: BackgroundMessagePort,
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
            raise ValueError("proactive communication tools are background-only")
        active = _ActiveBackgroundTick(
            user_id=str(payload.user_id),
            logical_run_id=str(payload.logical_run_id),
            tick_id=str(payload.tick_id),
            attempt_id=context.attempt_id,
        )
        with self._lock:
            if self._active is not None:
                raise RuntimeError(
                    "another background Tick already owns communications"
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
            name=COMMUNICATION_STATUS_TOOL_NAME,
            toolset=COMMUNICATION_TOOLSET,
            schema=COMMUNICATION_STATUS_TOOL_SCHEMA,
            handler=self.status,
            scope=scope,
        )
        active_registry.register(
            name=COMMUNICATION_SEND_TOOL_NAME,
            toolset=COMMUNICATION_TOOLSET,
            schema=COMMUNICATION_SEND_TOOL_SCHEMA,
            handler=self.send,
            scope=scope,
        )

    def prompt_context(self, user_id: str) -> dict[str, object]:
        """Return current user-side invocation context without changing tools."""
        permission, permission_stale = self._permission(user_id)
        return {
            "allowance": self._dispatcher.allowance_snapshot(user_id).to_dict(),
            "notification_permission": permission.to_dict(),
            "permission_stale": permission_stale,
            "proactive_message_delivery": (
                "Persists one assistant message and automatically requests one push "
                "whose body is the exact same message. Push permission/failure does "
                "not change message-persistence success."
            ),
        }

    def status(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if args:
            return self._json({"ok": False, "error_code": "unexpected_arguments"})
        active = self._active_tick()
        if active is None:
            return self._json({"ok": False, "error_code": "no_active_background_tick"})
        return self._json({"ok": True, **self.prompt_context(active.user_id)})

    def send(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if set(args) != {"message"} or not isinstance(args.get("message"), str):
            return self._json({"ok": False, "error_code": "invalid_message"})
        active = self._active_tick()
        if active is None:
            return self._json({"ok": False, "error_code": "no_active_background_tick"})
        message = str(args["message"])
        try:
            record = self._dispatcher.reserve_background_message(
                user_id=active.user_id,
                logical_run_id=active.logical_run_id,
                tick_id=active.tick_id,
                attempt_id=active.attempt_id,
                message_text=message,
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
            return self._json({
                "ok": False,
                "error_code": "action_intent_conflict",
            })
        except ValueError as exc:
            return self._json({"ok": False, "error_code": str(exc)})

        if record.state != "succeeded":
            record = self._deliver(record)
        return self._tool_result(record)

    def reconcile_due(self, user_id: str) -> tuple[str, ...]:
        """Retry transport only; this never starts or resumes model reasoning."""
        completed: list[str] = []
        for record in self._dispatcher.pending_actions(
            user_id, action_kind="background_message"
        ):
            reconciled = self._deliver(record)
            if reconciled.state == "succeeded":
                completed.append(reconciled.action_id)
        return tuple(completed)

    def reconcile_one(self, user_id: str, action_id: str) -> CommunicationActionRecord:
        """Retry exactly one background-message transport reservation."""
        record = self._dispatcher.record(action_id)
        if record.user_id != user_id or record.action_kind != "background_message":
            raise KeyError(action_id)
        if record.state not in {"reserved", "executing", "retry_pending"}:
            raise ValueError("communication action is not retryable")
        return self._deliver(record)

    def _deliver(self, record: CommunicationActionRecord) -> CommunicationActionRecord:
        if record.state == "succeeded":
            return record
        self._dispatcher.mark_executing(record.action_id)
        try:
            outcome = self._backend.deliver(record.notification_intent)
        except Exception as exc:
            return self._dispatcher.mark_retry_pending(
                record.action_id,
                error_code=f"transport:{type(exc).__name__}",
            )
        return self._dispatcher.complete(action_id=record.action_id, outcome=outcome)

    def _tool_result(self, record: CommunicationActionRecord) -> str:
        allowance = self._dispatcher.allowance_snapshot(record.user_id).to_dict()
        if record.state != "succeeded":
            return self._json({
                "ok": True,
                "status": "queued_for_retry",
                "message_persistence_state": "unknown_pending_reconciliation",
                "action_id": record.action_id,
                "assistant_message_id": record.assistant_message_id,
                "allowance": allowance,
            })
        assert record.outcome is not None and record.receipt is not None
        assert record.assistant_message_id is not None
        outcome = record.outcome.payload
        return self._json({
            "ok": True,
            "status": "succeeded",
            "message_persisted": True,
            "action_id": record.action_id,
            "assistant_message_id": record.assistant_message_id,
            "push_status": outcome.outcome,
            "push_provider_correlation_id": outcome.provider_correlation_id,
            "notification_permission": outcome.permission_state,
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
                    "schema_version": _VERSION,
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
    "BackendCommunicationClient",
    "BackgroundMessagePort",
    "COMMUNICATION_SEND_TOOL_NAME",
    "COMMUNICATION_SEND_TOOL_SCHEMA",
    "COMMUNICATION_STATUS_TOOL_NAME",
    "COMMUNICATION_STATUS_TOOL_SCHEMA",
    "COMMUNICATION_TOOLSET",
    "CommunicationToolBinding",
    "StandaloneNotificationPort",
]
