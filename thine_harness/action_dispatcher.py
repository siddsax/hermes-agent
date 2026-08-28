"""Durable action identity and Communication Allowance ownership.

The dispatcher records an intent and reserves the shared unsolicited allowance
in one SQLite transaction before any backend call.  Delivery and inference are
therefore independent: an ambiguous transport result remains reconcilable under
the same stable action identity without asking the model to decide again.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable, Literal, cast
import uuid

from .contracts.action import ActionIntent, ActionReceipt
from .contracts.notifications import (
    NotificationIntent,
    NotificationOutcome,
    NotificationPermission,
)
from .run_state import DurableRunState, DurableStateError, ReceiptConflict


_VERSION = {"major": 1, "minor": 0}
COMMUNICATION_ALLOWANCE_WINDOW_MS = 30 * 60 * 1000


class CommunicationAllowanceUnavailable(DurableStateError):
    """Another unsolicited communication currently owns the allowance."""


@dataclass(frozen=True)
class CommunicationAllowanceSnapshot:
    status: Literal["available", "reserved", "consumed"]
    remaining: Literal[0, 1]
    reservation_action_id: str | None
    last_consumed_action_id: str | None
    next_eligible_at_ms: int | None
    generated_at_ms: int

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "remaining": self.remaining,
            "reservation_action_id": self.reservation_action_id,
            "last_consumed_action_id": self.last_consumed_action_id,
            "next_eligible_at_ms": self.next_eligible_at_ms,
            "generated_at_ms": self.generated_at_ms,
        }


@dataclass(frozen=True)
class CommunicationActionRecord:
    user_id: str
    logical_run_id: str
    tick_id: str
    action_id: str
    effect_ordinal: int
    action_kind: Literal["background_message", "standalone_notification"]
    intent_fingerprint: str
    assistant_message_id: str | None
    title: str
    message_text: str
    navigation_template: str | None
    action_intent: ActionIntent
    notification_intent: NotificationIntent
    state: str
    outcome: NotificationOutcome | None
    receipt: ActionReceipt | None
    last_error_code: str | None
    created_at_ms: int
    updated_at_ms: int


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class ActionDispatcher:
    """Own proactive-message intents, receipts, and the shared allowance."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._state = state
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def reserve_background_message(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        tick_id: str,
        attempt_id: str,
        message_text: str,
    ) -> CommunicationActionRecord:
        if not user_id or not logical_run_id or not tick_id or not attempt_id:
            raise ValueError("active Tick identity is required")
        if not message_text or not message_text.strip():
            raise ValueError("message must not be empty")
        if len(message_text) > 500:
            raise ValueError("message exceeds the frozen 500-character push body")

        effect_ordinal = 1
        action_id = f"{tick_id}:effect:{effect_ordinal}"
        if len(action_id) > 128:
            raise ValueError("Tick identity is too long for a frozen action_id")
        assistant_message_id = "message:" + str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"thine-proactive:{tick_id}:effect:{effect_ordinal}",
            )
        )
        fingerprint_source = {
            "kind": "background_message",
            "logical_run_id": logical_run_id,
            "effect_ordinal": effect_ordinal,
            "assistant_message_id": assistant_message_id,
            "message_text": message_text,
            "navigation_template": "route.chat",
            "push_required_for_background_message": True,
        }
        intent_fingerprint = hashlib.sha256(
            _canonical_json(fingerprint_source).encode("utf-8")
        ).hexdigest()
        now_ms = self._clock_ms()
        action_intent = ActionIntent.from_dict({
            "schema_version": _VERSION,
            "identity": {
                "action_id": action_id,
                "logical_run_id": logical_run_id,
                "effect_ordinal": effect_ordinal,
                "intent_fingerprint": intent_fingerprint,
            },
            "kind": "background_message",
            "status": "pending",
            "created_at_ms": now_ms,
            "safe_payload": {
                "payload_kind": "background_message",
                "reference_id": assistant_message_id,
            },
            "extensions": {},
        })
        notification_intent = NotificationIntent.from_dict({
            "schema_version": _VERSION,
            "action_id": action_id,
            "kind": "background_message_push",
            "title": "Thine",
            "body": message_text,
            "persisted_message_id": assistant_message_id,
            "navigation_template": "route.chat",
            "push_required_for_background_message": True,
            "created_at_ms": now_ms,
            "extensions": {},
        })
        return self._reserve_communication(
            user_id=user_id,
            logical_run_id=logical_run_id,
            tick_id=tick_id,
            attempt_id=attempt_id,
            effect_ordinal=effect_ordinal,
            action_kind="background_message",
            assistant_message_id=assistant_message_id,
            title="Thine",
            message_text=message_text,
            navigation_template="route.chat",
            intent_fingerprint=intent_fingerprint,
            action_intent=action_intent,
            notification_intent=notification_intent,
        )

    def reserve_standalone_notification(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        tick_id: str,
        attempt_id: str,
        title: str,
        body: str,
        navigation_template: str | None,
    ) -> CommunicationActionRecord:
        if not user_id or not logical_run_id or not tick_id or not attempt_id:
            raise ValueError("active Tick identity is required")
        if not title or not title.strip():
            raise ValueError("title must not be empty")
        if len(title) > 120:
            raise ValueError("title exceeds the frozen 120-character limit")
        if not body or not body.strip():
            raise ValueError("body must not be empty")
        if len(body) > 500:
            raise ValueError("body exceeds the frozen 500-character limit")
        allowed_navigation = {
            None,
            "route.chat",
            "route.connectors",
            "route.speakers-list",
            "route.notification-settings",
            "route.permission-other",
            "route.live-transcript",
            "route.profile",
        }
        if navigation_template not in allowed_navigation:
            raise ValueError("unsupported navigation_template")

        effect_ordinal = 2
        action_id = f"{tick_id}:effect:{effect_ordinal}"
        if len(action_id) > 128:
            raise ValueError("Tick identity is too long for a frozen action_id")
        fingerprint_source = {
            "kind": "standalone_notification",
            "logical_run_id": logical_run_id,
            "effect_ordinal": effect_ordinal,
            "title": title,
            "body": body,
            "navigation_template": navigation_template,
        }
        intent_fingerprint = hashlib.sha256(
            _canonical_json(fingerprint_source).encode("utf-8")
        ).hexdigest()
        now_ms = self._clock_ms()
        action_intent = ActionIntent.from_dict({
            "schema_version": _VERSION,
            "identity": {
                "action_id": action_id,
                "logical_run_id": logical_run_id,
                "effect_ordinal": effect_ordinal,
                "intent_fingerprint": intent_fingerprint,
            },
            "kind": "standalone_notification",
            "status": "pending",
            "created_at_ms": now_ms,
            "safe_payload": {
                "payload_kind": "standalone_notification",
                "reference_id": action_id,
            },
            "extensions": {},
        })
        notification_intent = NotificationIntent.from_dict({
            "schema_version": _VERSION,
            "action_id": action_id,
            "kind": "standalone_notification",
            "title": title,
            "body": body,
            "persisted_message_id": None,
            "navigation_template": navigation_template,
            "push_required_for_background_message": True,
            "created_at_ms": now_ms,
            "extensions": {},
        })
        return self._reserve_communication(
            user_id=user_id,
            logical_run_id=logical_run_id,
            tick_id=tick_id,
            attempt_id=attempt_id,
            effect_ordinal=effect_ordinal,
            action_kind="standalone_notification",
            assistant_message_id=None,
            title=title,
            message_text=body,
            navigation_template=navigation_template,
            intent_fingerprint=intent_fingerprint,
            action_intent=action_intent,
            notification_intent=notification_intent,
        )

    def _reserve_communication(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        tick_id: str,
        attempt_id: str,
        effect_ordinal: int,
        action_kind: Literal["background_message", "standalone_notification"],
        assistant_message_id: str | None,
        title: str,
        message_text: str,
        navigation_template: str | None,
        intent_fingerprint: str,
        action_intent: ActionIntent,
        notification_intent: NotificationIntent,
    ) -> CommunicationActionRecord:
        action_id = f"{tick_id}:effect:{effect_ordinal}"
        if len(action_id) > 128:
            raise ValueError("Tick identity is too long for a frozen action_id")
        now_ms = self._clock_ms()
        with self._state._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM communication_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
            if existing is not None:
                record = self._record_from_row(existing)
                if (
                    record.user_id != user_id
                    or record.logical_run_id != logical_run_id
                    or record.action_kind != action_kind
                    or record.intent_fingerprint != intent_fingerprint
                ):
                    raise ReceiptConflict(
                        "action identity was reused with a different intent"
                    )
                return record

            active = connection.execute(
                """
                SELECT l.*, a.logical_run_id
                FROM communication_allowance_ledger l
                JOIN communication_actions a ON a.action_id = l.action_id
                WHERE l.user_id = ?
                  AND (
                    l.state = 'reserved'
                    OR (l.state = 'consumed' AND l.consumed_at_ms > ?)
                  )
                ORDER BY l.reserved_at_ms DESC LIMIT 1
                """,
                (user_id, now_ms - COMMUNICATION_ALLOWANCE_WINDOW_MS),
            ).fetchone()
            if active is not None:
                raise CommunicationAllowanceUnavailable(str(active["action_id"]))

            queue = connection.execute(
                """
                SELECT q.state, a.status AS attempt_status
                FROM queue_items q
                JOIN attempts a
                  ON a.user_id = q.user_id
                 AND a.logical_run_id = q.logical_run_id
                 AND a.attempt_id = ?
                WHERE q.user_id = ? AND q.logical_run_id = ?
                """,
                (attempt_id, user_id, logical_run_id),
            ).fetchone()
            if (
                queue is None
                or queue["state"] != "running"
                or queue["attempt_status"] != "running"
            ):
                raise DurableStateError(
                    "communication intent requires the active global coordinator run"
                )

            connection.execute(
                """
                INSERT INTO communication_actions (
                    action_id, user_id, logical_run_id, tick_id, effect_ordinal,
                    action_kind, intent_fingerprint, assistant_message_id, title,
                    message_text, navigation_template, action_intent_json,
                    notification_intent_json, state, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?)
                """,
                (
                    action_id,
                    user_id,
                    logical_run_id,
                    tick_id,
                    effect_ordinal,
                    action_kind,
                    intent_fingerprint,
                    assistant_message_id,
                    title,
                    message_text,
                    navigation_template,
                    action_intent.to_json(),
                    notification_intent.to_json(),
                    now_ms,
                    now_ms,
                ),
            )
            connection.execute(
                """
                INSERT INTO communication_allowance_ledger (
                    action_id, user_id, state, reserved_at_ms
                ) VALUES (?, ?, 'reserved', ?)
                """,
                (action_id, user_id, now_ms),
            )
            row = connection.execute(
                "SELECT * FROM communication_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        assert row is not None
        return self._record_from_row(row)

    def mark_executing(self, action_id: str) -> CommunicationActionRecord:
        now_ms = self._clock_ms()
        with self._state._transaction() as connection:
            connection.execute(
                """
                UPDATE communication_actions
                SET state = CASE WHEN state = 'succeeded' THEN state ELSE 'executing' END,
                    updated_at_ms = ?
                WHERE action_id = ?
                """,
                (now_ms, action_id),
            )
            row = connection.execute(
                "SELECT * FROM communication_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        if row is None:
            raise KeyError(action_id)
        return self._record_from_row(row)

    def mark_retry_pending(
        self, action_id: str, *, error_code: str
    ) -> CommunicationActionRecord:
        now_ms = self._clock_ms()
        with self._state._transaction() as connection:
            connection.execute(
                """
                UPDATE communication_actions
                SET state = CASE WHEN state = 'succeeded' THEN state ELSE 'retry_pending' END,
                    last_error_code = CASE WHEN state = 'succeeded' THEN last_error_code ELSE ? END,
                    updated_at_ms = ?
                WHERE action_id = ?
                """,
                (error_code[:128], now_ms, action_id),
            )
            row = connection.execute(
                "SELECT * FROM communication_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        if row is None:
            raise KeyError(action_id)
        return self._record_from_row(row)

    def complete(
        self,
        *,
        action_id: str,
        outcome: NotificationOutcome,
    ) -> CommunicationActionRecord:
        now_ms = self._clock_ms()
        payload = outcome.payload
        with self._state._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM communication_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
            if row is None:
                raise KeyError(action_id)
            existing = self._record_from_row(row)
            if existing.state in {"succeeded", "terminal_negative"}:
                if (
                    existing.outcome is None
                    or existing.outcome.to_json() != outcome.to_json()
                ):
                    raise ReceiptConflict(
                        "backend replay changed the durable communication outcome"
                    )
                return existing
            if payload.action_id != action_id:
                raise ReceiptConflict(
                    "backend outcome does not match the reserved communication"
                )
            if existing.action_kind == "background_message":
                valid_outcome = (
                    payload.communication_kind == "background_message"
                    and payload.persisted_message_id == existing.assistant_message_id
                    and payload.allowance_consumed is True
                )
                terminal_negative = False
            else:
                terminal_negative = payload.outcome in {
                    "permission_required",
                    "capability_disabled",
                    "failed_terminal",
                }
                valid_outcome = (
                    payload.communication_kind == "standalone_notification"
                    and payload.persisted_message_id is None
                    and (
                        (payload.outcome == "accepted" and payload.allowance_consumed)
                        or (terminal_negative and not payload.allowance_consumed)
                    )
                )
            if not valid_outcome:
                raise ReceiptConflict(
                    "backend outcome does not match the reserved communication"
                )
            identity = existing.action_intent.to_dict()["identity"]
            receipt = ActionReceipt.from_dict({
                "schema_version": _VERSION,
                "receipt_id": f"receipt:{action_id}",
                "action_identity": identity,
                "status": "failed_terminal" if terminal_negative else "succeeded",
                "provider_correlation_id": payload.provider_correlation_id,
                "accepted_at_ms": (
                    None if terminal_negative else payload.completed_at_ms
                ),
                "reconciliation_state": (
                    "confirmed_not_applied"
                    if terminal_negative
                    else "confirmed_applied"
                ),
                "extensions": {},
            })
            receipt_id = f"receipt:{action_id}"
            provider_reference = str(
                payload.provider_correlation_id
                or payload.persisted_message_id
                or payload.outcome
            )
            durable_receipt = connection.execute(
                "SELECT * FROM tool_receipts WHERE receipt_id = ?",
                (receipt_id,),
            ).fetchone()
            if durable_receipt is not None and (
                str(durable_receipt["user_id"]) != existing.user_id
                or str(durable_receipt["logical_run_id"]) != existing.logical_run_id
                or str(durable_receipt["action_id"]) != action_id
                or str(durable_receipt["intent_fingerprint"])
                != existing.intent_fingerprint
                or str(durable_receipt["provider_reference"]) != provider_reference
                or str(durable_receipt["result_json"]) != outcome.to_json()
            ):
                raise ReceiptConflict(
                    "communication action conflicts with an existing tool receipt"
                )
            connection.execute(
                """
                UPDATE communication_actions
                SET state = ?, outcome_json = ?, receipt_json = ?,
                    last_error_code = NULL, updated_at_ms = ?
                WHERE action_id = ?
                """,
                (
                    "terminal_negative" if terminal_negative else "succeeded",
                    outcome.to_json(),
                    receipt.to_json(),
                    now_ms,
                    action_id,
                ),
            )
            connection.execute(
                """
                UPDATE communication_allowance_ledger
                SET state = ?, consumed_at_ms = ?, released_at_ms = ?
                WHERE action_id = ?
                """,
                (
                    "released" if terminal_negative else "consumed",
                    None if terminal_negative else int(payload.completed_at_ms),
                    int(payload.completed_at_ms) if terminal_negative else None,
                    action_id,
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO tool_receipts (
                    receipt_id, user_id, logical_run_id, action_id,
                    intent_fingerprint, provider_reference, result_json,
                    acknowledged_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    existing.user_id,
                    existing.logical_run_id,
                    action_id,
                    existing.intent_fingerprint,
                    provider_reference,
                    outcome.to_json(),
                    int(payload.completed_at_ms),
                ),
            )
            completed = connection.execute(
                "SELECT * FROM communication_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        assert completed is not None
        return self._record_from_row(completed)

    def allowance_snapshot(
        self, user_id: str, *, now_ms: int | None = None
    ) -> CommunicationAllowanceSnapshot:
        generated_at_ms = self._clock_ms() if now_ms is None else now_ms
        with self._state._connect() as connection:
            reserved = connection.execute(
                """
                SELECT action_id FROM communication_allowance_ledger
                WHERE user_id = ? AND state = 'reserved'
                ORDER BY reserved_at_ms DESC LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if reserved is not None:
                return CommunicationAllowanceSnapshot(
                    status="reserved",
                    remaining=0,
                    reservation_action_id=str(reserved["action_id"]),
                    last_consumed_action_id=None,
                    next_eligible_at_ms=None,
                    generated_at_ms=generated_at_ms,
                )
            consumed = connection.execute(
                """
                SELECT action_id, consumed_at_ms
                FROM communication_allowance_ledger
                WHERE user_id = ? AND state = 'consumed'
                ORDER BY consumed_at_ms DESC LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        if consumed is None:
            return CommunicationAllowanceSnapshot(
                status="available",
                remaining=1,
                reservation_action_id=None,
                last_consumed_action_id=None,
                next_eligible_at_ms=None,
                generated_at_ms=generated_at_ms,
            )
        next_eligible = int(consumed["consumed_at_ms"]) + (
            COMMUNICATION_ALLOWANCE_WINDOW_MS
        )
        if generated_at_ms >= next_eligible:
            return CommunicationAllowanceSnapshot(
                status="available",
                remaining=1,
                reservation_action_id=None,
                last_consumed_action_id=str(consumed["action_id"]),
                next_eligible_at_ms=None,
                generated_at_ms=generated_at_ms,
            )
        return CommunicationAllowanceSnapshot(
            status="consumed",
            remaining=0,
            reservation_action_id=None,
            last_consumed_action_id=str(consumed["action_id"]),
            next_eligible_at_ms=next_eligible,
            generated_at_ms=generated_at_ms,
        )

    def pending_actions(
        self,
        user_id: str,
        *,
        action_kind: Literal["background_message", "standalone_notification"]
        | None = None,
    ) -> tuple[CommunicationActionRecord, ...]:
        with self._state._connect() as connection:
            if action_kind is None:
                rows = connection.execute(
                    """
                    SELECT * FROM communication_actions
                    WHERE user_id = ?
                      AND state IN ('reserved', 'executing', 'retry_pending')
                    ORDER BY created_at_ms, action_id
                    """,
                    (user_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM communication_actions
                    WHERE user_id = ? AND action_kind = ?
                      AND state IN ('reserved', 'executing', 'retry_pending')
                    ORDER BY created_at_ms, action_id
                    """,
                    (user_id, action_kind),
                ).fetchall()
        return tuple(self._record_from_row(row) for row in rows)

    def recent_actions(
        self,
        user_id: str,
        *,
        action_kind: Literal["background_message", "standalone_notification"]
        | None = None,
        limit: int = 10,
    ) -> tuple[CommunicationActionRecord, ...]:
        if not 1 <= limit <= 50:
            raise ValueError("recent action limit must be between 1 and 50")
        with self._state._connect() as connection:
            if action_kind is None:
                rows = connection.execute(
                    """
                    SELECT * FROM communication_actions
                    WHERE user_id = ?
                    ORDER BY created_at_ms DESC, action_id DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM communication_actions
                    WHERE user_id = ? AND action_kind = ?
                    ORDER BY created_at_ms DESC, action_id DESC
                    LIMIT ?
                    """,
                    (user_id, action_kind, limit),
                ).fetchall()
        return tuple(self._record_from_row(row) for row in rows)

    def record(self, action_id: str) -> CommunicationActionRecord:
        with self._state._connect() as connection:
            row = connection.execute(
                "SELECT * FROM communication_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        if row is None:
            raise KeyError(action_id)
        return self._record_from_row(row)

    def record_permission_for_user(
        self, user_id: str, permission: NotificationPermission
    ) -> None:
        with self._state._transaction() as connection:
            connection.execute(
                """
                INSERT INTO communication_permission_observations (
                    user_id, permission_json, observed_at_ms
                ) VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    permission_json = excluded.permission_json,
                    observed_at_ms = excluded.observed_at_ms
                WHERE excluded.observed_at_ms >= communication_permission_observations.observed_at_ms
                """,
                (user_id, permission.to_json(), int(permission.payload.observed_at_ms)),
            )

    def latest_permission(self, user_id: str) -> NotificationPermission | None:
        with self._state._connect() as connection:
            row = connection.execute(
                """
                SELECT permission_json FROM communication_permission_observations
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        return (
            None
            if row is None
            else NotificationPermission.from_json(str(row["permission_json"]))
        )

    @staticmethod
    def _record_from_row(row: Any) -> CommunicationActionRecord:
        return CommunicationActionRecord(
            user_id=str(row["user_id"]),
            logical_run_id=str(row["logical_run_id"]),
            tick_id=str(row["tick_id"]),
            action_id=str(row["action_id"]),
            effect_ordinal=int(row["effect_ordinal"]),
            action_kind=cast(
                Literal["background_message", "standalone_notification"],
                str(row["action_kind"]),
            ),
            intent_fingerprint=str(row["intent_fingerprint"]),
            assistant_message_id=(
                None
                if row["assistant_message_id"] is None
                else str(row["assistant_message_id"])
            ),
            title=str(row["title"]),
            message_text=str(row["message_text"]),
            navigation_template=(
                None
                if row["navigation_template"] is None
                else str(row["navigation_template"])
            ),
            action_intent=ActionIntent.from_json(str(row["action_intent_json"])),
            notification_intent=NotificationIntent.from_json(
                str(row["notification_intent_json"])
            ),
            state=str(row["state"]),
            outcome=(
                None
                if row["outcome_json"] is None
                else NotificationOutcome.from_json(str(row["outcome_json"]))
            ),
            receipt=(
                None
                if row["receipt_json"] is None
                else ActionReceipt.from_json(str(row["receipt_json"]))
            ),
            last_error_code=(
                None if row["last_error_code"] is None else str(row["last_error_code"])
            ),
            created_at_ms=int(row["created_at_ms"]),
            updated_at_ms=int(row["updated_at_ms"]),
        )


__all__ = [
    "ActionDispatcher",
    "COMMUNICATION_ALLOWANCE_WINDOW_MS",
    "CommunicationActionRecord",
    "CommunicationAllowanceSnapshot",
    "CommunicationAllowanceUnavailable",
]
