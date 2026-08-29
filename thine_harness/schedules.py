"""Durable agent-owned one-shot schedules for the local Thine Harness.

This module deliberately does not reuse Hermes cron.  A schedule is a single
future P2 Tick in the same SQLite queue as chat and background inputs.  The
active-to-enqueued transition and queue insertion share one transaction, so a
restart can repeat the scan without creating another Logical Run.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import threading
import time
from typing import Any, Callable, Iterator, Literal, Mapping, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import uuid

from .contracts.runtime import Tick
from .contracts.schedule import Schedule
from .run_coordinator import (
    ActiveRunLease,
    InvocationContext,
    InvocationControl,
    InvocationOutcome,
    RunFinalizationResult,
)
from .run_state import DurableRunState, DurableStateError, ReceiptConflict
from .runtime import (
    BackgroundCheckpointPayload,
    HermesAIAgentSession,
    InvocationControl as ProviderInvocationControl,
    InvocationKind,
    InvocationRequest,
    RuntimeModelConfig,
    build_background_invocation_request,
)
from .working_memory import (
    CacheIdentity,
    HermesCachedStopHookContext,
    StopHookOutcomeKind,
    StopHookRunner,
    WorkingMemorySnapshot,
)


_VERSION = {"major": 1, "minor": 0}
_OVERDUE_PROMOTION_MS = 10 * 60 * 1000
SCHEDULE_TOOLSET = "local-thine-transcripts"

SCHEDULE_CREATE_TOOL_NAME = "thine_schedules_create"
SCHEDULE_LIST_TOOL_NAME = "thine_schedules_list"
SCHEDULE_INSPECT_TOOL_NAME = "thine_schedules_inspect"
SCHEDULE_EDIT_TOOL_NAME = "thine_schedules_edit"
SCHEDULE_CANCEL_TOOL_NAME = "thine_schedules_cancel"
SCHEDULE_RUN_NOW_TOOL_NAME = "thine_schedules_run_now"

_DUE_PROPERTIES = {
    "due_at": {
        "type": "string",
        "minLength": 1,
        "maxLength": 64,
        "description": (
            "RFC3339/ISO-8601 timestamp with an explicit UTC offset, for example "
            "2026-08-29T18:00:00+05:30."
        ),
    },
    "timezone": {
        "type": "string",
        "minLength": 1,
        "maxLength": 64,
        "description": "IANA timezone whose offset must match due_at.",
    },
}
_REQUEST_KEY = {
    "type": "string",
    "minLength": 1,
    "maxLength": 64,
    "description": (
        "Stable semantic key for this operation within the current Tick. Reuse it "
        "only when retrying the exact same intent."
    ),
}

SCHEDULE_CREATE_TOOL_SCHEMA = {
    "name": SCHEDULE_CREATE_TOOL_NAME,
    "description": (
        "Create one durable one-shot wakeup. Schedule creation is always available. "
        "A required human-readable reason explains why the wakeup exists."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **_DUE_PROPERTIES,
            "reason": {"type": "string", "minLength": 1, "maxLength": 1000},
            "request_key": _REQUEST_KEY,
        },
        "required": ["due_at", "timezone", "reason", "request_key"],
        "additionalProperties": False,
    },
}
SCHEDULE_LIST_TOOL_SCHEMA = {
    "name": SCHEDULE_LIST_TOOL_NAME,
    "description": "List this user's one-shot schedules in deterministic order.",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": [
                    "active",
                    "enqueued",
                    "cancelled",
                    "completed",
                    "failed_terminal",
                ],
            }
        },
        "additionalProperties": False,
    },
}
SCHEDULE_INSPECT_TOOL_SCHEMA = {
    "name": SCHEDULE_INSPECT_TOOL_NAME,
    "description": "Inspect one durable one-shot schedule and its normalized time.",
    "parameters": {
        "type": "object",
        "properties": {
            "schedule_id": {"type": "string", "minLength": 1, "maxLength": 128}
        },
        "required": ["schedule_id"],
        "additionalProperties": False,
    },
}
SCHEDULE_EDIT_TOOL_SCHEMA = {
    "name": SCHEDULE_EDIT_TOOL_NAME,
    "description": (
        "Edit the due time and/or reason of an active one-shot schedule. Enqueued "
        "or terminal schedules cannot be rewritten."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "schedule_id": {"type": "string", "minLength": 1, "maxLength": 128},
            **_DUE_PROPERTIES,
            "reason": {"type": "string", "minLength": 1, "maxLength": 1000},
            "request_key": _REQUEST_KEY,
        },
        "required": ["schedule_id", "request_key"],
        "additionalProperties": False,
    },
}
SCHEDULE_CANCEL_TOOL_SCHEMA = {
    "name": SCHEDULE_CANCEL_TOOL_NAME,
    "description": "Cancel an active one-shot schedule before queue ownership begins.",
    "parameters": {
        "type": "object",
        "properties": {
            "schedule_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "request_key": _REQUEST_KEY,
        },
        "required": ["schedule_id", "request_key"],
        "additionalProperties": False,
    },
}
SCHEDULE_RUN_NOW_TOOL_SCHEMA = {
    "name": SCHEDULE_RUN_NOW_TOOL_NAME,
    "description": (
        "Atomically enqueue an active one-shot schedule now. Queue priority and "
        "single-flight ownership still apply."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "schedule_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "request_key": _REQUEST_KEY,
        },
        "required": ["schedule_id", "request_key"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class ScheduleRecord:
    schedule_id: str
    user_id: str
    due_at_ms: int
    timezone_name: str
    due_time_input: str
    created_at_ms: int
    updated_at_ms: int
    status: str
    reason: str
    creator_tick_id: str
    originating_run_id: str
    originating_action_id: str
    intent_fingerprint: str
    enqueued_tick_id: str | None
    enqueued_logical_run_id: str | None
    enqueued_at_ms: int | None
    promoted_to_p1_at_ms: int | None
    completed_at_ms: int | None
    attempt_ordinal: int

    def contract(self) -> Schedule:
        return Schedule.from_dict({
            "schema_version": _VERSION,
            "schedule_id": self.schedule_id,
            "user_id": self.user_id,
            "due_at_ms": self.due_at_ms,
            "created_at_ms": self.created_at_ms,
            "status": self.status,
            "reason": self.reason,
            "originating_run_id": self.originating_run_id,
            "originating_action_id": self.originating_action_id,
            "attempt_ordinal": self.attempt_ordinal,
            "promoted_to_p1_at_ms": self.promoted_to_p1_at_ms,
            "extensions": {},
        })

    def to_tool_dict(self) -> dict[str, object]:
        return {
            "schedule": self.contract().to_dict(),
            "timezone": self.timezone_name,
            "due_time_input": self.due_time_input,
            "due_at_utc": datetime
            .fromtimestamp(self.due_at_ms / 1000, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "creator_tick_id": self.creator_tick_id,
            "enqueued_tick_id": self.enqueued_tick_id,
            "enqueued_logical_run_id": self.enqueued_logical_run_id,
            "enqueued_at_ms": self.enqueued_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "completed_at_ms": self.completed_at_ms,
        }


@dataclass(frozen=True)
class PreparedScheduleInput:
    schedule: ScheduleRecord


@dataclass(frozen=True)
class ScheduleAgentArtifact:
    agent: Any
    result: Any
    current_memory: WorkingMemorySnapshot
    cache_identity: CacheIdentity
    provider: str
    model: str
    api_mode: str
    reasoning_effort: str
    tool_discoveries: tuple[str, ...]


@dataclass(frozen=True)
class _ActiveTick:
    tick_id: str
    logical_run_id: str
    attempt_id: str


class _StagedMemory:
    def __init__(self) -> None:
        self.markdown: str | None = None
        self.token_count: int | None = None
        self.marked_unchanged = False

    def commit(
        self, *, expected_version: int, markdown: str, token_count: int, run_id: str
    ) -> int:
        del run_id
        self.markdown = markdown
        self.token_count = token_count
        return expected_version + 1

    def mark_unchanged(self, *, expected_version: int, run_id: str) -> None:
        del expected_version, run_id
        self.marked_unchanged = True


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def normalize_due_time(due_at: str, timezone_name: str) -> tuple[int, str]:
    """Normalize an offset-aware time and verify its IANA-zone interpretation."""
    if not isinstance(due_at, str) or not due_at or len(due_at) > 64:
        raise ValueError(
            "due_at must be a non-empty timestamp of at most 64 characters"
        )
    if (
        not isinstance(timezone_name, str)
        or not timezone_name
        or len(timezone_name) > 64
    ):
        raise ValueError(
            "timezone must be a non-empty IANA name of at most 64 characters"
        )
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("timezone must be a valid IANA timezone") from exc
    normalized_input = due_at[:-1] + "+00:00" if due_at.endswith("Z") else due_at
    try:
        parsed = datetime.fromisoformat(normalized_input)
    except ValueError as exc:
        raise ValueError("due_at must be a valid RFC3339/ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("due_at must include an explicit UTC offset")
    utc = parsed.astimezone(timezone.utc)
    zoned = utc.astimezone(zone)
    if zoned.utcoffset() != parsed.utcoffset():
        raise ValueError("due_at offset does not match timezone at that instant")
    due_at_ms = int(utc.timestamp() * 1000)
    if due_at_ms < 0:
        raise ValueError("due_at must not precede the Unix epoch")
    return due_at_ms, utc.isoformat().replace("+00:00", "Z")


class OneShotScheduleService:
    """Transactional schedule state and deterministic queue promotion."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._state = state
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._change_callback: Callable[[], None] | None = None

    def set_change_callback(self, callback: Callable[[], None] | None) -> None:
        self._change_callback = callback

    def _changed(self) -> None:
        if self._change_callback is not None:
            self._change_callback()

    def create(
        self,
        *,
        user_id: str,
        creator_tick_id: str,
        originating_run_id: str,
        originating_action_id: str,
        due_at: str,
        timezone_name: str,
        reason: str,
    ) -> ScheduleRecord:
        reason = self._validate_reason(reason)
        due_at_ms, normalized = normalize_due_time(due_at, timezone_name)
        intent = {
            "operation": "create",
            "user_id": user_id,
            "originating_run_id": originating_run_id,
            "due_at_ms": due_at_ms,
            "timezone": timezone_name,
            "reason": reason,
        }
        intent_fingerprint = _fingerprint(intent)
        schedule_id = "schedule:" + str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"thine-one-shot:{user_id}:{originating_action_id}",
            )
        )
        now_ms = self._clock_ms()
        with self._state._transaction() as connection:
            existing = connection.execute(
                """
                SELECT * FROM one_shot_schedules
                WHERE user_id = ? AND originating_action_id = ?
                """,
                (user_id, originating_action_id),
            ).fetchone()
            if existing is not None:
                record = self._record(connection, existing)
                if record.intent_fingerprint != intent_fingerprint:
                    raise ReceiptConflict(
                        "schedule action identity was reused with different intent"
                    )
                return record
            same_intent = connection.execute(
                """
                SELECT * FROM one_shot_schedules
                WHERE user_id = ? AND originating_run_id = ?
                  AND intent_fingerprint = ?
                """,
                (user_id, originating_run_id, intent_fingerprint),
            ).fetchone()
            if same_intent is not None:
                return self._record(connection, same_intent)
            active = connection.execute(
                """
                SELECT q.tick_id, q.logical_run_id, a.attempt_id
                FROM queue_items q
                JOIN attempts a ON a.logical_run_id = q.logical_run_id
                WHERE q.user_id = ? AND q.tick_id = ? AND q.logical_run_id = ?
                  AND q.state = 'running' AND a.status = 'running'
                """,
                (user_id, creator_tick_id, originating_run_id),
            ).fetchone()
            if active is None:
                raise DurableStateError("schedule creation requires an active Tick")
            connection.execute(
                """
                INSERT INTO one_shot_schedules (
                    schedule_id, user_id, due_at_ms, timezone_name,
                    due_time_input, created_at_ms, updated_at_ms, status,
                    reason, creator_tick_id, originating_run_id,
                    originating_action_id, intent_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
                """,
                (
                    schedule_id,
                    user_id,
                    due_at_ms,
                    timezone_name,
                    normalized,
                    now_ms,
                    now_ms,
                    reason,
                    creator_tick_id,
                    originating_run_id,
                    originating_action_id,
                    intent_fingerprint,
                ),
            )
            row = connection.execute(
                "SELECT * FROM one_shot_schedules WHERE schedule_id = ?",
                (schedule_id,),
            ).fetchone()
            assert row is not None
            record = self._record(connection, row)
        self._changed()
        return record

    def list(
        self, user_id: str, *, status: str | None = None
    ) -> tuple[ScheduleRecord, ...]:
        if status is not None and status not in {
            "active",
            "enqueued",
            "cancelled",
            "completed",
            "failed_terminal",
        }:
            raise ValueError("invalid schedule status")
        with self._state._connect() as connection:
            if status is None:
                rows = connection.execute(
                    """
                    SELECT * FROM one_shot_schedules WHERE user_id = ?
                    ORDER BY due_at_ms, created_at_ms, schedule_id
                    """,
                    (user_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM one_shot_schedules
                    WHERE user_id = ? AND status = ?
                    ORDER BY due_at_ms, created_at_ms, schedule_id
                    """,
                    (user_id, status),
                ).fetchall()
            return tuple(self._record(connection, row) for row in rows)

    def inspect(self, user_id: str, schedule_id: str) -> ScheduleRecord:
        with self._state._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM one_shot_schedules
                WHERE user_id = ? AND schedule_id = ?
                """,
                (user_id, schedule_id),
            ).fetchone()
            if row is None:
                raise KeyError(schedule_id)
            return self._record(connection, row)

    def edit(
        self,
        *,
        user_id: str,
        schedule_id: str,
        action_id: str,
        due_at: str | None,
        timezone_name: str | None,
        reason: str | None,
    ) -> ScheduleRecord:
        if (due_at is None) != (timezone_name is None):
            raise ValueError("due_at and timezone must be edited together")
        if due_at is None and reason is None:
            raise ValueError("edit requires a due time and/or reason")
        normalized_reason = None if reason is None else self._validate_reason(reason)
        normalized_due: tuple[int, str] | None = None
        if due_at is not None and timezone_name is not None:
            normalized_due = normalize_due_time(due_at, timezone_name)
        intent = {
            "operation": "edit",
            "schedule_id": schedule_id,
            "due_at_ms": None if normalized_due is None else normalized_due[0],
            "timezone": timezone_name,
            "reason": normalized_reason,
        }
        return self._mutate(
            user_id=user_id,
            schedule_id=schedule_id,
            action_id=action_id,
            operation="edit",
            intent=intent,
            apply=lambda connection, row, now_ms: self._apply_edit(
                connection,
                row,
                now_ms=now_ms,
                normalized_due=normalized_due,
                timezone_name=timezone_name,
                normalized_reason=normalized_reason,
            ),
        )

    def cancel(
        self, *, user_id: str, schedule_id: str, action_id: str
    ) -> ScheduleRecord:
        return self._mutate(
            user_id=user_id,
            schedule_id=schedule_id,
            action_id=action_id,
            operation="cancel",
            intent={"operation": "cancel", "schedule_id": schedule_id},
            apply=self._apply_cancel,
        )

    def run_now(
        self, *, user_id: str, schedule_id: str, action_id: str
    ) -> ScheduleRecord:
        return self._mutate(
            user_id=user_id,
            schedule_id=schedule_id,
            action_id=action_id,
            operation="run_now",
            intent={"operation": "run_now", "schedule_id": schedule_id},
            apply=lambda connection, row, now_ms: self._enqueue_locked(
                connection, row=row, now_ms=now_ms
            ),
        )

    def fire_due_once(self, user_id: str, *, now_ms: int | None = None) -> str | None:
        fired_at_ms = self._clock_ms() if now_ms is None else now_ms
        with self._state._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM one_shot_schedules
                WHERE user_id = ? AND status = 'active' AND due_at_ms <= ?
                ORDER BY due_at_ms, created_at_ms, schedule_id LIMIT 1
                """,
                (user_id, fired_at_ms),
            ).fetchone()
            if row is None:
                return None
            record = self._enqueue_locked(connection, row=row, now_ms=fired_at_ms)
        self._changed()
        return record.schedule_id

    def promote_oldest_overdue(
        self, user_id: str, *, now_ms: int | None = None
    ) -> str | None:
        promoted_at_ms = self._clock_ms() if now_ms is None else now_ms
        with self._state._transaction() as connection:
            blocking = connection.execute(
                """
                SELECT 1
                FROM one_shot_schedules s
                JOIN queue_items q ON q.logical_run_id = s.enqueued_logical_run_id
                WHERE s.user_id = ? AND s.status = 'enqueued'
                  AND s.promoted_to_p1_at_ms IS NOT NULL
                  AND q.state NOT IN ('completed', 'quarantined', 'failed_terminal')
                  AND NOT EXISTS (
                      SELECT 1 FROM checkpoints c
                      WHERE c.logical_run_id = q.logical_run_id
                  )
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if blocking is not None:
                return None
            row = connection.execute(
                """
                SELECT s.*, q.tick_json, q.enqueue_sequence
                FROM one_shot_schedules s
                JOIN queue_items q ON q.logical_run_id = s.enqueued_logical_run_id
                WHERE s.user_id = ? AND s.status = 'enqueued'
                  AND s.promoted_to_p1_at_ms IS NULL
                  AND q.state = 'queued' AND q.priority_rank = 2
                  AND s.enqueued_at_ms <= ?
                ORDER BY s.enqueued_at_ms, s.due_at_ms, s.created_at_ms,
                         s.schedule_id LIMIT 1
                """,
                (user_id, promoted_at_ms - _OVERDUE_PROMOTION_MS),
            ).fetchone()
            if row is None:
                return None
            tick_dict = json.loads(str(row["tick_json"]))
            tick_dict["priority"] = "p1"
            promoted_tick = Tick.from_dict(tick_dict)
            next_sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(enqueue_sequence), 0) + 1 FROM queue_items"
                ).fetchone()[0]
            )
            connection.execute(
                """
                UPDATE queue_items
                SET priority = 'p1', priority_rank = 1, tick_json = ?,
                    enqueue_sequence = ?, updated_at_ms = ?
                WHERE logical_run_id = ? AND state = 'queued' AND priority_rank = 2
                """,
                (
                    promoted_tick.to_json(),
                    next_sequence,
                    promoted_at_ms,
                    row["enqueued_logical_run_id"],
                ),
            )
            connection.execute(
                """
                UPDATE one_shot_schedules
                SET promoted_to_p1_at_ms = ?, updated_at_ms = ?
                WHERE schedule_id = ? AND promoted_to_p1_at_ms IS NULL
                """,
                (promoted_at_ms, promoted_at_ms, row["schedule_id"]),
            )
            schedule_id = str(row["schedule_id"])
        return schedule_id

    def next_wake_at_ms(self, user_id: str) -> int | None:
        with self._state._connect() as connection:
            active = connection.execute(
                """
                SELECT MIN(due_at_ms) FROM one_shot_schedules
                WHERE user_id = ? AND status = 'active'
                """,
                (user_id,),
            ).fetchone()[0]
            ageing = connection.execute(
                """
                SELECT MIN(s.enqueued_at_ms + ?)
                FROM one_shot_schedules s
                JOIN queue_items q ON q.logical_run_id = s.enqueued_logical_run_id
                WHERE s.user_id = ? AND s.status = 'enqueued'
                  AND s.promoted_to_p1_at_ms IS NULL
                  AND q.state = 'queued' AND q.priority_rank = 2
                """,
                (_OVERDUE_PROMOTION_MS, user_id),
            ).fetchone()[0]
            promotion_blocked = connection.execute(
                """
                SELECT 1
                FROM one_shot_schedules s
                JOIN queue_items q ON q.logical_run_id = s.enqueued_logical_run_id
                WHERE s.user_id = ? AND s.status = 'enqueued'
                  AND s.promoted_to_p1_at_ms IS NOT NULL
                  AND q.state NOT IN ('completed', 'quarantined', 'failed_terminal')
                  AND NOT EXISTS (
                      SELECT 1 FROM checkpoints c
                      WHERE c.logical_run_id = q.logical_run_id
                  )
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        if promotion_blocked is not None and ageing is not None:
            ageing = self._clock_ms() + 1000
        values = [int(value) for value in (active, ageing) if value is not None]
        return min(values) if values else None

    def prepared_input(
        self, *, user_id: str, logical_run_id: str
    ) -> PreparedScheduleInput:
        with self._state._connect() as connection:
            row = connection.execute(
                """
                SELECT s.* FROM one_shot_schedules s
                JOIN queue_items q ON q.logical_run_id = s.enqueued_logical_run_id
                WHERE s.user_id = ? AND q.logical_run_id = ?
                  AND q.kind = 'p2_scheduled' AND s.status = 'enqueued'
                """,
                (user_id, logical_run_id),
            ).fetchone()
            if row is None:
                raise DurableStateError("scheduled Tick lost its one-shot schedule")
            return PreparedScheduleInput(self._record(connection, row))

    def _mutate(
        self,
        *,
        user_id: str,
        schedule_id: str,
        action_id: str,
        operation: Literal["edit", "cancel", "run_now"],
        intent: object,
        apply: Callable[[Any, Any, int], ScheduleRecord],
    ) -> ScheduleRecord:
        intent_fingerprint = _fingerprint(intent)
        now_ms = self._clock_ms()
        with self._state._transaction() as connection:
            replay = connection.execute(
                "SELECT * FROM one_shot_schedule_mutations WHERE action_id = ?",
                (action_id,),
            ).fetchone()
            if replay is not None:
                if (
                    str(replay["user_id"]) != user_id
                    or str(replay["schedule_id"]) != schedule_id
                    or str(replay["operation"]) != operation
                    or str(replay["intent_fingerprint"]) != intent_fingerprint
                ):
                    raise ReceiptConflict(
                        "schedule mutation identity was reused with different intent"
                    )
                return self._record_from_tool_result(
                    connection, json.loads(str(replay["result_json"]))
                )
            same_intent = connection.execute(
                """
                SELECT * FROM one_shot_schedule_mutations
                WHERE user_id = ? AND schedule_id = ? AND operation = ?
                  AND intent_fingerprint = ?
                """,
                (user_id, schedule_id, operation, intent_fingerprint),
            ).fetchone()
            if same_intent is not None:
                return self._record_from_tool_result(
                    connection, json.loads(str(same_intent["result_json"]))
                )
            row = connection.execute(
                """
                SELECT * FROM one_shot_schedules
                WHERE user_id = ? AND schedule_id = ?
                """,
                (user_id, schedule_id),
            ).fetchone()
            if row is None:
                raise KeyError(schedule_id)
            record = apply(connection, row, now_ms)
            connection.execute(
                """
                INSERT INTO one_shot_schedule_mutations (
                    action_id, user_id, schedule_id, operation,
                    intent_fingerprint, result_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    user_id,
                    schedule_id,
                    operation,
                    intent_fingerprint,
                    _canonical_json(record.to_tool_dict()),
                    now_ms,
                ),
            )
        self._changed()
        return record

    def _apply_edit(
        self,
        connection: Any,
        row: Any,
        *,
        now_ms: int,
        normalized_due: tuple[int, str] | None,
        timezone_name: str | None,
        normalized_reason: str | None,
    ) -> ScheduleRecord:
        if str(row["status"]) != "active":
            raise DurableStateError("only an active schedule can be edited")
        due_at_ms = int(row["due_at_ms"])
        due_input = str(row["due_time_input"])
        zone = str(row["timezone_name"])
        if normalized_due is not None and timezone_name is not None:
            due_at_ms, due_input = normalized_due
            zone = timezone_name
        reason = str(row["reason"]) if normalized_reason is None else normalized_reason
        connection.execute(
            """
            UPDATE one_shot_schedules
            SET due_at_ms = ?, timezone_name = ?, due_time_input = ?,
                reason = ?, updated_at_ms = ?
            WHERE schedule_id = ? AND status = 'active'
            """,
            (due_at_ms, zone, due_input, reason, now_ms, row["schedule_id"]),
        )
        updated = connection.execute(
            "SELECT * FROM one_shot_schedules WHERE schedule_id = ?",
            (row["schedule_id"],),
        ).fetchone()
        assert updated is not None
        return self._record(connection, updated)

    def _apply_cancel(self, connection: Any, row: Any, now_ms: int) -> ScheduleRecord:
        if str(row["status"]) != "active":
            raise DurableStateError("only an active schedule can be cancelled")
        connection.execute(
            """
            UPDATE one_shot_schedules
            SET status = 'cancelled', updated_at_ms = ?, completed_at_ms = ?
            WHERE schedule_id = ? AND status = 'active'
            """,
            (now_ms, now_ms, row["schedule_id"]),
        )
        updated = connection.execute(
            "SELECT * FROM one_shot_schedules WHERE schedule_id = ?",
            (row["schedule_id"],),
        ).fetchone()
        assert updated is not None
        return self._record(connection, updated)

    def _enqueue_locked(
        self, connection: Any, *, row: Any, now_ms: int
    ) -> ScheduleRecord:
        if str(row["status"]) != "active":
            if str(row["status"]) == "enqueued":
                return self._record(connection, row)
            raise DurableStateError("only an active schedule can be enqueued")
        tick = self._tick_for(row, queued_at_ms=now_ms)
        self._state._insert_tick_locked(connection, tick=tick, now_ms=now_ms)
        connection.execute(
            """
            UPDATE one_shot_schedules
            SET status = 'enqueued', enqueued_tick_id = ?,
                enqueued_logical_run_id = ?, enqueued_at_ms = ?, updated_at_ms = ?
            WHERE schedule_id = ? AND status = 'active'
            """,
            (
                tick.payload.tick_id,
                tick.payload.logical_run_id,
                now_ms,
                now_ms,
                row["schedule_id"],
            ),
        )
        updated = connection.execute(
            "SELECT * FROM one_shot_schedules WHERE schedule_id = ?",
            (row["schedule_id"],),
        ).fetchone()
        assert updated is not None
        return self._record(connection, updated)

    @staticmethod
    def _tick_for(row: Any, *, queued_at_ms: int) -> Tick:
        schedule_id = str(row["schedule_id"])
        tick_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"thine-schedule-tick:{schedule_id}")
        )
        logical_run_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"thine-schedule-run:{schedule_id}")
        )
        return Tick.from_dict({
            "schema_version": _VERSION,
            "tick_id": tick_id,
            "user_id": str(row["user_id"]),
            "logical_run_id": logical_run_id,
            "kind": "p2_scheduled",
            "priority": "p2",
            "occurred_at_ms": int(row["due_at_ms"]),
            "received_at_ms": queued_at_ms,
            "queued_at_ms": queued_at_ms,
            "source_ref": {"kind": "schedule", "id": schedule_id},
            "causation_id": str(row["creator_tick_id"]),
            "correlation_id": schedule_id,
            "attempt_ordinal": 1,
            "lease": None,
            "communication_allowance_snapshot": None,
            "payload": {"payload_kind": "schedule", "reference_id": schedule_id},
            "extensions": {},
        })

    @staticmethod
    def _validate_reason(reason: str) -> str:
        if not isinstance(reason, str):
            raise ValueError("reason must be a string")
        normalized = reason.strip()
        if not normalized:
            raise ValueError("reason must not be empty")
        if len(normalized) > 1000:
            raise ValueError("reason exceeds 1000 characters")
        return normalized

    @staticmethod
    def _record(connection: Any, row: Any) -> ScheduleRecord:
        attempt_ordinal = 1
        if row["enqueued_logical_run_id"] is not None:
            attempt = connection.execute(
                "SELECT COALESCE(MAX(ordinal), 1) FROM attempts WHERE logical_run_id = ?",
                (row["enqueued_logical_run_id"],),
            ).fetchone()
            attempt_ordinal = int(attempt[0])
        return ScheduleRecord(
            schedule_id=str(row["schedule_id"]),
            user_id=str(row["user_id"]),
            due_at_ms=int(row["due_at_ms"]),
            timezone_name=str(row["timezone_name"]),
            due_time_input=str(row["due_time_input"]),
            created_at_ms=int(row["created_at_ms"]),
            updated_at_ms=int(row["updated_at_ms"]),
            status=str(row["status"]),
            reason=str(row["reason"]),
            creator_tick_id=str(row["creator_tick_id"]),
            originating_run_id=str(row["originating_run_id"]),
            originating_action_id=str(row["originating_action_id"]),
            intent_fingerprint=str(row["intent_fingerprint"]),
            enqueued_tick_id=(
                None
                if row["enqueued_tick_id"] is None
                else str(row["enqueued_tick_id"])
            ),
            enqueued_logical_run_id=(
                None
                if row["enqueued_logical_run_id"] is None
                else str(row["enqueued_logical_run_id"])
            ),
            enqueued_at_ms=(
                None if row["enqueued_at_ms"] is None else int(row["enqueued_at_ms"])
            ),
            promoted_to_p1_at_ms=(
                None
                if row["promoted_to_p1_at_ms"] is None
                else int(row["promoted_to_p1_at_ms"])
            ),
            completed_at_ms=(
                None if row["completed_at_ms"] is None else int(row["completed_at_ms"])
            ),
            attempt_ordinal=attempt_ordinal,
        )

    @staticmethod
    def _record_from_tool_result(
        connection: Any, result: Mapping[str, object]
    ) -> ScheduleRecord:
        schedule = result.get("schedule")
        if not isinstance(schedule, dict) or not isinstance(
            schedule.get("schedule_id"), str
        ):
            raise DurableStateError("stored schedule mutation result is invalid")
        row = connection.execute(
            "SELECT * FROM one_shot_schedules WHERE schedule_id = ?",
            (schedule["schedule_id"],),
        ).fetchone()
        if row is None:
            raise DurableStateError("stored schedule mutation lost its schedule")
        return OneShotScheduleService._record(connection, row)


class ScheduleToolBinding:
    """Typed deferred schedule tools scoped to one configured local user."""

    def __init__(
        self,
        *,
        state: DurableRunState,
        service: OneShotScheduleService,
        user_id: str,
    ) -> None:
        self._state = state
        self._service = service
        self._user_id = user_id

    def register(self, *, registry_instance: Any | None = None) -> None:
        from tools.registry import registry

        active_registry = registry_instance or registry
        scope = active_registry.current_scope_key()
        for name, schema, handler in (
            (SCHEDULE_CREATE_TOOL_NAME, SCHEDULE_CREATE_TOOL_SCHEMA, self.create),
            (SCHEDULE_LIST_TOOL_NAME, SCHEDULE_LIST_TOOL_SCHEMA, self.list),
            (SCHEDULE_INSPECT_TOOL_NAME, SCHEDULE_INSPECT_TOOL_SCHEMA, self.inspect),
            (SCHEDULE_EDIT_TOOL_NAME, SCHEDULE_EDIT_TOOL_SCHEMA, self.edit),
            (SCHEDULE_CANCEL_TOOL_NAME, SCHEDULE_CANCEL_TOOL_SCHEMA, self.cancel),
            (SCHEDULE_RUN_NOW_TOOL_NAME, SCHEDULE_RUN_NOW_TOOL_SCHEMA, self.run_now),
        ):
            active_registry.register(
                name=name,
                toolset=SCHEDULE_TOOLSET,
                schema=schema,
                handler=handler,
                scope=scope,
            )

    def create(self, args: Mapping[str, object], **_kwargs: object) -> str:
        required = {"due_at", "timezone", "reason", "request_key"}
        if set(args) != required or not all(
            isinstance(args[key], str) for key in required
        ):
            return self._error("invalid_create_request")
        active = self._active_tick()
        if active is None:
            return self._error("no_active_tick")
        try:
            action_id = self._action_id(
                active.tick_id, "create", cast(str, args["request_key"])
            )
            record = self._service.create(
                user_id=self._user_id,
                creator_tick_id=active.tick_id,
                originating_run_id=active.logical_run_id,
                originating_action_id=action_id,
                due_at=cast(str, args["due_at"]),
                timezone_name=cast(str, args["timezone"]),
                reason=cast(str, args["reason"]),
            )
        except (ValueError, DurableStateError, ReceiptConflict) as exc:
            return self._error(type(exc).__name__, detail=str(exc))
        return self._ok(record)

    def list(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if set(args) - {"status"} or (
            "status" in args and not isinstance(args["status"], str)
        ):
            return self._error("invalid_list_request")
        try:
            records = self._service.list(
                self._user_id, status=cast(str | None, args.get("status"))
            )
        except ValueError as exc:
            return self._error("invalid_status", detail=str(exc))
        return _canonical_json({
            "ok": True,
            "schedules": [record.to_tool_dict() for record in records],
        })

    def inspect(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if set(args) != {"schedule_id"} or not isinstance(args["schedule_id"], str):
            return self._error("invalid_inspect_request")
        try:
            record = self._service.inspect(
                self._user_id, cast(str, args["schedule_id"])
            )
        except KeyError:
            return self._error("schedule_not_found")
        return self._ok(record)

    def edit(self, args: Mapping[str, object], **_kwargs: object) -> str:
        allowed = {"schedule_id", "due_at", "timezone", "reason", "request_key"}
        if (
            set(args) - allowed
            or not {"schedule_id", "request_key"}.issubset(args)
            or not all(isinstance(value, str) for value in args.values())
        ):
            return self._error("invalid_edit_request")
        active = self._active_tick()
        if active is None:
            return self._error("no_active_tick")
        try:
            action_id = self._action_id(
                active.tick_id, "edit", cast(str, args["request_key"])
            )
            record = self._service.edit(
                user_id=self._user_id,
                schedule_id=cast(str, args["schedule_id"]),
                action_id=action_id,
                due_at=cast(str | None, args.get("due_at")),
                timezone_name=cast(str | None, args.get("timezone")),
                reason=cast(str | None, args.get("reason")),
            )
        except KeyError:
            return self._error("schedule_not_found")
        except (ValueError, DurableStateError, ReceiptConflict) as exc:
            return self._error(type(exc).__name__, detail=str(exc))
        return self._ok(record)

    def cancel(self, args: Mapping[str, object], **_kwargs: object) -> str:
        return self._simple_mutation("cancel", args)

    def run_now(self, args: Mapping[str, object], **_kwargs: object) -> str:
        return self._simple_mutation("run_now", args)

    def _simple_mutation(self, operation: str, args: Mapping[str, object]) -> str:
        if set(args) != {"schedule_id", "request_key"} or not all(
            isinstance(value, str) for value in args.values()
        ):
            return self._error(f"invalid_{operation}_request")
        active = self._active_tick()
        if active is None:
            return self._error("no_active_tick")
        try:
            action_id = self._action_id(
                active.tick_id, operation, cast(str, args["request_key"])
            )
            if operation == "cancel":
                record = self._service.cancel(
                    user_id=self._user_id,
                    schedule_id=cast(str, args["schedule_id"]),
                    action_id=action_id,
                )
            else:
                record = self._service.run_now(
                    user_id=self._user_id,
                    schedule_id=cast(str, args["schedule_id"]),
                    action_id=action_id,
                )
        except KeyError:
            return self._error("schedule_not_found")
        except (ValueError, DurableStateError, ReceiptConflict) as exc:
            return self._error(type(exc).__name__, detail=str(exc))
        return self._ok(record)

    def _active_tick(self) -> _ActiveTick | None:
        with self._state._connect() as connection:
            rows = connection.execute(
                """
                SELECT q.tick_id, q.logical_run_id, a.attempt_id
                FROM queue_items q
                JOIN attempts a ON a.logical_run_id = q.logical_run_id
                WHERE q.user_id = ? AND q.state = 'running'
                  AND a.status = 'running'
                ORDER BY q.enqueue_sequence
                """,
                (self._user_id,),
            ).fetchall()
        if len(rows) != 1:
            return None
        row = rows[0]
        return _ActiveTick(
            tick_id=str(row["tick_id"]),
            logical_run_id=str(row["logical_run_id"]),
            attempt_id=str(row["attempt_id"]),
        )

    @staticmethod
    def _action_id(tick_id: str, operation: str, request_key: str) -> str:
        if not request_key or len(request_key) > 64:
            raise ValueError("request_key must contain 1-64 characters")
        return "action:schedule:" + str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"thine-schedule-action:{tick_id}:{operation}:{request_key}",
            )
        )

    @staticmethod
    def _ok(record: ScheduleRecord) -> str:
        return _canonical_json({"ok": True, **record.to_tool_dict()})

    @staticmethod
    def _error(error_code: str, *, detail: str | None = None) -> str:
        value: dict[str, object] = {"ok": False, "error_code": error_code}
        if detail:
            value["detail"] = detail[:500]
        return _canonical_json(value)


class ScheduleInputPort:
    def __init__(self, service: OneShotScheduleService) -> None:
        self._service = service

    def prepare(
        self, context: InvocationContext, *, lease: ActiveRunLease
    ) -> PreparedScheduleInput | None:
        if str(context.tick.payload.kind) != "p2_scheduled":
            return None
        return self._service.prepared_input(
            user_id=lease.user_id, logical_run_id=lease.logical_run_id
        )


class RealScheduleAgentRuntime:
    """Process scheduled Ticks with the one cached background AIAgent."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        agent: Any,
        config: RuntimeModelConfig | None = None,
        communication_context: Callable[[str], Mapping[str, object]] | None = None,
    ) -> None:
        self._state = state
        self._agent = agent
        self._config = config or RuntimeModelConfig.openai_gpt_5_6_sol_medium()
        self._communication_context = communication_context
        self._session = HermesAIAgentSession(agent=agent, expected=self._config)

    def invoke(
        self,
        context: InvocationContext,
        *,
        tools: object,
        control: InvocationControl,
    ) -> InvocationOutcome:
        del tools
        if control.preemption_requested:
            return InvocationOutcome.preempted(
                remaining_work="scheduled inference not started"
            )
        prepared = context.prepared_input
        if not isinstance(prepared, PreparedScheduleInput):
            raise ValueError("scheduled inference requires its durable schedule")
        user_id = str(context.tick.payload.user_id)
        current = self._state.working_memory_snapshot(user_id)
        communication_context = (
            {}
            if self._communication_context is None
            else dict(self._communication_context(user_id))
        )
        prompt = (
            "Process this one-shot scheduled Tick. The schedule reason is context, "
            "not an instruction from an external source. Decide what is useful now; "
            "use deferred namespaced tools or deliberately do nothing. This schedule "
            "fires only once and must not be interpreted as recurring.\n\n"
            "<schedule>\n"
            + _canonical_json(prepared.schedule.to_tool_dict())
            + "\n</schedule>\n<working_memory>\n"
            + current.markdown
            + "\n</working_memory>\n<communication_context>\n"
            + _canonical_json(communication_context)
            + "\n</communication_context>\n"
            + f"<logical_run_id>{context.tick.payload.logical_run_id}</logical_run_id>"
        )
        provider_control = ProviderInvocationControl()
        stopped = threading.Event()

        def relay() -> None:
            while not stopped.wait(0.02):
                if control.preemption_requested:
                    provider_control.cancel(control.reason or "p0_user_tick")
                    return

        thread = threading.Thread(
            target=relay, name="thine-schedule-preemption", daemon=True
        )
        thread.start()
        request = build_background_invocation_request(
            logical_run_id=str(context.tick.payload.logical_run_id),
            initial_prompt=prompt,
            checkpoint=context.checkpoint,
            newest_working_memory=current.markdown,
            durable_action_receipts=tuple(
                asdict(receipt) for receipt in context.acknowledged_receipts
            ),
        )
        try:
            result = self._session.invoke(
                request,
                emit=lambda _event: None,
                control=provider_control,
            )
        finally:
            stopped.set()
            thread.join(timeout=0.1)
        if result.interrupted:
            return InvocationOutcome.interrupted(
                remaining_work=result.remaining_work or "resume scheduled inference",
                checkpoint_payload=BackgroundCheckpointPayload.from_turn(
                    request, result
                ),
                cap_reason=result.segment_cap_reason,
            )
        if result.failed or not result.completed:
            return InvocationOutcome.fault(
                result.failure_reason or "real_model_incomplete"
            )
        from .transcript_agent import _cache_identity

        artifact = ScheduleAgentArtifact(
            agent=self._agent,
            result=result,
            current_memory=current,
            cache_identity=_cache_identity(self._agent),
            provider=self._config.provider,
            model=self._config.model,
            api_mode=self._config.api_mode,
            reasoning_effort=self._config.reasoning_effort,
            tool_discoveries=tuple(
                str(message.get("tool_name") or message.get("name") or "")
                for message in result.context_messages
                if isinstance(message, dict) and message.get("role") == "tool"
            ),
        )
        return InvocationOutcome.no_action(finalization_context=artifact)


class FakeScheduleNoActionRuntime:
    def __init__(self, outcomes: list[InvocationOutcome] | None = None) -> None:
        self._outcomes = list(outcomes or [])
        self.invocations: list[InvocationContext] = []

    def invoke(
        self, context: InvocationContext, *, tools: object, control: InvocationControl
    ) -> InvocationOutcome:
        del tools
        if control.preemption_requested:
            return InvocationOutcome.preempted(
                remaining_work="scheduled inference not started"
            )
        if not isinstance(context.prepared_input, PreparedScheduleInput):
            raise ValueError("scheduled inference requires its durable schedule")
        self.invocations.append(context)
        return (
            self._outcomes.pop(0) if self._outcomes else InvocationOutcome.no_action()
        )


class ScheduleRunFinalizer:
    """Atomically publish Working Memory and complete one scheduled Tick."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._state = state
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def resume_pending(self, user_id: str) -> RunFinalizationResult | None:
        return self.finalize_quarantine(user_id)

    def finalize_quarantine(self, user_id: str) -> RunFinalizationResult | None:
        now_ms = self._clock_ms()
        with self._state._transaction() as connection:
            row = connection.execute(
                """
                SELECT s.schedule_id, q.tick_id, q.logical_run_id,
                       COALESCE(MAX(a.ordinal), 3) AS attempt_ordinal
                FROM one_shot_schedules s
                JOIN queue_items q ON q.logical_run_id = s.enqueued_logical_run_id
                LEFT JOIN attempts a ON a.logical_run_id = q.logical_run_id
                WHERE s.user_id = ? AND s.status = 'enqueued'
                  AND q.state IN ('quarantined', 'failed_terminal')
                GROUP BY s.schedule_id, q.tick_id, q.logical_run_id
                ORDER BY s.due_at_ms, s.created_at_ms LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE one_shot_schedules
                SET status = 'failed_terminal', updated_at_ms = ?, completed_at_ms = ?
                WHERE schedule_id = ? AND status = 'enqueued'
                """,
                (now_ms, now_ms, row["schedule_id"]),
            )
        return RunFinalizationResult(
            tick_id=str(row["tick_id"]),
            logical_run_id=str(row["logical_run_id"]),
            attempt_ordinal=int(row["attempt_ordinal"]),
            status="failed_terminal",
        )

    def finalize(
        self,
        context: InvocationContext,
        outcome: InvocationOutcome,
        *,
        lease: ActiveRunLease,
    ) -> RunFinalizationResult:
        if str(context.tick.payload.kind) != "p2_scheduled":
            raise ValueError("schedule finalizer received another Tick kind")
        if not isinstance(context.prepared_input, PreparedScheduleInput):
            raise ValueError("schedule finalizer requires its durable schedule")
        if outcome.status != "completed":
            raise ValueError("only completed scheduled inference can finalize")
        now_ms = self._clock_ms()
        artifact = outcome.finalization_context
        staged = _StagedMemory()
        hook_outcome = None
        if isinstance(artifact, ScheduleAgentArtifact):
            hook_outcome = StopHookRunner().finalize(
                run_id=lease.logical_run_id,
                current=artifact.current_memory,
                context=HermesCachedStopHookContext(
                    agent=artifact.agent,
                    conversation_history=list(artifact.result.context_messages),
                    cache_identity=artifact.cache_identity,
                ),
                store=staged,
                interrupted=False,
            )
        else:
            current = self._state.working_memory_snapshot(lease.user_id)
            staged.mark_unchanged(
                expected_version=current.version, run_id=lease.logical_run_id
            )
        with self._state._transaction() as connection:
            item = self._state._require_active_owner(
                connection,
                user_id=lease.user_id,
                logical_run_id=lease.logical_run_id,
                owner=lease.owner,
                attempt_id=lease.attempt_id,
                lease_token=lease.lease_token,
                now_ms=now_ms,
            )
            schedule = connection.execute(
                """
                SELECT * FROM one_shot_schedules
                WHERE user_id = ? AND enqueued_logical_run_id = ?
                  AND status = 'enqueued'
                """,
                (lease.user_id, lease.logical_run_id),
            ).fetchone()
            if item["kind"] != "p2_scheduled" or schedule is None:
                raise DurableStateError("scheduled run lost its durable schedule")
            memory_version, memory_token_count, memory_outcome = (
                self._commit_memory_locked(
                    connection,
                    lease=lease,
                    artifact=artifact,
                    staged=staged,
                    hook_outcome=hook_outcome,
                    now_ms=now_ms,
                )
            )
            connection.execute(
                """
                INSERT INTO decision_outcomes (
                    decision_receipt_id, user_id, logical_run_id, outcome,
                    visible_action_intent_count, recorded_at_ms
                ) VALUES (?, ?, ?, 'no_action', 0, ?)
                """,
                (
                    f"decision-receipt:{lease.logical_run_id}",
                    lease.user_id,
                    lease.logical_run_id,
                    now_ms,
                ),
            )
            if isinstance(artifact, ScheduleAgentArtifact):
                assert hook_outcome is not None
                connection.execute(
                    """
                    INSERT INTO agent_run_inspections (
                        user_id, logical_run_id, attempt_id, provider, model,
                        api_mode, reasoning_effort, decision_outcome, final_output,
                        tool_discoveries_json, usage_json, stop_hook_outcome,
                        stop_hook_cache_identity_json, memory_version,
                        memory_token_count, recorded_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'no_action', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lease.user_id,
                        lease.logical_run_id,
                        lease.attempt_id,
                        artifact.provider,
                        artifact.model,
                        artifact.api_mode,
                        artifact.reasoning_effort,
                        str(artifact.result.final_output or ""),
                        _canonical_json(list(artifact.tool_discoveries)),
                        _canonical_json(dict(artifact.result.usage)),
                        hook_outcome.kind.value,
                        _canonical_json(asdict(hook_outcome.cache_identity)),
                        memory_version,
                        memory_token_count,
                        now_ms,
                    ),
                )
            connection.execute(
                """
                UPDATE attempts SET status = 'succeeded', finished_at_ms = ?
                WHERE attempt_id = ? AND status = 'running'
                """,
                (now_ms, lease.attempt_id),
            )
            connection.execute(
                """
                UPDATE queue_items
                SET state = 'completed', completed_at_ms = ?, updated_at_ms = ?,
                    lease_owner = NULL, lease_token = NULL,
                    lease_expires_at_ms = NULL
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (now_ms, now_ms, lease.user_id, lease.logical_run_id),
            )
            connection.execute(
                """
                UPDATE one_shot_schedules
                SET status = 'completed', updated_at_ms = ?, completed_at_ms = ?
                WHERE schedule_id = ? AND status = 'enqueued'
                """,
                (
                    now_ms,
                    now_ms,
                    context.prepared_input.schedule.schedule_id,
                ),
            )
        return RunFinalizationResult(
            tick_id=str(context.tick.payload.tick_id),
            logical_run_id=lease.logical_run_id,
            attempt_ordinal=lease.attempt_ordinal,
            status="completed",
        )

    @staticmethod
    def _commit_memory_locked(
        connection: Any,
        *,
        lease: ActiveRunLease,
        artifact: object,
        staged: _StagedMemory,
        hook_outcome: object,
        now_ms: int,
    ) -> tuple[int, int | None, str]:
        connection.execute(
            """
            INSERT OR IGNORE INTO working_memory_state (
                user_id, version, markdown, token_count, last_run_id
            ) VALUES (?, 0, '', NULL, NULL)
            """,
            (lease.user_id,),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO working_memory_versions (
                user_id, version, markdown, configured_model_token_count,
                tokenizer_status, logical_run_id, committed_at_ms
            ) VALUES (?, 0, '', NULL, 'unresolved_fail_closed', NULL, 0)
            """,
            (lease.user_id,),
        )
        memory = connection.execute(
            "SELECT * FROM working_memory_state WHERE user_id = ?",
            (lease.user_id,),
        ).fetchone()
        assert memory is not None
        memory_version = int(memory["version"])
        if isinstance(artifact, ScheduleAgentArtifact) and (
            memory_version != artifact.current_memory.version
        ):
            raise DurableStateError("working memory changed during schedule Stop Hook")
        memory_outcome = "unchanged"
        memory_token_count = (
            None if memory["token_count"] is None else int(memory["token_count"])
        )
        if staged.markdown is not None:
            if (
                staged.token_count is None
                or not hasattr(hook_outcome, "kind")
                or cast(Any, hook_outcome).kind is not StopHookOutcomeKind.COMMITTED
            ):
                raise DurableStateError(
                    "changed schedule memory requires exact configured-model tokens"
                )
            if not 0 <= staged.token_count <= 16_000:
                raise DurableStateError("changed schedule memory exceeds 16K tokens")
            memory_version += 1
            memory_token_count = staged.token_count
            connection.execute(
                """
                INSERT INTO working_memory_versions (
                    user_id, version, markdown, configured_model_token_count,
                    tokenizer_status, logical_run_id, committed_at_ms
                ) VALUES (?, ?, ?, ?, 'exact', ?, ?)
                """,
                (
                    lease.user_id,
                    memory_version,
                    staged.markdown,
                    staged.token_count,
                    lease.logical_run_id,
                    now_ms,
                ),
            )
            updated = connection.execute(
                """
                UPDATE working_memory_state
                SET version = ?, markdown = ?, token_count = ?, last_run_id = ?
                WHERE user_id = ? AND version = ?
                """,
                (
                    memory_version,
                    staged.markdown,
                    staged.token_count,
                    lease.logical_run_id,
                    lease.user_id,
                    memory_version - 1,
                ),
            )
            if updated.rowcount != 1:
                raise DurableStateError("working memory changed during finalization")
            memory_outcome = "written"
        else:
            connection.execute(
                """
                INSERT INTO working_memory_unchanged (
                    marker_id, user_id, logical_run_id, expected_version, recorded_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    f"memory-unchanged:{lease.logical_run_id}",
                    lease.user_id,
                    lease.logical_run_id,
                    memory_version,
                    now_ms,
                ),
            )
            connection.execute(
                """
                UPDATE working_memory_state SET last_run_id = ?
                WHERE user_id = ? AND version = ?
                """,
                (lease.logical_run_id, lease.user_id, memory_version),
            )
        return memory_version, memory_token_count, memory_outcome


class OneShotScheduleDriver:
    """Own one re-armed timer for due firing and deterministic P1 ageing."""

    def __init__(
        self,
        *,
        service: OneShotScheduleService,
        user_id: str,
        wake_coordinator: Callable[[], None],
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._service = service
        self._user_id = user_id
        self._wake_coordinator = wake_coordinator
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._changed = threading.Event()
        self._closed = threading.Event()
        self._service.set_change_callback(self.notify_changed)
        self._thread = threading.Thread(
            target=self._run,
            name="thine-one-shot-schedule-driver",
            daemon=True,
        )
        self._thread.start()

    def notify_changed(self) -> None:
        self._changed.set()

    def close(self) -> None:
        self._closed.set()
        self._changed.set()
        self._thread.join(timeout=2)
        self._service.set_change_callback(None)

    def scan_now(self) -> tuple[str | None, str | None]:
        now_ms = self._clock_ms()
        fired = self._service.fire_due_once(self._user_id, now_ms=now_ms)
        promoted = self._service.promote_oldest_overdue(self._user_id, now_ms=now_ms)
        if fired is not None or promoted is not None:
            self._wake_coordinator()
        return fired, promoted

    def _run(self) -> None:
        while not self._closed.is_set():
            self._changed.clear()
            try:
                self.scan_now()
                wake_at = self._service.next_wake_at_ms(self._user_id)
            except Exception:
                wake_at = self._clock_ms() + 1000
            if self._closed.is_set():
                return
            if wake_at is None:
                self._changed.wait()
            else:
                delay = max((wake_at - self._clock_ms()) / 1000, 0.01)
                self._changed.wait(delay)


__all__ = [
    "FakeScheduleNoActionRuntime",
    "OneShotScheduleDriver",
    "OneShotScheduleService",
    "PreparedScheduleInput",
    "RealScheduleAgentRuntime",
    "SCHEDULE_CANCEL_TOOL_NAME",
    "SCHEDULE_CREATE_TOOL_NAME",
    "SCHEDULE_EDIT_TOOL_NAME",
    "SCHEDULE_INSPECT_TOOL_NAME",
    "SCHEDULE_LIST_TOOL_NAME",
    "SCHEDULE_RUN_NOW_TOOL_NAME",
    "ScheduleInputPort",
    "ScheduleRecord",
    "ScheduleRunFinalizer",
    "ScheduleToolBinding",
    "normalize_due_time",
]
