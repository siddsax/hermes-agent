"""Durable SQLite state for one local Thine Harness profile.

This module owns queue, lease, Attempt, checkpoint, fake-tool receipt, and
quarantine persistence.  It intentionally knows nothing about feature
databases or Hermes' model loop; callers interact through transactional
methods scoped by ``user_id``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import secrets
import sqlite3
from typing import Any, Iterator, Literal

from hermes_constants import get_hermes_home

from .contracts.runtime import Tick


SCHEMA_VERSION = 1


class DurableStateError(RuntimeError):
    """The durable Harness state could not honor a requested transition."""


class ReceiptConflict(DurableStateError):
    """A stable action identity was reused for a different intent."""


@dataclass(frozen=True)
class ToolReceiptRecord:
    receipt_id: str
    user_id: str
    logical_run_id: str
    action_id: str
    intent_fingerprint: str
    provider_reference: str
    result: dict[str, Any]
    acknowledged_at_ms: int


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    logical_run_id: str
    cause: str
    remaining_work: str
    completed_receipt_ids: tuple[str, ...]
    updated_at_ms: int


@dataclass(frozen=True)
class LeasedRun:
    tick: Tick
    attempt_id: str
    attempt_ordinal: int
    lease_token: str
    checkpoint: CheckpointRecord | None
    acknowledged_receipts: tuple[ToolReceiptRecord, ...]


@dataclass(frozen=True)
class QueueDiagnostic:
    tick_id: str
    logical_run_id: str
    kind: str
    priority: str
    state: str
    enqueue_sequence: int


@dataclass(frozen=True)
class LeaseDiagnostic:
    logical_run_id: str
    owner: str
    expires_at_ms: int
    state: str


@dataclass(frozen=True)
class AttemptDiagnostic:
    attempt_id: str
    logical_run_id: str
    ordinal: int
    status: str
    failure_code: str | None
    started_at_ms: int
    finished_at_ms: int | None


@dataclass(frozen=True)
class QuarantineDiagnostic:
    quarantine_id: str
    logical_run_id: str
    tick_id: str
    source_kind: str
    source_id: str
    attempt_ordinal: int
    failure_code: str
    quarantined_at_ms: int


@dataclass(frozen=True)
class StateDiagnostics:
    queue: tuple[QueueDiagnostic, ...]
    leases: tuple[LeaseDiagnostic, ...]
    attempts: tuple[AttemptDiagnostic, ...]
    checkpoints: tuple[CheckpointRecord, ...]
    receipts: tuple[ToolReceiptRecord, ...]
    quarantines: tuple[QuarantineDiagnostic, ...]


def default_database_path() -> Path:
    return get_hermes_home() / "thine-harness" / "run-state.sqlite3"


class DurableRunState:
    """Transactional state repository for the single-flight coordinator."""

    def __init__(
        self, path: str | Path | None = None, *, lease_duration_ms: int = 30_000
    ):
        if lease_duration_ms <= 0:
            raise ValueError("lease_duration_ms must be positive")
        self.path = Path(path) if path is not None else default_database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lease_duration_ms = lease_duration_ms
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=5,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise DurableStateError(
                    f"run-state schema {version} is newer than supported {SCHEMA_VERSION}"
                )
            if version == SCHEMA_VERSION:
                return
            connection.executescript(
                f"""
                BEGIN IMMEDIATE;
                CREATE TABLE queue_items (
                    enqueue_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    tick_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    logical_run_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    priority_rank INTEGER NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    tick_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_token TEXT,
                    lease_expires_at_ms INTEGER,
                    completed_at_ms INTEGER,
                    updated_at_ms INTEGER NOT NULL
                );
                CREATE INDEX queue_eligible
                    ON queue_items(user_id, state, priority_rank, enqueue_sequence);
                CREATE UNIQUE INDEX queue_one_active_per_user
                    ON queue_items(user_id)
                    WHERE state IN ('leased', 'running');

                CREATE TABLE attempts (
                    attempt_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    logical_run_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL CHECK (ordinal BETWEEN 1 AND 3),
                    status TEXT NOT NULL,
                    failure_code TEXT,
                    started_at_ms INTEGER NOT NULL,
                    finished_at_ms INTEGER,
                    UNIQUE(logical_run_id, ordinal),
                    FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
                );
                CREATE INDEX attempts_by_user
                    ON attempts(user_id, logical_run_id, ordinal);

                CREATE TABLE checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    logical_run_id TEXT NOT NULL,
                    cause TEXT NOT NULL,
                    remaining_work TEXT NOT NULL,
                    completed_receipt_ids_json TEXT NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
                );
                CREATE INDEX checkpoints_by_user_run
                    ON checkpoints(user_id, logical_run_id, updated_at_ms DESC);

                CREATE TABLE tool_receipts (
                    receipt_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    logical_run_id TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    intent_fingerprint TEXT NOT NULL,
                    provider_reference TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    acknowledged_at_ms INTEGER NOT NULL,
                    PRIMARY KEY(user_id, receipt_id),
                    UNIQUE(user_id, action_id),
                    FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
                );
                CREATE INDEX receipts_by_user_run
                    ON tool_receipts(user_id, logical_run_id, acknowledged_at_ms);

                CREATE TABLE quarantines (
                    quarantine_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    logical_run_id TEXT NOT NULL UNIQUE,
                    tick_id TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    attempt_ordinal INTEGER NOT NULL CHECK (attempt_ordinal = 3),
                    failure_code TEXT NOT NULL,
                    quarantined_at_ms INTEGER NOT NULL,
                    FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
                );
                CREATE INDEX quarantines_by_user
                    ON quarantines(user_id, quarantined_at_ms);
                PRAGMA user_version = {SCHEMA_VERSION};
                COMMIT;
                """
            )
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def enqueue(self, tick: Tick, *, now_ms: int) -> str:
        payload = tick.payload
        priority_rank = {"p0": 0, "p1": 1, "p2": 2}[str(payload.priority)]
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT tick_json FROM queue_items WHERE tick_id = ? AND user_id = ?",
                (payload.tick_id, payload.user_id),
            ).fetchone()
            if existing is not None:
                if json.loads(existing["tick_json"]) != tick.to_dict():
                    raise DurableStateError(
                        "tick_id was reused with a different payload"
                    )
                return str(payload.tick_id)
            connection.execute(
                """
                INSERT INTO queue_items (
                    tick_id, user_id, logical_run_id, kind, priority, priority_rank,
                    source_kind, source_id, tick_json, state, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)
                """,
                (
                    payload.tick_id,
                    payload.user_id,
                    payload.logical_run_id,
                    payload.kind,
                    payload.priority,
                    priority_rank,
                    payload.source_ref.kind,
                    payload.source_ref.id,
                    tick.to_json(),
                    now_ms,
                ),
            )
        return str(payload.tick_id)

    def lease_next(self, user_id: str, *, owner: str, now_ms: int) -> LeasedRun | None:
        with self._transaction() as connection:
            self._recover_expired_locked(connection, user_id=user_id, now_ms=now_ms)
            active = connection.execute(
                """
                SELECT 1 FROM queue_items
                WHERE user_id = ? AND state IN ('leased', 'running')
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active is not None:
                return None
            item = connection.execute(
                """
                SELECT * FROM queue_items
                WHERE user_id = ? AND state = 'queued'
                ORDER BY priority_rank ASC, enqueue_sequence ASC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if item is None:
                return None
            running_attempt = connection.execute(
                """
                SELECT * FROM attempts
                WHERE user_id = ? AND logical_run_id = ? AND status = 'running'
                """,
                (user_id, item["logical_run_id"]),
            ).fetchone()
            if running_attempt is None:
                failed_count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM attempts
                        WHERE user_id = ? AND logical_run_id = ? AND status = 'failed_fault'
                        """,
                        (user_id, item["logical_run_id"]),
                    ).fetchone()[0]
                )
                ordinal = failed_count + 1
                if ordinal > 3:
                    raise DurableStateError("automatic fourth Attempt is forbidden")
                attempt_id = f"{item['logical_run_id']}:attempt:{ordinal}"
                connection.execute(
                    """
                    INSERT INTO attempts (
                        attempt_id, user_id, logical_run_id, ordinal, status,
                        started_at_ms
                    ) VALUES (?, ?, ?, ?, 'running', ?)
                    """,
                    (attempt_id, user_id, item["logical_run_id"], ordinal, now_ms),
                )
            else:
                attempt_id = str(running_attempt["attempt_id"])
                ordinal = int(running_attempt["ordinal"])
            lease_token = secrets.token_hex(16)
            connection.execute(
                """
                UPDATE queue_items
                SET state = 'running', lease_owner = ?, lease_token = ?,
                    lease_expires_at_ms = ?, updated_at_ms = ?
                WHERE logical_run_id = ? AND user_id = ? AND state = 'queued'
                """,
                (
                    owner,
                    lease_token,
                    now_ms + self.lease_duration_ms,
                    now_ms,
                    item["logical_run_id"],
                    user_id,
                ),
            )
            checkpoint = self._latest_checkpoint_locked(
                connection, user_id=user_id, logical_run_id=item["logical_run_id"]
            )
            receipts = self._receipts_locked(
                connection, user_id=user_id, logical_run_id=item["logical_run_id"]
            )
            tick = Tick.from_json(item["tick_json"])
            return LeasedRun(
                tick=tick,
                attempt_id=attempt_id,
                attempt_ordinal=ordinal,
                lease_token=lease_token,
                checkpoint=checkpoint,
                acknowledged_receipts=receipts,
            )

    def save_checkpoint_and_requeue(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        owner: str,
        attempt_id: str,
        lease_token: str,
        cause: str,
        remaining_work: str,
        completed_receipt_ids: tuple[str, ...],
        now_ms: int,
    ) -> CheckpointRecord:
        receipt_ids = tuple(dict.fromkeys(completed_receipt_ids))
        with self._transaction() as connection:
            self._require_active_owner(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                owner=owner,
                attempt_id=attempt_id,
                lease_token=lease_token,
                now_ms=now_ms,
            )
            checkpoint_ordinal = (
                int(
                    connection.execute(
                        """
                    SELECT COUNT(*) FROM checkpoints
                    WHERE user_id = ? AND logical_run_id = ?
                    """,
                        (user_id, logical_run_id),
                    ).fetchone()[0]
                )
                + 1
            )
            checkpoint_id = f"{logical_run_id}:checkpoint:{checkpoint_ordinal}"
            connection.execute(
                """
                INSERT INTO checkpoints (
                    checkpoint_id, user_id, logical_run_id, cause, remaining_work,
                    completed_receipt_ids_json, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    user_id,
                    logical_run_id,
                    cause,
                    remaining_work,
                    json.dumps(receipt_ids, separators=(",", ":")),
                    now_ms,
                ),
            )
            connection.execute(
                """
                UPDATE queue_items
                SET state = 'queued', lease_owner = NULL, lease_token = NULL,
                    lease_expires_at_ms = NULL, updated_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (now_ms, user_id, logical_run_id),
            )
        return CheckpointRecord(
            checkpoint_id=checkpoint_id,
            logical_run_id=logical_run_id,
            cause=cause,
            remaining_work=remaining_work,
            completed_receipt_ids=receipt_ids,
            updated_at_ms=now_ms,
        )

    def renew_lease(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        owner: str,
        attempt_id: str,
        lease_token: str,
        now_ms: int,
    ) -> bool:
        """Extend one still-owned live lease without changing run state."""
        with self._transaction() as connection:
            changed = connection.execute(
                """
                UPDATE queue_items
                SET lease_expires_at_ms = ?, updated_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ? AND state = 'running'
                  AND lease_owner = ? AND lease_token = ?
                  AND lease_expires_at_ms > ?
                  AND EXISTS (
                    SELECT 1 FROM attempts
                    WHERE attempts.user_id = queue_items.user_id
                      AND attempts.logical_run_id = queue_items.logical_run_id
                      AND attempts.attempt_id = ? AND attempts.status = 'running'
                  )
                """,
                (
                    now_ms + self.lease_duration_ms,
                    now_ms,
                    user_id,
                    logical_run_id,
                    owner,
                    lease_token,
                    now_ms,
                    attempt_id,
                ),
            ).rowcount
        return changed == 1

    def complete(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        owner: str,
        attempt_id: str,
        lease_token: str,
        now_ms: int,
    ) -> None:
        with self._transaction() as connection:
            self._require_active_owner(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                owner=owner,
                attempt_id=attempt_id,
                lease_token=lease_token,
                now_ms=now_ms,
            )
            connection.execute(
                """
                UPDATE attempts
                SET status = 'succeeded', finished_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ? AND attempt_id = ?
                  AND status = 'running'
                """,
                (now_ms, user_id, logical_run_id, attempt_id),
            )
            connection.execute(
                """
                UPDATE queue_items
                SET state = 'completed', completed_at_ms = ?, lease_owner = NULL,
                    lease_token = NULL, lease_expires_at_ms = NULL, updated_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (now_ms, now_ms, user_id, logical_run_id),
            )

    def record_fault(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        owner: str,
        attempt_id: str,
        lease_token: str,
        failure_code: str,
        now_ms: int,
    ) -> Literal["failed_retryable", "failed_terminal", "quarantined"]:
        with self._transaction() as connection:
            item = self._require_active_owner(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                owner=owner,
                attempt_id=attempt_id,
                lease_token=lease_token,
                now_ms=now_ms,
            )
            attempt = connection.execute(
                """
                SELECT * FROM attempts
                WHERE user_id = ? AND logical_run_id = ? AND attempt_id = ?
                  AND status = 'running'
                """,
                (user_id, logical_run_id, attempt_id),
            ).fetchone()
            if attempt is None:
                raise DurableStateError("active run has no running Attempt")
            ordinal = int(attempt["ordinal"])
            connection.execute(
                """
                UPDATE attempts
                SET status = 'failed_fault', failure_code = ?, finished_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ? AND attempt_id = ?
                """,
                (failure_code, now_ms, user_id, logical_run_id, attempt_id),
            )
            if ordinal < 3:
                state = "failed_retryable"
                connection.execute(
                    """
                    UPDATE queue_items
                    SET state = 'queued', lease_owner = NULL, lease_token = NULL,
                        lease_expires_at_ms = NULL, updated_at_ms = ?
                    WHERE user_id = ? AND logical_run_id = ?
                    """,
                    (now_ms, user_id, logical_run_id),
                )
                return state
            if item["kind"] == "p0_user_chat":
                state = "failed_terminal"
            else:
                state = "quarantined"
                quarantine_id = f"{logical_run_id}:quarantine"
                connection.execute(
                    """
                    INSERT INTO quarantines (
                        quarantine_id, user_id, logical_run_id, tick_id,
                        source_kind, source_id, attempt_ordinal, failure_code,
                        quarantined_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, 3, ?, ?)
                    """,
                    (
                        quarantine_id,
                        user_id,
                        logical_run_id,
                        item["tick_id"],
                        item["source_kind"],
                        item["source_id"],
                        failure_code,
                        now_ms,
                    ),
                )
            connection.execute(
                """
                UPDATE queue_items
                SET state = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at_ms = NULL, updated_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (state, now_ms, user_id, logical_run_id),
            )
            return state

    def get_receipt(
        self, *, user_id: str, logical_run_id: str, action_id: str
    ) -> ToolReceiptRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM tool_receipts
                WHERE user_id = ? AND logical_run_id = ? AND action_id = ?
                """,
                (user_id, logical_run_id, action_id),
            ).fetchone()
        return self._receipt_from_row(row) if row is not None else None

    def receipts_for_run(
        self, *, user_id: str, logical_run_id: str
    ) -> tuple[ToolReceiptRecord, ...]:
        with self._connect() as connection:
            return self._receipts_locked(
                connection, user_id=user_id, logical_run_id=logical_run_id
            )

    def record_or_get_receipt(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        action_id: str,
        intent_fingerprint: str,
        owner: str,
        attempt_id: str,
        lease_token: str,
        provider_reference: str,
        result: dict[str, Any],
        acknowledged_at_ms: int,
    ) -> ToolReceiptRecord:
        receipt_id = f"receipt:{action_id}"
        with self._transaction() as connection:
            self._require_active_owner(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                owner=owner,
                attempt_id=attempt_id,
                lease_token=lease_token,
                now_ms=acknowledged_at_ms,
            )
            existing = connection.execute(
                """
                SELECT * FROM tool_receipts
                WHERE user_id = ? AND action_id = ?
                """,
                (user_id, action_id),
            ).fetchone()
            if existing is not None:
                if existing["logical_run_id"] != logical_run_id:
                    raise ReceiptConflict(
                        "action identity belongs to another Logical Run"
                    )
                if existing["intent_fingerprint"] != intent_fingerprint:
                    raise ReceiptConflict(
                        "action identity was reused with a different intent"
                    )
                return self._receipt_from_row(existing)
            owns_run = connection.execute(
                """
                SELECT 1 FROM queue_items
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (user_id, logical_run_id),
            ).fetchone()
            if owns_run is None:
                raise DurableStateError(
                    "cannot attach a receipt to an unknown Logical Run"
                )
            connection.execute(
                """
                INSERT INTO tool_receipts (
                    receipt_id, user_id, logical_run_id, action_id,
                    intent_fingerprint, provider_reference, result_json,
                    acknowledged_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    user_id,
                    logical_run_id,
                    action_id,
                    intent_fingerprint,
                    provider_reference,
                    json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                    acknowledged_at_ms,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM tool_receipts
                WHERE user_id = ? AND logical_run_id = ? AND receipt_id = ?
                """,
                (user_id, logical_run_id, receipt_id),
            ).fetchone()
            if row is None:
                raise DurableStateError("receipt insert was not observable")
            return self._receipt_from_row(row)

    def diagnostics(self, user_id: str) -> StateDiagnostics:
        with self._connect() as connection:
            queue_rows = connection.execute(
                """
                SELECT * FROM queue_items
                WHERE user_id = ? AND state NOT IN ('completed', 'failed_terminal', 'quarantined')
                ORDER BY priority_rank, enqueue_sequence
                """,
                (user_id,),
            ).fetchall()
            lease_rows = connection.execute(
                """
                SELECT * FROM queue_items
                WHERE user_id = ? AND lease_owner IS NOT NULL
                ORDER BY enqueue_sequence
                """,
                (user_id,),
            ).fetchall()
            attempt_rows = connection.execute(
                """
                SELECT * FROM attempts WHERE user_id = ?
                ORDER BY started_at_ms, logical_run_id, ordinal
                """,
                (user_id,),
            ).fetchall()
            checkpoint_rows = connection.execute(
                """
                SELECT * FROM checkpoints WHERE user_id = ?
                ORDER BY rowid DESC
                """,
                (user_id,),
            ).fetchall()
            receipt_rows = connection.execute(
                """
                SELECT * FROM tool_receipts WHERE user_id = ?
                ORDER BY acknowledged_at_ms, receipt_id
                """,
                (user_id,),
            ).fetchall()
            quarantine_rows = connection.execute(
                """
                SELECT * FROM quarantines WHERE user_id = ?
                ORDER BY quarantined_at_ms, quarantine_id
                """,
                (user_id,),
            ).fetchall()
        return StateDiagnostics(
            queue=tuple(
                QueueDiagnostic(
                    tick_id=str(row["tick_id"]),
                    logical_run_id=str(row["logical_run_id"]),
                    kind=str(row["kind"]),
                    priority=str(row["priority"]),
                    state=str(row["state"]),
                    enqueue_sequence=int(row["enqueue_sequence"]),
                )
                for row in queue_rows
            ),
            leases=tuple(
                LeaseDiagnostic(
                    logical_run_id=str(row["logical_run_id"]),
                    owner=str(row["lease_owner"]),
                    expires_at_ms=int(row["lease_expires_at_ms"]),
                    state=str(row["state"]),
                )
                for row in lease_rows
            ),
            attempts=tuple(
                AttemptDiagnostic(
                    attempt_id=str(row["attempt_id"]),
                    logical_run_id=str(row["logical_run_id"]),
                    ordinal=int(row["ordinal"]),
                    status=str(row["status"]),
                    failure_code=(
                        str(row["failure_code"])
                        if row["failure_code"] is not None
                        else None
                    ),
                    started_at_ms=int(row["started_at_ms"]),
                    finished_at_ms=(
                        int(row["finished_at_ms"])
                        if row["finished_at_ms"] is not None
                        else None
                    ),
                )
                for row in attempt_rows
            ),
            checkpoints=tuple(
                self._checkpoint_from_row(row) for row in checkpoint_rows
            ),
            receipts=tuple(self._receipt_from_row(row) for row in receipt_rows),
            quarantines=tuple(
                QuarantineDiagnostic(
                    quarantine_id=str(row["quarantine_id"]),
                    logical_run_id=str(row["logical_run_id"]),
                    tick_id=str(row["tick_id"]),
                    source_kind=str(row["source_kind"]),
                    source_id=str(row["source_id"]),
                    attempt_ordinal=int(row["attempt_ordinal"]),
                    failure_code=str(row["failure_code"]),
                    quarantined_at_ms=int(row["quarantined_at_ms"]),
                )
                for row in quarantine_rows
            ),
        )

    def _recover_expired_locked(
        self, connection: sqlite3.Connection, *, user_id: str, now_ms: int
    ) -> None:
        expired = connection.execute(
            """
            SELECT * FROM queue_items
            WHERE user_id = ? AND state IN ('leased', 'running')
              AND lease_expires_at_ms <= ?
            ORDER BY enqueue_sequence
            """,
            (user_id, now_ms),
        ).fetchall()
        for item in expired:
            if item["state"] == "leased":
                connection.execute(
                    """
                    UPDATE queue_items
                    SET state = 'queued', lease_owner = NULL, lease_token = NULL,
                        lease_expires_at_ms = NULL, updated_at_ms = ?
                    WHERE logical_run_id = ? AND user_id = ?
                    """,
                    (now_ms, item["logical_run_id"], user_id),
                )
                continue
            attempt = connection.execute(
                """
                SELECT * FROM attempts
                WHERE user_id = ? AND logical_run_id = ? AND status = 'running'
                """,
                (user_id, item["logical_run_id"]),
            ).fetchone()
            if attempt is None:
                raise DurableStateError("expired running item has no Attempt")
            ordinal = int(attempt["ordinal"])
            connection.execute(
                """
                UPDATE attempts
                SET status = 'failed_fault',
                    failure_code = 'crash_discarded_uncheckpointed_inference',
                    finished_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ? AND attempt_id = ?
                """,
                (now_ms, user_id, item["logical_run_id"], attempt["attempt_id"]),
            )
            if ordinal < 3:
                connection.execute(
                    """
                    UPDATE queue_items
                    SET state = 'queued', lease_owner = NULL, lease_token = NULL,
                        lease_expires_at_ms = NULL, updated_at_ms = ?
                    WHERE logical_run_id = ? AND user_id = ?
                    """,
                    (now_ms, item["logical_run_id"], user_id),
                )
                continue
            terminal = (
                "failed_terminal" if item["kind"] == "p0_user_chat" else "quarantined"
            )
            if terminal == "quarantined":
                connection.execute(
                    """
                    INSERT OR IGNORE INTO quarantines (
                        quarantine_id, user_id, logical_run_id, tick_id,
                        source_kind, source_id, attempt_ordinal, failure_code,
                        quarantined_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, 3,
                              'crash_discarded_uncheckpointed_inference', ?)
                    """,
                    (
                        f"{item['logical_run_id']}:quarantine",
                        user_id,
                        item["logical_run_id"],
                        item["tick_id"],
                        item["source_kind"],
                        item["source_id"],
                        now_ms,
                    ),
                )
            connection.execute(
                """
                UPDATE queue_items
                SET state = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at_ms = NULL, updated_at_ms = ?
                WHERE logical_run_id = ? AND user_id = ?
                """,
                (terminal, now_ms, item["logical_run_id"], user_id),
            )

    @staticmethod
    def _require_active_owner(
        connection: sqlite3.Connection,
        *,
        user_id: str,
        logical_run_id: str,
        owner: str,
        attempt_id: str,
        lease_token: str,
        now_ms: int,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT queue_items.* FROM queue_items
            JOIN attempts
              ON attempts.user_id = queue_items.user_id
             AND attempts.logical_run_id = queue_items.logical_run_id
            WHERE queue_items.user_id = ? AND queue_items.logical_run_id = ?
              AND queue_items.state = 'running' AND queue_items.lease_owner = ?
              AND queue_items.lease_token = ? AND attempts.attempt_id = ?
              AND queue_items.lease_expires_at_ms > ?
              AND attempts.status = 'running'
            """,
            (user_id, logical_run_id, owner, lease_token, attempt_id, now_ms),
        ).fetchone()
        if row is None:
            raise DurableStateError("Logical Run is not owned by this active lease")
        return row

    def assert_active_lease(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        owner: str,
        attempt_id: str,
        lease_token: str,
        now_ms: int,
    ) -> None:
        """Fail closed unless this exact, unexpired acquisition is still active."""
        with self._connect() as connection:
            self._require_active_owner(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                owner=owner,
                attempt_id=attempt_id,
                lease_token=lease_token,
                now_ms=now_ms,
            )

    def has_queued_p0(self, user_id: str) -> bool:
        """Return whether this user has an eligible user-chat Tick queued."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM queue_items
                WHERE user_id = ? AND state = 'queued' AND priority = 'p0'
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return row is not None

    def _latest_checkpoint_locked(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        logical_run_id: str,
    ) -> CheckpointRecord | None:
        row = connection.execute(
            """
            SELECT * FROM checkpoints
            WHERE user_id = ? AND logical_run_id = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (user_id, logical_run_id),
        ).fetchone()
        return self._checkpoint_from_row(row) if row is not None else None

    def _receipts_locked(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        logical_run_id: str,
    ) -> tuple[ToolReceiptRecord, ...]:
        rows = connection.execute(
            """
            SELECT * FROM tool_receipts
            WHERE user_id = ? AND logical_run_id = ?
            ORDER BY acknowledged_at_ms, receipt_id
            """,
            (user_id, logical_run_id),
        ).fetchall()
        return tuple(self._receipt_from_row(row) for row in rows)

    @staticmethod
    def _checkpoint_from_row(row: sqlite3.Row) -> CheckpointRecord:
        return CheckpointRecord(
            checkpoint_id=str(row["checkpoint_id"]),
            logical_run_id=str(row["logical_run_id"]),
            cause=str(row["cause"]),
            remaining_work=str(row["remaining_work"]),
            completed_receipt_ids=tuple(json.loads(row["completed_receipt_ids_json"])),
            updated_at_ms=int(row["updated_at_ms"]),
        )

    @staticmethod
    def _receipt_from_row(row: sqlite3.Row) -> ToolReceiptRecord:
        return ToolReceiptRecord(
            receipt_id=str(row["receipt_id"]),
            user_id=str(row["user_id"]),
            logical_run_id=str(row["logical_run_id"]),
            action_id=str(row["action_id"]),
            intent_fingerprint=str(row["intent_fingerprint"]),
            provider_reference=str(row["provider_reference"]),
            result=dict(json.loads(row["result_json"])),
            acknowledged_at_ms=int(row["acknowledged_at_ms"]),
        )


def diagnostics_as_dict(
    diagnostics: StateDiagnostics,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "queue": [asdict(item) for item in diagnostics.queue],
        "leases": [asdict(item) for item in diagnostics.leases],
        "attempts": [asdict(item) for item in diagnostics.attempts],
        "checkpoints": [asdict(item) for item in diagnostics.checkpoints],
        "receipts": [asdict(item) for item in diagnostics.receipts],
        "quarantines": [asdict(item) for item in diagnostics.quarantines],
    }


__all__ = [
    "AttemptDiagnostic",
    "CheckpointRecord",
    "DurableRunState",
    "DurableStateError",
    "LeaseDiagnostic",
    "LeasedRun",
    "QueueDiagnostic",
    "QuarantineDiagnostic",
    "ReceiptConflict",
    "SCHEMA_VERSION",
    "StateDiagnostics",
    "ToolReceiptRecord",
    "default_database_path",
    "diagnostics_as_dict",
]
