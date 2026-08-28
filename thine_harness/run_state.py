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
import uuid

from hermes_constants import get_hermes_home

from .contracts.runtime import Tick
from .contracts.recovery import ExplicitRetry, InputGap, QuarantineRecord
from .contracts.runtime import InputReceipt, RunFinalization, RunReceipt
from .contracts.transcripts import TranscriptAck, TranscriptClaim
from .working_memory import WorkingMemorySnapshot


SCHEMA_VERSION = 4


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


@dataclass(frozen=True)
class StoredTranscriptClaim:
    user_id: str
    tick_id: str
    logical_run_id: str
    claim_request_id: str
    claim_id: str | None
    claim: TranscriptClaim | None
    state: str


@dataclass(frozen=True)
class PendingTranscriptAck:
    user_id: str
    tick_id: str
    logical_run_id: str
    attempt_ordinal: int
    claim_id: str
    memory_version: int
    finalization_id: str


@dataclass(frozen=True)
class TranscriptRunRecord:
    queue_state: str
    attempts_total: int
    decision_outcome: str | None
    visible_action_intent_count: int | None
    working_memory_outcome: str | None
    memory_version: int | None
    finalization_phase: str | None
    ack_id: str | None
    input_receipt_id: str | None
    run_receipt_id: str | None
    canonical_transcript_retained: bool | None


@dataclass(frozen=True)
class PendingTranscriptQuarantine:
    user_id: str
    tick_id: str
    logical_run_id: str
    attempt_ordinal: int
    claim_id: str
    quarantine_id: str
    failure_code: str
    quarantined_at_ms: int


@dataclass(frozen=True)
class TranscriptQuarantineInspection:
    quarantine_id: str
    logical_run_id: str
    claim_id: str
    failure_code: str
    sync_state: str
    record: QuarantineRecord | None
    input_gap: InputGap | None
    retry_run_ids: tuple[str, ...]


@dataclass(frozen=True)
class AgentRunInspection:
    logical_run_id: str
    attempt_id: str
    provider: str
    model: str
    api_mode: str
    reasoning_effort: str
    decision_outcome: str
    final_output: str
    tool_discoveries: tuple[str, ...]
    usage: dict[str, int]
    stop_hook_outcome: str
    stop_hook_cache_identity: dict[str, str]
    memory_version: int
    memory_token_count: int | None
    recorded_at_ms: int


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
            if version == 0:
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
                PRAGMA user_version = 1;
                COMMIT;
                """
                )
                version = 1
            if version == 1:
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE working_memory_state (
                        user_id TEXT PRIMARY KEY,
                        version INTEGER NOT NULL,
                        markdown TEXT NOT NULL,
                        token_count INTEGER,
                        last_run_id TEXT
                    );

                    CREATE TABLE working_memory_unchanged (
                        marker_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        logical_run_id TEXT NOT NULL UNIQUE,
                        expected_version INTEGER NOT NULL,
                        recorded_at_ms INTEGER NOT NULL,
                        FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
                    );

                    CREATE TABLE transcript_claims (
                        user_id TEXT NOT NULL,
                        logical_run_id TEXT NOT NULL PRIMARY KEY,
                        tick_id TEXT NOT NULL UNIQUE,
                        claim_request_id TEXT NOT NULL,
                        claim_id TEXT,
                        claim_json TEXT,
                        ack_json TEXT,
                        state TEXT NOT NULL,
                        memory_version INTEGER,
                        finalization_id TEXT,
                        created_at_ms INTEGER NOT NULL,
                        updated_at_ms INTEGER NOT NULL,
                        UNIQUE(user_id, claim_request_id),
                        UNIQUE(user_id, claim_id),
                        FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
                    );
                    CREATE INDEX transcript_claims_by_user_state
                        ON transcript_claims(user_id, state, created_at_ms);

                    CREATE TABLE decision_outcomes (
                        decision_receipt_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        logical_run_id TEXT NOT NULL UNIQUE,
                        outcome TEXT NOT NULL,
                        visible_action_intent_count INTEGER NOT NULL,
                        recorded_at_ms INTEGER NOT NULL,
                        FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
                    );

                    CREATE TABLE run_finalizations (
                        finalization_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        logical_run_id TEXT NOT NULL UNIQUE,
                        tick_id TEXT NOT NULL,
                        phase TEXT NOT NULL,
                        working_memory_outcome TEXT NOT NULL,
                        source_ack_id TEXT,
                        finalization_json TEXT NOT NULL,
                        updated_at_ms INTEGER NOT NULL,
                        FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
                    );

                    CREATE TABLE input_receipts (
                        receipt_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        logical_run_id TEXT NOT NULL UNIQUE,
                        ack_id TEXT NOT NULL UNIQUE,
                        receipt_json TEXT NOT NULL,
                        recorded_at_ms INTEGER NOT NULL,
                        FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
                    );

                    CREATE TABLE run_receipts (
                        receipt_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        logical_run_id TEXT NOT NULL UNIQUE,
                        receipt_json TEXT NOT NULL,
                        recorded_at_ms INTEGER NOT NULL,
                        FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
                    );
                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                )
                version = 2
            if version == 2:
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE working_memory_versions (
                        user_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        markdown TEXT NOT NULL,
                        configured_model_token_count INTEGER,
                        tokenizer_status TEXT NOT NULL,
                        logical_run_id TEXT,
                        committed_at_ms INTEGER NOT NULL,
                        PRIMARY KEY (user_id, version)
                    );
                    INSERT INTO working_memory_versions (
                        user_id, version, markdown, configured_model_token_count,
                        tokenizer_status, logical_run_id, committed_at_ms
                    )
                    SELECT user_id, version, markdown, token_count,
                           CASE WHEN token_count IS NULL
                                THEN 'unresolved_fail_closed' ELSE 'exact' END,
                           last_run_id, 0
                    FROM working_memory_state;

                    CREATE TABLE agent_run_inspections (
                        user_id TEXT NOT NULL,
                        logical_run_id TEXT NOT NULL,
                        attempt_id TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        api_mode TEXT NOT NULL,
                        reasoning_effort TEXT NOT NULL,
                        decision_outcome TEXT NOT NULL,
                        final_output TEXT NOT NULL,
                        tool_discoveries_json TEXT NOT NULL,
                        usage_json TEXT NOT NULL,
                        stop_hook_outcome TEXT NOT NULL,
                        stop_hook_cache_identity_json TEXT NOT NULL,
                        memory_version INTEGER NOT NULL,
                        memory_token_count INTEGER,
                        recorded_at_ms INTEGER NOT NULL,
                        PRIMARY KEY (user_id, logical_run_id),
                        FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
                    );
                    CREATE INDEX agent_run_inspections_recent
                        ON agent_run_inspections(user_id, recorded_at_ms DESC);
                    PRAGMA user_version = 3;
                    COMMIT;
                    """
                )
                version = 3
            if version == 3:
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE attempt_execution_started (
                        attempt_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        logical_run_id TEXT NOT NULL,
                        started_at_ms INTEGER NOT NULL,
                        FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id),
                        FOREIGN KEY(logical_run_id) REFERENCES queue_items(logical_run_id)
                    );
                    INSERT INTO attempt_execution_started (
                        attempt_id, user_id, logical_run_id, started_at_ms
                    )
                    SELECT attempt_id, user_id, logical_run_id, started_at_ms
                    FROM attempts WHERE status = 'running';

                    CREATE TABLE transcript_quarantines (
                        quarantine_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        original_logical_run_id TEXT NOT NULL UNIQUE,
                        claim_id TEXT NOT NULL UNIQUE,
                        failure_code TEXT NOT NULL,
                        quarantined_at_ms INTEGER NOT NULL,
                        sync_state TEXT NOT NULL,
                        record_json TEXT,
                        input_gap_json TEXT,
                        gap_delivery_run_id TEXT,
                        synchronized_at_ms INTEGER,
                        FOREIGN KEY(quarantine_id) REFERENCES quarantines(quarantine_id),
                        FOREIGN KEY(original_logical_run_id)
                            REFERENCES queue_items(logical_run_id),
                        FOREIGN KEY(gap_delivery_run_id)
                            REFERENCES queue_items(logical_run_id)
                    );
                    CREATE INDEX transcript_quarantine_sync
                        ON transcript_quarantines(user_id, sync_state, quarantined_at_ms);

                    CREATE TABLE transcript_explicit_retries (
                        retry_run_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        quarantine_id TEXT NOT NULL,
                        retry_request_id TEXT NOT NULL UNIQUE,
                        explicit_retry_json TEXT NOT NULL,
                        state TEXT NOT NULL,
                        created_at_ms INTEGER NOT NULL,
                        completed_at_ms INTEGER,
                        FOREIGN KEY(retry_run_id) REFERENCES queue_items(logical_run_id),
                        FOREIGN KEY(quarantine_id)
                            REFERENCES transcript_quarantines(quarantine_id)
                    );
                    CREATE INDEX transcript_retries_by_quarantine
                        ON transcript_explicit_retries(user_id, quarantine_id, created_at_ms);
                    PRAGMA user_version = 4;
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
        with self._transaction() as connection:
            self._insert_tick_locked(connection, tick=tick, now_ms=now_ms)
        return str(payload.tick_id)

    def enqueue_transcript_availability(self, tick: Tick, *, now_ms: int) -> str:
        """Insert at most one outstanding transcript availability Tick per user."""
        payload = tick.payload
        if payload.kind != "p1_transcript":
            raise ValueError("transcript availability requires a p1_transcript Tick")
        with self._transaction() as connection:
            replay = connection.execute(
                """
                SELECT tick_id FROM queue_items
                WHERE user_id = ? AND tick_id = ? AND kind = 'p1_transcript'
                """,
                (payload.user_id, payload.tick_id),
            ).fetchone()
            if replay is not None:
                return str(replay["tick_id"])
            existing = connection.execute(
                """
                SELECT tick_id FROM queue_items
                WHERE user_id = ? AND kind = 'p1_transcript'
                  AND state IN ('queued', 'running', 'awaiting_audio_ack')
                ORDER BY enqueue_sequence
                LIMIT 1
                """,
                (payload.user_id,),
            ).fetchone()
            if existing is not None:
                return str(existing["tick_id"])
            self._insert_tick_locked(connection, tick=tick, now_ms=now_ms)
        return str(payload.tick_id)

    @staticmethod
    def _insert_tick_locked(
        connection: sqlite3.Connection,
        *,
        tick: Tick,
        now_ms: int,
    ) -> None:
        payload = tick.payload
        existing = connection.execute(
            "SELECT tick_json FROM queue_items WHERE tick_id = ? AND user_id = ?",
            (payload.tick_id, payload.user_id),
        ).fetchone()
        if existing is not None:
            if json.loads(existing["tick_json"]) != tick.to_dict():
                raise DurableStateError("tick_id was reused with a different payload")
            return
        priority_rank = {"p0": 0, "p1": 1, "p2": 2}[str(payload.priority)]
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
            connection.execute(
                "DELETE FROM attempt_execution_started WHERE attempt_id = ?",
                (attempt_id,),
            )
        return CheckpointRecord(
            checkpoint_id=checkpoint_id,
            logical_run_id=logical_run_id,
            cause=cause,
            remaining_work=remaining_work,
            completed_receipt_ids=receipt_ids,
            updated_at_ms=now_ms,
        )

    def requeue_input_transport_failure(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        owner: str,
        attempt_id: str,
        lease_token: str,
        now_ms: int,
    ) -> None:
        """Retry input delivery without consuming or replacing the Attempt."""
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
            started = connection.execute(
                "SELECT 1 FROM attempt_execution_started WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if started is not None:
                raise DurableStateError(
                    "input transport cannot requeue after inference started"
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

    def mark_inference_started(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        owner: str,
        attempt_id: str,
        lease_token: str,
        now_ms: int,
    ) -> None:
        """Mark the boundary after which a crash consumes this Attempt."""
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
                INSERT OR IGNORE INTO attempt_execution_started (
                    attempt_id, user_id, logical_run_id, started_at_ms
                ) VALUES (?, ?, ?, ?)
                """,
                (attempt_id, user_id, logical_run_id, now_ms),
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

    def ensure_transcript_claim_request(
        self,
        *,
        user_id: str,
        tick_id: str,
        logical_run_id: str,
        claim_request_id: str,
        owner: str,
        attempt_id: str,
        lease_token: str,
        now_ms: int,
    ) -> StoredTranscriptClaim:
        """Persist the idempotency key before the first Dataplane claim call."""
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
            if item["kind"] != "p1_transcript" or item["tick_id"] != tick_id:
                raise DurableStateError("claim request does not belong to this Tick")
            connection.execute(
                """
                INSERT OR IGNORE INTO transcript_claims (
                    user_id, logical_run_id, tick_id, claim_request_id,
                    state, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, 'requested', ?, ?)
                """,
                (
                    user_id,
                    logical_run_id,
                    tick_id,
                    claim_request_id,
                    now_ms,
                    now_ms,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM transcript_claims
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (user_id, logical_run_id),
            ).fetchone()
            if row is None:
                raise DurableStateError("transcript claim request was not durable")
            if row["claim_request_id"] != claim_request_id:
                raise DurableStateError(
                    "Logical Run already owns a different claim request"
                )
            return self._stored_transcript_claim_from_row(row)

    def record_transcript_claim(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        claim: TranscriptClaim,
        owner: str,
        attempt_id: str,
        lease_token: str,
        now_ms: int,
    ) -> StoredTranscriptClaim:
        """Attach one frozen claim envelope to its leased Logical Run."""
        payload = claim.payload
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
            row = connection.execute(
                """
                SELECT * FROM transcript_claims
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (user_id, logical_run_id),
            ).fetchone()
            if row is None:
                raise DurableStateError(
                    "claim response arrived before request persistence"
                )
            if payload.claim_request_id != row["claim_request_id"]:
                raise DurableStateError("claim response request identity mismatch")
            if payload.lease_owner != logical_run_id:
                raise DurableStateError("claim response lease owner mismatch")
            canonical = claim.to_json()
            if row["claim_json"] is not None and row["claim_json"] != canonical:
                raise DurableStateError("claim lookup changed its frozen envelope")
            connection.execute(
                """
                UPDATE transcript_claims
                SET claim_id = ?, claim_json = ?, state = 'claimed', updated_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (
                    payload.claim_id,
                    canonical,
                    now_ms,
                    user_id,
                    logical_run_id,
                ),
            )
            stored = connection.execute(
                """
                SELECT * FROM transcript_claims
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (user_id, logical_run_id),
            ).fetchone()
            if stored is None:
                raise DurableStateError("claim response was not durable")
            return self._stored_transcript_claim_from_row(stored)

    def next_pending_transcript_quarantine(
        self, user_id: str
    ) -> PendingTranscriptQuarantine | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT tq.*, q.tick_id, MAX(a.ordinal) AS attempt_ordinal
                FROM transcript_quarantines tq
                JOIN queue_items q
                  ON q.user_id = tq.user_id
                 AND q.logical_run_id = tq.original_logical_run_id
                JOIN attempts a
                  ON a.user_id = tq.user_id
                 AND a.logical_run_id = tq.original_logical_run_id
                WHERE tq.user_id = ? AND tq.sync_state = 'pending'
                GROUP BY tq.quarantine_id, q.tick_id
                ORDER BY tq.quarantined_at_ms, tq.quarantine_id
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return PendingTranscriptQuarantine(
            user_id=user_id,
            tick_id=str(row["tick_id"]),
            logical_run_id=str(row["original_logical_run_id"]),
            attempt_ordinal=int(row["attempt_ordinal"]),
            claim_id=str(row["claim_id"]),
            quarantine_id=str(row["quarantine_id"]),
            failure_code=str(row["failure_code"]),
            quarantined_at_ms=int(row["quarantined_at_ms"]),
        )

    def complete_transcript_quarantine(self, result: Any) -> None:
        """Commit the backend-confirmed immutable record and typed source gap."""
        gap = result.input_gap
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT tq.*, t.claim_json
                FROM transcript_quarantines tq
                JOIN transcript_claims t
                  ON t.user_id = tq.user_id
                 AND t.logical_run_id = tq.original_logical_run_id
                WHERE tq.quarantine_id = ?
                """,
                (result.quarantine_id,),
            ).fetchone()
            if row is None:
                raise DurableStateError("unknown transcript quarantine")
            if row["sync_state"] == "synchronized":
                if row["input_gap_json"] != gap.to_json():
                    raise DurableStateError("quarantine replay changed its input gap")
                return
            if (
                result.claim_id != row["claim_id"]
                or result.logical_run_id != row["original_logical_run_id"]
                or result.source_identity != row["claim_id"]
                or result.failure_code != row["failure_code"]
                or result.fault_attempts_total != 3
                or result.quarantined_at_ms != row["quarantined_at_ms"]
                or result.status != "quarantined"
                or not result.input_retained
                or not result.normal_cursor_advanced
                or not result.canonical_transcript_retained
                or gap.payload.quarantine_id != row["quarantine_id"]
                or gap.payload.source_kind != "transcript"
                or gap.payload.source_identity != row["claim_id"]
                or not gap.payload.normal_cursor_advanced
            ):
                raise DurableStateError("backend quarantine identity mismatch")
            claim = TranscriptClaim.from_json(str(row["claim_json"])).payload
            entries = claim.entries
            if not entries:
                raise DurableStateError("cannot quarantine an empty transcript claim")
            sequence_numbers = [
                int(entry.sequence_number)
                for entry in entries
                if entry.sequence_number is not None
            ]
            expected_buffer_ids = tuple(
                int(entry.aggregation_buffer_id) for entry in entries
            )
            expected_sequences = tuple(
                None if entry.sequence_number is None else int(entry.sequence_number)
                for entry in entries
            )
            expected_provenance = tuple(str(entry.provenance) for entry in entries)
            if (
                result.aggregation_buffer_ids != expected_buffer_ids
                or result.sequence_numbers != expected_sequences
                or result.provenance != expected_provenance
                or len(result.adoption_kinds) != len(entries)
            ):
                raise DurableStateError("backend quarantine range mismatch")
            record = QuarantineRecord.from_dict({
                "schema_version": {"major": 1, "minor": 0},
                "quarantine_id": str(row["quarantine_id"]),
                "source_kind": "transcript",
                "source_identity": str(row["claim_id"]),
                "immutable_range": {
                    "range_kind": "transcript_entries",
                    "aggregation_buffer_ids": [
                        int(entry.aggregation_buffer_id) for entry in entries
                    ],
                    "first_sequence_number": (
                        min(sequence_numbers) if sequence_numbers else None
                    ),
                    "last_sequence_number": (
                        max(sequence_numbers) if sequence_numbers else None
                    ),
                },
                "logical_run_id": str(row["original_logical_run_id"]),
                "fault_attempts_total": 3,
                "normal_cursor_advanced": True,
                "created_at_ms": int(row["quarantined_at_ms"]),
                "extensions": {},
            })
            connection.execute(
                """
                UPDATE transcript_quarantines
                SET sync_state = 'synchronized', record_json = ?,
                    input_gap_json = ?, synchronized_at_ms = ?
                WHERE quarantine_id = ? AND sync_state = 'pending'
                """,
                (
                    record.to_json(),
                    gap.to_json(),
                    int(gap.payload.recorded_at_ms),
                    result.quarantine_id,
                ),
            )

    def attach_pending_transcript_gaps(
        self, *, user_id: str, logical_run_id: str
    ) -> tuple[InputGap, ...]:
        """Attach each source gap once to the next normal transcript Logical Run."""
        with self._transaction() as connection:
            is_retry = connection.execute(
                """
                SELECT 1 FROM transcript_explicit_retries
                WHERE user_id = ? AND retry_run_id = ?
                """,
                (user_id, logical_run_id),
            ).fetchone()
            if is_retry is not None:
                return ()
            rows = connection.execute(
                """
                SELECT quarantine_id, input_gap_json
                FROM transcript_quarantines
                WHERE user_id = ? AND sync_state = 'synchronized'
                  AND (gap_delivery_run_id IS NULL OR gap_delivery_run_id = ?)
                ORDER BY quarantined_at_ms, quarantine_id
                """,
                (user_id, logical_run_id),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE transcript_quarantines SET gap_delivery_run_id = ?
                    WHERE quarantine_id = ? AND gap_delivery_run_id IS NULL
                    """,
                    (logical_run_id, row["quarantine_id"]),
                )
        return tuple(InputGap.from_json(str(row["input_gap_json"])) for row in rows)

    def enqueue_transcript_retry(
        self,
        *,
        user_id: str,
        quarantine_id: str,
        retry_run_id: str,
        created_at_ms: int,
    ) -> ExplicitRetry:
        """Create separately identified retry work without mutating the old stream."""
        if not user_id or not quarantine_id or not retry_run_id:
            raise ValueError("user_id, quarantine_id, and retry_run_id are required")
        retry_request_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"thine-transcript-quarantine-retry:{retry_run_id}",
            )
        )
        tick_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"thine-transcript-quarantine-retry-tick:{retry_run_id}",
            )
        )
        with self._transaction() as connection:
            quarantine = connection.execute(
                """
                SELECT * FROM transcript_quarantines
                WHERE user_id = ? AND quarantine_id = ?
                  AND sync_state = 'synchronized'
                """,
                (user_id, quarantine_id),
            ).fetchone()
            if quarantine is None:
                raise DurableStateError("retry requires a synchronized quarantine")
            receipt_rows = connection.execute(
                """
                SELECT receipt_id FROM tool_receipts
                WHERE user_id = ? AND logical_run_id = ?
                ORDER BY acknowledged_at_ms, receipt_id
                """,
                (user_id, quarantine["original_logical_run_id"]),
            ).fetchall()
            retry = ExplicitRetry.from_dict({
                "schema_version": {"major": 1, "minor": 0},
                "retry_run_id": retry_run_id,
                "quarantine_id": quarantine_id,
                "source_identity": str(quarantine["claim_id"]),
                "preserved_receipt_ids": [
                    str(row["receipt_id"]) for row in receipt_rows
                ],
                "created_at_ms": created_at_ms,
                "rewinds_normal_cursor": False,
                "extensions": {},
            })
            existing = connection.execute(
                """
                SELECT explicit_retry_json FROM transcript_explicit_retries
                WHERE user_id = ? AND retry_run_id = ?
                """,
                (user_id, retry_run_id),
            ).fetchone()
            if existing is not None:
                if str(existing["explicit_retry_json"]) != retry.to_json():
                    raise DurableStateError(
                        "retry_run_id was reused with a different quarantine"
                    )
                return ExplicitRetry.from_json(str(existing["explicit_retry_json"]))
            tick = Tick.from_dict({
                "schema_version": {"major": 1, "minor": 0},
                "tick_id": tick_id,
                "user_id": user_id,
                "logical_run_id": retry_run_id,
                "kind": "p1_transcript",
                "priority": "p1",
                "occurred_at_ms": created_at_ms,
                "received_at_ms": created_at_ms,
                "queued_at_ms": created_at_ms,
                "source_ref": {
                    "kind": "transcript_availability",
                    "id": quarantine_id,
                },
                "causation_id": str(quarantine["original_logical_run_id"]),
                "correlation_id": tick_id,
                "attempt_ordinal": 1,
                "lease": None,
                "communication_allowance_snapshot": None,
                "payload": {
                    "payload_kind": "transcript_availability",
                    "reference_id": quarantine_id,
                },
                "extensions": {},
            })
            self._insert_tick_locked(connection, tick=tick, now_ms=created_at_ms)
            connection.execute(
                """
                INSERT INTO transcript_explicit_retries (
                    retry_run_id, user_id, quarantine_id, retry_request_id,
                    explicit_retry_json, state, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?)
                """,
                (
                    retry_run_id,
                    user_id,
                    quarantine_id,
                    retry_request_id,
                    retry.to_json(),
                    created_at_ms,
                ),
            )
        return retry

    def explicit_transcript_retry(
        self, *, user_id: str, retry_run_id: str
    ) -> ExplicitRetry | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT explicit_retry_json FROM transcript_explicit_retries
                WHERE user_id = ? AND retry_run_id = ?
                """,
                (user_id, retry_run_id),
            ).fetchone()
        return (
            None
            if row is None
            else ExplicitRetry.from_json(str(row["explicit_retry_json"]))
        )

    def transcript_retry_request_id(self, *, user_id: str, retry_run_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT retry_request_id FROM transcript_explicit_retries
                WHERE user_id = ? AND retry_run_id = ?
                """,
                (user_id, retry_run_id),
            ).fetchone()
        if row is None:
            raise KeyError(retry_run_id)
        return str(row["retry_request_id"])

    def inspect_transcript_quarantine(
        self, *, user_id: str, quarantine_id: str
    ) -> TranscriptQuarantineInspection:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM transcript_quarantines
                WHERE user_id = ? AND quarantine_id = ?
                """,
                (user_id, quarantine_id),
            ).fetchone()
            if row is None:
                raise KeyError(quarantine_id)
            retry_rows = connection.execute(
                """
                SELECT retry_run_id FROM transcript_explicit_retries
                WHERE user_id = ? AND quarantine_id = ?
                ORDER BY created_at_ms, retry_run_id
                """,
                (user_id, quarantine_id),
            ).fetchall()
        return TranscriptQuarantineInspection(
            quarantine_id=quarantine_id,
            logical_run_id=str(row["original_logical_run_id"]),
            claim_id=str(row["claim_id"]),
            failure_code=str(row["failure_code"]),
            sync_state=str(row["sync_state"]),
            record=(
                None
                if row["record_json"] is None
                else QuarantineRecord.from_json(str(row["record_json"]))
            ),
            input_gap=(
                None
                if row["input_gap_json"] is None
                else InputGap.from_json(str(row["input_gap_json"]))
            ),
            retry_run_ids=tuple(str(item["retry_run_id"]) for item in retry_rows),
        )

    def working_memory_snapshot(self, user_id: str) -> WorkingMemorySnapshot:
        """Load the one automatically injected profile-scoped memory document."""
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO working_memory_state (
                    user_id, version, markdown, token_count, last_run_id
                ) VALUES (?, 0, '', NULL, NULL)
                """,
                (user_id,),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO working_memory_versions (
                    user_id, version, markdown, configured_model_token_count,
                    tokenizer_status, logical_run_id, committed_at_ms
                ) VALUES (?, 0, '', NULL, 'unresolved_fail_closed', NULL, 0)
                """,
                (user_id,),
            )
            row = connection.execute(
                """
                SELECT version, markdown, token_count
                FROM working_memory_state WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            raise DurableStateError("working memory singleton is missing")
        return WorkingMemorySnapshot(
            version=int(row["version"]),
            markdown=str(row["markdown"]),
            token_count=(
                None if row["token_count"] is None else int(row["token_count"])
            ),
        )

    def finalize_transcript_no_action(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        owner: str,
        attempt_id: str,
        lease_token: str,
        now_ms: int,
        expected_memory_version: int | None = None,
        memory_markdown: str | None = None,
        memory_token_count: int | None = None,
        tokenizer_status: str = "unresolved_fail_closed",
        agent_inspection: dict[str, Any] | None = None,
    ) -> PendingTranscriptAck:
        """Atomically finalize no-action memory and enter the ack-only suffix."""
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
            if item["kind"] != "p1_transcript":
                raise DurableStateError(
                    "transcript finalizer received another Tick kind"
                )
            claim = connection.execute(
                """
                SELECT * FROM transcript_claims
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (user_id, logical_run_id),
            ).fetchone()
            if (
                claim is None
                or claim["state"] != "claimed"
                or claim["claim_id"] is None
                or claim["claim_json"] is None
            ):
                raise DurableStateError("transcript run has no durable claimed input")
            claim_payload = TranscriptClaim.from_json(claim["claim_json"]).payload
            if not claim_payload.entries:
                raise DurableStateError(
                    "empty transcript claim cannot enter inference finalization"
                )

            connection.execute(
                """
                INSERT OR IGNORE INTO working_memory_state (
                    user_id, version, markdown, token_count, last_run_id
                ) VALUES (?, 0, '', NULL, NULL)
                """,
                (user_id,),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO working_memory_versions (
                    user_id, version, markdown, configured_model_token_count,
                    tokenizer_status, logical_run_id, committed_at_ms
                ) VALUES (?, 0, '', NULL, 'unresolved_fail_closed', NULL, 0)
                """,
                (user_id,),
            )
            memory = connection.execute(
                """
                SELECT version, markdown, token_count
                FROM working_memory_state WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            if memory is None:
                raise DurableStateError("working memory singleton is missing")
            memory_version = int(memory["version"])
            if memory_markdown is None:
                memory_token_count = (
                    None
                    if memory["token_count"] is None
                    else int(memory["token_count"])
                )
            if (
                expected_memory_version is not None
                and memory_version != expected_memory_version
            ):
                raise DurableStateError(
                    "working memory version changed during the Stop Hook"
                )
            memory_outcome = "unchanged"
            if memory_markdown is not None:
                if tokenizer_status != "exact" or memory_token_count is None:
                    raise DurableStateError(
                        "changed working memory requires exact configured-model tokens"
                    )
                if memory_token_count < 0 or memory_token_count > 16_000:
                    raise DurableStateError(
                        "changed working memory exceeds the 16K-token contract"
                    )
                memory_version += 1
                connection.execute(
                    """
                    INSERT INTO working_memory_versions (
                        user_id, version, markdown, configured_model_token_count,
                        tokenizer_status, logical_run_id, committed_at_ms
                    ) VALUES (?, ?, ?, ?, 'exact', ?, ?)
                    """,
                    (
                        user_id,
                        memory_version,
                        memory_markdown,
                        memory_token_count,
                        logical_run_id,
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
                        memory_markdown,
                        memory_token_count,
                        logical_run_id,
                        user_id,
                        memory_version - 1,
                    ),
                )
                if updated.rowcount != 1:
                    raise DurableStateError(
                        "working memory version changed during finalization"
                    )
                memory_outcome = "written"
            finalization_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"thine-transcript-finalization:{logical_run_id}",
                )
            )
            finalization = RunFinalization.from_dict({
                "schema_version": {"major": 1, "minor": 0},
                "finalization_id": finalization_id,
                "logical_run_id": logical_run_id,
                "tick_id": str(item["tick_id"]),
                "tick_kind": "p1_transcript",
                "phase": "awaiting_audio_ack",
                "source_ack_id": None,
                "final_reply_receipt_id": None,
                "recovery_mode": "ack_only",
                "inference_allowed": False,
                "restream_allowed": False,
                "working_memory_outcome": memory_outcome,
                "finalized_at_ms": None,
                "extensions": {},
            })
            if memory_outcome == "unchanged":
                connection.execute(
                    """
                    INSERT INTO working_memory_unchanged (
                        marker_id, user_id, logical_run_id, expected_version,
                        recorded_at_ms
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        f"memory-unchanged:{logical_run_id}",
                        user_id,
                        logical_run_id,
                        memory_version,
                        now_ms,
                    ),
                )
                connection.execute(
                    """
                    UPDATE working_memory_state SET last_run_id = ?
                    WHERE user_id = ? AND version = ?
                    """,
                    (logical_run_id, user_id, memory_version),
                )
            connection.execute(
                """
                INSERT INTO decision_outcomes (
                    decision_receipt_id, user_id, logical_run_id, outcome,
                    visible_action_intent_count, recorded_at_ms
                ) VALUES (?, ?, ?, 'no_action', 0, ?)
                """,
                (
                    f"decision-receipt:{logical_run_id}",
                    user_id,
                    logical_run_id,
                    now_ms,
                ),
            )
            connection.execute(
                """
                INSERT INTO run_finalizations (
                    finalization_id, user_id, logical_run_id, tick_id, phase,
                    working_memory_outcome, source_ack_id, finalization_json,
                    updated_at_ms
                ) VALUES (?, ?, ?, ?, 'awaiting_audio_ack', ?, NULL, ?, ?)
                """,
                (
                    finalization_id,
                    user_id,
                    logical_run_id,
                    item["tick_id"],
                    memory_outcome,
                    finalization.to_json(),
                    now_ms,
                ),
            )
            if agent_inspection is not None:
                required = {
                    "provider",
                    "model",
                    "api_mode",
                    "reasoning_effort",
                    "final_output",
                    "tool_discoveries",
                    "usage",
                    "stop_hook_outcome",
                    "stop_hook_cache_identity",
                }
                if set(agent_inspection) != required:
                    raise DurableStateError("agent inspection fields are incomplete")
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
                        user_id,
                        logical_run_id,
                        attempt_id,
                        str(agent_inspection["provider"]),
                        str(agent_inspection["model"]),
                        str(agent_inspection["api_mode"]),
                        str(agent_inspection["reasoning_effort"]),
                        str(agent_inspection["final_output"]),
                        json.dumps(
                            agent_inspection["tool_discoveries"], separators=(",", ":")
                        ),
                        json.dumps(
                            agent_inspection["usage"],
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        str(agent_inspection["stop_hook_outcome"]),
                        json.dumps(
                            agent_inspection["stop_hook_cache_identity"],
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        memory_version,
                        memory_token_count,
                        now_ms,
                    ),
                )
            connection.execute(
                """
                UPDATE transcript_claims
                SET state = 'awaiting_ack', memory_version = ?,
                    finalization_id = ?, updated_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (
                    memory_version,
                    finalization_id,
                    now_ms,
                    user_id,
                    logical_run_id,
                ),
            )
            connection.execute(
                """
                UPDATE attempts SET status = 'succeeded', finished_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ? AND attempt_id = ?
                  AND status = 'running'
                """,
                (now_ms, user_id, logical_run_id, attempt_id),
            )
            connection.execute(
                """
                UPDATE queue_items
                SET state = 'awaiting_audio_ack', lease_owner = NULL,
                    lease_token = NULL, lease_expires_at_ms = NULL, updated_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (now_ms, user_id, logical_run_id),
            )
            attempt_ordinal = int(
                connection.execute(
                    "SELECT ordinal FROM attempts WHERE attempt_id = ?",
                    (attempt_id,),
                ).fetchone()[0]
            )
            return PendingTranscriptAck(
                user_id=user_id,
                tick_id=str(item["tick_id"]),
                logical_run_id=logical_run_id,
                attempt_ordinal=attempt_ordinal,
                claim_id=str(claim["claim_id"]),
                memory_version=memory_version,
                finalization_id=finalization_id,
            )

    def next_pending_transcript_ack(self, user_id: str) -> PendingTranscriptAck | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT q.tick_id, q.logical_run_id, t.claim_id,
                       t.memory_version, t.finalization_id,
                       MAX(a.ordinal) AS attempt_ordinal
                FROM queue_items q
                JOIN transcript_claims t
                  ON t.user_id = q.user_id
                 AND t.logical_run_id = q.logical_run_id
                JOIN attempts a
                  ON a.user_id = q.user_id
                 AND a.logical_run_id = q.logical_run_id
                WHERE q.user_id = ? AND q.state = 'awaiting_audio_ack'
                  AND t.state = 'awaiting_ack'
                GROUP BY q.tick_id, q.logical_run_id, t.claim_id,
                         t.memory_version, t.finalization_id, q.enqueue_sequence
                ORDER BY q.enqueue_sequence
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return PendingTranscriptAck(
            user_id=user_id,
            tick_id=str(row["tick_id"]),
            logical_run_id=str(row["logical_run_id"]),
            attempt_ordinal=int(row["attempt_ordinal"]),
            claim_id=str(row["claim_id"]),
            memory_version=int(row["memory_version"]),
            finalization_id=str(row["finalization_id"]),
        )

    def complete_transcript_ack(
        self,
        *,
        pending: PendingTranscriptAck,
        acknowledgement: TranscriptAck,
    ) -> None:
        """Commit the frozen input/run receipts after Dataplane cleanup."""
        ack = acknowledgement.payload
        if (
            ack.claim_id != pending.claim_id
            or ack.run_id != pending.logical_run_id
            or ack.memory_version != str(pending.memory_version)
            or not ack.durable_receipt_written
            or not ack.canonical_transcript_retained
        ):
            raise DurableStateError("Dataplane acknowledgement identity mismatch")
        recorded_at_ms = int(ack.acknowledged_at_ms)
        input_receipt = InputReceipt.from_dict({
            "schema_version": {"major": 1, "minor": 0},
            "receipt_id": f"input-receipt:{ack.ack_id}",
            "source_kind": "transcript",
            "source_identity": pending.claim_id,
            "logical_run_id": pending.logical_run_id,
            "ack_id": ack.ack_id,
            "disposition": "acknowledged",
            "recorded_at_ms": recorded_at_ms,
            "extensions": {},
        })
        with self._connect() as connection:
            outcome_row = connection.execute(
                """
                SELECT working_memory_outcome FROM run_finalizations
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (pending.user_id, pending.logical_run_id),
            ).fetchone()
        if outcome_row is None:
            raise DurableStateError("transcript finalization is missing")
        memory_outcome = str(outcome_row["working_memory_outcome"])
        finalization = RunFinalization.from_dict({
            "schema_version": {"major": 1, "minor": 0},
            "finalization_id": pending.finalization_id,
            "logical_run_id": pending.logical_run_id,
            "tick_id": pending.tick_id,
            "tick_kind": "p1_transcript",
            "phase": "completed",
            "source_ack_id": ack.ack_id,
            "final_reply_receipt_id": None,
            "recovery_mode": "ack_only",
            "inference_allowed": False,
            "restream_allowed": False,
            "working_memory_outcome": memory_outcome,
            "finalized_at_ms": recorded_at_ms,
            "extensions": {},
        })
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT q.state, t.state AS claim_state
                FROM queue_items q JOIN transcript_claims t
                  ON t.user_id = q.user_id
                 AND t.logical_run_id = q.logical_run_id
                WHERE q.user_id = ? AND q.logical_run_id = ?
                """,
                (pending.user_id, pending.logical_run_id),
            ).fetchone()
            if row is None:
                raise DurableStateError("acknowledgement targets an unknown run")
            if row["state"] == "completed" and row["claim_state"] == "acknowledged":
                return
            if (
                row["state"] != "awaiting_audio_ack"
                or row["claim_state"] != "awaiting_ack"
            ):
                raise DurableStateError(
                    "run is not awaiting transcript acknowledgement"
                )
            attempts_total = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM attempts
                    WHERE user_id = ? AND logical_run_id = ?
                    """,
                    (pending.user_id, pending.logical_run_id),
                ).fetchone()[0]
            )
            run_receipt = RunReceipt.from_dict({
                "schema_version": {"major": 1, "minor": 0},
                "receipt_id": f"run-receipt:{pending.logical_run_id}",
                "logical_run_id": pending.logical_run_id,
                "tick_id": pending.tick_id,
                "outcome": "completed",
                "attempts_total": attempts_total,
                "finalization_id": pending.finalization_id,
                "recorded_at_ms": recorded_at_ms,
                "extensions": {},
            })
            connection.execute(
                """
                INSERT INTO input_receipts (
                    receipt_id, user_id, logical_run_id, ack_id,
                    receipt_json, recorded_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    input_receipt.payload.receipt_id,
                    pending.user_id,
                    pending.logical_run_id,
                    ack.ack_id,
                    input_receipt.to_json(),
                    recorded_at_ms,
                ),
            )
            connection.execute(
                """
                INSERT INTO run_receipts (
                    receipt_id, user_id, logical_run_id, receipt_json, recorded_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_receipt.payload.receipt_id,
                    pending.user_id,
                    pending.logical_run_id,
                    run_receipt.to_json(),
                    recorded_at_ms,
                ),
            )
            connection.execute(
                """
                UPDATE run_finalizations
                SET phase = 'completed', source_ack_id = ?, finalization_json = ?,
                    updated_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (
                    ack.ack_id,
                    finalization.to_json(),
                    recorded_at_ms,
                    pending.user_id,
                    pending.logical_run_id,
                ),
            )
            connection.execute(
                """
                UPDATE transcript_claims
                SET state = 'acknowledged', ack_json = ?, updated_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (
                    acknowledgement.to_json(),
                    recorded_at_ms,
                    pending.user_id,
                    pending.logical_run_id,
                ),
            )
            connection.execute(
                """
                UPDATE queue_items
                SET state = 'completed', completed_at_ms = ?, updated_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (
                    recorded_at_ms,
                    recorded_at_ms,
                    pending.user_id,
                    pending.logical_run_id,
                ),
            )
            connection.execute(
                """
                UPDATE transcript_explicit_retries
                SET state = 'completed', completed_at_ms = ?
                WHERE user_id = ? AND retry_run_id = ?
                """,
                (
                    recorded_at_ms,
                    pending.user_id,
                    pending.logical_run_id,
                ),
            )

    def transcript_run_record(
        self, *, user_id: str, logical_run_id: str
    ) -> TranscriptRunRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT q.state AS queue_state,
                       COUNT(a.attempt_id) AS attempts_total,
                       d.outcome AS decision_outcome,
                       d.visible_action_intent_count,
                       f.working_memory_outcome,
                       f.phase AS finalization_phase,
                       t.memory_version,
                       t.ack_json,
                       i.ack_id,
                       i.receipt_id AS input_receipt_id,
                       r.receipt_id AS run_receipt_id
                FROM queue_items q
                LEFT JOIN attempts a
                  ON a.user_id = q.user_id AND a.logical_run_id = q.logical_run_id
                LEFT JOIN decision_outcomes d
                  ON d.user_id = q.user_id AND d.logical_run_id = q.logical_run_id
                LEFT JOIN run_finalizations f
                  ON f.user_id = q.user_id AND f.logical_run_id = q.logical_run_id
                LEFT JOIN transcript_claims t
                  ON t.user_id = q.user_id AND t.logical_run_id = q.logical_run_id
                LEFT JOIN input_receipts i
                  ON i.user_id = q.user_id AND i.logical_run_id = q.logical_run_id
                LEFT JOIN run_receipts r
                  ON r.user_id = q.user_id AND r.logical_run_id = q.logical_run_id
                WHERE q.user_id = ? AND q.logical_run_id = ?
                GROUP BY q.state, d.outcome, d.visible_action_intent_count,
                         f.working_memory_outcome, f.phase, t.memory_version,
                         t.ack_json, i.ack_id, i.receipt_id, r.receipt_id
                """,
                (user_id, logical_run_id),
            ).fetchone()
        if row is None:
            raise KeyError(logical_run_id)
        canonical_retained: bool | None = None
        if row["ack_json"] is not None:
            canonical_retained = bool(
                TranscriptAck.from_json(
                    row["ack_json"]
                ).payload.canonical_transcript_retained
            )
        return TranscriptRunRecord(
            queue_state=str(row["queue_state"]),
            attempts_total=int(row["attempts_total"]),
            decision_outcome=(
                str(row["decision_outcome"])
                if row["decision_outcome"] is not None
                else None
            ),
            visible_action_intent_count=(
                int(row["visible_action_intent_count"])
                if row["visible_action_intent_count"] is not None
                else None
            ),
            working_memory_outcome=(
                str(row["working_memory_outcome"])
                if row["working_memory_outcome"] is not None
                else None
            ),
            memory_version=(
                int(row["memory_version"])
                if row["memory_version"] is not None
                else None
            ),
            finalization_phase=(
                str(row["finalization_phase"])
                if row["finalization_phase"] is not None
                else None
            ),
            ack_id=str(row["ack_id"]) if row["ack_id"] is not None else None,
            input_receipt_id=(
                str(row["input_receipt_id"])
                if row["input_receipt_id"] is not None
                else None
            ),
            run_receipt_id=(
                str(row["run_receipt_id"])
                if row["run_receipt_id"] is not None
                else None
            ),
            canonical_transcript_retained=canonical_retained,
        )

    def inspect_agent_run(
        self, *, user_id: str, logical_run_id: str
    ) -> AgentRunInspection:
        """Read one explicit model/tool/receipt/Stop-Hook diagnostic record."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM agent_run_inspections
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (user_id, logical_run_id),
            ).fetchone()
        if row is None:
            raise KeyError(logical_run_id)
        return AgentRunInspection(
            logical_run_id=str(row["logical_run_id"]),
            attempt_id=str(row["attempt_id"]),
            provider=str(row["provider"]),
            model=str(row["model"]),
            api_mode=str(row["api_mode"]),
            reasoning_effort=str(row["reasoning_effort"]),
            decision_outcome=str(row["decision_outcome"]),
            final_output=str(row["final_output"]),
            tool_discoveries=tuple(json.loads(row["tool_discoveries_json"])),
            usage={
                str(key): int(value)
                for key, value in json.loads(row["usage_json"]).items()
            },
            stop_hook_outcome=str(row["stop_hook_outcome"]),
            stop_hook_cache_identity={
                str(key): str(value)
                for key, value in json.loads(
                    row["stop_hook_cache_identity_json"]
                ).items()
            },
            memory_version=int(row["memory_version"]),
            memory_token_count=(
                None
                if row["memory_token_count"] is None
                else int(row["memory_token_count"])
            ),
            recorded_at_ms=int(row["recorded_at_ms"]),
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
            if state == "quarantined" and item["kind"] == "p1_transcript":
                self._insert_pending_transcript_quarantine_locked(
                    connection,
                    user_id=user_id,
                    logical_run_id=logical_run_id,
                    quarantine_id=quarantine_id,
                    failure_code=failure_code,
                    quarantined_at_ms=now_ms,
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
            execution_started = connection.execute(
                "SELECT 1 FROM attempt_execution_started WHERE attempt_id = ?",
                (attempt["attempt_id"],),
            ).fetchone()
            if execution_started is None:
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
                if item["kind"] == "p1_transcript":
                    self._insert_pending_transcript_quarantine_locked(
                        connection,
                        user_id=user_id,
                        logical_run_id=str(item["logical_run_id"]),
                        quarantine_id=f"{item['logical_run_id']}:quarantine",
                        failure_code="crash_discarded_uncheckpointed_inference",
                        quarantined_at_ms=now_ms,
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
    def _insert_pending_transcript_quarantine_locked(
        connection: sqlite3.Connection,
        *,
        user_id: str,
        logical_run_id: str,
        quarantine_id: str,
        failure_code: str,
        quarantined_at_ms: int,
    ) -> None:
        claim = connection.execute(
            """
            SELECT claim_id, claim_json FROM transcript_claims
            WHERE user_id = ? AND logical_run_id = ?
              AND claim_id IS NOT NULL AND claim_json IS NOT NULL
            """,
            (user_id, logical_run_id),
        ).fetchone()
        if claim is None:
            # Generic coordinator tests and non-transcript adapters may use the
            # p1_transcript kind without the real Input Pump. There is no
            # cross-store suffix to synchronize when no Dataplane claim exists.
            return
        connection.execute(
            """
            INSERT OR IGNORE INTO transcript_quarantines (
                quarantine_id, user_id, original_logical_run_id, claim_id,
                failure_code, quarantined_at_ms, sync_state
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                quarantine_id,
                user_id,
                logical_run_id,
                claim["claim_id"],
                failure_code,
                quarantined_at_ms,
            ),
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

    @staticmethod
    def _stored_transcript_claim_from_row(row: sqlite3.Row) -> StoredTranscriptClaim:
        claim = (
            TranscriptClaim.from_json(row["claim_json"])
            if row["claim_json"] is not None
            else None
        )
        return StoredTranscriptClaim(
            user_id=str(row["user_id"]),
            tick_id=str(row["tick_id"]),
            logical_run_id=str(row["logical_run_id"]),
            claim_request_id=str(row["claim_request_id"]),
            claim_id=(str(row["claim_id"]) if row["claim_id"] is not None else None),
            claim=claim,
            state=str(row["state"]),
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
    "AgentRunInspection",
    "AttemptDiagnostic",
    "CheckpointRecord",
    "DurableRunState",
    "DurableStateError",
    "LeaseDiagnostic",
    "LeasedRun",
    "PendingTranscriptAck",
    "PendingTranscriptQuarantine",
    "QueueDiagnostic",
    "QuarantineDiagnostic",
    "ReceiptConflict",
    "SCHEMA_VERSION",
    "StateDiagnostics",
    "StoredTranscriptClaim",
    "TranscriptRunRecord",
    "TranscriptQuarantineInspection",
    "ToolReceiptRecord",
    "default_database_path",
    "diagnostics_as_dict",
]
