"""Authoritative retention, inspection, and confirmation-bound reset services.

The model receives read-only helpers from this module. Destructive maintenance
is intentionally a separate local-operator boundary: a reset is first planned,
then executed only while the Harness is stopped and the exact plan is confirmed.
Neither path opens or mutates Thine's backend-owned databases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import time
from typing import Any, Callable, Literal, Mapping, cast
import uuid

from .contracts.reset import ResetCommand, ResetResult
from .home_state import HomeStateProjector
from .run_state import DurableRunState, DurableStateError, diagnostics_as_dict
from .schedules import OneShotScheduleService
from .topics_preferences import TopicPreferenceService


DAY_MS = 24 * 60 * 60 * 1000
INTERACTION_CONTENT_RETENTION_MS = 7 * DAY_MS
COMPLETED_RUN_RETENTION_MS = 30 * DAY_MS
RECEIPT_RETENTION_MS = 90 * DAY_MS
RESOLVED_TOPIC_RETENTION_MS = 90 * DAY_MS
HISTORY_LIMIT = 50
_VERSION = {"major": 1, "minor": 0}
_PRESERVED_AUTHORITIES = [
    "canonical_transcripts",
    "speaker_mappings_and_relationships",
    "aggregation_buffer",
    "quarantine",
]

ResetScope = Literal[
    "working_memory_topics",
    "queues_schedules_receipts",
    "home_state",
    "all_hermes_state",
]


@dataclass(frozen=True)
class ResetPlan:
    reset_id: str
    user_id: str
    scope: ResetScope
    confirmation: str
    preference_revision: int
    explicit_preferences_to_delete: tuple[dict[str, object], ...]
    targets: dict[str, int]
    preserved: tuple[str, ...]
    state_fingerprint: str
    created_at_ms: int

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["explicit_preferences_to_delete"] = [
            dict(item) for item in self.explicit_preferences_to_delete
        ]
        value["preserved"] = list(self.preserved)
        return value


@dataclass(frozen=True)
class RetentionResult:
    user_id: str
    completed_at_ms: int
    changed: dict[str, int]
    preserved_quarantines: int
    preserved_explicit_retries: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AuthoritativeStateReader:
    """Read directly from authoritative repositories, never dashboard projections."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        home: HomeStateProjector,
        topics: TopicPreferenceService | None = None,
        schedules: OneShotScheduleService | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._state = state
        self._home = home
        self._topics = topics or TopicPreferenceService(state)
        self._schedules = schedules or OneShotScheduleService(state)
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def snapshot(self, user_id: str) -> dict[str, object]:
        with self._state._connect() as connection:
            current_memory = connection.execute(
                """
                SELECT version, markdown, token_count, last_run_id
                FROM working_memory_state WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            unchanged_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM working_memory_unchanged WHERE user_id = ?",
                    (user_id,),
                ).fetchone()[0]
            )
            interaction_clock = connection.execute(
                "SELECT * FROM interaction_clock_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            permission = connection.execute(
                """
                SELECT permission_json, observed_at_ms
                FROM communication_permission_observations WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            counts = {
                "transcript_claims": self._count(
                    connection, "transcript_claims", user_id
                ),
                "working_memory_versions": self._count(
                    connection, "working_memory_versions", user_id
                ),
                "tool_receipts": self._count(connection, "tool_receipts", user_id),
                "communication_actions": self._count(
                    connection, "communication_actions", user_id
                ),
                "communication_allowance": self._count(
                    connection, "communication_allowance_ledger", user_id
                ),
                "agent_run_inspections": self._count(
                    connection, "agent_run_inspections", user_id
                ),
            }
        memory = (
            {"version": 0, "markdown": "", "token_count": None, "last_run_id": None}
            if current_memory is None
            else dict(current_memory)
        )
        return {
            "authoritative": True,
            "generated_at_ms": int(self._clock_ms()),
            "queue_state": diagnostics_as_dict(self._state.diagnostics(user_id)),
            "working_memory": {**memory, "unchanged_markers": unchanged_count},
            "home": self._home.current(user_id).to_dict(),
            "schedules": [
                item.to_tool_dict() for item in self._schedules.list(user_id)
            ],
            "topics_preferences": self._topics.inspect(user_id),
            "interaction_clock": (
                None if interaction_clock is None else dict(interaction_clock)
            ),
            "notification_permission": (
                None
                if permission is None
                else {
                    "permission": json.loads(str(permission["permission_json"])),
                    "observed_at_ms": int(permission["observed_at_ms"]),
                }
            ),
            "authoritative_counts": counts,
        }

    def working_memory_history(
        self, user_id: str, *, limit: int = HISTORY_LIMIT
    ) -> list[dict[str, object]]:
        bounded = _bounded_limit(limit)
        with self._state._connect() as connection:
            rows = connection.execute(
                """
                SELECT version, markdown, configured_model_token_count,
                       tokenizer_status, logical_run_id, committed_at_ms
                FROM working_memory_versions WHERE user_id = ?
                ORDER BY version DESC LIMIT ?
                """,
                (user_id, bounded),
            ).fetchall()
        return [dict(row) for row in rows]

    def working_memory_current(self, user_id: str) -> dict[str, object]:
        """Return the authoritative current Working Memory projection."""
        with self._state._connect() as connection:
            row = connection.execute(
                """
                SELECT version, markdown, token_count, last_run_id
                FROM working_memory_state WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            unchanged = int(
                connection.execute(
                    "SELECT COUNT(*) FROM working_memory_unchanged WHERE user_id = ?",
                    (user_id,),
                ).fetchone()[0]
            )
        current = (
            {"version": 0, "markdown": "", "token_count": None, "last_run_id": None}
            if row is None
            else dict(row)
        )
        return {**current, "unchanged_markers": unchanged}

    def home_history(self, user_id: str) -> dict[str, object]:
        """Return bounded Home history through the Home owner Interface."""
        return cast(dict[str, object], self._home.history(user_id).to_dict())

    def home_current(self, user_id: str) -> dict[str, object]:
        """Return current Home state through the Home owner Interface."""
        return cast(dict[str, object], self._home.current(user_id).to_dict())

    def debug_invocations(
        self, user_id: str, *, limit: int = HISTORY_LIMIT
    ) -> list[dict[str, object]]:
        bounded = _bounded_limit(limit)
        with self._state._connect() as connection:
            rows = connection.execute(
                """
                SELECT logical_run_id, attempt_id, provider, model, api_mode,
                       reasoning_effort, decision_outcome, usage_json,
                       stop_hook_outcome, memory_version, memory_token_count,
                       recorded_at_ms
                FROM agent_run_inspections WHERE user_id = ?
                ORDER BY recorded_at_ms DESC, logical_run_id DESC LIMIT ?
                """,
                (user_id, bounded),
            ).fetchall()
        return [
            {
                **dict(row),
                "usage": json.loads(str(row["usage_json"])),
                "redacted": True,
            }
            for row in rows
        ]

    def quarantines(self, user_id: str) -> dict[str, object]:
        with self._state._connect() as connection:
            generic = connection.execute(
                """
                SELECT quarantine_id, logical_run_id, tick_id, source_kind,
                       source_id, attempt_ordinal, failure_code, quarantined_at_ms
                FROM quarantines WHERE user_id = ?
                ORDER BY quarantined_at_ms, quarantine_id
                """,
                (user_id,),
            ).fetchall()
            transcript = connection.execute(
                """
                SELECT quarantine_id, original_logical_run_id, claim_id,
                       failure_code, quarantined_at_ms, sync_state
                FROM transcript_quarantines WHERE user_id = ?
                ORDER BY quarantined_at_ms, quarantine_id
                """,
                (user_id,),
            ).fetchall()
            speaker = connection.execute(
                """
                SELECT quarantine_id, event_id, original_logical_run_id, cursor,
                       state, updated_at_ms
                FROM speaker_mapping_inputs
                WHERE user_id = ? AND quarantine_id IS NOT NULL
                ORDER BY cursor, event_id
                """,
                (user_id,),
            ).fetchall()
            interaction = connection.execute(
                """
                SELECT quarantine_id, original_logical_run_id, batch_id,
                       first_cursor, last_cursor, failure_code,
                       quarantined_at_ms, sync_state
                FROM interaction_quarantines WHERE user_id = ?
                ORDER BY quarantined_at_ms, quarantine_id
                """,
                (user_id,),
            ).fetchall()
            retries: dict[str, list[dict[str, object]]] = {}
            for table in (
                "transcript_explicit_retries",
                "speaker_explicit_retries",
                "interaction_explicit_retries",
            ):
                rows = connection.execute(
                    f"""
                    SELECT retry_run_id, quarantine_id, state, created_at_ms,
                           completed_at_ms
                    FROM {table} WHERE user_id = ?
                    ORDER BY created_at_ms, retry_run_id
                    """,
                    (user_id,),
                ).fetchall()
                retries[table.removesuffix("_explicit_retries")] = [
                    dict(row) for row in rows
                ]
        return {
            "authoritative": True,
            "quarantine_is_immutable": True,
            "normal_cursor_rewind_allowed": False,
            "generic": [dict(row) for row in generic],
            "transcript": [dict(row) for row in transcript],
            "speaker": [dict(row) for row in speaker],
            "interaction": [dict(row) for row in interaction],
            "explicit_retries": retries,
        }

    @staticmethod
    def _count(connection: Any, table: str, user_id: str) -> int:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
        )


class RetentionResetService:
    """Apply time-bounded cleanup and exact, local-operator-confirmed resets."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        home: HomeStateProjector,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._state = state
        self._home = home
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def retention_policy(self) -> dict[str, object]:
        return {
            "interaction_payload_days": 7,
            "completed_attempt_checkpoint_days": 30,
            "action_and_communication_receipt_days": 90,
            "resolved_topic_days": 90,
            "working_memory_versions": HISTORY_LIMIT,
            "home_revisions": HISTORY_LIMIT,
            "debug_invocations": HISTORY_LIMIT,
            "active_schedules": "until_terminal",
            "unresolved_topics": "until_resolved",
            "explicit_preferences": "until_user_reverses_or_confirmed_full_reset",
            "quarantines_and_retry_provenance": "indefinite",
            "canonical_transcripts_and_speakers": "backend_owned_not_touched",
        }

    def cleanup(self, user_id: str) -> RetentionResult:
        now_ms = int(self._clock_ms())
        interaction_cutoff = now_ms - INTERACTION_CONTENT_RETENTION_MS
        run_cutoff = now_ms - COMPLETED_RUN_RETENTION_MS
        receipt_cutoff = now_ms - RECEIPT_RETENTION_MS
        changed: dict[str, int] = {}
        with self._state._transaction() as connection:
            changed["interaction_payloads_redacted"] = connection.execute(
                """
                UPDATE interaction_claims SET batch_json = NULL
                WHERE user_id = ? AND updated_at_ms < ?
                  AND state IN ('completed', 'quarantined')
                  AND logical_run_id NOT IN (
                      SELECT original_logical_run_id FROM interaction_quarantines
                      WHERE user_id = ?
                  )
                """,
                (user_id, interaction_cutoff, user_id),
            ).rowcount
            changed["checkpoint_rows"] = connection.execute(
                """
                DELETE FROM checkpoints WHERE user_id = ?
                  AND updated_at_ms < ?
                  AND logical_run_id IN (
                      SELECT logical_run_id FROM queue_items
                      WHERE user_id = ? AND state IN ('completed', 'failed_terminal')
                  )
                """,
                (user_id, run_cutoff, user_id),
            ).rowcount
            connection.execute(
                """
                DELETE FROM attempt_execution_started
                WHERE user_id = ? AND started_at_ms < ?
                  AND attempt_id IN (
                      SELECT attempt_id FROM attempts WHERE user_id = ?
                        AND status != 'running'
                  )
                """,
                (user_id, run_cutoff, user_id),
            )
            changed["attempt_rows"] = connection.execute(
                """
                DELETE FROM attempts WHERE user_id = ?
                  AND finished_at_ms IS NOT NULL AND finished_at_ms < ?
                  AND logical_run_id NOT IN (
                      SELECT logical_run_id FROM quarantines WHERE user_id = ?
                  )
                  AND logical_run_id NOT IN (
                      SELECT retry_run_id FROM transcript_explicit_retries
                      WHERE user_id = ?
                  )
                  AND logical_run_id NOT IN (
                      SELECT retry_run_id FROM speaker_explicit_retries
                      WHERE user_id = ?
                  )
                  AND logical_run_id NOT IN (
                      SELECT retry_run_id FROM interaction_explicit_retries
                      WHERE user_id = ?
                  )
                """,
                (user_id, run_cutoff, user_id, user_id, user_id, user_id),
            ).rowcount
            changed["working_memory_unchanged_rows"] = connection.execute(
                """
                DELETE FROM working_memory_unchanged
                WHERE user_id = ? AND recorded_at_ms < ?
                  AND logical_run_id NOT IN (
                      SELECT logical_run_id FROM checkpoints WHERE user_id = ?
                  )
                """,
                (user_id, run_cutoff, user_id),
            ).rowcount
            changed["working_memory_versions"] = self._prune_memory_versions(
                connection, user_id
            )
            changed["debug_invocations"] = self._prune_latest(
                connection,
                table="agent_run_inspections",
                user_id=user_id,
                order_by="recorded_at_ms DESC, logical_run_id DESC",
            )
            changed["resolved_topics"] = connection.execute(
                """
                DELETE FROM durable_topics WHERE user_id = ?
                  AND state IN ('resolved', 'expired') AND updated_at_ms < ?
                  AND do_not_ask = 0
                """,
                (user_id, receipt_cutoff),
            ).rowcount
            terminal_schedules = connection.execute(
                """
                SELECT schedule_id FROM one_shot_schedules
                WHERE user_id = ? AND status IN (
                    'cancelled', 'completed', 'failed_terminal'
                ) AND COALESCE(completed_at_ms, updated_at_ms) < ?
                """,
                (user_id, receipt_cutoff),
            ).fetchall()
            schedule_ids = [str(row["schedule_id"]) for row in terminal_schedules]
            changed["terminal_schedules"] = 0
            for schedule_id in schedule_ids:
                connection.execute(
                    """
                    DELETE FROM one_shot_schedule_mutations
                    WHERE user_id = ? AND schedule_id = ?
                    """,
                    (user_id, schedule_id),
                )
                changed["terminal_schedules"] += connection.execute(
                    """
                    DELETE FROM one_shot_schedules
                    WHERE user_id = ? AND schedule_id = ?
                    """,
                    (user_id, schedule_id),
                ).rowcount
            changed["communication_receipts"] = self._prune_communications(
                connection, user_id=user_id, cutoff_ms=receipt_cutoff
            )
            changed["tool_receipts"] = connection.execute(
                """
                DELETE FROM tool_receipts WHERE user_id = ?
                  AND acknowledged_at_ms < ?
                  AND logical_run_id NOT IN (
                      SELECT logical_run_id FROM checkpoints WHERE user_id = ?
                  )
                  AND logical_run_id NOT IN (
                      SELECT logical_run_id FROM quarantines WHERE user_id = ?
                  )
                  AND logical_run_id NOT IN (
                      SELECT retry_run_id FROM transcript_explicit_retries
                      WHERE user_id = ?
                  )
                  AND logical_run_id NOT IN (
                      SELECT retry_run_id FROM speaker_explicit_retries
                      WHERE user_id = ?
                  )
                  AND logical_run_id NOT IN (
                      SELECT retry_run_id FROM interaction_explicit_retries
                      WHERE user_id = ?
                  )
                """,
                (
                    user_id,
                    receipt_cutoff,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                ),
            ).rowcount
            changed["topic_preference_receipts"] = connection.execute(
                """
                DELETE FROM topic_preference_receipts WHERE user_id = ?
                  AND created_at_ms < ?
                  AND logical_run_id NOT IN (
                      SELECT logical_run_id FROM checkpoints WHERE user_id = ?
                  )
                  AND logical_run_id NOT IN (
                      SELECT logical_run_id FROM quarantines WHERE user_id = ?
                  )
                  AND logical_run_id NOT IN (
                      SELECT retry_run_id FROM transcript_explicit_retries
                      WHERE user_id = ?
                  )
                  AND logical_run_id NOT IN (
                      SELECT retry_run_id FROM speaker_explicit_retries
                      WHERE user_id = ?
                  )
                  AND logical_run_id NOT IN (
                      SELECT retry_run_id FROM interaction_explicit_retries
                      WHERE user_id = ?
                  )
                """,
                (
                    user_id,
                    receipt_cutoff,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                ),
            ).rowcount
            quarantine_count, retry_count = self._preserved_counts(connection, user_id)
            event_id = f"retention:{uuid.uuid4()}"
            connection.execute(
                """
                INSERT INTO maintenance_events (
                    event_id, user_id, event_kind, subject_id,
                    details_json, recorded_at_ms
                ) VALUES (?, ?, 'retention_cleanup', ?, ?, ?)
                """,
                (
                    event_id,
                    user_id,
                    event_id,
                    _canonical({
                        "changed": changed,
                        "preserved_quarantines": quarantine_count,
                        "preserved_explicit_retries": retry_count,
                    }),
                    now_ms,
                ),
            )
        return RetentionResult(
            user_id=user_id,
            completed_at_ms=now_ms,
            changed=changed,
            preserved_quarantines=quarantine_count,
            preserved_explicit_retries=retry_count,
        )

    def plan_reset(self, user_id: str, scope: ResetScope) -> ResetPlan:
        if scope not in {
            "working_memory_topics",
            "queues_schedules_receipts",
            "home_state",
            "all_hermes_state",
        }:
            raise ValueError("unsupported reset scope")
        now_ms = int(self._clock_ms())
        reset_id = f"reset:{uuid.uuid4()}"
        snapshot = self._reset_snapshot(user_id, scope)
        plan = ResetPlan(
            reset_id=reset_id,
            user_id=user_id,
            scope=scope,
            confirmation=f"CONFIRM {reset_id}",
            preference_revision=cast(int, snapshot["preference_revision"]),
            explicit_preferences_to_delete=tuple(
                cast(list[dict[str, object]], snapshot["preferences"])
                if scope == "all_hermes_state"
                else []
            ),
            targets=cast(dict[str, int], snapshot["targets"]),
            preserved=tuple(_PRESERVED_AUTHORITIES),
            state_fingerprint=str(snapshot["state_fingerprint"]),
            created_at_ms=now_ms,
        )
        with self._state._transaction() as connection:
            connection.execute(
                """
                INSERT INTO maintenance_plans (
                    reset_id, user_id, scope, plan_json, preference_revision,
                    status, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, 'planned', ?)
                """,
                (
                    reset_id,
                    user_id,
                    scope,
                    _canonical(plan.to_dict()),
                    plan.preference_revision,
                    now_ms,
                ),
            )
        return plan

    def execute_reset(
        self,
        *,
        reset_id: str,
        confirmation: str,
        harness_stopped: bool,
    ) -> ResetResult:
        now_ms = int(self._clock_ms())
        plan = self._load_plan(reset_id)
        status = self._plan_status(reset_id)
        if status == "completed":
            return self._reset_result(
                plan,
                status="completed",
                execution_requested=True,
                completed_at_ms=self._plan_completed_at(reset_id),
            )
        live_work_count = self.live_work_count(plan.user_id)
        if confirmation != plan.confirmation:
            return self._reset_result(
                plan,
                status="confirmation_required",
                execution_requested=False,
                completed_at_ms=None,
            )
        if not harness_stopped or live_work_count:
            return self._reset_result(
                plan,
                status="refused_live_work",
                execution_requested=True,
                completed_at_ms=None,
            )
        if status == "planned":
            current = self._reset_snapshot(plan.user_id, plan.scope)
            if str(current["state_fingerprint"]) != plan.state_fingerprint:
                with self._state._transaction() as connection:
                    connection.execute(
                        """
                        UPDATE maintenance_plans SET status = 'superseded'
                        WHERE reset_id = ? AND status = 'planned'
                        """,
                        (reset_id,),
                    )
                return self._reset_result(
                    plan,
                    status="confirmation_required",
                    execution_requested=False,
                    completed_at_ms=None,
                )
            command = ResetCommand.from_dict({
                "schema_version": _VERSION,
                "reset_id": plan.reset_id,
                "scope": plan.scope,
                "confirmed": True,
                "confirmed_preferences_revision": (
                    plan.preference_revision
                    if plan.scope == "all_hermes_state"
                    else None
                ),
                "current_preferences_revision": (
                    cast(int, current["preference_revision"])
                    if plan.scope == "all_hermes_state"
                    else None
                ),
                "explicit_preferences_to_delete": [
                    dict(item) for item in plan.explicit_preferences_to_delete
                ],
                "execute": True,
                "harness_stopped": True,
                "live_work_count": 0,
                "requested_at_ms": plan.created_at_ms,
                "extensions": {},
            })
            del command  # Contract validation is the destructive-boundary gate.
            with self._state._transaction() as connection:
                transitioned = connection.execute(
                    """
                    UPDATE maintenance_plans SET status = 'executing'
                    WHERE reset_id = ? AND status = 'planned'
                    """,
                    (reset_id,),
                ).rowcount
            if transitioned != 1:
                raise DurableStateError("reset plan execution ownership was lost")
        self._apply_reset(plan, now_ms=now_ms)
        return self._reset_result(
            plan,
            status="completed",
            execution_requested=True,
            completed_at_ms=now_ms,
        )

    def live_work_count(self, user_id: str) -> int:
        with self._state._connect() as connection:
            queue = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM queue_items WHERE user_id = ?
                      AND state IN (
                          'leased', 'running', 'memory_finalizing',
                          'awaiting_audio_ack', 'awaiting_speaker_cursor_ack',
                          'awaiting_interaction_ack', 'reply_delivery_pending',
                          'reply_persisted', 'memory_finalization_pending'
                      )
                    """,
                    (user_id,),
                ).fetchone()[0]
            )
            actions = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM communication_actions WHERE user_id = ?
                      AND state IN ('reserved', 'executing', 'reconciling', 'retryable')
                    """,
                    (user_id,),
                ).fetchone()[0]
            )
        return queue + actions

    def _apply_reset(self, plan: ResetPlan, *, now_ms: int) -> None:
        reset_memory = plan.scope in {"working_memory_topics", "all_hermes_state"}
        reset_operations = plan.scope in {
            "queues_schedules_receipts",
            "all_hermes_state",
        }
        reset_home = plan.scope in {"home_state", "all_hermes_state"}
        with self._state._transaction() as connection:
            quarantine_count, retry_count = self._preserved_counts(
                connection, plan.user_id
            )
            if reset_memory:
                connection.execute(
                    "DELETE FROM working_memory_unchanged WHERE user_id = ?",
                    (plan.user_id,),
                )
                connection.execute(
                    "DELETE FROM working_memory_versions WHERE user_id = ?",
                    (plan.user_id,),
                )
                connection.execute(
                    "DELETE FROM working_memory_state WHERE user_id = ?",
                    (plan.user_id,),
                )
                if plan.scope == "all_hermes_state":
                    connection.execute(
                        "DELETE FROM durable_topics WHERE user_id = ?",
                        (plan.user_id,),
                    )
                else:
                    connection.execute(
                        """
                        DELETE FROM durable_topics
                        WHERE user_id = ? AND do_not_ask = 0
                        """,
                        (plan.user_id,),
                    )
                if plan.scope == "all_hermes_state":
                    connection.execute(
                        "DELETE FROM explicit_preferences WHERE user_id = ?",
                        (plan.user_id,),
                    )
                    connection.execute(
                        "DELETE FROM explicit_corrections WHERE user_id = ?",
                        (plan.user_id,),
                    )
            if reset_operations:
                connection.execute(
                    "DELETE FROM checkpoints WHERE user_id = ?",
                    (plan.user_id,),
                )
                connection.execute(
                    """
                    UPDATE queue_items
                    SET state = 'failed_terminal', lease_owner = NULL,
                        lease_token = NULL, lease_expires_at_ms = NULL,
                        completed_at_ms = COALESCE(completed_at_ms, ?),
                        updated_at_ms = ?
                    WHERE user_id = ?
                      AND state NOT IN ('completed', 'failed_terminal', 'quarantined')
                      AND logical_run_id NOT IN (
                          SELECT logical_run_id FROM quarantines WHERE user_id = ?
                      )
                    """,
                    (now_ms, now_ms, plan.user_id, plan.user_id),
                )
                connection.execute(
                    """
                    UPDATE attempts SET status = 'failed_fault',
                        failure_code = 'operator_scoped_reset',
                        finished_at_ms = COALESCE(finished_at_ms, ?)
                    WHERE user_id = ? AND status = 'running'
                    """,
                    (now_ms, plan.user_id),
                )
                connection.execute(
                    """
                    UPDATE one_shot_schedules
                    SET status = 'cancelled', completed_at_ms = ?, updated_at_ms = ?
                    WHERE user_id = ? AND status IN ('active', 'enqueued')
                    """,
                    (now_ms, now_ms, plan.user_id),
                )
            details = {
                "scope": plan.scope,
                "targets": plan.targets,
                "preserved_authorities": _PRESERVED_AUTHORITIES,
                "preserved_quarantines": quarantine_count,
                "preserved_explicit_retries": retry_count,
            }
            connection.execute(
                """
                INSERT OR IGNORE INTO maintenance_events (
                    event_id, user_id, event_kind, subject_id,
                    details_json, recorded_at_ms
                ) VALUES (?, ?, 'reset_completed', ?, ?, ?)
                """,
                (
                    f"event:{plan.reset_id}",
                    plan.user_id,
                    plan.reset_id,
                    _canonical(details),
                    now_ms,
                ),
            )
        if reset_home:
            self._home.reset_to_default(user_id=plan.user_id, reset_id=plan.reset_id)
        with self._state._transaction() as connection:
            completed = connection.execute(
                """
                UPDATE maintenance_plans SET status = 'completed', completed_at_ms = ?
                WHERE reset_id = ? AND status = 'executing'
                """,
                (now_ms, plan.reset_id),
            ).rowcount
        if completed != 1:
            raise DurableStateError("reset plan completion state was lost")

    def _reset_snapshot(self, user_id: str, scope: ResetScope) -> dict[str, object]:
        with self._state._connect() as connection:
            preferences = [
                {
                    "key": str(row["preference_key"]),
                    "current_value": bool(row["preference_value"]),
                }
                for row in connection.execute(
                    """
                    SELECT preference_key, preference_value
                    FROM explicit_preferences WHERE user_id = ?
                    ORDER BY preference_key
                    """,
                    (user_id,),
                ).fetchall()
            ]
            preference_revision = max(
                1,
                int(
                    connection.execute(
                        """
                        SELECT COALESCE(MAX(revision), 0)
                        FROM explicit_preferences WHERE user_id = ?
                        """,
                        (user_id,),
                    ).fetchone()[0]
                ),
            )
            all_targets = {
                "working_memory_state": self._count(
                    connection, "working_memory_state", user_id
                ),
                "working_memory_versions": self._count(
                    connection, "working_memory_versions", user_id
                ),
                "working_memory_unchanged": self._count(
                    connection, "working_memory_unchanged", user_id
                ),
                "routine_topics": int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM durable_topics
                        WHERE user_id = ? AND (? = 1 OR do_not_ask = 0)
                        """,
                        (user_id, int(scope == "all_hermes_state")),
                    ).fetchone()[0]
                ),
                "explicit_preferences": (
                    len(preferences) if scope == "all_hermes_state" else 0
                ),
                "explicit_corrections": (
                    self._count(connection, "explicit_corrections", user_id)
                    if scope == "all_hermes_state"
                    else 0
                ),
                "nonterminal_queue_items": int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM queue_items WHERE user_id = ?
                          AND state NOT IN (
                              'completed', 'failed_terminal', 'quarantined'
                          )
                        """,
                        (user_id,),
                    ).fetchone()[0]
                ),
                "checkpoints": self._count(connection, "checkpoints", user_id),
                "active_schedules": int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM one_shot_schedules
                        WHERE user_id = ? AND status = 'active'
                        """,
                        (user_id,),
                    ).fetchone()[0]
                ),
            }
            quarantine_count, retry_count = self._preserved_counts(connection, user_id)
        home_counts = self._home.maintenance_snapshot(user_id)
        all_targets.update({f"home_{key}": value for key, value in home_counts.items()})
        keys_by_scope = {
            "working_memory_topics": {
                "working_memory_state",
                "working_memory_versions",
                "working_memory_unchanged",
                "routine_topics",
            },
            "queues_schedules_receipts": {
                "nonterminal_queue_items",
                "checkpoints",
                "active_schedules",
            },
            "home_state": {
                "home_current_rows",
                "home_revision_rows",
                "home_action_receipt_rows",
            },
            "all_hermes_state": set(all_targets),
        }
        targets = {
            key: value
            for key, value in all_targets.items()
            if key in keys_by_scope[scope]
        }
        relevant = {
            "scope": scope,
            "targets": targets,
            "preferences": preferences if scope == "all_hermes_state" else [],
            "preference_revision": (
                preference_revision if scope == "all_hermes_state" else 1
            ),
            "preserved_quarantines": quarantine_count,
            "preserved_explicit_retries": retry_count,
        }
        return {
            **relevant,
            "state_fingerprint": hashlib.sha256(
                _canonical(relevant).encode("utf-8")
            ).hexdigest(),
        }

    def _load_plan(self, reset_id: str) -> ResetPlan:
        with self._state._connect() as connection:
            row = connection.execute(
                """
                SELECT plan_json, status FROM maintenance_plans WHERE reset_id = ?
                """,
                (reset_id,),
            ).fetchone()
        if row is None:
            raise KeyError(reset_id)
        if str(row["status"]) == "completed":
            payload = cast(dict[str, Any], json.loads(str(row["plan_json"])))
            return _plan_from_dict(payload)
        if str(row["status"]) not in {"planned", "executing"}:
            raise DurableStateError("reset plan is stale; create a new plan")
        return _plan_from_dict(cast(dict[str, Any], json.loads(str(row["plan_json"]))))

    def _plan_status(self, reset_id: str) -> str:
        with self._state._connect() as connection:
            row = connection.execute(
                "SELECT status FROM maintenance_plans WHERE reset_id = ?",
                (reset_id,),
            ).fetchone()
        if row is None:
            raise KeyError(reset_id)
        return str(row["status"])

    def _plan_completed_at(self, reset_id: str) -> int:
        with self._state._connect() as connection:
            row = connection.execute(
                "SELECT completed_at_ms FROM maintenance_plans WHERE reset_id = ?",
                (reset_id,),
            ).fetchone()
        if row is None or row["completed_at_ms"] is None:
            raise DurableStateError("completed reset lost its completion timestamp")
        return int(row["completed_at_ms"])

    def _reset_result(
        self,
        plan: ResetPlan,
        *,
        status: Literal["completed", "refused_live_work", "confirmation_required"],
        execution_requested: bool,
        completed_at_ms: int | None,
    ) -> ResetResult:
        return ResetResult.from_dict({
            "schema_version": _VERSION,
            "reset_id": plan.reset_id,
            "status": status,
            "execution_requested": execution_requested,
            "preconditions_satisfied": status == "completed",
            "preserved_authorities": _PRESERVED_AUTHORITIES,
            "default_home_recreated": (
                status == "completed"
                and plan.scope in {"home_state", "all_hermes_state"}
            ),
            "completed_at_ms": completed_at_ms,
            "extensions": {},
        })

    @staticmethod
    def _count(connection: Any, table: str, user_id: str) -> int:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
        )

    @staticmethod
    def _preserved_counts(connection: Any, user_id: str) -> tuple[int, int]:
        quarantines = int(
            connection.execute(
                "SELECT COUNT(*) FROM quarantines WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
        )
        retries = sum(
            int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (user_id,)
                ).fetchone()[0]
            )
            for table in (
                "transcript_explicit_retries",
                "speaker_explicit_retries",
                "interaction_explicit_retries",
            )
        )
        return quarantines, retries

    @staticmethod
    def _prune_latest(
        connection: Any,
        *,
        table: str,
        user_id: str,
        order_by: str,
    ) -> int:
        return connection.execute(
            f"""
            DELETE FROM {table} WHERE user_id = ? AND rowid NOT IN (
                SELECT rowid FROM {table} WHERE user_id = ?
                ORDER BY {order_by} LIMIT ?
            )
            """,
            (user_id, user_id, HISTORY_LIMIT),
        ).rowcount

    @staticmethod
    def _prune_memory_versions(connection: Any, user_id: str) -> int:
        protected = {
            int(row[0])
            for row in connection.execute(
                """
                SELECT memory_version FROM agent_run_inspections
                WHERE user_id = ? AND logical_run_id IN (
                    SELECT logical_run_id FROM checkpoints WHERE user_id = ?
                )
                """,
                (user_id, user_id),
            ).fetchall()
        }
        current = connection.execute(
            "SELECT version FROM working_memory_state WHERE user_id = ?", (user_id,)
        ).fetchone()
        if current is not None:
            protected.add(int(current[0]))
        keep = {
            int(row[0])
            for row in connection.execute(
                """
                SELECT version FROM working_memory_versions
                WHERE user_id = ? ORDER BY version DESC LIMIT ?
                """,
                (user_id, HISTORY_LIMIT),
            ).fetchall()
        }
        keep.update(protected)
        if not keep:
            return 0
        placeholders = ",".join("?" for _ in keep)
        return connection.execute(
            f"""
            DELETE FROM working_memory_versions
            WHERE user_id = ? AND version NOT IN ({placeholders})
            """,
            (user_id, *sorted(keep)),
        ).rowcount

    @staticmethod
    def _prune_communications(connection: Any, *, user_id: str, cutoff_ms: int) -> int:
        rows = connection.execute(
            """
            SELECT action_id FROM communication_actions
            WHERE user_id = ? AND updated_at_ms < ?
              AND state NOT IN ('reserved', 'executing', 'reconciling', 'retryable')
              AND logical_run_id NOT IN (
                  SELECT logical_run_id FROM checkpoints WHERE user_id = ?
              )
              AND logical_run_id NOT IN (
                  SELECT logical_run_id FROM quarantines WHERE user_id = ?
              )
              AND logical_run_id NOT IN (
                  SELECT retry_run_id FROM transcript_explicit_retries
                  WHERE user_id = ?
              )
              AND logical_run_id NOT IN (
                  SELECT retry_run_id FROM speaker_explicit_retries
                  WHERE user_id = ?
              )
              AND logical_run_id NOT IN (
                  SELECT retry_run_id FROM interaction_explicit_retries
                  WHERE user_id = ?
              )
            """,
            (
                user_id,
                cutoff_ms,
                user_id,
                user_id,
                user_id,
                user_id,
                user_id,
            ),
        ).fetchall()
        changed = 0
        for row in rows:
            action_id = str(row["action_id"])
            connection.execute(
                "DELETE FROM communication_allowance_ledger WHERE action_id = ?",
                (action_id,),
            )
            changed += connection.execute(
                "DELETE FROM communication_actions WHERE user_id = ? AND action_id = ?",
                (user_id, action_id),
            ).rowcount
        return changed


AUTHORITATIVE_STATE_TOOL_NAME = "thine_run_inspect_authoritative_state"
WORKING_MEMORY_HISTORY_TOOL_NAME = "thine_working_memory_inspect_history"
QUARANTINE_INSPECT_TOOL_NAME = "thine_run_inspect_quarantines"
DEBUG_INVOCATIONS_TOOL_NAME = "thine_run_inspect_recent_debug"
MAINTENANCE_TOOLSET = "local-thine-transcripts"


class AuthoritativeReadToolBinding:
    """Register only read helpers; reset and cleanup remain operator-only."""

    def __init__(self, reader: AuthoritativeStateReader, *, user_id: str) -> None:
        self._reader = reader
        self._user_id = user_id

    def register(self, *, registry_instance: Any | None = None) -> None:
        from tools.registry import registry

        active = registry_instance or registry
        scope = active.current_scope_key()
        for name, description, parameters, handler in (
            (
                AUTHORITATIVE_STATE_TOOL_NAME,
                "Read current authoritative Hermes queue, Working Memory, Home, schedule, topic, permission, and cursor state. This is not a dashboard projection.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                self.inspect_state,
            ),
            (
                WORKING_MEMORY_HISTORY_TOOL_NAME,
                "Read up to 50 immutable Working Memory debug versions. Versions are inspect-only and cannot be restored.",
                {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50}
                    },
                    "additionalProperties": False,
                },
                self.inspect_memory,
            ),
            (
                QUARANTINE_INSPECT_TOOL_NAME,
                "Read immutable transcript, speaker, and interaction quarantines plus explicit retry provenance. This cannot delete or rewind a source cursor.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                self.inspect_quarantines,
            ),
            (
                DEBUG_INVOCATIONS_TOOL_NAME,
                "Read up to 50 recent redacted Hermes invocation records. Final output and raw tool payloads are omitted.",
                {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50}
                    },
                    "additionalProperties": False,
                },
                self.inspect_debug,
            ),
        ):
            active.register(
                name=name,
                toolset=MAINTENANCE_TOOLSET,
                schema={
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
                handler=handler,
                scope=scope,
            )

    def inspect_state(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if args:
            return _canonical({"ok": False, "error_code": "unexpected_arguments"})
        return _canonical({"ok": True, "state": self._reader.snapshot(self._user_id)})

    def inspect_memory(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if set(args) - {"limit"}:
            return _canonical({"ok": False, "error_code": "unexpected_arguments"})
        try:
            limit = _bounded_limit(args.get("limit", HISTORY_LIMIT))
        except ValueError as exc:
            return _canonical({"ok": False, "error_code": str(exc)})
        return _canonical({
            "ok": True,
            "restore_available": False,
            "versions": self._reader.working_memory_history(self._user_id, limit=limit),
        })

    def inspect_quarantines(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if args:
            return _canonical({"ok": False, "error_code": "unexpected_arguments"})
        return _canonical({
            "ok": True,
            "quarantines": self._reader.quarantines(self._user_id),
        })

    def inspect_debug(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if set(args) - {"limit"}:
            return _canonical({"ok": False, "error_code": "unexpected_arguments"})
        try:
            limit = _bounded_limit(args.get("limit", HISTORY_LIMIT))
        except ValueError as exc:
            return _canonical({"ok": False, "error_code": str(exc)})
        return _canonical({
            "ok": True,
            "redacted": True,
            "invocations": self._reader.debug_invocations(self._user_id, limit=limit),
        })


def _bounded_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 50:
        raise ValueError("limit_must_be_between_1_and_50")
    return value


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _plan_from_dict(value: dict[str, Any]) -> ResetPlan:
    return ResetPlan(
        reset_id=str(value["reset_id"]),
        user_id=str(value["user_id"]),
        scope=cast(ResetScope, value["scope"]),
        confirmation=str(value["confirmation"]),
        preference_revision=int(value["preference_revision"]),
        explicit_preferences_to_delete=tuple(
            cast(list[dict[str, object]], value["explicit_preferences_to_delete"])
        ),
        targets={str(key): int(item) for key, item in value["targets"].items()},
        preserved=tuple(str(item) for item in value["preserved"]),
        state_fingerprint=str(value["state_fingerprint"]),
        created_at_ms=int(value["created_at_ms"]),
    )


__all__ = [
    "AUTHORITATIVE_STATE_TOOL_NAME",
    "AuthoritativeReadToolBinding",
    "AuthoritativeStateReader",
    "DEBUG_INVOCATIONS_TOOL_NAME",
    "HISTORY_LIMIT",
    "QUARANTINE_INSPECT_TOOL_NAME",
    "ResetPlan",
    "ResetScope",
    "RetentionResetService",
    "RetentionResult",
    "WORKING_MEMORY_HISTORY_TOOL_NAME",
]
