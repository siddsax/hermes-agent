from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thine_harness.dashboard_read_models import DashboardReadModelService
from thine_harness.contracts.runtime import Tick
from thine_harness.home_state import HomeStateProjector
from thine_harness.maintenance import (
    AuthoritativeReadToolBinding,
    AuthoritativeStateReader,
    RetentionResetService,
)
from thine_harness.maintenance_cli import main as maintenance_main
from thine_harness.run_state import DurableRunState, SCHEMA_VERSION


USER = "daily-user"
NOW = 1_900_000_000_000
DAY_MS = 24 * 60 * 60 * 1000


def _tick(tick_id: str) -> Tick:
    return Tick.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "tick_id": tick_id,
        "user_id": USER,
        "logical_run_id": f"run:{tick_id}",
        "kind": "p1_transcript",
        "priority": "p1",
        "occurred_at_ms": 1,
        "received_at_ms": 1,
        "queued_at_ms": 1,
        "source_ref": {"kind": "transcript_availability", "id": tick_id},
        "causation_id": None,
        "correlation_id": tick_id,
        "attempt_ordinal": 1,
        "lease": None,
        "communication_allowance_snapshot": None,
        "payload": {
            "payload_kind": "transcript_availability",
            "reference_id": tick_id,
        },
        "extensions": {},
    })


def _seed_queue(state: DurableRunState, tick_id: str, *, state_name: str) -> str:
    tick = _tick(tick_id)
    logical_run_id = f"run:{tick_id}"
    state.enqueue(tick, now_ms=1)
    with state._transaction() as connection:
        connection.execute(
            """
            UPDATE queue_items SET state = ?, completed_at_ms = ?, updated_at_ms = ?
            WHERE logical_run_id = ?
            """,
            (
                state_name,
                1 if state_name in {"completed", "failed_terminal"} else None,
                1,
                logical_run_id,
            ),
        )
    return logical_run_id


def _seed_preference(
    state: DurableRunState,
    *,
    key: str = "notifications_enabled",
    value: bool = False,
    revision: int = 1,
) -> None:
    with state._transaction() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO explicit_preferences (
                user_id, preference_key, preference_value,
                authorizing_message_id, updated_at_ms, revision
            ) VALUES (?, ?, ?, 'message-authorized', ?, ?)
            """,
            (USER, key, int(value), NOW, revision),
        )


def _seed_quarantine(state: DurableRunState, quarantine_id: str) -> None:
    logical_run_id = _seed_queue(state, quarantine_id, state_name="quarantined")
    with state._transaction() as connection:
        connection.execute(
            """
            INSERT INTO quarantines (
                quarantine_id, user_id, logical_run_id, tick_id, source_kind,
                source_id, attempt_ordinal, failure_code, quarantined_at_ms
            ) VALUES (?, ?, ?, ?, 'transcript', ?, 3, 'provider_fault', ?)
            """,
            (
                quarantine_id,
                USER,
                logical_run_id,
                quarantine_id,
                quarantine_id,
                NOW - 365 * DAY_MS,
            ),
        )


def _seed_speaker_retry(
    state: DurableRunState, *, quarantine_id: str, retry_id: str
) -> None:
    with state._transaction() as connection:
        original_run_id = str(
            connection.execute(
                "SELECT logical_run_id FROM quarantines WHERE quarantine_id = ?",
                (quarantine_id,),
            ).fetchone()[0]
        )
        connection.execute(
            """
            INSERT INTO speaker_mapping_inputs (
                event_id, user_id, cursor, original_logical_run_id, event_json,
                state, quarantine_id, quarantine_record_json,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, 1, ?, '{}', 'quarantined', ?, '{}', ?, ?)
            """,
            (
                f"event:{quarantine_id}",
                USER,
                original_run_id,
                quarantine_id,
                NOW,
                NOW,
            ),
        )
        retry_run_id = _seed_queue_in_connection(
            connection, retry_id, state_name="completed"
        )
        connection.execute(
            """
            INSERT INTO speaker_explicit_retries (
                retry_run_id, user_id, quarantine_id, event_id,
                explicit_retry_json, state, created_at_ms, completed_at_ms
            ) VALUES (?, ?, ?, ?, '{}', 'completed', ?, ?)
            """,
            (
                retry_run_id,
                USER,
                quarantine_id,
                f"event:{quarantine_id}",
                NOW,
                NOW,
            ),
        )


def test_v10_database_migrates_without_changing_existing_state(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    state = DurableRunState(database)
    _seed_queue(state, "preserved", state_name="completed")
    _seed_preference(state)
    with state._connect() as connection:
        connection.executescript(
            """
            DROP TABLE maintenance_events;
            DROP TABLE maintenance_plans;
            PRAGMA user_version = 10;
            """
        )

    upgraded = DurableRunState(database)

    with upgraded._connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert (
            connection.execute(
                "SELECT state FROM queue_items WHERE logical_run_id = 'run:preserved'"
            ).fetchone()[0]
            == "completed"
        )
        assert (
            connection.execute(
                """
            SELECT preference_value FROM explicit_preferences
            WHERE user_id = ? AND preference_key = 'notifications_enabled'
            """,
                (USER,),
            ).fetchone()[0]
            == 0
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_operator_snapshot_returns_bounded_recent_maintenance_owner_state(
    tmp_path: Path,
) -> None:
    state = DurableRunState(tmp_path / "state.sqlite3")
    home = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: NOW)
    service = RetentionResetService(state, home=home, clock_ms=lambda: NOW)
    older = service.plan_reset(USER, "working_memory_topics")
    newer = service.plan_reset(USER, "home_state")
    _seed_queue(state, "live", state_name="running")
    with state._transaction() as connection:
        connection.execute(
            "UPDATE maintenance_plans SET created_at_ms = ? WHERE reset_id = ?",
            (NOW - 20, older.reset_id),
        )
        connection.execute(
            "UPDATE maintenance_plans SET created_at_ms = ? WHERE reset_id = ?",
            (NOW - 10, newer.reset_id),
        )
        connection.execute(
            """
            INSERT INTO maintenance_events (
                event_id, user_id, event_kind, subject_id,
                details_json, recorded_at_ms
            ) VALUES
                ('event-old', ?, 'retention_cleanup', 'cleanup-old', '{}', ?),
                ('event-new', ?, 'retention_cleanup', 'cleanup-new',
                 '{"changed":{"tool_receipts":2}}', ?)
            """,
            (USER, NOW - 30, USER, NOW - 5),
        )

    snapshot = service.operator_snapshot(USER, limit=1)

    assert snapshot["authoritative"] is True
    assert snapshot["read_at_ms"] == NOW
    assert snapshot["owner_observed_at_ms"] == NOW - 5
    assert snapshot["live_work_count"] == 1
    assert snapshot["policy"] == service.retention_policy()
    assert snapshot["events"] == [
        {
            "event_id": "event-new",
            "event_kind": "retention_cleanup",
            "subject_id": "cleanup-new",
            "details": {"changed": {"tool_receipts": 2}},
            "recorded_at_ms": NOW - 5,
        }
    ]
    assert len(snapshot["reset_plans"]) == 1
    assert snapshot["reset_plans"][0]["reset_id"] == newer.reset_id
    assert snapshot["reset_plans"][0]["scope"] == "home_state"
    assert snapshot["reset_plans"][0]["status"] == "planned"
    assert snapshot["reset_plans"][0]["plan"]["targets"] == newer.targets


def test_authoritative_quarantine_helper_bounds_each_recent_owner_collection(
    tmp_path: Path,
) -> None:
    state = DurableRunState(tmp_path / "state.sqlite3")
    home = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: NOW)
    for index in range(3):
        quarantine_id = f"quarantine-{index}"
        _seed_quarantine(state, quarantine_id)
        with state._transaction() as connection:
            connection.execute(
                "UPDATE quarantines SET quarantined_at_ms = ? WHERE quarantine_id = ?",
                (NOW + index, quarantine_id),
            )

    snapshot = AuthoritativeStateReader(state, home=home).quarantines(USER, limit=2)

    assert snapshot["retention_limit"] == 2
    assert [item["quarantine_id"] for item in snapshot["generic"]] == [
        "quarantine-2",
        "quarantine-1",
    ]


def test_time_frozen_retention_matrix_prunes_only_expired_derived_state(
    tmp_path: Path,
) -> None:
    state = DurableRunState(tmp_path / "state.sqlite3")
    home = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: NOW)
    old_run = _seed_queue(state, "old", state_name="completed")
    fresh_run = _seed_queue(state, "fresh", state_name="completed")
    _seed_preference(state)
    _seed_quarantine(state, "quarantine-old")
    _seed_speaker_retry(
        state, quarantine_id="quarantine-old", retry_id="quarantine-old-retry"
    )
    with state._transaction() as connection:
        for version in range(56):
            connection.execute(
                """
                INSERT INTO working_memory_versions (
                    user_id, version, markdown, configured_model_token_count,
                    tokenizer_status, logical_run_id, committed_at_ms
                ) VALUES (?, ?, ?, 1, 'exact', NULL, ?)
                """,
                (USER, version, f"memory {version}", version),
            )
        connection.execute(
            """
            INSERT INTO working_memory_state (
                user_id, version, markdown, token_count, last_run_id
            ) VALUES (?, 55, 'memory 55', 1, ?)
            """,
            (USER, fresh_run),
        )
        for index in range(55):
            logical_run_id = _seed_queue_in_connection(
                connection, f"debug-{index}", state_name="completed"
            )
            connection.execute(
                """
                INSERT INTO agent_run_inspections (
                    user_id, logical_run_id, attempt_id, provider, model,
                    api_mode, reasoning_effort, decision_outcome, final_output,
                    tool_discoveries_json, usage_json, stop_hook_outcome,
                    stop_hook_cache_identity_json, memory_version,
                    memory_token_count, recorded_at_ms
                ) VALUES (?, ?, ?, 'openai-codex', 'gpt-5.6-sol', 'responses',
                          'medium', 'no_action', '', '[]', '{}', 'unchanged',
                          '{}', 55, 1, ?)
                """,
                (USER, logical_run_id, f"attempt:{index}", index),
            )
        connection.execute(
            """
            INSERT INTO checkpoints (
                checkpoint_id, user_id, logical_run_id, cause, remaining_work,
                completed_receipt_ids_json, updated_at_ms, original_input,
                context_messages_json, completed_tool_results_json,
                successful_action_receipts_json,
                partial_visible_assistant_output
            ) VALUES ('checkpoint-old', ?, ?, 'continuation', 'done', '[]', ?,
                      '', '[]', '[]', '[]', '')
            """,
            (USER, old_run, NOW - 31 * DAY_MS),
        )
        connection.execute(
            """
            INSERT INTO attempts (
                attempt_id, user_id, logical_run_id, ordinal, status,
                failure_code, started_at_ms, finished_at_ms
            ) VALUES ('attempt-old', ?, ?, 1, 'succeeded', NULL, ?, ?)
            """,
            (USER, old_run, NOW - 31 * DAY_MS, NOW - 31 * DAY_MS),
        )
        for logical_run_id, updated_at, batch_id in (
            (old_run, NOW - 8 * DAY_MS, "batch-old"),
            (fresh_run, NOW - 6 * DAY_MS, "batch-fresh"),
        ):
            connection.execute(
                """
                INSERT INTO interaction_claims (
                    user_id, logical_run_id, tick_id, claim_request_id,
                    boundary_start_ms, boundary_end_ms, batch_id, batch_json,
                    state, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, 1, 2, ?, '{"events":["redact-me"]}',
                          'completed', ?, ?)
                """,
                (
                    USER,
                    logical_run_id,
                    f"interaction:{batch_id}",
                    f"claim:{batch_id}",
                    batch_id,
                    updated_at,
                    updated_at,
                ),
            )
        connection.execute(
            """
            INSERT INTO durable_topics (
                topic_id, user_id, topic_key, state, evidence_refs_json,
                receipt_refs_json, do_not_ask, updated_at_ms
            ) VALUES ('topic-old', ?, 'old_resolved', 'resolved', '[]', '[]', 0, ?)
            """,
            (USER, NOW - 91 * DAY_MS),
        )
        connection.execute(
            """
            INSERT INTO durable_topics (
                topic_id, user_id, topic_key, state, evidence_refs_json,
                receipt_refs_json, do_not_ask, updated_at_ms
            ) VALUES ('topic-live', ?, 'live_topic', 'asked', '[]', '[]', 0, ?)
            """,
            (USER, NOW - 365 * DAY_MS),
        )
        for receipt_id, logical_run_id, acknowledged_at_ms in (
            ("receipt-old", old_run, NOW - 91 * DAY_MS),
            ("receipt-fresh", fresh_run, NOW - 89 * DAY_MS),
            (
                "receipt-quarantine",
                "run:quarantine-old",
                NOW - 365 * DAY_MS,
            ),
        ):
            connection.execute(
                """
                INSERT INTO tool_receipts (
                    receipt_id, user_id, logical_run_id, action_id,
                    intent_fingerprint, provider_reference, result_json,
                    acknowledged_at_ms
                ) VALUES (?, ?, ?, ?, ?, 'provider', '{}', ?)
                """,
                (
                    receipt_id,
                    USER,
                    logical_run_id,
                    f"action:{receipt_id}",
                    f"intent:{receipt_id}",
                    acknowledged_at_ms,
                ),
            )
        for receipt_id, logical_run_id, created_at_ms in (
            ("topic-receipt-old", old_run, NOW - 91 * DAY_MS),
            ("topic-receipt-fresh", fresh_run, NOW - 89 * DAY_MS),
        ):
            connection.execute(
                """
                INSERT INTO topic_preference_receipts (
                    receipt_id, user_id, logical_run_id, operation,
                    subject_key, intent_fingerprint, result_json, created_at_ms
                ) VALUES (?, ?, ?, 'record_topic_action', 'topic', ?, '{}', ?)
                """,
                (
                    receipt_id,
                    USER,
                    logical_run_id,
                    f"intent:{receipt_id}",
                    created_at_ms,
                ),
            )

    result = RetentionResetService(state, home=home, clock_ms=lambda: NOW).cleanup(USER)

    with state._connect() as connection:
        assert (
            connection.execute(
                "SELECT batch_json FROM interaction_claims WHERE batch_id = 'batch-old'"
            ).fetchone()[0]
            is None
        )
        assert (
            connection.execute(
                "SELECT batch_json FROM interaction_claims WHERE batch_id = 'batch-fresh'"
            ).fetchone()[0]
            is not None
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM working_memory_versions WHERE user_id = ?",
                (USER,),
            ).fetchone()[0]
            == 50
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM agent_run_inspections WHERE user_id = ?", (USER,)
            ).fetchone()[0]
            == 50
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE checkpoint_id = 'checkpoint-old'"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE attempt_id = 'attempt-old'"
            ).fetchone()[0]
            == 0
        )
        assert [
            str(row["topic_key"])
            for row in connection.execute(
                "SELECT topic_key FROM durable_topics WHERE user_id = ?", (USER,)
            ).fetchall()
        ] == ["live_topic"]
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM explicit_preferences WHERE user_id = ?", (USER,)
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM quarantines WHERE user_id = ?", (USER,)
            ).fetchone()[0]
            == 1
        )
        assert [
            str(row["receipt_id"])
            for row in connection.execute(
                """
                SELECT receipt_id FROM tool_receipts
                WHERE user_id = ? ORDER BY receipt_id
                """,
                (USER,),
            ).fetchall()
        ] == ["receipt-fresh", "receipt-quarantine"]
        assert [
            str(row["receipt_id"])
            for row in connection.execute(
                """
                SELECT receipt_id FROM topic_preference_receipts
                WHERE user_id = ? ORDER BY receipt_id
                """,
                (USER,),
            ).fetchall()
        ] == ["topic-receipt-fresh"]
    assert result.preserved_quarantines == 1
    assert result.preserved_explicit_retries == 1
    assert result.changed["working_memory_versions"] == 6
    assert result.changed["debug_invocations"] == 5
    assert result.changed["tool_receipts"] == 1
    assert result.changed["topic_preference_receipts"] == 1


def test_scoped_reset_preserves_preferences_corrections_audit_and_quarantine(
    tmp_path: Path,
) -> None:
    state = DurableRunState(tmp_path / "state.sqlite3")
    home = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: NOW)
    service = RetentionResetService(state, home=home, clock_ms=lambda: NOW)
    _seed_preference(state)
    _seed_quarantine(state, "quarantine-preserved")
    _seed_speaker_retry(
        state,
        quarantine_id="quarantine-preserved",
        retry_id="quarantine-preserved-retry",
    )
    with state._transaction() as connection:
        connection.execute(
            """
            INSERT INTO explicit_corrections (
                user_id, correction_key, corrected_value,
                authorizing_message_id, updated_at_ms
            ) VALUES (?, 'name', 'Sid', 'message-correction', ?)
            """,
            (USER, NOW),
        )
        connection.execute(
            """
            INSERT INTO durable_topics (
                topic_id, user_id, topic_key, state, evidence_refs_json,
                receipt_refs_json, authorizing_message_id, do_not_ask,
                updated_at_ms
            ) VALUES ('topic-do-not-ask', ?, 'never_repeat', 'resolved',
                      '[]', '[]', 'message-do-not-ask', 1, ?)
            """,
            (USER, NOW),
        )
        connection.execute(
            """
            INSERT INTO working_memory_state (
                user_id, version, markdown, token_count, last_run_id
            ) VALUES (?, 1, 'recent work', 2, NULL)
            """,
            (USER,),
        )
        connection.execute(
            """
            INSERT INTO working_memory_versions (
                user_id, version, markdown, configured_model_token_count,
                tokenizer_status, logical_run_id, committed_at_ms
            ) VALUES (?, 1, 'recent work', 2, 'exact', NULL, ?)
            """,
            (USER, NOW),
        )
        connection.execute(
            """
            INSERT INTO durable_topics (
                topic_id, user_id, topic_key, state, evidence_refs_json,
                receipt_refs_json, do_not_ask, updated_at_ms
            ) VALUES ('topic', ?, 'follow_up', 'asked', '[]', '[]', 0, ?)
            """,
            (USER, NOW),
        )

    plan = service.plan_reset(USER, "working_memory_topics")
    assert plan.explicit_preferences_to_delete == ()
    refused = service.execute_reset(
        reset_id=plan.reset_id,
        confirmation="wrong",
        harness_stopped=True,
    )
    assert refused.payload.status == "confirmation_required"

    completed = service.execute_reset(
        reset_id=plan.reset_id,
        confirmation=plan.confirmation,
        harness_stopped=True,
    )

    assert completed.payload.status == "completed"
    with state._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM working_memory_state WHERE user_id = ?", (USER,)
            ).fetchone()[0]
            == 0
        )
        assert [
            str(row["topic_key"])
            for row in connection.execute(
                "SELECT topic_key FROM durable_topics WHERE user_id = ?",
                (USER,),
            ).fetchall()
        ] == ["never_repeat"]
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM explicit_preferences WHERE user_id = ?", (USER,)
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM explicit_corrections WHERE user_id = ?", (USER,)
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM quarantines WHERE user_id = ?", (USER,)
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM speaker_explicit_retries WHERE user_id = ?",
                (USER,),
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM maintenance_events WHERE user_id = ?", (USER,)
            ).fetchone()[0]
            == 1
        )


def test_full_reset_is_revision_bound_enumerates_preferences_and_recreates_home(
    tmp_path: Path,
) -> None:
    state = DurableRunState(tmp_path / "state.sqlite3")
    home = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: NOW)
    service = RetentionResetService(state, home=home, clock_ms=lambda: NOW)
    _seed_preference(state, revision=1)
    _seed_preference(
        state,
        key="speaker_tag_nudges_enabled",
        value=False,
        revision=2,
    )
    _seed_quarantine(state, "quarantine-full")
    current = home.current(USER)
    home.replace_current(
        user_id=USER,
        expected_revision=current.payload.revision,
        nodes=[
            {
                "node_id": "greeting",
                "component_id": "home.hero.greeting",
                "visible": True,
                "content": {"title": "Hello", "body": "Before reset", "count": None},
            }
        ],
        reason="Seed non-default Home.",
        originating_run_id="run-home",
        source_tick_id="tick-home",
        author="hermes_agent",
        action_id="action-home",
    )
    stale = service.plan_reset(USER, "all_hermes_state")
    assert stale.explicit_preferences_to_delete == (
        {"key": "notifications_enabled", "current_value": False},
        {"key": "speaker_tag_nudges_enabled", "current_value": False},
    )
    _seed_preference(state, value=True, revision=3)

    stale_result = service.execute_reset(
        reset_id=stale.reset_id,
        confirmation=stale.confirmation,
        harness_stopped=True,
    )
    assert stale_result.payload.status == "confirmation_required"
    assert service.plan_reset(USER, "all_hermes_state").preference_revision == 3
    plan = service.plan_reset(USER, "all_hermes_state")

    completed = service.execute_reset(
        reset_id=plan.reset_id,
        confirmation=plan.confirmation,
        harness_stopped=True,
    )

    assert completed.payload.status == "completed"
    assert completed.payload.default_home_recreated is True
    assert home.current(USER).payload.nodes == ()
    with state._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM explicit_preferences WHERE user_id = ?", (USER,)
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM quarantines WHERE user_id = ?", (USER,)
            ).fetchone()[0]
            == 1
        )
    replay = service.execute_reset(
        reset_id=plan.reset_id,
        confirmation=plan.confirmation,
        harness_stopped=True,
    )
    assert replay.to_dict() == completed.to_dict()


def test_reset_refuses_a_live_lease_and_authoritative_helpers_are_not_projections(
    tmp_path: Path,
) -> None:
    state = DurableRunState(tmp_path / "state.sqlite3")
    home = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: NOW)
    state.enqueue(_tick("live"), now_ms=NOW)
    assert state.lease_next(USER, owner="worker", now_ms=NOW) is not None
    service = RetentionResetService(state, home=home, clock_ms=lambda: NOW)
    plan = service.plan_reset(USER, "queues_schedules_receipts")

    result = service.execute_reset(
        reset_id=plan.reset_id,
        confirmation=plan.confirmation,
        harness_stopped=True,
    )

    assert result.payload.status == "refused_live_work"
    reader = AuthoritativeStateReader(state, home=home, clock_ms=lambda: NOW)
    binding = AuthoritativeReadToolBinding(reader, user_id=USER)
    helper = json.loads(binding.inspect_state({}))
    memory = json.loads(binding.inspect_memory({"limit": 50}))
    debug = json.loads(binding.inspect_debug({"limit": 50}))
    dashboard = DashboardReadModelService(reader, clock_ms=lambda: NOW).snapshot(USER)
    assert helper["state"]["authoritative"] is True
    assert memory["restore_available"] is False
    assert debug["redacted"] is True
    assert dashboard.payload.authoritative is False
    assert dashboard.payload.binding == "mac_loopback_only"


def test_queued_operational_reset_cancels_work_but_preserves_receipts(
    tmp_path: Path,
) -> None:
    state = DurableRunState(tmp_path / "state.sqlite3")
    home = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: NOW)
    logical_run_id = _seed_queue(state, "queued", state_name="queued")
    with state._transaction() as connection:
        connection.execute(
            """
            INSERT INTO one_shot_schedules (
                schedule_id, user_id, due_at_ms, timezone_name, due_time_input,
                created_at_ms, updated_at_ms, status, reason, creator_tick_id,
                originating_run_id, originating_action_id, intent_fingerprint
            ) VALUES ('schedule-reset', ?, ?, 'Asia/Kolkata',
                      '2031-01-01T12:00:00+05:30', ?, ?, 'active',
                      'Wake up after the requested pause.', 'queued', ?,
                      'action:schedule', 'intent:schedule')
            """,
            (USER, NOW + DAY_MS, NOW, NOW, logical_run_id),
        )
        connection.execute(
            """
            INSERT INTO checkpoints (
                checkpoint_id, user_id, logical_run_id, cause, remaining_work,
                completed_receipt_ids_json, updated_at_ms, original_input,
                context_messages_json, completed_tool_results_json,
                successful_action_receipts_json,
                partial_visible_assistant_output
            ) VALUES ('checkpoint-queued', ?, ?, 'continuation', 'resume',
                      '["receipt-preserved"]', ?, '', '[]', '[]', '[]', '')
            """,
            (USER, logical_run_id, NOW),
        )
        connection.execute(
            """
            INSERT INTO tool_receipts (
                receipt_id, user_id, logical_run_id, action_id,
                intent_fingerprint, provider_reference, result_json,
                acknowledged_at_ms
            ) VALUES ('receipt-preserved', ?, ?, 'action:preserved',
                      'intent:preserved', 'provider', '{}', ?)
            """,
            (USER, logical_run_id, NOW),
        )
    service = RetentionResetService(state, home=home, clock_ms=lambda: NOW)
    plan = service.plan_reset(USER, "queues_schedules_receipts")

    result = service.execute_reset(
        reset_id=plan.reset_id,
        confirmation=plan.confirmation,
        harness_stopped=True,
    )

    assert result.payload.status == "completed"
    with state._connect() as connection:
        assert (
            connection.execute(
                "SELECT state FROM queue_items WHERE logical_run_id = ?",
                (logical_run_id,),
            ).fetchone()[0]
            == "failed_terminal"
        )
        assert (
            connection.execute(
                "SELECT status FROM one_shot_schedules WHERE schedule_id = ?",
                ("schedule-reset",),
            ).fetchone()[0]
            == "cancelled"
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE logical_run_id = ?",
                (logical_run_id,),
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM tool_receipts WHERE receipt_id = ?",
                ("receipt-preserved",),
            ).fetchone()[0]
            == 1
        )


def test_executing_home_reset_resumes_without_duplicate_revision(
    tmp_path: Path,
) -> None:
    state = DurableRunState(tmp_path / "state.sqlite3")
    home = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: NOW)
    service = RetentionResetService(state, home=home, clock_ms=lambda: NOW)
    plan = service.plan_reset(USER, "home_state")
    with state._transaction() as connection:
        connection.execute(
            "UPDATE maintenance_plans SET status = 'executing' WHERE reset_id = ?",
            (plan.reset_id,),
        )
    first = home.reset_to_default(user_id=USER, reset_id=plan.reset_id)

    completed = service.execute_reset(
        reset_id=plan.reset_id,
        confirmation=plan.confirmation,
        harness_stopped=True,
    )

    assert completed.payload.status == "completed"
    assert completed.payload.default_home_recreated is True
    assert home.current(USER).payload.revision == first.payload.revision
    assert home.maintenance_snapshot(USER)["action_receipt_rows"] == 2


def test_operator_cli_exposes_read_and_plan_boundaries(
    tmp_path: Path, capsys: Any
) -> None:
    state_db = tmp_path / "state.sqlite3"
    home_db = tmp_path / "home.sqlite3"
    assert (
        maintenance_main([
            "--user-id",
            USER,
            "--state-db",
            str(state_db),
            "--home-db",
            str(home_db),
            "retention-policy",
        ])
        == 0
    )
    policy = json.loads(capsys.readouterr().out)
    assert policy["interaction_payload_days"] == 7
    assert policy["quarantines_and_retry_provenance"] == "indefinite"
    assert (
        maintenance_main([
            "--user-id",
            USER,
            "--state-db",
            str(state_db),
            "--home-db",
            str(home_db),
            "reset-plan",
            "--scope",
            "all_hermes_state",
        ])
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["confirmation"] == f"CONFIRM {plan['reset_id']}"
    assert plan["preserved"] == [
        "canonical_transcripts",
        "speaker_mappings_and_relationships",
        "aggregation_buffer",
        "quarantine",
    ]


def _seed_queue_in_connection(connection: Any, tick_id: str, *, state_name: str) -> str:
    logical_run_id = f"run:{tick_id}"
    connection.execute(
        """
        INSERT INTO queue_items (
            tick_id, user_id, logical_run_id, kind, priority, priority_rank,
            source_kind, source_id, tick_json, state, completed_at_ms,
            updated_at_ms
        ) VALUES (?, ?, ?, 'p1_transcript', 'p1', 1,
                  'transcript_availability', ?, '{}', ?, 1, 1)
        """,
        (tick_id, USER, logical_run_id, tick_id, state_name),
    )
    return logical_run_id
