from __future__ import annotations

import json
from pathlib import Path

import pytest

from thine_harness.contracts.runtime import Tick
from thine_harness.deferred_tools import DeferredNamespaceCatalog
from thine_harness.run_coordinator import InvocationOutcome, RunCoordinator
from thine_harness.run_state import DurableRunState, ReceiptConflict, SCHEMA_VERSION
from thine_harness.schedules import (
    FakeScheduleNoActionRuntime,
    OneShotScheduleService,
    SCHEDULE_CREATE_TOOL_SCHEMA,
    SCHEDULE_CREATE_TOOL_NAME,
    SCHEDULE_CANCEL_TOOL_NAME,
    SCHEDULE_EDIT_TOOL_NAME,
    SCHEDULE_EDIT_TOOL_SCHEMA,
    SCHEDULE_INSPECT_TOOL_NAME,
    SCHEDULE_LIST_TOOL_NAME,
    SCHEDULE_RUN_NOW_TOOL_NAME,
    ScheduleInputPort,
    ScheduleRunFinalizer,
    ScheduleToolBinding,
    normalize_due_time,
)
from tools.registry import registry


def _tick(tick_id: str, *, kind: str = "p1_transcript", queued_at_ms: int = 1) -> Tick:
    priority = "p0" if kind == "p0_user_chat" else "p1"
    source_kind = (
        "user_message" if kind == "p0_user_chat" else "transcript_availability"
    )
    return Tick.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "tick_id": tick_id,
        "user_id": "daily-user",
        "logical_run_id": f"run:{tick_id}",
        "kind": kind,
        "priority": priority,
        "occurred_at_ms": queued_at_ms,
        "received_at_ms": queued_at_ms,
        "queued_at_ms": queued_at_ms,
        "source_ref": {"kind": source_kind, "id": tick_id},
        "causation_id": None,
        "correlation_id": f"correlation:{tick_id}",
        "attempt_ordinal": 1,
        "lease": None,
        "communication_allowance_snapshot": None,
        "payload": {"payload_kind": source_kind, "reference_id": tick_id},
        "extensions": {},
    })


class _NoopFeature:
    def apply(self, command):
        raise AssertionError(f"unexpected fake feature call: {command}")


def _active_origin(state: DurableRunState, tick_id: str, *, now_ms: int = 10):
    tick = _tick(tick_id, queued_at_ms=now_ms)
    state.enqueue(tick, now_ms=now_ms)
    leased = state.lease_next("daily-user", owner="test", now_ms=now_ms)
    assert leased is not None
    return tick, leased


def _complete_origin(
    state: DurableRunState, tick: Tick, leased, *, now_ms: int = 20
) -> None:
    state.mark_inference_started(
        user_id="daily-user",
        logical_run_id=str(tick.payload.logical_run_id),
        owner="test",
        attempt_id=leased.attempt_id,
        lease_token=leased.lease_token,
        now_ms=now_ms,
    )
    state.complete(
        user_id="daily-user",
        logical_run_id=str(tick.payload.logical_run_id),
        owner="test",
        attempt_id=leased.attempt_id,
        lease_token=leased.lease_token,
        now_ms=now_ms,
    )


def _create(
    service: OneShotScheduleService,
    *,
    tick_id: str,
    action_id: str,
    due_at: str = "2026-08-29T12:00:00+05:30",
    reason: str = "Follow up after the promised review window",
):
    return service.create(
        user_id="daily-user",
        creator_tick_id=tick_id,
        originating_run_id=f"run:{tick_id}",
        originating_action_id=action_id,
        due_at=due_at,
        timezone_name="Asia/Kolkata",
        reason=reason,
    )


def test_schema_upgrade_and_time_normalization_are_deterministic(tmp_path: Path):
    state = DurableRunState(tmp_path / "state.sqlite3")
    with state._connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    due_ms, normalized = normalize_due_time("2026-08-29T12:00:00+05:30", "Asia/Kolkata")

    assert due_ms == 1_787_985_000_000
    assert normalized == "2026-08-29T06:30:00Z"
    with pytest.raises(ValueError, match="offset does not match"):
        normalize_due_time("2026-08-29T12:00:00+00:00", "Asia/Kolkata")
    with pytest.raises(ValueError, match="explicit UTC offset"):
        normalize_due_time("2026-08-29T12:00:00", "Asia/Kolkata")


def test_schema_nine_database_upgrades_schedules_in_place(tmp_path: Path):
    database = tmp_path / "state.sqlite3"
    state = DurableRunState(database)
    with state._connect() as connection:
        connection.executescript(
            """
            DROP TABLE one_shot_schedule_mutations;
            DROP TABLE one_shot_schedules;
            PRAGMA user_version = 9;
            """
        )

    DurableRunState(database)

    with state._connect() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert {"one_shot_schedules", "one_shot_schedule_mutations"} <= tables


def test_schema_eight_database_upgrades_topics_and_schedules_without_data_loss(
    tmp_path: Path,
):
    database = tmp_path / "state.sqlite3"
    state = DurableRunState(database)
    state.enqueue(_tick("preserved-before-upgrade"), now_ms=1)
    with state._connect() as connection:
        connection.executescript(
            """
            DROP TABLE one_shot_schedule_mutations;
            DROP TABLE one_shot_schedules;
            DROP TABLE topic_preference_receipts;
            DROP TABLE explicit_corrections;
            DROP TABLE explicit_preferences;
            DROP TABLE durable_topics;
            PRAGMA user_version = 8;
            """
        )

    DurableRunState(database)

    with state._connect() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert (
            connection.execute(
                "SELECT tick_id FROM queue_items WHERE tick_id = ?",
                ("preserved-before-upgrade",),
            ).fetchone()[0]
            == "preserved-before-upgrade"
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert {
        "durable_topics",
        "explicit_preferences",
        "explicit_corrections",
        "topic_preference_receipts",
        "one_shot_schedules",
        "one_shot_schedule_mutations",
    } <= tables


def test_create_is_idempotent_and_conflicting_action_identity_fails(tmp_path: Path):
    state = DurableRunState(tmp_path / "state.sqlite3")
    tick, _leased = _active_origin(state, "creator")
    service = OneShotScheduleService(state, clock_ms=lambda: 100)

    first = _create(
        service, tick_id=str(tick.payload.tick_id), action_id="action:create"
    )
    replay = _create(
        service, tick_id=str(tick.payload.tick_id), action_id="action:create"
    )
    same_intent_new_action = _create(
        service,
        tick_id=str(tick.payload.tick_id),
        action_id="action:create-after-provider-retry",
    )

    assert replay == first
    assert same_intent_new_action == first
    assert first.status == "active"
    assert first.creator_tick_id == "creator"
    assert first.reason == "Follow up after the promised review window"
    assert first.contract().payload.reason == first.reason
    assert first.timezone_name == "Asia/Kolkata"
    with pytest.raises(ReceiptConflict):
        _create(
            service,
            tick_id=str(tick.payload.tick_id),
            action_id="action:create",
            reason="A different intent under the same action",
        )
    with pytest.raises(ValueError, match="1000"):
        _create(
            service,
            tick_id=str(tick.payload.tick_id),
            action_id="action:too-long",
            reason="x" * 1001,
        )


def test_typed_tools_create_edit_list_cancel_and_run_now(tmp_path: Path):
    now = [100]
    state = DurableRunState(tmp_path / "state.sqlite3")
    _active_origin(state, "tool-turn")
    service = OneShotScheduleService(state, clock_ms=lambda: now[0])
    binding = ScheduleToolBinding(state=state, service=service, user_id="daily-user")

    created = json.loads(
        binding.create({
            "due_at": "2026-08-29T12:00:00+05:30",
            "timezone": "Asia/Kolkata",
            "reason": "Check the local daily-driver after lunch",
            "request_key": "after-lunch-check",
        })
    )
    assert created["ok"] is True
    schedule_id = created["schedule"]["schedule_id"]

    edited = json.loads(
        binding.edit({
            "schedule_id": schedule_id,
            "due_at": "2026-08-29T12:30:00+05:30",
            "timezone": "Asia/Kolkata",
            "reason": "Check the local daily-driver after the next half hour",
            "request_key": "move-after-lunch-check",
        })
    )
    assert edited["ok"] is True
    assert edited["schedule"]["reason"].endswith("next half hour")
    assert (
        json.loads(binding.list({}))["schedules"][0]["schedule"]["schedule_id"]
        == schedule_id
    )

    run_now = json.loads(
        binding.run_now({
            "schedule_id": schedule_id,
            "request_key": "run-after-lunch-check-now",
        })
    )
    assert run_now["schedule"]["status"] == "enqueued"
    assert (
        json.loads(
            binding.run_now({
                "schedule_id": schedule_id,
                "request_key": "run-after-lunch-check-now",
            })
        )
        == run_now
    )
    cancelled_after_enqueue = json.loads(
        binding.cancel({
            "schedule_id": schedule_id,
            "request_key": "cancel-after-enqueue",
        })
    )
    assert cancelled_after_enqueue["ok"] is False

    second = json.loads(
        binding.create({
            "due_at": "2026-08-30T12:00:00+05:30",
            "timezone": "Asia/Kolkata",
            "reason": "Cancel this disposable one-shot",
            "request_key": "disposable",
        })
    )
    cancelled = json.loads(
        binding.cancel({
            "schedule_id": second["schedule"]["schedule_id"],
            "request_key": "cancel-disposable",
        })
    )
    assert cancelled["schedule"]["status"] == "cancelled"


def test_due_scan_and_restart_insert_exactly_one_p2_tick(tmp_path: Path):
    database = tmp_path / "state.sqlite3"
    state = DurableRunState(database)
    tick, leased = _active_origin(state, "creator")
    service = OneShotScheduleService(state, clock_ms=lambda: 200)
    schedule = _create(
        service,
        tick_id=str(tick.payload.tick_id),
        action_id="action:create",
        due_at="1970-01-01T05:30:00+05:30",
    )
    _complete_origin(state, tick, leased)

    assert service.fire_due_once("daily-user", now_ms=200) == schedule.schedule_id
    restarted = OneShotScheduleService(DurableRunState(database), clock_ms=lambda: 300)
    assert restarted.fire_due_once("daily-user", now_ms=300) is None

    with state._connect() as connection:
        rows = connection.execute(
            "SELECT * FROM queue_items WHERE kind = 'p2_scheduled'"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["source_id"] == schedule.schedule_id
    assert rows[0]["priority"] == "p2"


def test_run_now_executes_once_through_global_coordinator_and_completes(tmp_path: Path):
    now = [100]
    state = DurableRunState(tmp_path / "state.sqlite3")
    tick, leased = _active_origin(state, "creator")
    service = OneShotScheduleService(state, clock_ms=lambda: now[0])
    schedule = _create(service, tick_id="creator", action_id="action:create")
    _complete_origin(state, tick, leased, now_ms=110)
    service.run_now(
        user_id="daily-user",
        schedule_id=schedule.schedule_id,
        action_id="action:run-now",
    )
    runtime = FakeScheduleNoActionRuntime()
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=_NoopFeature(),
        input_port=ScheduleInputPort(service),
        finalizer=ScheduleRunFinalizer(state, clock_ms=lambda: now[0]),
        clock_ms=lambda: now[0],
    )

    result = coordinator.run_next("daily-user")

    assert result is not None and result.status == "completed"
    assert len(runtime.invocations) == 1
    assert service.inspect("daily-user", schedule.schedule_id).status == "completed"
    assert coordinator.run_next("daily-user") is None


def test_p0_stays_ahead_of_a_due_schedule_in_the_same_global_queue(tmp_path: Path):
    state = DurableRunState(tmp_path / "state.sqlite3")
    tick, leased = _active_origin(state, "creator")
    service = OneShotScheduleService(state, clock_ms=lambda: 100)
    _create(
        service,
        tick_id="creator",
        action_id="action:create",
        due_at="1970-01-01T05:30:00+05:30",
    )
    _complete_origin(state, tick, leased, now_ms=110)
    assert service.fire_due_once("daily-user", now_ms=120) is not None
    state.enqueue(_tick("user-chat", kind="p0_user_chat", queued_at_ms=121), now_ms=121)

    next_run = state.lease_next("daily-user", owner="test", now_ms=122)

    assert next_run is not None
    assert next_run.tick.payload.tick_id == "user-chat"


def test_third_scheduled_inference_fault_is_terminal_and_never_runs_a_fourth_time(
    tmp_path: Path,
):
    now = [100]
    state = DurableRunState(tmp_path / "state.sqlite3")
    tick, leased = _active_origin(state, "creator")
    service = OneShotScheduleService(state, clock_ms=lambda: now[0])
    schedule = _create(service, tick_id="creator", action_id="action:create")
    _complete_origin(state, tick, leased, now_ms=110)
    service.run_now(
        user_id="daily-user",
        schedule_id=schedule.schedule_id,
        action_id="action:run-now",
    )
    runtime = FakeScheduleNoActionRuntime([
        InvocationOutcome.fault("provider_fault"),
        InvocationOutcome.fault("provider_fault"),
        InvocationOutcome.fault("provider_fault"),
    ])
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=_NoopFeature(),
        input_port=ScheduleInputPort(service),
        finalizer=ScheduleRunFinalizer(state, clock_ms=lambda: now[0]),
        clock_ms=lambda: now[0],
    )

    outcomes = []
    for attempt in range(3):
        now[0] += 1
        result = coordinator.run_next("daily-user")
        assert result is not None
        outcomes.append(result.status)

    assert outcomes == ["failed_retryable", "failed_retryable", "failed_terminal"]
    assert len(runtime.invocations) == 3
    assert (
        service.inspect("daily-user", schedule.schedule_id).status == "failed_terminal"
    )
    assert coordinator.run_next("daily-user") is None


def test_only_oldest_ten_minute_overdue_schedule_promotes_to_p1_tail(tmp_path: Path):
    state = DurableRunState(tmp_path / "state.sqlite3")
    service = OneShotScheduleService(state, clock_ms=lambda: 0)
    schedules = []
    for ordinal in (1, 2):
        tick, leased = _active_origin(state, f"creator-{ordinal}", now_ms=ordinal)
        schedules.append(
            _create(
                service,
                tick_id=str(tick.payload.tick_id),
                action_id=f"action:create:{ordinal}",
                due_at="1970-01-01T05:30:00+05:30",
                reason=f"Overdue schedule {ordinal}",
            )
        )
        _complete_origin(state, tick, leased, now_ms=10 + ordinal)
        assert service.fire_due_once("daily-user", now_ms=20 + ordinal) is not None
    state.enqueue(_tick("existing-p1", queued_at_ms=30), now_ms=30)

    promoted = service.promote_oldest_overdue("daily-user", now_ms=600_100)
    second = service.promote_oldest_overdue("daily-user", now_ms=600_101)

    assert promoted == schedules[0].schedule_id
    assert second is None
    with state._connect() as connection:
        rows = connection.execute(
            """
            SELECT source_id, priority, enqueue_sequence FROM queue_items
            WHERE kind = 'p2_scheduled' ORDER BY enqueue_sequence
            """
        ).fetchall()
        existing_sequence = connection.execute(
            "SELECT enqueue_sequence FROM queue_items WHERE tick_id = 'existing-p1'"
        ).fetchone()[0]
    promoted_row = next(row for row in rows if row["source_id"] == promoted)
    other_row = next(row for row in rows if row["source_id"] != promoted)
    assert promoted_row["priority"] == "p1"
    assert promoted_row["enqueue_sequence"] > existing_sequence
    assert other_row["priority"] == "p2"


def test_tool_schemas_do_not_accept_recurring_semantics():
    create_properties = SCHEDULE_CREATE_TOOL_SCHEMA["parameters"]["properties"]
    edit_properties = SCHEDULE_EDIT_TOOL_SCHEMA["parameters"]["properties"]

    assert "cron" not in create_properties
    assert "recurrence" not in create_properties
    assert "interval" not in create_properties
    assert "cron" not in edit_properties


def test_schedule_helpers_are_deferred_and_searchable_in_the_schedules_namespace(
    tmp_path: Path,
):
    names = {
        SCHEDULE_CREATE_TOOL_NAME,
        SCHEDULE_LIST_TOOL_NAME,
        SCHEDULE_INSPECT_TOOL_NAME,
        SCHEDULE_EDIT_TOOL_NAME,
        SCHEDULE_CANCEL_TOOL_NAME,
        SCHEDULE_RUN_NOW_TOOL_NAME,
    }
    state = DurableRunState(tmp_path / "state.sqlite3")
    binding = ScheduleToolBinding(
        state=state,
        service=OneShotScheduleService(state),
        user_id="daily-user",
    )
    binding.register()
    try:
        definitions = registry.get_definitions(names, quiet=True)
        catalog = DeferredNamespaceCatalog(definitions, context_length=272_000)

        assert names.isdisjoint({
            item["function"]["name"] for item in catalog.model_tool_definitions()
        })
        matches = catalog.search("create one shot wakeup with a reason", limit=10)
        assert any(match["name"] == SCHEDULE_CREATE_TOOL_NAME for match in matches)
        assert all(match["namespace"] == "schedules" for match in matches)
    finally:
        for name in names:
            registry.deregister(name)
