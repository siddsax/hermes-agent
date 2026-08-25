from __future__ import annotations

import threading
import time
from pathlib import Path
import sqlite3

import pytest

from thine_harness.contracts.runtime import Tick
from thine_harness.run_coordinator import (
    FakeFeatureAcknowledgement,
    FakeFeatureCommand,
    HarnessRuntimeConfiguration,
    InvocationContext,
    InvocationOutcome,
    RunCoordinator,
    RunResult,
)
from thine_harness.run_state import (
    DurableRunState,
    DurableStateError,
    ReceiptConflict,
    default_database_path,
)
from thine_harness.runtime import RuntimeModelConfig


def _tick(
    tick_id: str,
    *,
    kind: str = "p1_transcript",
    queued_at_ms: int = 1,
    user_id: str = "daily-user",
) -> Tick:
    priority = {
        "p0_user_chat": "p0",
        "p1_transcript": "p1",
        "p1_speaker": "p1",
        "p1_interaction": "p1",
        "p2_scheduled": "p2",
    }[kind]
    source_kind = {
        "p0_user_chat": "user_message",
        "p1_transcript": "transcript_availability",
        "p1_speaker": "speaker_mapping",
        "p1_interaction": "interaction_window",
        "p2_scheduled": "schedule",
    }[kind]
    return Tick.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "tick_id": tick_id,
        "user_id": user_id,
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


def _require_result(coordinator: RunCoordinator) -> RunResult:
    result = coordinator.run_next("daily-user")
    assert result is not None
    return result


class _RecordingFeature:
    def __init__(self) -> None:
        self.commands: list[FakeFeatureCommand] = []

    def apply(self, command: FakeFeatureCommand) -> FakeFeatureAcknowledgement:
        self.commands.append(command)
        return FakeFeatureAcknowledgement(
            provider_reference=f"feature:{command.action_id}",
            result={"saved": command.payload["value"]},
        )


class _CheckpointAfterFeature:
    def __init__(self) -> None:
        self.contexts: list[InvocationContext] = []

    def invoke(self, context, *, tools, control):
        self.contexts.append(context)
        receipt = tools.execute_once(
            FakeFeatureCommand(
                action_id=f"{context.tick.payload.tick_id}:effect:1",
                intent_fingerprint="a" * 64,
                payload={"value": "stored-once"},
            )
        )
        return InvocationOutcome.checkpointed(
            remaining_work="finish after restart",
            completed_receipt_ids=(receipt.receipt_id,),
        )


class _ResumeAndComplete:
    def __init__(self) -> None:
        self.contexts: list[InvocationContext] = []

    def invoke(self, context, *, tools, control):
        self.contexts.append(context)
        assert context.checkpoint is not None
        assert context.checkpoint.remaining_work == "finish after restart"
        assert len(context.acknowledged_receipts) == 1
        replay = tools.execute_once(
            FakeFeatureCommand(
                action_id=f"{context.tick.payload.tick_id}:effect:1",
                intent_fingerprint="a" * 64,
                payload={"value": "stored-once"},
            )
        )
        assert replay.receipt_id == context.acknowledged_receipts[0].receipt_id
        return InvocationOutcome.completed()


def test_checkpoint_and_acknowledged_feature_receipt_survive_restart(tmp_path: Path):
    database = tmp_path / "harness.sqlite3"
    feature = _RecordingFeature()
    first = RunCoordinator(
        DurableRunState(database),
        runtime=_CheckpointAfterFeature(),
        feature_port=feature,
        clock_ms=lambda: 100,
    )
    first.enqueue(_tick("background"))

    checkpointed = first.run_next("daily-user")

    assert checkpointed is not None
    assert checkpointed.status == "checkpointed"
    restarted_runtime = _ResumeAndComplete()
    restarted = RunCoordinator(
        DurableRunState(database),
        runtime=restarted_runtime,
        feature_port=feature,
        clock_ms=lambda: 200,
    )

    completed = restarted.run_next("daily-user")

    assert completed is not None
    assert completed.status == "completed"
    assert len(feature.commands) == 1
    assert restarted_runtime.contexts[0].attempt_ordinal == 1
    diagnostics = restarted.diagnostics("daily-user")
    assert diagnostics.checkpoints[0].remaining_work == "finish after restart"
    assert diagnostics.receipts[0].provider_reference == "feature:background:effect:1"


class _CompleteInOrder:
    def __init__(self) -> None:
        self.order: list[str] = []

    def invoke(self, context, *, tools, control):
        self.order.append(context.tick.payload.tick_id)
        return InvocationOutcome.completed()


def test_priority_and_fifo_choose_p0_then_p1_then_p2(tmp_path: Path):
    runtime = _CompleteInOrder()
    coordinator = RunCoordinator(
        DurableRunState(tmp_path / "state.sqlite3"),
        runtime=runtime,
        feature_port=_RecordingFeature(),
        clock_ms=lambda: 50,
    )
    for tick in (
        _tick("p1-first", queued_at_ms=1),
        _tick("p2", kind="p2_scheduled", queued_at_ms=2),
        _tick("p1-second", kind="p1_speaker", queued_at_ms=3),
        _tick("p0-first", kind="p0_user_chat", queued_at_ms=4),
        _tick("p0-second", kind="p0_user_chat", queued_at_ms=5),
    ):
        coordinator.enqueue(tick)

    while coordinator.run_next("daily-user") is not None:
        pass

    assert runtime.order == ["p0-first", "p0-second", "p1-first", "p1-second", "p2"]


def test_duplicate_tick_enqueue_is_idempotent_across_json_member_order(tmp_path: Path):
    original = _tick("same-tick")
    reordered_payload = original.to_dict()
    reordered = Tick.from_dict({
        key: reordered_payload[key] for key in reversed(reordered_payload)
    })
    coordinator = RunCoordinator(
        DurableRunState(tmp_path / "state.sqlite3"),
        runtime=_CompleteInOrder(),
        feature_port=_RecordingFeature(),
        clock_ms=lambda: 10,
    )

    assert coordinator.enqueue(original) == "same-tick"
    assert coordinator.enqueue(reordered) == "same-tick"
    assert len(coordinator.diagnostics("daily-user").queue) == 1


class _BlockingRuntime:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.maximum_active = 0

    def invoke(self, context, *, tools, control):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.entered.set()
        assert self.release.wait(2)
        self.active -= 1
        return InvocationOutcome.completed()


def test_only_one_invocation_runs_for_a_user(tmp_path: Path):
    runtime = _BlockingRuntime()
    coordinator = RunCoordinator(
        DurableRunState(tmp_path / "state.sqlite3"),
        runtime=runtime,
        feature_port=_RecordingFeature(),
    )
    coordinator.enqueue(_tick("one"))
    coordinator.enqueue(_tick("two", queued_at_ms=2))
    first_results = []
    worker = threading.Thread(
        target=lambda: first_results.append(coordinator.run_next("daily-user"))
    )
    worker.start()
    assert runtime.entered.wait(2)

    assert coordinator.run_next("daily-user") is None
    runtime.release.set()
    worker.join(2)

    assert runtime.maximum_active == 1
    assert first_results[0] is not None
    assert coordinator.run_next("daily-user") is not None


def test_live_invocation_renews_its_lease_across_coordinator_instances(tmp_path: Path):
    database = tmp_path / "state.sqlite3"
    runtime = _BlockingRuntime()
    first = RunCoordinator(
        DurableRunState(database, lease_duration_ms=60),
        runtime=runtime,
        feature_port=_RecordingFeature(),
    )
    competitor = RunCoordinator(
        DurableRunState(database, lease_duration_ms=60),
        runtime=_CompleteInOrder(),
        feature_port=_RecordingFeature(),
        lease_owner="competing-harness",
    )
    first.enqueue(_tick("one"))
    worker = threading.Thread(target=lambda: first.run_next("daily-user"))
    worker.start()
    assert runtime.entered.wait(2)

    time.sleep(0.15)

    assert competitor.run_next("daily-user") is None
    runtime.release.set()
    worker.join(2)
    assert not worker.is_alive()


class _PreemptThenComplete:
    def __init__(self) -> None:
        self.background_started = threading.Event()
        self.background_invocations = 0

    def invoke(self, context, *, tools, control):
        if context.tick.payload.kind == "p0_user_chat":
            return InvocationOutcome.completed()
        self.background_invocations += 1
        if self.background_invocations == 1:
            self.background_started.set()
            assert control.wait_for_preemption(2)
            return InvocationOutcome.preempted(remaining_work="resume background")
        assert context.checkpoint is not None
        return InvocationOutcome.completed()


class _LeaseBarrierState(DurableRunState):
    """Pause after leasing so a P0 can land before active registration."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.background_leased = threading.Event()
        self.release_lease = threading.Event()

    def lease_next(self, user_id: str, *, owner: str, now_ms: int):
        leased = super().lease_next(user_id, owner=owner, now_ms=now_ms)
        if leased is not None and leased.tick.payload.kind != "p0_user_chat":
            self.background_leased.set()
            assert self.release_lease.wait(2)
        return leased


class _RequireEarlyPreemption:
    def invoke(self, context, *, tools, control):
        assert control.preemption_requested
        return InvocationOutcome.preempted(remaining_work="resume after p0")


def test_p0_arriving_between_lease_and_activation_still_preempts(tmp_path: Path):
    state = _LeaseBarrierState(tmp_path / "state.sqlite3")
    coordinator = RunCoordinator(
        state,
        runtime=_RequireEarlyPreemption(),
        feature_port=_RecordingFeature(),
        clock_ms=lambda: 100,
    )
    coordinator.enqueue(_tick("background"))
    results: list[RunResult | None] = []
    worker = threading.Thread(
        target=lambda: results.append(coordinator.run_next("daily-user"))
    )
    worker.start()
    assert state.background_leased.wait(2)

    coordinator.enqueue(_tick("chat", kind="p0_user_chat", queued_at_ms=2))
    state.release_lease.set()
    worker.join(2)

    assert results and results[0] is not None
    assert results[0].status == "checkpointed"


def test_p0_preemption_queues_and_resumes_without_consuming_attempt(tmp_path: Path):
    runtime = _PreemptThenComplete()
    coordinator = RunCoordinator(
        DurableRunState(tmp_path / "state.sqlite3"),
        runtime=runtime,
        feature_port=_RecordingFeature(),
        clock_ms=lambda: 100,
    )
    coordinator.enqueue(_tick("background"))
    result_holder = []
    worker = threading.Thread(
        target=lambda: result_holder.append(coordinator.run_next("daily-user"))
    )
    worker.start()
    assert runtime.background_started.wait(2)

    coordinator.enqueue(_tick("chat", kind="p0_user_chat", queued_at_ms=2))
    worker.join(2)

    assert result_holder[0].status == "checkpointed"
    assert _require_result(coordinator).tick_id == "chat"
    assert _require_result(coordinator).tick_id == "background"
    attempts = coordinator.diagnostics("daily-user").attempts
    background_attempts = [a for a in attempts if a.logical_run_id == "run:background"]
    assert [(a.ordinal, a.status) for a in background_attempts] == [(1, "succeeded")]


class _YieldContinuationThenComplete:
    def __init__(self) -> None:
        self.outcomes = [
            InvocationOutcome.cancelled(remaining_work="cancelled cooperatively"),
            InvocationOutcome.yielded(remaining_work="yielded safely"),
            InvocationOutcome.continuation(remaining_work="bounded continuation"),
            InvocationOutcome.completed(),
        ]

    def invoke(self, context, *, tools, control):
        return self.outcomes.pop(0)


def test_cancellation_yield_and_bounded_continuation_do_not_consume_attempt(
    tmp_path: Path,
):
    coordinator = RunCoordinator(
        DurableRunState(tmp_path / "state.sqlite3"),
        runtime=_YieldContinuationThenComplete(),
        feature_port=_RecordingFeature(),
    )
    coordinator.enqueue(_tick("background"))

    assert _require_result(coordinator).status == "checkpointed"
    assert _require_result(coordinator).status == "checkpointed"
    assert _require_result(coordinator).status == "checkpointed"
    assert _require_result(coordinator).status == "completed"

    attempts = coordinator.diagnostics("daily-user").attempts
    assert [(attempt.ordinal, attempt.status) for attempt in attempts] == [
        (1, "succeeded")
    ]


class _FaultFirstTick:
    def __init__(self) -> None:
        self.invoked: list[str] = []

    def invoke(self, context, *, tools, control):
        tick_id = context.tick.payload.tick_id
        self.invoked.append(tick_id)
        if tick_id == "poison":
            return InvocationOutcome.fault("fake_runtime_fault")
        return InvocationOutcome.completed()


def test_third_fault_quarantines_item_and_later_work_proceeds(tmp_path: Path):
    runtime = _FaultFirstTick()
    database = tmp_path / "state.sqlite3"
    coordinator = RunCoordinator(
        DurableRunState(database),
        runtime=runtime,
        feature_port=_RecordingFeature(),
        clock_ms=lambda: 100,
    )
    coordinator.enqueue(_tick("poison"))
    coordinator.enqueue(_tick("later", queued_at_ms=2))

    assert _require_result(coordinator).status == "failed_retryable"
    coordinator = RunCoordinator(
        DurableRunState(database),
        runtime=runtime,
        feature_port=_RecordingFeature(),
        clock_ms=lambda: 200,
    )
    assert _require_result(coordinator).status == "failed_retryable"
    coordinator = RunCoordinator(
        DurableRunState(database),
        runtime=runtime,
        feature_port=_RecordingFeature(),
        clock_ms=lambda: 300,
    )
    third = _require_result(coordinator)
    assert third.status == "quarantined"
    later = _require_result(coordinator)

    assert later.tick_id == "later"
    assert later.status == "completed"
    assert runtime.invoked == ["poison", "poison", "poison", "later"]
    diagnostics = coordinator.diagnostics("daily-user")
    assert [
        attempt.ordinal
        for attempt in diagnostics.attempts
        if attempt.logical_run_id == "run:poison"
    ] == [1, 2, 3]
    assert diagnostics.quarantines[0].source_id == "poison"
    assert coordinator.run_next("daily-user") is None


def test_expired_uncheckpointed_invocation_consumes_one_durable_attempt(tmp_path: Path):
    database = tmp_path / "state.sqlite3"
    state = DurableRunState(database, lease_duration_ms=10)
    state.enqueue(_tick("crashed"), now_ms=0)
    first = state.lease_next("daily-user", owner="dead-process", now_ms=0)
    assert first is not None
    assert first.attempt_ordinal == 1

    restarted = DurableRunState(database, lease_duration_ms=10)
    resumed = restarted.lease_next("daily-user", owner="restart", now_ms=11)

    assert resumed is not None
    assert resumed.attempt_ordinal == 2
    attempts = restarted.diagnostics("daily-user").attempts
    assert [(item.ordinal, item.status, item.failure_code) for item in attempts] == [
        (1, "failed_fault", "crash_discarded_uncheckpointed_inference"),
        (2, "running", None),
    ]


def test_stale_lease_cannot_finalize_a_recovered_attempt(tmp_path: Path):
    state = DurableRunState(tmp_path / "state.sqlite3", lease_duration_ms=10)
    state.enqueue(_tick("stale"), now_ms=0)
    first = state.lease_next("daily-user", owner="local-harness", now_ms=0)
    assert first is not None
    second = state.lease_next("daily-user", owner="local-harness", now_ms=11)
    assert second is not None
    assert second.attempt_ordinal == 2

    with pytest.raises(DurableStateError):
        state.complete(
            user_id="daily-user",
            logical_run_id="run:stale",
            owner="local-harness",
            attempt_id=first.attempt_id,
            lease_token=first.lease_token,
            now_ms=12,
        )

    attempts = state.diagnostics("daily-user").attempts
    assert [(attempt.ordinal, attempt.status) for attempt in attempts] == [
        (1, "failed_fault"),
        (2, "running"),
    ]


def test_same_action_id_is_isolated_between_users(tmp_path: Path):
    state = DurableRunState(tmp_path / "state.sqlite3")
    state.enqueue(_tick("first", user_id="user-a"), now_ms=1)
    state.enqueue(_tick("second", user_id="user-b"), now_ms=2)

    first = state.record_or_get_receipt(
        user_id="user-a",
        logical_run_id="run:first",
        action_id="shared-action",
        intent_fingerprint="a" * 64,
        provider_reference="feature:first",
        result={"user": "a"},
        acknowledged_at_ms=3,
    )
    second = state.record_or_get_receipt(
        user_id="user-b",
        logical_run_id="run:second",
        action_id="shared-action",
        intent_fingerprint="b" * 64,
        provider_reference="feature:second",
        result={"user": "b"},
        acknowledged_at_ms=4,
    )

    assert first.receipt_id == second.receipt_id
    assert (
        state.get_receipt(
            user_id="user-a", logical_run_id="run:first", action_id="shared-action"
        )
        == first
    )
    assert (
        state.get_receipt(
            user_id="user-b", logical_run_id="run:second", action_id="shared-action"
        )
        == second
    )


class _CheckpointWithUntrustedReceiptIds:
    def invoke(self, context, *, tools, control):
        receipt = tools.execute_once(
            FakeFeatureCommand(
                action_id="current-action",
                intent_fingerprint="c" * 64,
                payload={"value": "current"},
            )
        )
        return InvocationOutcome.checkpointed(
            remaining_work="resume safely",
            completed_receipt_ids=(
                receipt.receipt_id,
                "receipt:bogus",
                "receipt:other",
            ),
        )


def test_checkpoint_receipts_are_derived_from_current_run_only(tmp_path: Path):
    state = DurableRunState(tmp_path / "state.sqlite3")
    state.enqueue(_tick("other", user_id="other-user"), now_ms=1)
    state.record_or_get_receipt(
        user_id="other-user",
        logical_run_id="run:other",
        action_id="other",
        intent_fingerprint="d" * 64,
        provider_reference="feature:other",
        result={"value": "other"},
        acknowledged_at_ms=2,
    )
    coordinator = RunCoordinator(
        state,
        runtime=_CheckpointWithUntrustedReceiptIds(),
        feature_port=_RecordingFeature(),
        clock_ms=lambda: 3,
    )
    coordinator.enqueue(_tick("current", queued_at_ms=2))

    result = _require_result(coordinator)

    assert result.tick_id == "current"
    checkpoint = state.diagnostics("daily-user").checkpoints[0]
    assert checkpoint.completed_receipt_ids == ("receipt:current-action",)


def test_state_path_is_profile_scoped_and_future_schema_fails_closed(
    tmp_path: Path, monkeypatch
):
    profile = tmp_path / "daily-driver-profile"
    monkeypatch.setenv("HERMES_HOME", str(profile))
    assert default_database_path() == profile / "thine-harness" / "run-state.sqlite3"

    database = tmp_path / "future.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 2")

    with pytest.raises(DurableStateError, match="newer than supported"):
        DurableRunState(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_receipt_reuse_with_changed_intent_fails_closed(tmp_path: Path):
    state = DurableRunState(tmp_path / "state.sqlite3")
    coordinator = RunCoordinator(
        state,
        runtime=_CheckpointAfterFeature(),
        feature_port=_RecordingFeature(),
    )
    coordinator.enqueue(_tick("background"))
    coordinator.run_next("daily-user")

    with pytest.raises(ReceiptConflict, match="different intent"):
        state.record_or_get_receipt(
            user_id="daily-user",
            logical_run_id="run:background",
            action_id="background:effect:1",
            intent_fingerprint="b" * 64,
            provider_reference="different",
            result={"saved": "different"},
            acknowledged_at_ms=2,
        )


def test_diagnostics_expose_durable_state_and_accepted_runtime_configuration(
    tmp_path: Path,
):
    coordinator = RunCoordinator(
        DurableRunState(tmp_path / "state.sqlite3"),
        runtime=_CheckpointAfterFeature(),
        feature_port=_RecordingFeature(),
        runtime_configuration=HarnessRuntimeConfiguration.from_model_config(
            RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
            tool_search_enabled=True,
            tool_search_listing=False,
        ),
        clock_ms=lambda: 101,
    )
    coordinator.enqueue(_tick("background"))
    coordinator.run_next("daily-user")

    diagnostics = coordinator.diagnostics("daily-user").as_dict()

    assert diagnostics["runtime"] == {
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "api_mode": "codex_responses",
        "reasoning_effort": "medium",
        "context_window_tokens": 272_000,
        "tool_search_enabled": True,
        "tool_search_listing": False,
        "tool_namespaces": [
            "transcripts",
            "speakers",
            "communications",
            "ui.state",
            "schedules",
            "working_memory",
            "topics",
            "permissions",
            "run",
        ],
    }
    assert diagnostics["queue"][0]["state"] == "queued"
    assert diagnostics["leases"] == []
    assert diagnostics["attempts"][0]["status"] == "running"
    assert diagnostics["checkpoints"][0]["logical_run_id"] == "run:background"
    assert diagnostics["receipts"][0]["action_id"] == "background:effect:1"
