from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import cast

from thine_harness.action_dispatcher import ActionDispatcher
from thine_harness.contracts.notifications import (
    NotificationIntent,
    NotificationOutcome,
    NotificationPermission,
)
from thine_harness.contracts.runtime import Tick
from thine_harness.deferred_tools import DeferredNamespaceCatalog
from thine_harness.p0_chat import (
    BackendPrivateChatPort,
    P0ChatStore,
    P0CoordinatorRuntime,
    ResolvedSubmission,
)
from thine_harness.run_coordinator import InvocationContext, InvocationControl
from thine_harness.run_state import DurableRunState, LeasedRun, SCHEMA_VERSION
from thine_harness.runtime import AgentTurnResult, HermesInvocationRuntime
from thine_harness.standalone_notifications import StandaloneNotificationToolBinding
from thine_harness.topics_preferences import (
    TOPIC_INSPECT_TOOL_NAME,
    TOPIC_INSPECT_TOOL_SCHEMA,
    TOPIC_UPDATE_TOOL_NAME,
    TOPIC_UPDATE_TOOL_SCHEMA,
    TopicPreferenceService,
    TopicPreferenceToolBinding,
)


_VERSION = {"major": 1, "minor": 0}
HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS


class _Clock:
    def __init__(self, value: int = 1_800_000_000_000) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


class _NotificationBackend:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.deliveries: list[NotificationIntent] = []

    def permission(self) -> NotificationPermission:
        return NotificationPermission.from_dict({
            "schema_version": _VERSION,
            "user_preference": "enabled",
            "os_permission": "authorized",
            "last_permission_ask_at_ms": None,
            "last_permission_ask_topic_id": None,
            "observed_at_ms": self.clock(),
            "extensions": {},
        })

    def deliver_standalone(self, intent: NotificationIntent) -> NotificationOutcome:
        self.deliveries.append(intent)
        raise AssertionError("explicit notifications-disabled must block transport")

    def standalone_receipt(self, action_id: str) -> NotificationOutcome | None:
        del action_id
        return None


class _P0Backend:
    def __init__(self) -> None:
        self.receipts = []

    def resolve_submission(self, **_kwargs: object) -> ResolvedSubmission:
        return ResolvedSubmission(
            user_message_id="message-p0-context",
            text="Please turn off notifications.",
        )

    def record_queue_receipt(self, receipt: object) -> None:
        self.receipts.append(receipt)


class _CapturingRuntime:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, request, *, emit) -> AgentTurnResult:
        del emit
        self.prompts.append(request.prompt)
        return AgentTurnResult(final_output="Done")


class _P0ActivationProbe:
    def __init__(self) -> None:
        self.seen: tuple[str, str] | None = None

    @contextmanager
    def activate_p0(
        self,
        _context: InvocationContext,
        *,
        user_message_id: str,
        user_message_text: str,
    ):
        self.seen = (user_message_id, user_message_text)
        yield


def _tick(tick_id: str, *, kind: str = "p1_transcript") -> Tick:
    priority = "p0" if kind == "p0_user_chat" else "p1"
    source_kind = (
        "user_message" if kind == "p0_user_chat" else "transcript_availability"
    )
    return Tick.from_dict({
        "schema_version": _VERSION,
        "tick_id": tick_id,
        "user_id": "daily-user",
        "logical_run_id": f"run:{tick_id}",
        "kind": kind,
        "priority": priority,
        "occurred_at_ms": 1,
        "received_at_ms": 1,
        "queued_at_ms": 1,
        "source_ref": {"kind": source_kind, "id": tick_id},
        "causation_id": None,
        "correlation_id": f"correlation:{tick_id}",
        "attempt_ordinal": 1,
        "lease": None,
        "communication_allowance_snapshot": None,
        "payload": {
            "payload_kind": source_kind,
            "reference_id": tick_id,
        },
        "extensions": {},
    })


def _active(
    state: DurableRunState, clock: _Clock, tick_id: str, *, kind: str = "p1_transcript"
) -> tuple[InvocationContext, LeasedRun]:
    state.enqueue(_tick(tick_id, kind=kind), now_ms=clock())
    lease = state.lease_next("daily-user", owner="test", now_ms=clock())
    assert lease is not None
    state.mark_inference_started(
        user_id="daily-user",
        logical_run_id=f"run:{tick_id}",
        owner="test",
        attempt_id=lease.attempt_id,
        lease_token=lease.lease_token,
        now_ms=clock(),
    )
    return (
        InvocationContext(
            tick=lease.tick,
            attempt_id=lease.attempt_id,
            attempt_ordinal=lease.attempt_ordinal,
            checkpoint=lease.checkpoint,
            acknowledged_receipts=lease.acknowledged_receipts,
        ),
        lease,
    )


def test_schema_migrates_to_durable_topics_preferences_corrections_and_receipts(
    tmp_path: Path,
) -> None:
    state = DurableRunState(tmp_path / "state.sqlite3")
    with state._connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {
        "durable_topics",
        "explicit_preferences",
        "explicit_corrections",
        "topic_preference_receipts",
    } <= tables


def test_topic_repeat_is_blocked_for_24_hours_without_new_evidence_and_receipted(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    service = TopicPreferenceService(
        DurableRunState(tmp_path / "state.sqlite3"), clock_ms=clock
    )

    first = service.record_topic_action(
        user_id="daily-user",
        logical_run_id="run:first",
        topic_key="tag_unknown_speakers",
        last_action="asked the user to tag an unknown speaker",
        evidence_refs=("speaker:unknown-1",),
        evidence_at_ms=clock(),
        receipt_refs=("action:notification-1",),
    )
    replay = service.record_topic_action(
        user_id="daily-user",
        logical_run_id="run:first",
        topic_key="tag_unknown_speakers",
        last_action="asked the user to tag an unknown speaker",
        evidence_refs=("speaker:unknown-1",),
        evidence_at_ms=clock(),
        receipt_refs=("action:notification-1",),
    )
    assert replay.receipt_id == first.receipt_id
    assert replay.topic.lifecycle.to_dict() == first.topic.lifecycle.to_dict()
    assert first.topic.last_evidence_at_ms == clock()
    assert first.topic.last_action == "asked the user to tag an unknown speaker"
    assert first.topic.receipt_refs == ("action:notification-1",)

    clock.value += HOUR_MS
    blocked = service.repeat_eligibility(
        user_id="daily-user", topic_key="tag_unknown_speakers"
    )
    assert blocked.eligible is False
    assert blocked.reason == "cooldown_requires_fresh_evidence"

    fresh = service.repeat_eligibility(
        user_id="daily-user",
        topic_key="tag_unknown_speakers",
        proposed_evidence_refs=("speaker:unknown-2",),
        proposed_evidence_at_ms=clock(),
    )
    assert fresh.eligible is True
    assert fresh.reason == "fresh_evidence"

    clock.value += DAY_MS
    elapsed = service.repeat_eligibility(
        user_id="daily-user", topic_key="tag_unknown_speakers"
    )
    assert elapsed.eligible is True
    assert elapsed.reason == "cooldown_elapsed"


def test_topic_action_rejects_future_evidence_timestamp(tmp_path: Path) -> None:
    clock = _Clock()
    service = TopicPreferenceService(
        DurableRunState(tmp_path / "state.sqlite3"), clock_ms=clock
    )

    try:
        service.record_topic_action(
            user_id="daily-user",
            logical_run_id="run:future",
            topic_key="tag_unknown_speakers",
            last_action="asked for a speaker tag",
            evidence_refs=("speaker:unknown-1",),
            evidence_at_ms=clock() + 1,
            receipt_refs=("action:1",),
        )
    except ValueError as exc:
        assert str(exc) == "evidence_at_ms cannot be in the future"
    else:
        raise AssertionError("future evidence must not enter durable Topic state")


def test_recent_working_memory_blocks_immediate_repeat_until_fresh_evidence(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    state = DurableRunState(tmp_path / "state.sqlite3")
    service = TopicPreferenceService(state, clock_ms=clock)
    service.record_topic_action(
        user_id="daily-user",
        logical_run_id="run:first",
        topic_key="tag_unknown_speakers",
        last_action="asked for a speaker tag",
        evidence_refs=("speaker:unknown-1",),
        evidence_at_ms=clock(),
        receipt_refs=("action:1",),
    )
    with state._transaction() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO working_memory_state "
            "(user_id, version, markdown, token_count, last_run_id) "
            "VALUES (?, 1, ?, 9, ?)",
            (
                "daily-user",
                "- topic_key: tag_unknown_speakers\n- Asked once; await the user.",
                "run:first",
            ),
        )

    context = cast(
        dict[str, object], service.prompt_context("daily-user")["topics_preferences"]
    )
    topic = cast(list[dict[str, object]], context["topics"])[0]
    assert topic["recent_working_memory_mentions_topic"] is True
    assert topic["repeat_eligibility"] == {
        "eligible": False,
        "reason": "recent_working_memory",
    }

    clock.value += 1
    fresh = service.repeat_eligibility(
        user_id="daily-user",
        topic_key="tag_unknown_speakers",
        proposed_evidence_refs=("speaker:unknown-2",),
        proposed_evidence_at_ms=clock(),
    )
    assert fresh.eligible is True
    assert fresh.reason == "fresh_evidence"


def test_permission_topic_never_repeats_before_seven_days(tmp_path: Path) -> None:
    clock = _Clock()
    service = TopicPreferenceService(
        DurableRunState(tmp_path / "state.sqlite3"), clock_ms=clock
    )
    service.record_topic_action(
        user_id="daily-user",
        logical_run_id="run:permission",
        topic_key="enable_notifications",
        last_action="asked the user to enable notifications",
        evidence_refs=("permission:not_determined",),
        evidence_at_ms=clock(),
        receipt_refs=("message:permission-ask",),
    )

    clock.value += 6 * DAY_MS
    early = service.repeat_eligibility(
        user_id="daily-user",
        topic_key="enable_notifications",
        proposed_evidence_refs=("permission:denied",),
        proposed_evidence_at_ms=clock(),
    )
    assert early.eligible is False
    assert early.reason == "permission_cadence"

    clock.value += DAY_MS
    due = service.repeat_eligibility(
        user_id="daily-user", topic_key="enable_notifications"
    )
    assert due.eligible is True
    assert due.reason == "permission_cadence_elapsed"


def test_topic_tracks_open_snoozed_and_resolved_lifecycle(tmp_path: Path) -> None:
    clock = _Clock()
    service = TopicPreferenceService(
        DurableRunState(tmp_path / "state.sqlite3"), clock_ms=clock
    )
    proposed, _receipt = service.set_topic_state(
        user_id="daily-user",
        logical_run_id="run:state",
        topic_key="follow_up_name_tagging",
        state="proposed",
    )
    assert proposed.lifecycle.payload.state == "proposed"

    snoozed, _receipt = service.set_topic_state(
        user_id="daily-user",
        logical_run_id="run:state",
        topic_key="follow_up_name_tagging",
        state="snoozed",
        snooze_until_ms=clock() + DAY_MS,
    )
    assert snoozed.lifecycle.payload.state == "snoozed"
    assert snoozed.lifecycle.payload.next_eligible_at_ms == clock() + DAY_MS
    assert (
        service.repeat_eligibility(
            user_id="daily-user", topic_key="follow_up_name_tagging"
        ).reason
        == "topic_snoozed"
    )

    resolved, _receipt = service.set_topic_state(
        user_id="daily-user",
        logical_run_id="run:state",
        topic_key="follow_up_name_tagging",
        state="resolved",
    )
    assert resolved.lifecycle.payload.state == "resolved"
    assert (
        service.repeat_eligibility(
            user_id="daily-user", topic_key="follow_up_name_tagging"
        ).reason
        == "topic_resolved"
    )


def test_only_explicit_p0_can_change_the_two_narrow_preferences(tmp_path: Path) -> None:
    clock = _Clock()
    state = DurableRunState(tmp_path / "state.sqlite3")
    service = TopicPreferenceService(state, clock_ms=clock)
    binding = TopicPreferenceToolBinding(service=service)

    background, background_lease = _active(state, clock, "tick-background")
    with binding.activate(background):
        denied = json.loads(
            binding.update({
                "operation": "set_preference",
                "preference_key": "notifications_enabled",
                "preference_value": False,
            })
        )
    assert denied["error_code"] == "explicit_p0_authorization_required"
    state.complete(
        user_id="daily-user",
        logical_run_id="run:tick-background",
        owner="test",
        attempt_id=background_lease.attempt_id,
        lease_token=background_lease.lease_token,
        now_ms=clock(),
    )

    clock.value += 1
    p0, _p0_lease = _active(state, clock, "tick-p0", kind="p0_user_chat")
    with binding.activate_p0(
        p0,
        user_message_id="message-user-1",
        user_message_text="Please turn off notifications.",
    ):
        changed = json.loads(
            binding.update({
                "operation": "set_preference",
                "preference_key": "notifications_enabled",
                "preference_value": False,
            })
        )
        impossible = json.loads(
            binding.update({
                "operation": "set_preference",
                "preference_key": "background_inference_enabled",
                "preference_value": False,
            })
        )

    assert changed["ok"] is True
    assert changed["preferences"]["narrow_preferences"] == [
        {
            "key": "notifications_enabled",
            "value": False,
            "switchable_by_agent_on_explicit_user_request": True,
            "authorizing_message_id": "message-user-1",
            "updated_at_ms": clock(),
        }
    ]
    assert impossible["error_code"] == "preference_not_switchable"
    state.complete(
        user_id="daily-user",
        logical_run_id="run:tick-p0",
        owner="test",
        attempt_id=_p0_lease.attempt_id,
        lease_token=_p0_lease.lease_token,
        now_ms=clock(),
    )

    clock.value += 1
    p0_reversal, _p0_reversal_lease = _active(
        state, clock, "tick-p0-reversal", kind="p0_user_chat"
    )
    with binding.activate_p0(
        p0_reversal,
        user_message_id="message-user-2",
        user_message_text=(
            "Please turn on notifications, but don't ask me to tag speakers."
        ),
    ):
        notifications_on = json.loads(
            binding.update({
                "operation": "set_preference",
                "preference_key": "notifications_enabled",
                "preference_value": True,
            })
        )
        speaker_nudges_off = json.loads(
            binding.update({
                "operation": "set_preference",
                "preference_key": "speaker_tag_nudges_enabled",
                "preference_value": False,
            })
        )

    assert notifications_on["ok"] is True
    assert speaker_nudges_off["ok"] is True
    narrow = {
        item["key"]: item["value"]
        for item in speaker_nudges_off["preferences"]["narrow_preferences"]
    }
    assert narrow == {
        "notifications_enabled": True,
        "speaker_tag_nudges_enabled": False,
    }
    assert (
        service.repeat_eligibility(
            user_id="daily-user", topic_key="tag_unknown_speakers"
        ).reason
        == "explicit_do_not_ask"
    )
    assert set(speaker_nudges_off["preferences"]["non_disableable_capabilities"]) == {
        "background_inference",
        "home_mutation",
        "schedule_creation",
        "proactive_chat",
    }
    policy = cast(
        dict[str, object], service.prompt_context("daily-user")["topics_preferences"]
    )
    assert policy["preferences"] == speaker_nudges_off["preferences"]


def test_explicit_correction_survives_memory_change_and_is_authoritative_context(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    state = DurableRunState(tmp_path / "state.sqlite3")
    service = TopicPreferenceService(state, clock_ms=clock)
    binding = TopicPreferenceToolBinding(service=service)
    p0, _lease = _active(state, clock, "tick-correction", kind="p0_user_chat")
    with binding.activate_p0(
        p0,
        user_message_id="message-user-2",
        user_message_text="Actually, my preferred name is Sid.",
    ):
        result = json.loads(
            binding.update({
                "operation": "record_correction",
                "correction_key": "preferred_name",
                "corrected_value": "Sid",
            })
        )
    assert result["ok"] is True

    with state._transaction() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO working_memory_state "
            "(user_id, version, markdown, token_count, last_run_id) "
            "VALUES (?, 9, ?, 4, ?)",
            ("daily-user", "- Compacted unrelated recent actions.", "run:later"),
        )

    context = cast(
        dict[str, object], service.prompt_context("daily-user")["topics_preferences"]
    )
    assert context["explicit_corrections"] == [
        {
            "correction_key": "preferred_name",
            "corrected_value": "Sid",
            "authorizing_message_id": "message-user-2",
            "updated_at_ms": clock(),
        }
    ]
    assert context["precedence"] == (
        "explicit_preferences_and_corrections_override_inferred_state"
    )


def test_explicit_notification_preference_blocks_standalone_transport(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    state = DurableRunState(tmp_path / "state.sqlite3")
    service = TopicPreferenceService(state, clock_ms=clock)
    service.set_preference(
        user_id="daily-user",
        logical_run_id="run:preference",
        preference_key="notifications_enabled",
        value=False,
        authorizing_message_id="message-user-1",
    )
    backend = _NotificationBackend(clock)
    notification = StandaloneNotificationToolBinding(
        dispatcher=ActionDispatcher(state, clock_ms=clock),
        backend=backend,
        clock_ms=clock,
        preference_lookup=lambda user_id: service.preference_value(
            user_id, "notifications_enabled"
        ),
    )
    background, _lease = _active(state, clock, "tick-notification-block")
    with notification.activate(background):
        result = json.loads(
            notification.send({"title": "Blocked", "body": "Do not deliver"})
        )
    assert result["error_code"] == "capability_disabled"
    assert result["durable_repeat_guard"]["source"] == "explicit_preference"
    assert backend.deliveries == []


def test_p0_runtime_activates_authorization_context_and_injects_policy(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    store = P0ChatStore(tmp_path / "state.sqlite3")
    receipt = store.admit(
        user_id="daily-user",
        user_message_id="message-p0-context",
        idempotency_key="user-message:message-p0-context",
        submission_ref="p0-submission:message-p0-context",
        now_ms=clock(),
    )
    lease = store.run_state.lease_next("daily-user", owner="test", now_ms=clock())
    assert lease is not None
    store.run_state.mark_inference_started(
        user_id="daily-user",
        logical_run_id=receipt.logical_run_id,
        owner="test",
        attempt_id=lease.attempt_id,
        lease_token=lease.lease_token,
        now_ms=clock(),
    )
    context = InvocationContext(
        tick=lease.tick,
        attempt_id=lease.attempt_id,
        attempt_ordinal=lease.attempt_ordinal,
        checkpoint=lease.checkpoint,
        acknowledged_receipts=lease.acknowledged_receipts,
    )
    backend = _P0Backend()
    model = _CapturingRuntime()
    activation = _P0ActivationProbe()
    runtime = P0CoordinatorRuntime(
        store=store,
        backend=cast(BackendPrivateChatPort, backend),
        runtime_loader=lambda: cast(HermesInvocationRuntime, model),
        now_ms=clock,
        heartbeat_interval_seconds=60,
        context_bindings=(activation,),
        policy_context=lambda _user_id: {
            "topics_preferences": {
                "preferences": {"notifications_enabled": False},
                "precedence": "explicit_over_inferred",
            }
        },
    )

    outcome = runtime.invoke(context, tools=None, control=InvocationControl())

    assert outcome.status == "completed"
    assert activation.seen == (
        "message-p0-context",
        "Please turn off notifications.",
    )
    assert "explicit_over_inferred" in model.prompts[0]
    assert "Current user message:\nPlease turn off notifications." in model.prompts[0]


def test_tools_are_deferred_in_topics_namespace_and_offer_no_memory_restore(
    tmp_path: Path,
) -> None:
    binding = TopicPreferenceToolBinding(
        service=TopicPreferenceService(DurableRunState(tmp_path / "state.sqlite3"))
    )
    from tools.registry import registry

    binding.register(registry_instance=registry)
    try:
        definitions = [
            {"type": "function", "function": TOPIC_INSPECT_TOOL_SCHEMA},
            {"type": "function", "function": TOPIC_UPDATE_TOOL_SCHEMA},
        ]
        catalog = DeferredNamespaceCatalog(definitions, context_length=272_000)
        eager = {tool["function"]["name"] for tool in catalog.model_tool_definitions()}
        assert TOPIC_INSPECT_TOOL_NAME not in eager
        assert TOPIC_UPDATE_TOOL_NAME not in eager
        matches = catalog.search("inspect topics and preferences")
        assert any(
            match["name"] == TOPIC_INSPECT_TOOL_NAME and match["namespace"] == "topics"
            for match in matches
        )
        assert all("restore" not in match["name"] for match in matches)
    finally:
        registry.deregister(TOPIC_INSPECT_TOOL_NAME)
        registry.deregister(TOPIC_UPDATE_TOOL_NAME)
