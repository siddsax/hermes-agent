from __future__ import annotations

import json
from pathlib import Path
import threading

import httpx

from thine_harness.contracts.speakers import (
    SpeakerCursorOutcome,
    SpeakerMappingEvent,
)
from thine_harness.contracts.runtime import Tick
from thine_harness.run_coordinator import (
    InvocationOutcome,
    RunCoordinator,
    RunFinalizationResult,
)
from thine_harness.run_state import DurableRunState
from thine_harness.p0_chat import HarnessCoordinatorDriver
from thine_harness.speaker_mappings import (
    BackendSpeakerMappingClient,
    BackgroundFinalizerDispatcher,
    BackgroundInputDispatcher,
    BackgroundRuntimeDispatcher,
    FakeSpeakerMappingNoActionRuntime,
    INSPECT_ACTIVE_MAPPING_TOOL_NAME,
    PreparedSpeakerMappingInput,
    RealSpeakerMappingAgentRuntime,
    SpeakerMappingInspectionToolBinding,
    SpeakerMappingFinalizer,
    SpeakerMappingInputPump,
    SpeakerMappingToolBinding,
)


def _event(
    event_id: str,
    cursor: int,
    *,
    kind: str = "rename",
    source_speaker_ids: list[str] | None = None,
    canonical_speaker_id: str | None = None,
    old_name: str | None = None,
    new_name: str | None = "Taylor",
) -> SpeakerMappingEvent:
    source_ids = source_speaker_ids or [f"SPEAKER_{cursor}"]
    return SpeakerMappingEvent.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "event_id": event_id,
        "user_id": "daily-user",
        "cursor": cursor,
        "kind": kind,
        "source_speaker_ids": source_ids,
        "canonical_speaker_id": canonical_speaker_id or f"speaker-{cursor}",
        "old_name": old_name,
        "new_name": new_name,
        "changed_at_ms": 100 + cursor,
        "transcript_text_included": False,
        "extensions": {},
    })


def _outcome(
    event: SpeakerMappingEvent,
    *,
    outcome: str = "acknowledged",
    quarantine_id: str | None = None,
) -> SpeakerCursorOutcome:
    return SpeakerCursorOutcome.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "ack_id": f"speaker-ack:{event.payload.event_id}:{outcome}",
        "event_id": event.payload.event_id,
        "cursor": event.payload.cursor,
        "outcome": outcome,
        "quarantine_id": quarantine_id,
        "normal_cursor": event.payload.cursor,
        "acknowledged_at_ms": 500 + event.payload.cursor,
        "extensions": {},
    })


class _SpeakerPort:
    def __init__(self, events: list[SpeakerMappingEvent]) -> None:
        self.events = events
        self.fetches: list[tuple[str, int | None]] = []
        self.acks: list[tuple[str, int]] = []
        self.quarantines: list[tuple[str, int, str]] = []
        self.lose_first_ack_response = False
        self.lose_first_quarantine_response = False

    def next(self, user_id: str, after_cursor: int | None = None):
        self.fetches.append((user_id, after_cursor))
        cursor = after_cursor or 0
        return next(
            (event for event in self.events if event.payload.cursor > cursor),
            None,
        )

    def acknowledge(self, event_id, cursor):
        request = (str(event_id), int(cursor))
        self.acks.append(request)
        event = next(
            event for event in self.events if event.payload.event_id == request[0]
        )
        outcome = _outcome(event)
        if self.lose_first_ack_response and len(self.acks) == 1:
            raise TimeoutError("ack response lost after backend commit")
        return outcome

    def quarantine_and_advance(self, event_id, cursor, quarantine_id):
        request = (str(event_id), int(cursor), str(quarantine_id))
        self.quarantines.append(request)
        event = next(
            event for event in self.events if event.payload.event_id == request[0]
        )
        outcome = _outcome(
            event,
            outcome="quarantined_and_advanced",
            quarantine_id=request[2],
        )
        if self.lose_first_quarantine_response and len(self.quarantines) == 1:
            raise TimeoutError("quarantine response lost after backend commit")
        return outcome


class _NoEffects:
    def apply(self, command):  # pragma: no cover - mapping no-op has no effect
        raise AssertionError(command)


def test_unknown_mapping_is_durably_enqueued_after_cursor_and_acknowledged(
    tmp_path: Path,
) -> None:
    unknown = _event(
        "speaker-event-1",
        1,
        source_speaker_ids=["SPEAKER_UNKNOWN_1"],
        canonical_speaker_id="speaker-1",
        old_name=None,
        new_name="Taylor",
    )
    port = _SpeakerPort([unknown])
    state = DurableRunState(tmp_path / "state.sqlite3")
    pump = SpeakerMappingInputPump(state, speaker_port=port, clock_ms=lambda: 200)
    runtime = FakeSpeakerMappingNoActionRuntime()
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=_NoEffects(),
        input_port=pump,
        finalizer=SpeakerMappingFinalizer(
            state, speaker_port=port, clock_ms=lambda: 500
        ),
        clock_ms=lambda: 200,
    )

    tick_id = pump.enqueue_next("daily-user", coordinator=coordinator)
    result = coordinator.run_next("daily-user")

    assert tick_id == "speaker-tick:speaker-event-1"
    assert result is not None and result.status == "completed"
    assert len(runtime.invocations) == 1
    prepared = runtime.invocations[0].prepared_input
    assert isinstance(prepared, PreparedSpeakerMappingInput)
    assert prepared.event.to_dict() == unknown.to_dict()
    assert prepared.event.payload.source_speaker_ids == ("SPEAKER_UNKNOWN_1",)
    assert port.fetches == [("daily-user", None)]
    assert port.acks == [("speaker-event-1", 1)]
    inspection = state.inspect_speaker_mapping(
        user_id="daily-user", event_id="speaker-event-1"
    )
    assert inspection.state == "acknowledged"
    assert inspection.normal_cursor == 1
    assert inspection.event.to_dict() == unknown.to_dict()


def test_lost_ack_response_retries_only_suffix_without_new_attempt(
    tmp_path: Path,
) -> None:
    event = _event("speaker-event-1", 1)
    port = _SpeakerPort([event])
    port.lose_first_ack_response = True
    state = DurableRunState(tmp_path / "state.sqlite3")
    pump = SpeakerMappingInputPump(state, speaker_port=port, clock_ms=lambda: 200)
    runtime = FakeSpeakerMappingNoActionRuntime()
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=_NoEffects(),
        input_port=pump,
        finalizer=SpeakerMappingFinalizer(
            state, speaker_port=port, clock_ms=lambda: 500
        ),
        clock_ms=lambda: 200,
    )
    pump.enqueue_next("daily-user", coordinator=coordinator)

    awaiting = coordinator.run_next("daily-user")
    recovered = coordinator.run_next("daily-user")

    assert awaiting is not None
    assert awaiting.status == "awaiting_speaker_cursor_ack"
    assert recovered is not None and recovered.status == "completed"
    assert len(runtime.invocations) == 1
    assert port.acks == [("speaker-event-1", 1), ("speaker-event-1", 1)]
    attempts = state.diagnostics("daily-user").attempts
    assert [(item.ordinal, item.status) for item in attempts] == [(1, "succeeded")]


def test_backend_client_uses_only_strict_mapping_helpers() -> None:
    event = _event("speaker-event-1", 1)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/speaker-mappings/next":
            return httpx.Response(200, json=event.to_dict())
        if request.url.path == "/v1/speaker-mappings/ack":
            return httpx.Response(200, json=_outcome(event).to_dict())
        if request.url.path == "/v1/speaker-mappings/quarantine":
            return httpx.Response(
                200,
                json=_outcome(
                    event,
                    outcome="quarantined_and_advanced",
                    quarantine_id="q-1",
                ).to_dict(),
            )
        raise AssertionError(request.url.path)

    client = BackendSpeakerMappingClient(
        origin="http://127.0.0.1:8790",
        credential="private-token",
        firebase_uid="daily-user",
        transport=httpx.MockTransport(handler),
    )

    assert client.next("daily-user") is not None
    client.acknowledge("speaker-event-1", 1)
    client.quarantine_and_advance("speaker-event-1", 1, "q-1")
    client.close()

    assert [request.url.path for request in requests] == [
        "/v1/speaker-mappings/next",
        "/v1/speaker-mappings/ack",
        "/v1/speaker-mappings/quarantine",
    ]
    bodies = [json.loads(request.content) for request in requests]
    assert bodies == [
        {"after_cursor": None},
        {
            "event_id": "speaker-event-1",
            "cursor": 1,
            "idempotency_key": "speaker-ack:speaker-event-1:1",
        },
        {
            "event_id": "speaker-event-1",
            "cursor": 1,
            "quarantine_id": "q-1",
            "idempotency_key": "speaker-quarantine:q-1",
        },
    ]
    assert all(
        request.headers["authorization"] == "Bearer private-token"
        and request.headers["x-thine-firebase-uid"] == "daily-user"
        for request in requests
    )


class _FaultFirstEvent:
    def __init__(self) -> None:
        self.invocations = []

    def invoke(self, context, *, tools, control):
        del tools, control
        self.invocations.append(context)
        prepared = context.prepared_input
        assert isinstance(prepared, PreparedSpeakerMappingInput)
        if prepared.event.payload.event_id == "speaker-event-bad":
            return InvocationOutcome.fault("provider_timeout")
        return InvocationOutcome.no_action()


class _FaultOriginalThenSucceedRetry(_FaultFirstEvent):
    def invoke(self, context, *, tools, control):
        prepared = context.prepared_input
        assert isinstance(prepared, PreparedSpeakerMappingInput)
        if prepared.explicit_retry is not None:
            self.invocations.append(context)
            return InvocationOutcome.no_action()
        return super().invoke(context, tools=tools, control=control)


def test_third_real_fault_quarantines_only_mapping_then_later_mapping_runs(
    tmp_path: Path,
) -> None:
    bad = _event("speaker-event-bad", 1)
    later = _event(
        "speaker-event-later",
        2,
        kind="merge",
        source_speaker_ids=["S1", "S2"],
        new_name="Morgan",
    )
    port = _SpeakerPort([bad, later])
    state = DurableRunState(tmp_path / "state.sqlite3")
    pump = SpeakerMappingInputPump(state, speaker_port=port, clock_ms=lambda: 200)
    runtime = _FaultFirstEvent()
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=_NoEffects(),
        input_port=pump,
        finalizer=SpeakerMappingFinalizer(
            state, speaker_port=port, clock_ms=lambda: 500
        ),
        clock_ms=lambda: 200,
    )
    pump.enqueue_next("daily-user", coordinator=coordinator)

    first = coordinator.run_next("daily-user")
    second = coordinator.run_next("daily-user")
    third = coordinator.run_next("daily-user")
    later_tick = pump.enqueue_next("daily-user", coordinator=coordinator)
    later_result = coordinator.run_next("daily-user")

    assert [first.status, second.status, third.status] == [
        "failed_retryable",
        "failed_retryable",
        "quarantined",
    ]
    assert later_tick == "speaker-tick:speaker-event-later"
    assert later_result is not None and later_result.status == "completed"
    assert port.quarantines == [
        ("speaker-event-bad", 1, "run:speaker-event-bad:quarantine")
    ]
    assert port.acks == [("speaker-event-later", 2)]
    assert state.speaker_cursor("daily-user") == 2
    quarantined = state.inspect_speaker_mapping(
        user_id="daily-user", event_id="speaker-event-bad"
    )
    assert quarantined.state == "quarantined"
    assert quarantined.normal_cursor == 2


def test_lost_quarantine_response_retries_suffix_without_fourth_attempt(
    tmp_path: Path,
) -> None:
    bad = _event("speaker-event-bad", 1)
    port = _SpeakerPort([bad])
    port.lose_first_quarantine_response = True
    database = tmp_path / "state.sqlite3"
    state = DurableRunState(database)
    pump = SpeakerMappingInputPump(state, speaker_port=port, clock_ms=lambda: 200)
    runtime = _FaultFirstEvent()
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=_NoEffects(),
        input_port=pump,
        finalizer=SpeakerMappingFinalizer(
            state, speaker_port=port, clock_ms=lambda: 500
        ),
        clock_ms=lambda: 200,
    )
    pump.enqueue_next("daily-user", coordinator=coordinator)

    results = [coordinator.run_next("daily-user") for _ in range(3)]
    restarted_state = DurableRunState(database)
    restarted = RunCoordinator(
        restarted_state,
        runtime=runtime,
        feature_port=_NoEffects(),
        input_port=SpeakerMappingInputPump(
            restarted_state, speaker_port=port, clock_ms=lambda: 200
        ),
        finalizer=SpeakerMappingFinalizer(
            restarted_state, speaker_port=port, clock_ms=lambda: 500
        ),
        clock_ms=lambda: 200,
    )
    recovered = restarted.run_next("daily-user")

    assert [result.status for result in results if result is not None] == [
        "failed_retryable",
        "failed_retryable",
        "quarantine_pending",
    ]
    assert recovered is not None and recovered.status == "quarantined"
    assert len(runtime.invocations) == 3
    assert port.quarantines == [
        ("speaker-event-bad", 1, "run:speaker-event-bad:quarantine"),
        ("speaker-event-bad", 1, "run:speaker-event-bad:quarantine"),
    ]
    attempts = restarted_state.diagnostics("daily-user").attempts
    assert [(item.ordinal, item.status) for item in attempts] == [
        (1, "failed_fault"),
        (2, "failed_fault"),
        (3, "failed_fault"),
    ]


def test_explicit_retry_uses_immutable_event_without_rewinding_normal_cursor(
    tmp_path: Path,
) -> None:
    bad = _event("speaker-event-bad", 1)
    later = _event("speaker-event-later", 2)
    port = _SpeakerPort([bad, later])
    state = DurableRunState(tmp_path / "state.sqlite3")
    pump = SpeakerMappingInputPump(state, speaker_port=port, clock_ms=lambda: 200)
    runtime = _FaultFirstEvent()
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=_NoEffects(),
        input_port=pump,
        finalizer=SpeakerMappingFinalizer(
            state, speaker_port=port, clock_ms=lambda: 500
        ),
        clock_ms=lambda: 200,
    )
    pump.enqueue_next("daily-user", coordinator=coordinator)
    for _ in range(3):
        coordinator.run_next("daily-user")
    pump.enqueue_next("daily-user", coordinator=coordinator)
    coordinator.run_next("daily-user")
    quarantine = state.inspect_speaker_mapping(
        user_id="daily-user", event_id="speaker-event-bad"
    )

    retry_tick = pump.enqueue_explicit_retry(
        user_id="daily-user",
        quarantine_id=quarantine.quarantine_id,
        coordinator=coordinator,
    )
    retried = coordinator.run_next("daily-user")

    assert retry_tick == "speaker-retry-tick:run:speaker-event-bad:quarantine:1"
    assert retried is not None and retried.status == "failed_retryable"
    prepared = runtime.invocations[-1].prepared_input
    assert isinstance(prepared, PreparedSpeakerMappingInput)
    assert prepared.event.to_dict() == bad.to_dict()
    assert prepared.explicit_retry is not None
    assert prepared.explicit_retry.payload.rewinds_normal_cursor is False
    assert state.speaker_cursor("daily-user") == 2


def test_successful_explicit_retry_completes_without_backend_cursor_mutation(
    tmp_path: Path,
) -> None:
    bad = _event("speaker-event-bad", 1)
    later = _event("speaker-event-later", 2)
    port = _SpeakerPort([bad, later])
    state = DurableRunState(tmp_path / "state.sqlite3")
    pump = SpeakerMappingInputPump(state, speaker_port=port, clock_ms=lambda: 200)
    runtime = _FaultOriginalThenSucceedRetry()
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=_NoEffects(),
        input_port=pump,
        finalizer=SpeakerMappingFinalizer(
            state, speaker_port=port, clock_ms=lambda: 500
        ),
        clock_ms=lambda: 200,
    )
    pump.enqueue_next("daily-user", coordinator=coordinator)
    for _ in range(3):
        coordinator.run_next("daily-user")
    pump.enqueue_next("daily-user", coordinator=coordinator)
    coordinator.run_next("daily-user")
    quarantine = state.inspect_speaker_mapping(
        user_id="daily-user", event_id="speaker-event-bad"
    )
    acks_before_retry = list(port.acks)
    quarantines_before_retry = list(port.quarantines)
    pump.enqueue_explicit_retry(
        user_id="daily-user",
        quarantine_id=quarantine.quarantine_id,
        coordinator=coordinator,
    )

    retried = coordinator.run_next("daily-user")

    assert retried is not None and retried.status == "completed"
    assert state.speaker_cursor("daily-user") == 2
    assert port.acks == acks_before_retry
    assert port.quarantines == quarantines_before_retry
    inspected = state.inspect_speaker_quarantine(
        user_id="daily-user", quarantine_id=quarantine.quarantine_id
    )
    assert inspected.retry_run_ids == (retried.logical_run_id,)


class _WireTransport:
    def convert_tools(self, tools):
        return list(tools)


class _SpeakerGPTAgent:
    provider = "openai-codex"
    model = "gpt-5.6-sol"
    api_mode = "codex_responses"
    reasoning_config = {"enabled": True, "effort": "medium"}
    context_compressor = type("Context", (), {"context_length": 272_000})()
    skip_memory = True
    skip_background_review = True
    _fallback_chain = []
    _fallback_model = None
    _cached_system_prompt = "stable harness prefix"
    ephemeral_system_prompt = "stable background policy"

    def __init__(self) -> None:
        self.session_id = "thine-background:daily-user"
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": name,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in ("tool_search", "tool_describe", "tool_call")
        ]
        self._persist_disabled = False
        self.primary_prompts: list[str] = []
        self._output_tokens = 0
        self._reasoning_tokens = 0

    def _get_transport(self):
        return _WireTransport()

    def run_conversation(self, prompt, **kwargs):
        history = list(kwargs.get("conversation_history") or [])
        if prompt.startswith("Stop Hook:"):
            self._output_tokens += 20
            response = (
                '{"worth_remembering":true,"markdown":"# Recent operations\\n\\n'
                '- Processed speaker-event-1; preserve Unknown attribution."}'
            )
            return {
                "final_response": response,
                "completed": True,
                "messages": [*history, {"role": "assistant", "content": response}],
                "output_tokens": self._output_tokens,
                "reasoning_tokens": self._reasoning_tokens,
            }
        if prompt.startswith("Configured-model token measurement."):
            candidate = (
                "# Recent operations\n\n"
                "- Processed speaker-event-1; preserve Unknown attribution."
            )
            self._output_tokens += 30
            return {
                "final_response": candidate,
                "completed": True,
                "messages": [*history, {"role": "assistant", "content": candidate}],
                "output_tokens": self._output_tokens,
                "reasoning_tokens": self._reasoning_tokens,
            }
        self.primary_prompts.append(prompt)
        self._output_tokens += 10
        return {
            "final_response": "No user-visible action is needed.",
            "messages": [
                {"role": "tool", "tool_name": "tool_search", "content": "found"},
                {
                    "role": "tool",
                    "tool_name": INSPECT_ACTIVE_MAPPING_TOOL_NAME,
                    "content": "mapping",
                },
                {"role": "assistant", "content": "No action."},
            ],
            "completed": True,
            "failed": False,
            "interrupted": False,
            "input_tokens": 100,
            "output_tokens": self._output_tokens,
            "reasoning_tokens": self._reasoning_tokens,
            "cache_read_tokens": 80,
            "cache_write_tokens": 0,
        }


def test_real_gpt_mapping_path_uses_cached_stop_hook_and_inspection_helpers(
    tmp_path: Path,
) -> None:
    event = _event(
        "speaker-event-1",
        1,
        source_speaker_ids=["SPEAKER_UNKNOWN_1"],
        old_name=None,
        new_name="Taylor",
    )
    port = _SpeakerPort([event])
    state = DurableRunState(tmp_path / "state.sqlite3")
    pump = SpeakerMappingInputPump(state, speaker_port=port, clock_ms=lambda: 200)
    agent = _SpeakerGPTAgent()
    binding = SpeakerMappingToolBinding()
    runtime = RealSpeakerMappingAgentRuntime(
        state,
        agent=agent,
        binding=binding,
    )
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=_NoEffects(),
        input_port=pump,
        finalizer=SpeakerMappingFinalizer(
            state, speaker_port=port, clock_ms=lambda: 500
        ),
        clock_ms=lambda: 200,
    )
    pump.enqueue_next("daily-user", coordinator=coordinator)

    result = coordinator.run_next("daily-user")

    assert result is not None and result.status == "completed"
    assert '"old_name":null' in agent.primary_prompts[0]
    assert "never infer an identity" in agent.primary_prompts[0]
    memory = state.working_memory_snapshot("daily-user")
    assert memory.version == 1
    assert "preserve Unknown attribution" in memory.markdown
    inspection = state.inspect_agent_run(
        user_id="daily-user", logical_run_id=result.logical_run_id
    )
    assert inspection.model == "gpt-5.6-sol"
    assert inspection.stop_hook_outcome == "committed"
    assert inspection.tool_discoveries == (
        "tool_search",
        INSPECT_ACTIVE_MAPPING_TOOL_NAME,
    )
    history = json.loads(
        SpeakerMappingInspectionToolBinding(state=state, user_id="daily-user").inspect({
            "event_id": "speaker-event-1"
        })
    )
    assert history["ok"] is True
    assert history["mapping"]["event"]["old_name"] is None
    assert json.loads(binding.inspect({}))["error_code"] == (
        "no_active_speaker_mapping"
    )


class _TranscriptInput:
    def prepare(self, context, *, lease):
        del context, lease
        return {"eligible": True}


class _TranscriptRuntime:
    def __init__(self) -> None:
        self.runs: list[str] = []

    def invoke(self, context, *, tools, control):
        del tools, control
        self.runs.append(str(context.tick.payload.logical_run_id))
        return InvocationOutcome.no_action()


class _TranscriptFinalizer:
    def __init__(self, state: DurableRunState) -> None:
        self.state = state

    def resume_pending(self, user_id):
        del user_id
        return None

    def finalize(self, context, outcome, *, lease):
        del outcome
        self.state.complete(
            user_id=lease.user_id,
            logical_run_id=lease.logical_run_id,
            owner=lease.owner,
            attempt_id=lease.attempt_id,
            lease_token=lease.lease_token,
            now_ms=500,
        )
        return RunFinalizationResult(
            tick_id=str(context.tick.payload.tick_id),
            logical_run_id=lease.logical_run_id,
            attempt_ordinal=lease.attempt_ordinal,
            status="completed",
        )


def _transcript_tick() -> Tick:
    return Tick.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "tick_id": "transcript-tick-1",
        "user_id": "daily-user",
        "logical_run_id": "transcript-run-1",
        "kind": "p1_transcript",
        "priority": "p1",
        "occurred_at_ms": 110,
        "received_at_ms": 110,
        "queued_at_ms": 210,
        "source_ref": {"kind": "transcript_availability", "id": "source-1"},
        "causation_id": None,
        "correlation_id": "transcript-tick-1",
        "attempt_ordinal": 1,
        "lease": None,
        "communication_allowance_snapshot": None,
        "payload": {
            "payload_kind": "transcript_availability",
            "reference_id": "source-1",
        },
        "extensions": {},
    })


def test_one_global_coordinator_runs_transcript_after_bad_mapping_quarantine(
    tmp_path: Path,
) -> None:
    bad = _event("speaker-event-bad", 1)
    port = _SpeakerPort([bad])
    state = DurableRunState(tmp_path / "state.sqlite3")
    speaker_pump = SpeakerMappingInputPump(
        state, speaker_port=port, clock_ms=lambda: 200
    )
    speaker_runtime = _FaultFirstEvent()
    transcript_runtime = _TranscriptRuntime()
    coordinator = RunCoordinator(
        state,
        runtime=BackgroundRuntimeDispatcher(
            p1_speaker=speaker_runtime,
            p1_transcript=transcript_runtime,
        ),
        feature_port=_NoEffects(),
        input_port=BackgroundInputDispatcher(
            p1_speaker=speaker_pump,
            p1_transcript=_TranscriptInput(),
        ),
        finalizer=BackgroundFinalizerDispatcher(
            p1_speaker=SpeakerMappingFinalizer(
                state, speaker_port=port, clock_ms=lambda: 500
            ),
            p1_transcript=_TranscriptFinalizer(state),
        ),
        clock_ms=lambda: 200,
    )
    speaker_pump.enqueue_next("daily-user", coordinator=coordinator)
    coordinator.enqueue(_transcript_tick())

    results = [coordinator.run_next("daily-user") for _ in range(4)]

    assert [result.status for result in results if result is not None] == [
        "failed_retryable",
        "failed_retryable",
        "quarantined",
        "completed",
    ]
    assert transcript_runtime.runs == ["transcript-run-1"]
    assert state.speaker_cursor("daily-user") == 1


def test_bounded_mapping_scan_runs_inside_existing_coordinator_driver() -> None:
    scanned = threading.Event()
    thread_names: list[str] = []

    class _IdleCoordinator:
        def run_next(self, user_id):
            del user_id
            return None

    coordinator = _IdleCoordinator()

    def scan(user_id, active_coordinator):
        assert user_id == "daily-user"
        assert active_coordinator is coordinator
        thread_names.append(threading.current_thread().name)
        scanned.set()

    driver = HarnessCoordinatorDriver(
        coordinator=coordinator,  # type: ignore[arg-type]
        user_id="daily-user",
        retry_delay_seconds=0,
        result_callback=lambda *_args: None,
        background_scan=scan,
        background_scan_interval_seconds=0.01,
    )
    try:
        driver.wake()
        assert scanned.wait(1)
    finally:
        driver.close()

    assert thread_names
    assert set(thread_names) == {"thine-global-run-coordinator"}
