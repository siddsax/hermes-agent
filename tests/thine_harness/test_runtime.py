from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from thine_harness.runtime import (
    AgentTurnResult,
    BackgroundCheckpoint,
    HermesInvocationRuntime,
    HermesAIAgentSession,
    InvocationEvent,
    InvocationEventKind,
    InvocationControl,
    InvocationKind,
    InvocationRequest,
    RuntimeModelConfig,
    RuntimeSelectionError,
    SAFE_BOUNDARY_RESUME_PROMPT,
)


class _CheckpointStore:
    def __init__(self) -> None:
        self.saved: list[BackgroundCheckpoint] = []

    def save(self, checkpoint: BackgroundCheckpoint) -> None:
        self.saved.append(checkpoint)

    def load(self, resume_token: str) -> BackgroundCheckpoint:
        for checkpoint in reversed(self.saved):
            if checkpoint.resume_token == resume_token:
                return checkpoint
        raise KeyError(resume_token)


class _CompletingSession:
    def invoke(self, request, *, emit, control):
        emit(InvocationEvent.progress("provider", "Thinking"))
        emit(InvocationEvent.progress("tool", "Checking recent context"))
        return AgentTurnResult(final_output="Done", context_messages=[{"role": "assistant", "content": "Done"}])


def test_user_chat_emits_ephemeral_progress_before_one_final_output():
    events: list[InvocationEvent] = []
    runtime = HermesInvocationRuntime(
        session=_CompletingSession(),
        config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
    )

    result = runtime.invoke(
        InvocationRequest(
            logical_run_id="run-user-1",
            kind=InvocationKind.USER_CHAT,
            prompt="What changed?",
        ),
        emit=events.append,
    )

    assert result.final_output == "Done"
    assert [event.kind for event in events] == [
        InvocationEventKind.ACCEPTED,
        InvocationEventKind.STARTED,
        InvocationEventKind.PROGRESS,
        InvocationEventKind.PROGRESS,
        InvocationEventKind.FINAL,
    ]
    assert all(event.ephemeral for event in events[:-1])
    assert events[-1].ephemeral is False
    assert events[-1].text == "Done"
    assert runtime.diagnostics().as_dict() == {
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "api_mode": "codex_responses",
        "reasoning_effort": "medium",
        "context_window_tokens": 272_000,
    }


class _InterruptibleBackgroundSession:
    def __init__(self) -> None:
        self.entered_provider = threading.Event()

    def invoke(self, request, *, emit, control):
        emit(InvocationEvent.progress("provider", "Waiting for the model"))
        self.entered_provider.set()
        assert control.wait_cancelled(2), "test did not deliver P0 cancellation"
        return AgentTurnResult(
            interrupted=True,
            resume_token=request.resume_token,
            context_messages=[
                *request.context_messages,
                {"role": "tool", "tool_call_id": "receipt-1", "content": "saved"},
            ],
            remaining_work="Process the remaining background input",
            completed_tool_results=[
                {
                    "tool_call_id": "receipt-1",
                    "name": "write_record",
                    "content": "saved",
                    "effect_disposition": "applied",
                }
            ],
            successful_action_receipts=[
                {"tool_call_id": "receipt-1", "status": "completed"}
            ],
            partial_visible_assistant_output="Saved the first record.",
        )


def test_background_provider_wait_can_be_cancelled_and_returns_resume_token():
    session = _InterruptibleBackgroundSession()
    checkpoints = _CheckpointStore()
    events: list[InvocationEvent] = []
    runtime = HermesInvocationRuntime(
        session=session,
        config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
        checkpoint_store=checkpoints,
        clock=lambda: "2026-08-25T12:00:00Z",
    )
    control = InvocationControl()
    result_holder: list[AgentTurnResult] = []

    worker = threading.Thread(
        target=lambda: result_holder.append(
            runtime.invoke(
                InvocationRequest(
                    logical_run_id="run-background-1",
                    kind=InvocationKind.BACKGROUND,
                    prompt="Process the next background input",
                    resume_token="checkpoint:run-background-1",
                ),
                emit=events.append,
                control=control,
            )
        )
    )
    worker.start()
    assert session.entered_provider.wait(2)

    control.cancel("p0_user_tick")
    worker.join(2)

    assert not worker.is_alive()
    assert result_holder[0].interrupted is True
    assert result_holder[0].resume_token == "checkpoint:run-background-1"
    assert [event.kind for event in events] == [
        InvocationEventKind.ACCEPTED,
        InvocationEventKind.STARTED,
        InvocationEventKind.INTERRUPTED,
    ]
    assert events[-1].text == "p0_user_tick"
    checkpoint = checkpoints.load("checkpoint:run-background-1")
    assert checkpoint.logical_run_id == "run-background-1"
    assert checkpoint.input_prompt == "Process the next background input"
    assert checkpoint.remaining_work == "Process the remaining background input"
    assert checkpoint.completed_tool_results == [
        {
            "tool_call_id": "receipt-1",
            "name": "write_record",
            "content": "saved",
            "effect_disposition": "applied",
        }
    ]
    assert checkpoint.successful_action_receipts == [
        {"tool_call_id": "receipt-1", "status": "completed"}
    ]
    assert checkpoint.partial_visible_assistant_output == "Saved the first record."
    assert checkpoint.updated_at == "2026-08-25T12:00:00Z"


def test_background_invocation_requires_a_durable_resume_token():
    runtime = HermesInvocationRuntime(
        session=_CompletingSession(),
        config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
    )

    with pytest.raises(ValueError, match="resume_token"):
        runtime.invoke(
            InvocationRequest(
                logical_run_id="run-background-without-checkpoint",
                kind=InvocationKind.BACKGROUND,
                prompt="background",
            ),
            emit=lambda event: None,
        )


class _InterruptThenResumeSession:
    def __init__(self) -> None:
        self.background_started = threading.Event()
        self.requests: list[InvocationRequest] = []

    def invoke(self, request, *, emit, control):
        self.requests.append(request)
        if request.kind is InvocationKind.USER_CHAT:
            return AgentTurnResult(final_output="P0 complete")
        if len([item for item in self.requests if item.kind is InvocationKind.BACKGROUND]) == 1:
            self.background_started.set()
            assert control.wait_cancelled(2)
            return AgentTurnResult(
                interrupted=True,
                resume_token=request.resume_token,
                context_messages=[
                    {"role": "tool", "tool_call_id": "tool-1", "content": "receipt"}
                ],
                remaining_work="Continue from tool-1",
                completed_tool_results=[
                    {
                        "tool_call_id": "tool-1",
                        "name": "write_record",
                        "content": "receipt",
                        "effect_disposition": "applied",
                    }
                ],
                successful_action_receipts=[
                    {"tool_call_id": "tool-1", "status": "completed"}
                ],
                partial_visible_assistant_output="Step one is complete.",
            )
        assert request.prompt == "Continue from tool-1"
        assert request.original_input == "Initial background work"
        assert request.completed_tool_results == [
            {
                "tool_call_id": "tool-1",
                "name": "write_record",
                "content": "receipt",
                "effect_disposition": "applied",
            }
        ]
        assert request.successful_action_receipts == [
            {"tool_call_id": "tool-1", "status": "completed"}
        ]
        assert request.partial_visible_assistant_output == "Step one is complete."
        return AgentTurnResult(final_output="Background complete")


def test_interrupted_background_run_resumes_after_p0_from_saved_safe_boundary():
    session = _InterruptThenResumeSession()
    checkpoints = _CheckpointStore()
    runtime = HermesInvocationRuntime(
        session=session,
        config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
        checkpoint_store=checkpoints,
        clock=lambda: "2026-08-25T12:00:00Z",
    )
    control = InvocationControl()
    background_results: list[AgentTurnResult] = []
    worker = threading.Thread(
        target=lambda: background_results.append(
            runtime.invoke(
                InvocationRequest(
                    "logical-run-1",
                    InvocationKind.BACKGROUND,
                    "Initial background work",
                    resume_token="resume:logical-run-1",
                ),
                emit=lambda event: None,
                control=control,
            )
        )
    )
    worker.start()
    assert session.background_started.wait(2)
    control.cancel("p0_arrived")
    worker.join(2)
    assert background_results[0].interrupted is True

    p0 = runtime.invoke(
        InvocationRequest("p0-run", InvocationKind.USER_CHAT, "User question"),
        emit=lambda event: None,
    )
    assert p0.final_output == "P0 complete"

    resumed = runtime.resume(
        "resume:logical-run-1",
        emit=lambda event: None,
    )
    assert resumed.final_output == "Background complete"
    assert session.requests[-1].logical_run_id == "logical-run-1"


class _FakeAIAgent:
    provider = "openai-codex"
    model = "gpt-5.6-sol"
    api_mode = "codex_responses"
    reasoning_config = {"enabled": True, "effort": "medium"}
    context_compressor = type("Context", (), {"context_length": 272_000})()
    skip_memory = True
    skip_background_review = True

    def __init__(self) -> None:
        self.interrupts: list[str] = []
        self._executing_tools = False

    def interrupt(self, message=None, **kwargs):
        self.interrupts.append(message)

    def run_conversation(self, prompt, **kwargs):
        kwargs["stream_callback"]("working")
        return {
            "final_response": "answer",
            "messages": [{"role": "assistant", "content": "answer"}],
            "interrupted": False,
            "input_tokens": 100,
            "output_tokens": 5,
            "cache_read_tokens": 80,
            "cache_write_tokens": 0,
        }


def test_aiagent_session_fails_closed_on_model_drift_and_maps_usage():
    expected = RuntimeModelConfig.openai_gpt_5_6_sol_medium()
    bad_agent = _FakeAIAgent()
    bad_agent.model = "gpt-5.6-terra"

    with pytest.raises(RuntimeSelectionError, match="model"):
        HermesAIAgentSession(agent=bad_agent, expected=expected)

    review_agent = _FakeAIAgent()
    review_agent.skip_background_review = False
    with pytest.raises(RuntimeSelectionError, match="background review"):
        HermesAIAgentSession(agent=review_agent, expected=expected)

    agent = _FakeAIAgent()
    events: list[InvocationEvent] = []
    runtime = HermesInvocationRuntime(
        session=HermesAIAgentSession(agent=agent, expected=expected),
        config=expected,
    )
    result = runtime.invoke(
        InvocationRequest("run-2", InvocationKind.USER_CHAT, "hello"),
        emit=events.append,
    )

    assert result.final_output == "answer"
    assert result.usage == {
        "input_tokens": 100,
        "output_tokens": 5,
        "cache_read_tokens": 80,
        "cache_write_tokens": 0,
    }
    assert events[2].phase == "assistant_delta"


class _InterruptedMappingAIAgent(_FakeAIAgent):
    def run_conversation(self, prompt, **kwargs):
        kwargs["stream_callback"]("Visible partial")
        return {
            "final_response": None,
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "applied-1",
                    "tool_name": "write_record",
                    "content": "saved",
                    "effect_disposition": "applied",
                },
                {
                    "role": "tool",
                    "tool_call_id": "failed-1",
                    "tool_name": "send_message",
                    "content": "Error executing tool",
                    "effect_disposition": "unknown",
                },
            ],
            "interrupted": True,
        }


def test_aiagent_checkpoint_mapping_separates_results_from_success_receipts():
    session = HermesAIAgentSession(
        agent=_InterruptedMappingAIAgent(),
        expected=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
    )
    result = session.invoke(
        InvocationRequest(
            "mapping",
            InvocationKind.BACKGROUND,
            "Original work",
            resume_token="resume:mapping",
        ),
        emit=lambda event: None,
        control=InvocationControl(),
    )

    assert [item["tool_call_id"] for item in result.completed_tool_results] == [
        "applied-1",
        "failed-1",
    ]
    assert result.successful_action_receipts == [
        {
            "tool_call_id": "applied-1",
            "name": "write_record",
            "status": "applied",
        }
    ]
    assert result.remaining_work == SAFE_BOUNDARY_RESUME_PROMPT
    assert result.remaining_work != "Original work"
    assert result.partial_visible_assistant_output == "Visible partial"


class _ToolBoundaryAIAgent(_FakeAIAgent):
    def __init__(self) -> None:
        super().__init__()
        self.tool_started = threading.Event()
        self.release_tool = threading.Event()
        self.interrupt_delivered = threading.Event()
        self.persisted_receipt = False

    def interrupt(self, message=None, **kwargs):
        assert self.persisted_receipt is True
        super().interrupt(message, **kwargs)
        self.interrupt_delivered.set()

    def run_conversation(self, prompt, **kwargs):
        self._executing_tools = True
        self.tool_started.set()
        assert self.release_tool.wait(2)
        self.persisted_receipt = True
        self._executing_tools = False
        assert self.interrupt_delivered.wait(2)
        return {
            "final_response": None,
            "messages": [
                {"role": "tool", "tool_call_id": "tool-safe", "content": "saved"}
            ],
            "interrupted": True,
        }


def test_aiagent_cancellation_waits_until_active_tool_result_is_persisted():
    agent = _ToolBoundaryAIAgent()
    session = HermesAIAgentSession(
        agent=agent,
        expected=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
    )
    control = InvocationControl()
    results: list[AgentTurnResult] = []
    worker = threading.Thread(
        target=lambda: results.append(
            session.invoke(
                InvocationRequest(
                    "background-safe-boundary",
                    InvocationKind.BACKGROUND,
                    "work",
                    resume_token="resume:safe-boundary",
                ),
                emit=lambda event: None,
                control=control,
            )
        )
    )
    worker.start()
    assert agent.tool_started.wait(2)
    control.cancel("p0_arrived")
    assert control.wait_safe_boundary_deferred(2)
    assert agent.interrupts == []
    agent.release_tool.set()
    worker.join(2)

    assert not worker.is_alive()
    assert results[0].interrupted is True
    assert agent.interrupts == ["p0_arrived"]


def test_completed_aiagent_turn_unbinds_its_cancellation_control():
    agent = _FakeAIAgent()
    session = HermesAIAgentSession(
        agent=agent,
        expected=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
    )
    control = InvocationControl()
    session.invoke(
        InvocationRequest("completed", InvocationKind.USER_CHAT, "hello"),
        emit=lambda event: None,
        control=control,
    )

    control.cancel("too_late")
    assert agent.interrupts == []


def test_real_aiagent_persists_tool_result_before_safe_boundary_interrupt(
    monkeypatch,
    tmp_path,
):
    from agent import model_metadata
    from hermes_state import SessionDB
    import run_agent
    from run_agent import AIAgent
    from tools.registry import registry

    tool_name = "thine_test_blocking_receipt"
    tool_started = threading.Event()
    release_tool = threading.Event()
    interrupt_observed = threading.Event()
    db_path = tmp_path / "state.db"
    session_id = "thine-real-safe-boundary"
    schema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Block until the test releases a durable receipt.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    def blocking_handler(_args, **_kwargs):
        tool_started.set()
        assert release_tool.wait(2)
        return "durable-tool-result"

    registry.register(
        name=tool_name,
        toolset="thine-test-safe-boundary",
        schema=schema,
        handler=blocking_handler,
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **_kwargs: [schema])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run_agent, "OpenAI", MagicMock())
    monkeypatch.setattr(run_agent, "_hermes_home", tmp_path)
    monkeypatch.setattr(model_metadata, "fetch_model_metadata", lambda *_args, **_kwargs: {})

    db = SessionDB(db_path=db_path)
    try:
        db.create_session(session_id=session_id, source="test", model="test/model")
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openai",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            skip_background_review=True,
            session_id=session_id,
            session_db=db,
            max_iterations=2,
        )
        agent._session_db_created = True
        agent._last_flushed_db_idx = 0
        agent._flushed_db_message_ids = set()
        agent._flushed_db_message_session_id = None
        agent._persist_disabled = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.client = MagicMock()
        tool_call = SimpleNamespace(
            id="call-safe-boundary",
            type="function",
            function=SimpleNamespace(name=tool_name, arguments="{}"),
        )
        assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
        agent.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=assistant_message,
                    finish_reason="tool_calls",
                )
            ],
            model="test/model",
            usage=None,
        )

        original_interrupt = agent.interrupt

        def assert_durable_then_interrupt(message=None, **kwargs):
            reopened = SessionDB(db_path=db_path)
            try:
                durable = reopened.get_messages_as_conversation(session_id)
            finally:
                reopened.close()
            tool_rows = [row for row in durable if row.get("role") == "tool"]
            assert len(tool_rows) == 1
            assert tool_rows[0]["content"] == "durable-tool-result"
            interrupt_observed.set()
            original_interrupt(message, **kwargs)

        agent.interrupt = assert_durable_then_interrupt
        expected = RuntimeModelConfig(
            provider=str(agent.provider),
            model=str(agent.model),
            api_mode=str(agent.api_mode),
            reasoning_effort="",
            context_window_tokens=int(agent.context_compressor.context_length),
        )
        session = HermesAIAgentSession(agent=agent, expected=expected)
        control = InvocationControl()
        results: list[AgentTurnResult] = []
        worker = threading.Thread(
            target=lambda: results.append(
                session.invoke(
                    InvocationRequest("real-safe-boundary", InvocationKind.USER_CHAT, "run"),
                    emit=lambda event: None,
                    control=control,
                )
            )
        )
        worker.start()
        assert tool_started.wait(2)
        control.cancel("p0_arrived")
        assert control.wait_safe_boundary_deferred(2)
        release_tool.set()
        assert interrupt_observed.wait(2)
        worker.join(2)

        assert not worker.is_alive()
        assert results[0].interrupted is True
        assert results[0].remaining_work == SAFE_BOUNDARY_RESUME_PROMPT
        assert results[0].completed_tool_results == [
            {
                "tool_call_id": "call-safe-boundary",
                "name": tool_name,
                "content": "durable-tool-result",
                "effect_disposition": None,
            }
        ]
        assert results[0].successful_action_receipts == []
    finally:
        db.close()
        registry.deregister(tool_name)
