from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
import pytest

from thine_harness.contracts.chat import (
    ChatEvent,
    FinalReplyOutbox,
    FinalReplyReceipt,
    QueueReceipt,
)
from thine_harness.contracts.control import HermesControlRequest
from thine_harness.p0_chat import (
    BackendPrivateChatClient,
    P0FinalizationArtifact,
    P0ChatController,
    P0ChatStore,
    ResolvedSubmission,
    build_p0_runtime,
)
from thine_harness.private_service import create_private_service_app
from thine_harness.private_topology import load_private_service_config
from thine_harness.runtime import (
    AgentTurnResult,
    HermesInvocationRuntime,
    InvocationEvent,
    RuntimeModelConfig,
)
from thine_harness.run_coordinator import ActiveRunLease
from thine_harness.working_memory import (
    CONFIGURED_MODEL_TOKENIZER_LIMITATION,
    WorkingMemorySnapshot,
)


NOW_MS = 1_787_644_800_000


class _BlockedBackend:
    def __init__(self) -> None:
        self.resolve_started = threading.Event()
        self.release_resolve = threading.Event()

    def resolve_submission(self, **_kwargs) -> ResolvedSubmission:
        self.resolve_started.set()
        assert self.release_resolve.wait(2)
        return ResolvedSubmission(user_message_id="user-msg-1", text="Hello Hermes")

    def record_queue_receipt(self, _receipt) -> None:
        return None

    def publish_event(self, _event) -> None:
        return None

    def persist_final_reply(self, outbox) -> FinalReplyReceipt:
        return _final_receipt(outbox)


def _final_receipt(outbox) -> FinalReplyReceipt:
    payload = outbox.payload
    return FinalReplyReceipt.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "receipt_id": f"final-reply:{payload.assistant_message_id}",
        "assistant_message_id": payload.assistant_message_id,
        "user_message_id": payload.user_message_id,
        "backend_message_id": payload.assistant_message_id,
        "idempotency_key": payload.idempotency_key,
        "resolution": "persisted_now",
        "persisted_at_ms": NOW_MS + 2,
        "extensions": {},
    })


class _CompletingSession:
    def invoke(self, request, *, emit, control) -> AgentTurnResult:
        return AgentTurnResult(final_output=f"Reply to: {request.prompt}")


class _AlwaysFailingSession:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, request, *, emit, control) -> AgentTurnResult:
        self.calls += 1
        return AgentTurnResult(
            completed=False,
            failed=True,
            failure_reason="provider unavailable",
        )


class _CountingSession(_CompletingSession):
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, request, *, emit, control) -> AgentTurnResult:
        self.calls += 1
        return super().invoke(request, emit=emit, control=control)


class _RetryingBackend:
    def __init__(self, store: P0ChatStore) -> None:
        self.store = store
        self.events: list[str] = []
        self.heartbeat_seen = threading.Event()
        self.final_delivered = threading.Event()
        self.final_published = threading.Event()
        self.final_attempts = 0

    def resolve_submission(self, **_kwargs) -> ResolvedSubmission:
        return ResolvedSubmission(user_message_id="user-msg-1", text="Hello Hermes")

    def record_queue_receipt(self, _receipt) -> None:
        return None

    def publish_event(self, event) -> None:
        kind = event.payload.kind
        self.events.append(kind)
        if kind == "heartbeat":
            self.heartbeat_seen.set()
        if kind == "final":
            self.final_published.set()

    def persist_final_reply(self, outbox) -> FinalReplyReceipt:
        self.final_attempts += 1
        pending = self.store.final_outbox_contract(
            self.store.recoverable_receipt_ids(max_attempts=3)[0]
        )
        assert pending.to_dict() == outbox.to_dict()
        assert outbox.payload.status == "pending_backend_persistence"
        if self.final_attempts < 3:
            raise OSError("backend temporarily unavailable")
        self.final_delivered.set()
        return _final_receipt(outbox)


class _HeartbeatSession:
    def __init__(self, heartbeat_seen: threading.Event) -> None:
        self.heartbeat_seen = heartbeat_seen
        self.calls = 0

    def invoke(self, request, *, emit, control) -> AgentTurnResult:
        self.calls += 1
        emit(InvocationEvent.progress("assistant_delta", "Working"))
        assert self.heartbeat_seen.wait(1)
        return AgentTurnResult(final_output=f"Reply to: {request.prompt}")


class _HookStatusRuntime(HermesInvocationRuntime):
    def __init__(self) -> None:
        super().__init__(
            session=_CompletingSession(),
            config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
        )
        self.finalized = 0

    def finalize_working_memory(self, **_kwargs) -> str:
        self.finalized += 1
        return CONFIGURED_MODEL_TOKENIZER_LIMITATION


class _FailOnceHookRuntime(HermesInvocationRuntime):
    def __init__(self, trace: list[str]) -> None:
        self.session = _CountingSession()
        super().__init__(
            session=self.session,
            config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
        )
        self.trace = trace
        self.finalized = 0

    def finalize_working_memory(self, **kwargs) -> None:
        self.finalized += 1
        self.trace.append(f"hook:{self.finalized}")
        if self.finalized == 1:
            raise RuntimeError("injected hook failure")
        kwargs["store"].mark_unchanged(
            expected_version=kwargs["current"].version,
            run_id=kwargs["run_id"],
        )


class _RecoveryOnlySession:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, request, *, emit, control) -> AgentTurnResult:
        self.calls += 1
        raise AssertionError("reply-persisted recovery must not rerun inference")


class _RecoveryMemoryRuntime(HermesInvocationRuntime):
    def __init__(
        self,
        *,
        expected_memory: WorkingMemorySnapshot,
        expected_context: tuple[dict[str, object], ...],
    ) -> None:
        self.session = _RecoveryOnlySession()
        self.expected_memory = expected_memory
        self.expected_context = expected_context
        self.finalized = threading.Event()
        super().__init__(
            session=self.session,
            config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
        )

    def finalize_working_memory(self, **kwargs) -> None:
        assert kwargs["current"] == self.expected_memory
        assert tuple(kwargs["result"].context_messages) == self.expected_context
        assert kwargs["result"].final_output == "The durable reply"
        kwargs["store"].commit(
            expected_version=kwargs["current"].version,
            markdown="# Working Memory\n\n- Answered the user's offline message.",
            token_count=11,
            run_id=kwargs["run_id"],
        )
        self.finalized.set()


class _SuccessfulBackend:
    def __init__(self) -> None:
        self.events: list[ChatEvent] = []
        self.final_published = threading.Event()
        self.persist_attempts = 0

    def resolve_submission(self, **_kwargs) -> ResolvedSubmission:
        return ResolvedSubmission(user_message_id="user-msg-1", text="Hello Hermes")

    def record_queue_receipt(self, _receipt) -> None:
        return None

    def publish_event(self, event: ChatEvent) -> None:
        self.events.append(event)
        if event.payload.kind == "final":
            self.final_published.set()

    def persist_final_reply(self, outbox: FinalReplyOutbox) -> FinalReplyReceipt:
        self.persist_attempts += 1
        return _final_receipt(outbox)


class _TracingBackend(_SuccessfulBackend):
    def __init__(self, trace: list[str]) -> None:
        super().__init__()
        self.trace = trace

    def persist_final_reply(self, outbox: FinalReplyOutbox) -> FinalReplyReceipt:
        self.trace.append("reply_persisted")
        return super().persist_final_reply(outbox)


class _TerminalFailureBackend(_SuccessfulBackend):
    def __init__(self) -> None:
        super().__init__()
        self.failed = threading.Event()

    def publish_event(self, event: ChatEvent) -> None:
        super().publish_event(event)
        if event.payload.kind == "failed":
            self.failed.set()


class _AmbiguousTerminalEventBackend(_SuccessfulBackend):
    def __init__(self) -> None:
        super().__init__()
        self.final_event_attempts = 0
        self.final_event_payloads: list[dict[str, object]] = []

    def publish_event(self, event: ChatEvent) -> None:
        if event.payload.kind == "final":
            self.final_event_attempts += 1
            self.final_event_payloads.append(event.to_dict())
            if self.final_event_attempts == 1:
                raise OSError("event callback temporarily unavailable")
        super().publish_event(event)


def _private_config():
    return load_private_service_config(
        {
            "thine_harness": {
                "private_service": {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": 8789,
                    "firebase_uid": "firebase-user-1",
                    "request_timeout_seconds": 5,
                    "credential": {
                        "env": "HERMES_CONTROL_TOKEN",
                        "file": "",
                    },
                }
            }
        },
        environ={"HERMES_CONTROL_TOKEN": "private-test-token"},
    )


def _headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer private-test-token",
        "X-Thine-Firebase-UID": "firebase-user-1",
        "X-Request-ID": "request-1",
    }


def _submit_payload() -> dict[str, object]:
    return {
        "schema_version": {"major": 1, "minor": 0},
        "request_id": "request-1",
        "operation": "submit_p0",
        "user_id": "firebase-user-1",
        "deadline_at_ms": NOW_MS + 5_000,
        "timeout_ms": 5_000,
        "idempotency_key": "user-message:user-msg-1",
        "payload_ref": "p0-submission:submission-1",
        "created_at_ms": NOW_MS,
        "extensions": {},
    }


def _wait_for_completed_outbox(store: P0ChatStore, receipt_id: str) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        outbox = store.pending_final(receipt_id)
        if outbox is not None and outbox["finalization_phase"] == "completed":
            return
        time.sleep(0.01)
    raise AssertionError("P0 finalization did not complete")


def test_submit_p0_returns_a_durable_receipt_before_resolving_user_content(
    tmp_path,
) -> None:
    backend = _BlockedBackend()
    controller = P0ChatController(
        store=P0ChatStore(tmp_path / "p0-chat.sqlite3"),
        backend=backend,
        runtime=HermesInvocationRuntime(
            session=_CompletingSession(),
            config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
        ),
        now_ms=lambda: NOW_MS,
    )
    app = create_private_service_app(_private_config(), p0_control=controller)

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/control",
                headers=_headers(),
                json=_submit_payload(),
            )

            assert response.status_code == 200
            body = response.json()
            assert body == {
                "schema_version": {"major": 1, "minor": 0},
                "request_id": "request-1",
                "operation": "submit_p0",
                "idempotency_key": "user-message:user-msg-1",
                "deadline_at_ms": NOW_MS + 5_000,
                "timeout_ms": 5_000,
                "status": "succeeded",
                "result_ref": "queue-receipt:" + body["result_ref"].split(":", 1)[-1],
                "error_code": None,
                "responded_at_ms": NOW_MS,
                "extensions": {},
            }
            assert body["result_ref"].startswith("queue-receipt:")
            receipt_response = client.post(
                "/v1/chat/queue-receipts/resolve",
                headers={**_headers(), "X-Request-ID": "resolve-receipt-1"},
                json={
                    "user_id": "firebase-user-1",
                    "result_ref": body["result_ref"],
                },
            )
            assert receipt_response.status_code == 200
            assert receipt_response.json() == {
                "schema_version": {"major": 1, "minor": 0},
                "receipt_id": body["result_ref"].removeprefix("queue-receipt:"),
                "user_id": "firebase-user-1",
                "user_message_id": "user-msg-1",
                "idempotency_key": "user-message:user-msg-1",
                "logical_run_id": receipt_response.json()["logical_run_id"],
                "tick_id": receipt_response.json()["tick_id"],
                "resolution": "enqueued_now",
                "enqueued_at_ms": NOW_MS,
                "extensions": {},
            }
            assert backend.resolve_started.wait(1)
    finally:
        backend.release_resolve.set()
        controller.close()


@pytest.mark.parametrize(
    ("payload_update", "header_update", "status", "error_code"),
    [
        ({"request_id": "another-request"}, {}, "rejected", "request_id_mismatch"),
        ({"user_id": "another-user"}, {}, "rejected", "user_id_mismatch"),
        (
            {"created_at_ms": NOW_MS - 1, "deadline_at_ms": NOW_MS, "timeout_ms": 1},
            {},
            "timed_out",
            "deadline_expired",
        ),
        ({"operation": "append_interactions"}, {}, "rejected", "unsupported_operation"),
        (
            {"payload_ref": "submission-1"},
            {},
            "rejected",
            "invalid_payload_ref",
        ),
        (
            {"idempotency_key": "not-a-user-message"},
            {},
            "rejected",
            "invalid_idempotency_key",
        ),
    ],
)
def test_submit_p0_fails_closed_on_identity_deadline_and_semantic_ref_drift(
    tmp_path,
    payload_update,
    header_update,
    status,
    error_code,
) -> None:
    backend = _BlockedBackend()
    controller = P0ChatController(
        store=P0ChatStore(tmp_path / "p0-chat.sqlite3"),
        backend=backend,
        runtime=HermesInvocationRuntime(
            session=_CompletingSession(),
            config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
        ),
        now_ms=lambda: NOW_MS,
    )
    payload = {**_submit_payload(), **payload_update}
    headers = {**_headers(), **header_update}

    try:
        with TestClient(
            create_private_service_app(_private_config(), p0_control=controller)
        ) as client:
            response = client.post("/v1/control", headers=headers, json=payload)

        assert response.status_code == 200
        assert response.json()["status"] == status
        assert response.json()["error_code"] == error_code
        assert response.json()["result_ref"] is None
        assert not backend.resolve_started.is_set()
    finally:
        backend.release_resolve.set()
        controller.close()


def test_final_delivery_retries_from_the_durable_outbox_without_rerunning_inference(
    tmp_path,
) -> None:
    store = P0ChatStore(tmp_path / "p0-chat.sqlite3")
    backend = _RetryingBackend(store)
    session = _HeartbeatSession(backend.heartbeat_seen)
    controller = P0ChatController(
        store=store,
        backend=backend,
        runtime=HermesInvocationRuntime(
            session=session,
            config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
        ),
        now_ms=lambda: NOW_MS,
        heartbeat_interval_seconds=0.01,
        retry_delay_seconds=0,
    )

    try:
        with TestClient(
            create_private_service_app(_private_config(), p0_control=controller)
        ) as client:
            response = client.post(
                "/v1/control",
                headers=_headers(),
                json=_submit_payload(),
            )
            assert response.status_code == 200
            assert backend.final_delivered.wait(2)
            receipt_id = response.json()["result_ref"].removeprefix("queue-receipt:")
            _wait_for_completed_outbox(store, receipt_id)

        assert session.calls == 1
        assert backend.final_attempts == 3
        assert backend.events[:3] == ["accepted", "started", "assistant_delta"]
        assert "heartbeat" in backend.events
        assert backend.events[-1] == "final"
        assert [
            (item.ordinal, item.status)
            for item in store.run_state.diagnostics("firebase-user-1").attempts
        ] == [(1, "succeeded")]
    finally:
        controller.close()


def test_ambiguous_terminal_marker_replays_exactly_without_reply_content(
    tmp_path,
) -> None:
    backend = _AmbiguousTerminalEventBackend()
    session = _CompletingSession()
    store = P0ChatStore(tmp_path / "p0-chat.sqlite3")
    controller = P0ChatController(
        store=store,
        backend=backend,
        runtime=HermesInvocationRuntime(
            session=session,
            config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
        ),
        now_ms=lambda: NOW_MS,
        retry_delay_seconds=0,
    )

    try:
        with TestClient(
            create_private_service_app(_private_config(), p0_control=controller)
        ) as client:
            response = client.post(
                "/v1/control",
                headers=_headers(),
                json=_submit_payload(),
            )
            assert response.status_code == 200
            receipt_id = response.json()["result_ref"].removeprefix("queue-receipt:")
            _wait_for_completed_outbox(store, receipt_id)

        assert backend.persist_attempts == 1
        assert backend.final_event_attempts == 2
        assert backend.final_event_payloads[0] == backend.final_event_payloads[1]
        marker = backend.final_event_payloads[0]
        assert marker["kind"] == "final"
        assert marker["safe_display_text"] == ""
        assert marker["ephemeral"] is False
        assert marker["assistant_message_id"] is not None
        assert marker["final_reply_receipt_id"] is not None
        assert backend.final_published.is_set()
    finally:
        controller.close()


def test_provider_failure_retries_three_times_then_exposes_one_recoverable_error(
    tmp_path,
) -> None:
    backend = _TerminalFailureBackend()
    session = _AlwaysFailingSession()
    controller = P0ChatController(
        store=P0ChatStore(tmp_path / "p0-chat.sqlite3"),
        backend=backend,
        runtime=HermesInvocationRuntime(
            session=session,
            config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
        ),
        now_ms=lambda: NOW_MS,
        retry_delay_seconds=0,
        max_attempts=3,
    )

    try:
        with TestClient(
            create_private_service_app(_private_config(), p0_control=controller)
        ) as client:
            response = client.post(
                "/v1/control",
                headers=_headers(),
                json=_submit_payload(),
            )
            assert response.status_code == 200
            assert backend.failed.wait(2)

        assert session.calls == 3
        failures = [
            event.payload for event in backend.events if event.payload.kind == "failed"
        ]
        assert len(failures) == 1
        assert failures[0].phase == "retry_exhausted"
        assert "three attempts" in failures[0].safe_display_text
        assert backend.persist_attempts == 0
    finally:
        controller.close()


def test_stop_hook_tokenizer_gap_is_visible_but_does_not_block_final_reply(
    tmp_path,
) -> None:
    backend = _SuccessfulBackend()
    runtime = _HookStatusRuntime()
    controller = P0ChatController(
        store=P0ChatStore(tmp_path / "p0-chat.sqlite3"),
        backend=backend,
        runtime=runtime,
        now_ms=lambda: NOW_MS,
        retry_delay_seconds=0,
    )

    try:
        with TestClient(
            create_private_service_app(_private_config(), p0_control=controller)
        ) as client:
            response = client.post(
                "/v1/control",
                headers=_headers(),
                json=_submit_payload(),
            )
            assert response.status_code == 200
            assert backend.final_published.wait(2)

        assert runtime.finalized == 1
        assert backend.persist_attempts == 1
        hook_events = [
            event.payload
            for event in backend.events
            if event.payload.phase == "working_memory"
        ]
        assert len(hook_events) == 1
        assert hook_events[0].kind == "safe_status"
        assert hook_events[0].safe_display_text == CONFIGURED_MODEL_TOKENIZER_LIMITATION
        assert backend.events[-1].payload.kind == "final"
    finally:
        controller.close()


def test_stop_hook_failure_recovers_hook_only_after_reply_persistence(
    tmp_path,
) -> None:
    trace: list[str] = []
    store = P0ChatStore(tmp_path / "run-state.sqlite3")
    backend = _TracingBackend(trace)
    runtime = _FailOnceHookRuntime(trace)
    controller = P0ChatController(
        store=store,
        backend=backend,
        runtime=runtime,
        now_ms=lambda: NOW_MS,
        retry_delay_seconds=0,
    )

    try:
        with TestClient(
            create_private_service_app(_private_config(), p0_control=controller)
        ) as client:
            response = client.post(
                "/v1/control",
                headers=_headers(),
                json=_submit_payload(),
            )
            assert response.status_code == 200
            receipt_id = response.json()["result_ref"].removeprefix("queue-receipt:")
            _wait_for_completed_outbox(store, receipt_id)

        assert runtime.session.calls == 1
        assert runtime.finalized == 2
        assert trace == ["reply_persisted", "hook:1", "hook:2"]
        event_sequences = [
            int(event.payload.event_id.rsplit(":", 1)[1]) for event in backend.events
        ]
        assert event_sequences == sorted(event_sequences)
        final_events = [
            event for event in backend.events if event.payload.kind == "final"
        ]
        assert len(final_events) == 1
        assert final_events[0].payload.safe_display_text == ""
        attempts = store.run_state.diagnostics("firebase-user-1").attempts
        assert [(item.ordinal, item.status) for item in attempts] == [
            (1, "failed_fault"),
            (2, "succeeded"),
        ]
        receipt = store.receipt_for_run(
            user_id="firebase-user-1",
            logical_run_id=attempts[0].logical_run_id,
        )
        outbox = store.pending_final(receipt.receipt_id)
        assert outbox is not None
        assert outbox["finalization_phase"] == "completed"
    finally:
        controller.close()


def test_restart_after_persisted_reply_recovers_memory_without_resending_final(
    tmp_path,
) -> None:
    path = tmp_path / "run-state.sqlite3"
    first_store = P0ChatStore(path)
    backend = _SuccessfulBackend()
    receipt = first_store.admit(
        user_id="firebase-user-1",
        user_message_id="user-msg-1",
        idempotency_key="user-message:user-msg-1",
        submission_ref="p0-submission:submission-1",
        now_ms=NOW_MS,
    )
    leased = first_store.run_state.lease_next(
        "firebase-user-1", owner="crashed-process", now_ms=NOW_MS
    )
    assert leased is not None
    first_store.run_state.mark_inference_started(
        user_id="firebase-user-1",
        logical_run_id=receipt.logical_run_id,
        owner="crashed-process",
        attempt_id=leased.attempt_id,
        lease_token=leased.lease_token,
        now_ms=NOW_MS,
    )
    current_memory = first_store.working_memory_snapshot_for_user("firebase-user-1")
    durable_context: tuple[dict[str, object], ...] = (
        {"role": "user", "content": "Hello Hermes"},
        {"role": "assistant", "content": "The durable reply"},
    )
    first_store.persist_final(
        receipt=receipt,
        lease=ActiveRunLease(
            user_id="firebase-user-1",
            logical_run_id=receipt.logical_run_id,
            owner="crashed-process",
            attempt_id=leased.attempt_id,
            attempt_ordinal=leased.attempt_ordinal,
            lease_token=leased.lease_token,
        ),
        assistant_message_id="assistant-message-1",
        text="The durable reply",
        terminal_sequence=0,
        artifact=P0FinalizationArtifact(
            context_messages=durable_context,
            cache_identity=None,
            current_memory=current_memory,
        ),
        now_ms=NOW_MS,
    )
    backend_receipt = backend.persist_final_reply(
        first_store.final_outbox_contract(receipt.receipt_id)
    )
    first_store.mark_final_persisted(
        receipt_id=receipt.receipt_id,
        backend_receipt=backend_receipt,
        now_ms=NOW_MS + 1,
    )
    first_store.mark_memory_finalization_pending(receipt.receipt_id, now_ms=NOW_MS + 2)

    recovered_runtime = _RecoveryMemoryRuntime(
        expected_memory=current_memory,
        expected_context=durable_context,
    )
    recovered_store = P0ChatStore(path)
    recovered = P0ChatController(
        store=recovered_store,
        backend=backend,
        runtime=recovered_runtime,
        now_ms=lambda: NOW_MS + 3,
        retry_delay_seconds=0,
    )
    try:
        assert recovered_runtime.finalized.wait(2)
        _wait_for_completed_outbox(recovered_store, receipt.receipt_id)

        assert recovered_runtime.session.calls == 0
        assert backend.persist_attempts == 1
        assert backend.final_published.is_set()
        final_events = [
            event for event in backend.events if event.payload.kind == "final"
        ]
        assert len(final_events) == 1
        marker = final_events[0].payload
        assert marker.safe_display_text == ""
        assert marker.assistant_message_id == "assistant-message-1"
        assert marker.final_reply_receipt_id == backend_receipt.payload.receipt_id
        assert marker.ephemeral is False
        outbox = recovered_store.pending_final(receipt.receipt_id)
        assert outbox is not None
        assert outbox["finalization_phase"] == "completed"
        memory = recovered_store.working_memory_snapshot_for_user("firebase-user-1")
        assert memory.version == current_memory.version + 1
        assert memory.markdown == (
            "# Working Memory\n\n- Answered the user's offline message."
        )
        assert memory.token_count == 11
        attempts = recovered_store.run_state.diagnostics("firebase-user-1").attempts
        assert [(item.ordinal, item.status) for item in attempts] == [(1, "succeeded")]
    finally:
        recovered.close()


def test_backend_private_adapter_sends_only_frozen_receipts_events_and_outbox_refs() -> (
    None
):
    observed: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        observed.append((request.url.path, body))
        assert request.headers["Authorization"] == "Bearer backend-test-token"
        assert request.headers["X-Thine-Firebase-UID"] == "firebase-user-1"
        if request.url.path.endswith("/resolve"):
            return httpx.Response(
                200,
                json={
                    "submission_ref": "p0-submission:submission-1",
                    "user_message_id": "user-msg-1",
                    "text": "Hello Hermes",
                },
            )
        if request.url.path.endswith("/final"):
            return httpx.Response(200, json=_final_receipt(outbox).to_dict())
        return httpx.Response(204)

    queue_receipt = QueueReceipt.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "receipt_id": "receipt-1",
        "user_id": "firebase-user-1",
        "user_message_id": "user-msg-1",
        "idempotency_key": "user-message:user-msg-1",
        "logical_run_id": "logical-run-1",
        "tick_id": "tick-1",
        "resolution": "enqueued_now",
        "enqueued_at_ms": NOW_MS,
        "extensions": {},
    })
    event = ChatEvent.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "event_id": "event-1",
        "stream_id": "stream-1",
        "step_id": None,
        "user_message_id": "user-msg-1",
        "assistant_message_id": None,
        "final_reply_receipt_id": None,
        "kind": "accepted",
        "phase": "queue",
        "safe_display_text": "Accepted",
        "ephemeral": True,
        "origin": "user_initiated_chat",
        "emitted_at_ms": NOW_MS,
        "heartbeat_max_silence_ms": 5000,
        "extensions": {},
    })
    outbox = FinalReplyOutbox.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "outbox_id": "outbox-1",
        "assistant_message_id": "assistant-1",
        "user_message_id": "user-msg-1",
        "idempotency_key": "assistant-message:assistant-1",
        "logical_run_id": "logical-run-1",
        "content_ref": "assistant-content:assistant-1",
        "status": "pending_backend_persistence",
        "backend_receipt_id": None,
        "created_at_ms": NOW_MS,
        "updated_at_ms": NOW_MS,
        "extensions": {},
    })
    client = BackendPrivateChatClient(
        origin="http://127.0.0.1:8790",
        credential="backend-test-token",
        firebase_uid="firebase-user-1",
        transport=httpx.MockTransport(handler),
    )

    try:
        resolved = client.resolve_submission(
            user_id="firebase-user-1",
            submission_ref="p0-submission:submission-1",
        )
        client.record_queue_receipt(queue_receipt)
        client.publish_event(event)
        final_receipt = client.persist_final_reply(outbox)
    finally:
        client.close()

    assert resolved == ResolvedSubmission("user-msg-1", "Hello Hermes")
    assert final_receipt.payload.assistant_message_id == "assistant-1"
    assert observed == [
        (
            "/v1/chat/submissions/resolve",
            {
                "user_id": "firebase-user-1",
                "submission_ref": "p0-submission:submission-1",
            },
        ),
        ("/v1/chat/submissions/receipt", queue_receipt.to_dict()),
        ("/v1/chat/submissions/events", event.to_dict()),
        ("/v1/chat/submissions/final", outbox.to_dict()),
    ]
    assert "Hello Hermes" not in repr(observed[1:])


def test_product_runtime_is_one_long_lived_gpt_5_6_sol_medium_agent_without_proof_limits() -> (
    None
):
    constructed: list[dict[str, object]] = []

    def agent_factory(**kwargs):
        constructed.append(kwargs)
        return SimpleNamespace(
            provider="openai-codex",
            model="gpt-5.6-sol",
            api_mode="codex_responses",
            reasoning_config={"enabled": True, "effort": "medium"},
            context_compressor=SimpleNamespace(context_length=272_000),
            skip_memory=True,
            skip_background_review=True,
            _fallback_chain=[],
            _fallback_model=None,
        )

    first = build_p0_runtime(
        firebase_uid="firebase-user-1",
        token_loader=lambda: {"access_token": "codex-test-token"},
        agent_factory=agent_factory,
    )

    assert first.diagnostics().as_dict() == {
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "api_mode": "codex_responses",
        "reasoning_effort": "medium",
        "context_window_tokens": 272_000,
    }
    assert len(constructed) == 1
    arguments = constructed[0]
    assert arguments["session_id"] == "thine-p0:firebase-user-1"
    assert arguments["fallback_model"] is None
    assert arguments["skip_memory"] is True
    assert arguments["skip_background_review"] is True
    assert arguments["pass_session_id"] is True
    assert arguments["enabled_toolsets"] == ["local-thine-transcripts"]
    assert "max_iterations" not in arguments
    assert "max_tokens" not in arguments
    assert "proof" not in str(arguments["ephemeral_system_prompt"]).lower()
    assert (
        "never send a notification" in str(arguments["ephemeral_system_prompt"]).lower()
    )


def test_p0_admission_is_the_same_durable_tick_owned_by_the_global_coordinator(
    tmp_path,
) -> None:
    store = P0ChatStore(tmp_path / "run-state.sqlite3")
    backend = _BlockedBackend()
    controller = P0ChatController(
        store=store,
        backend=backend,
        runtime=HermesInvocationRuntime(
            session=_CompletingSession(),
            config=RuntimeModelConfig.openai_gpt_5_6_sol_medium(),
        ),
        now_ms=lambda: NOW_MS,
    )

    try:
        response = controller.admit(
            HermesControlRequest.from_dict(_submit_payload()),
            authenticated_user_id="firebase-user-1",
            transport_request_id="request-1",
        )

        receipt = controller.resolve_queue_receipt(
            user_id="firebase-user-1",
            result_ref=str(response.payload.result_ref),
        )
        queued = store.run_state.diagnostics("firebase-user-1").queue
        assert [
            (item.tick_id, item.logical_run_id, item.kind, item.state)
            for item in queued
        ] == [
            (
                receipt.payload.tick_id,
                receipt.payload.logical_run_id,
                "p0_user_chat",
                "queued",
            )
        ]
        assert not hasattr(controller, "_worker")
    finally:
        backend.release_resolve.set()
        controller.close()


def test_duplicate_p0_admission_reuses_the_original_tick_without_time_drift(
    tmp_path,
) -> None:
    store = P0ChatStore(tmp_path / "run-state.sqlite3")

    first = store.admit(
        user_id="firebase-user-1",
        user_message_id="user-msg-1",
        idempotency_key="user-message:user-msg-1",
        submission_ref="p0-submission:submission-1",
        now_ms=NOW_MS,
    )
    replay = store.admit(
        user_id="firebase-user-1",
        user_message_id="user-msg-1",
        idempotency_key="user-message:user-msg-1",
        submission_ref="p0-submission:submission-1",
        now_ms=NOW_MS + 60_000,
    )

    assert replay == first
    queued = store.run_state.diagnostics("firebase-user-1").queue
    assert len(queued) == 1
