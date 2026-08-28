"""Durable single-flight delivery for user-initiated Thine chat turns."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import queue
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

import httpx

from .contracts.chat import ChatEvent, FinalReplyOutbox, FinalReplyReceipt, QueueReceipt
from .contracts.control import HermesControlRequest, HermesControlResponse
from .runtime import (
    AgentTurnResult,
    HermesAIAgentSession,
    HermesInvocationRuntime,
    InvocationEvent,
    InvocationEventKind,
    InvocationKind,
    InvocationRequest,
    RuntimeModelConfig,
)
from .working_memory import (
    CONFIGURED_MODEL_TOKENIZER_LIMITATION,
    CacheIdentity,
    HermesCachedStopHookContext,
    StopHookRunner,
    WorkingMemorySnapshot,
    WorkingMemoryTokenizerUnavailable,
)


_VERSION = {"major": 1, "minor": 0}
_USER_MESSAGE_KEY_PREFIX = "user-message:"
_SUBMISSION_REF_PREFIX = "p0-submission:"
_QUEUE_RECEIPT_REF_PREFIX = "queue-receipt:"
_P0_SYSTEM_PROMPT = (
    "You are Hermes controlling the user's local Thine daily-driver. This is a "
    "user-initiated chat turn, so answer the user directly while using available "
    "tools whenever they are needed. Keep user-visible progress factual and concise. "
    "Treat Thine backend resources as authoritative, preserve prompt-cache stability, "
    "and leave durable Working Memory updates to the same-context Stop Hook."
)


@dataclass(frozen=True)
class ResolvedSubmission:
    user_message_id: str
    text: str


class BackendPrivateChatPort(Protocol):
    def resolve_submission(
        self, *, user_id: str, submission_ref: str
    ) -> ResolvedSubmission: ...

    def record_queue_receipt(self, receipt: QueueReceipt) -> None: ...

    def publish_event(self, event: ChatEvent) -> None: ...

    def persist_final_reply(self, outbox: FinalReplyOutbox) -> FinalReplyReceipt: ...


class BackendPrivateChatClient:
    """Authenticated explicit-resource adapter to the backend-private bridge."""

    def __init__(
        self,
        *,
        origin: str,
        credential: str,
        firebase_uid: str,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlparse(origin)
        try:
            address = ipaddress.ip_address(parsed.hostname or "")
        except ValueError as exc:
            raise ValueError(
                "backend private origin must use a loopback IP literal"
            ) from exc
        if (
            parsed.scheme != "http"
            or not address.is_loopback
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("backend private origin must be loopback-only HTTP")
        if not credential or not firebase_uid:
            raise ValueError("backend private credential and Firebase UID are required")
        self._credential = credential
        self._firebase_uid = firebase_uid
        self._client = httpx.Client(
            base_url=origin.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def resolve_submission(
        self, *, user_id: str, submission_ref: str
    ) -> ResolvedSubmission:
        body = self._post(
            "/v1/chat/submissions/resolve",
            {"user_id": user_id, "submission_ref": submission_ref},
        )
        if set(body) != {"submission_ref", "user_message_id", "text"} or (
            body.get("submission_ref") != submission_ref
            or not isinstance(body.get("user_message_id"), str)
            or not isinstance(body.get("text"), str)
        ):
            raise ValueError("backend returned an invalid resolved submission")
        return ResolvedSubmission(
            user_message_id=str(body["user_message_id"]),
            text=str(body["text"]),
        )

    def record_queue_receipt(self, receipt: QueueReceipt) -> None:
        self._post("/v1/chat/submissions/receipt", receipt.to_dict())

    def publish_event(self, event: ChatEvent) -> None:
        self._post("/v1/chat/submissions/events", event.to_dict())

    def persist_final_reply(self, outbox: FinalReplyOutbox) -> FinalReplyReceipt:
        body = self._post("/v1/chat/submissions/final", outbox.to_dict())
        return FinalReplyReceipt.from_dict(body)

    def _post(self, path: str, body: dict[str, object]) -> dict[str, object]:
        request_id = str(uuid.uuid4())
        response = self._client.post(
            path,
            headers={
                "Authorization": f"Bearer {self._credential}",
                "Content-Type": "application/json",
                "X-Thine-Firebase-UID": self._firebase_uid,
                "X-Request-ID": request_id,
            },
            json=body,
        )
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return {}
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("backend private response must be a JSON object")
        return payload


class ProductP0Runtime(HermesInvocationRuntime):
    """Long-lived product runtime plus its same-agent Working Memory Stop Hook."""

    def __init__(self, *, agent: Any, config: RuntimeModelConfig) -> None:
        self._agent = agent
        super().__init__(
            session=HermesAIAgentSession(agent=agent, expected=config),
            config=config,
        )

    def finalize_working_memory(
        self,
        *,
        run_id: str,
        current: WorkingMemorySnapshot,
        result: AgentTurnResult,
        store: P0ChatStore,
    ) -> str | None:
        """Run the memory-only continuation without changing the primary cache prefix."""
        from agent.prompt_cache_scope import resolve_prompt_cache_scope
        from agent.transports.codex import (
            _cache_scope_from_session_id,
            _content_cache_key,
        )

        agent = self._agent
        tools = list(getattr(agent, "tools", None) or [])
        transport = agent._get_transport()
        wire_tools = list(transport.convert_tools(tools) or [])
        system_prompt = str(getattr(agent, "_cached_system_prompt", "") or "")
        ephemeral_prompt = str(getattr(agent, "ephemeral_system_prompt", "") or "")
        if ephemeral_prompt:
            system_prompt = (system_prompt + "\n\n" + ephemeral_prompt).strip()
        cache_scope = _cache_scope_from_session_id(
            resolve_prompt_cache_scope(agent) or str(agent.session_id)
        )
        prompt_cache_key = (
            _content_cache_key(system_prompt, wire_tools, cache_scope) or cache_scope
        )
        context = HermesCachedStopHookContext(
            agent=agent,
            conversation_history=list(result.context_messages),
            cache_identity=CacheIdentity.from_request(
                session_id=str(agent.session_id),
                prompt_cache_key=prompt_cache_key,
                tools=wire_tools,
            ),
        )
        try:
            StopHookRunner().finalize(
                run_id=run_id,
                current=current,
                context=context,
                store=store,
                interrupted=result.interrupted,
            )
        except WorkingMemoryTokenizerUnavailable:
            return CONFIGURED_MODEL_TOKENIZER_LIMITATION
        return None


def build_p0_runtime(
    *,
    firebase_uid: str,
    token_loader: Callable[[], dict[str, Any] | None] | None = None,
    agent_factory: Callable[..., Any] | None = None,
) -> ProductP0Runtime:
    """Build one long-lived exact-model runtime for all local P0 chat turns."""
    from .probe import (
        CODEX_BASE_URL,
        CodexCredentialUnavailable,
        _aiagent_factory,
        _load_codex_cli_token,
    )

    resolved_token_loader = token_loader or _load_codex_cli_token
    resolved_agent_factory = agent_factory or _aiagent_factory
    credentials = resolved_token_loader()
    access_token = str((credentials or {}).get("access_token") or "")
    if not access_token:
        raise CodexCredentialUnavailable(
            "Codex CLI credential is missing, expired, or unavailable"
        )
    config = RuntimeModelConfig.openai_gpt_5_6_sol_medium()
    agent = resolved_agent_factory(
        base_url=CODEX_BASE_URL,
        api_key=access_token,
        provider=config.provider,
        requested_provider=config.provider,
        api_mode=config.api_mode,
        model=config.model,
        reasoning_config={"enabled": True, "effort": config.reasoning_effort},
        fallback_model=None,
        quiet_mode=True,
        session_id=f"thine-p0:{firebase_uid}",
        pass_session_id=True,
        platform="api",
        user_id=firebase_uid,
        ephemeral_system_prompt=_P0_SYSTEM_PROMPT,
        skip_memory=True,
        skip_background_review=True,
    )
    return ProductP0Runtime(agent=agent, config=config)


@dataclass(frozen=True)
class _QueueReceipt:
    receipt_id: str
    user_id: str
    user_message_id: str
    idempotency_key: str
    submission_ref: str
    logical_run_id: str
    tick_id: str
    enqueued_at_ms: int


class P0ChatStore:
    """Profile-local SQLite queue receipts and final reply outbox."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS p0_queue_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    user_message_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    submission_ref TEXT NOT NULL,
                    logical_run_id TEXT NOT NULL,
                    tick_id TEXT NOT NULL,
                    enqueued_at_ms INTEGER NOT NULL,
                    state TEXT NOT NULL DEFAULT 'queued',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(user_id, user_message_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS p0_final_reply_outbox (
                    outbox_id TEXT PRIMARY KEY,
                    queue_receipt_id TEXT NOT NULL UNIQUE,
                    assistant_message_id TEXT NOT NULL UNIQUE,
                    user_message_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL,
                    content_ref TEXT NOT NULL UNIQUE,
                    terminal_sequence INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    backend_receipt_id TEXT,
                    backend_message_id TEXT,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    FOREIGN KEY(queue_receipt_id) REFERENCES p0_queue_receipts(receipt_id)
                );
                CREATE TABLE IF NOT EXISTS p0_working_memory (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    version INTEGER NOT NULL,
                    markdown TEXT NOT NULL,
                    token_count INTEGER,
                    last_run_id TEXT
                );
                INSERT OR IGNORE INTO p0_working_memory (
                    singleton, version, markdown, token_count, last_run_id
                ) VALUES (1, 0, '', NULL, NULL);
                """
            )

    def working_memory_snapshot(self) -> WorkingMemorySnapshot:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT version, markdown, token_count
                FROM p0_working_memory WHERE singleton = 1
                """
            ).fetchone()
        if row is None:
            raise RuntimeError("working memory singleton is missing")
        return WorkingMemorySnapshot(
            version=int(row["version"]),
            markdown=str(row["markdown"]),
            token_count=(
                None if row["token_count"] is None else int(row["token_count"])
            ),
        )

    def commit(
        self,
        *,
        expected_version: int,
        markdown: str,
        token_count: int,
        run_id: str,
    ) -> int:
        next_version = expected_version + 1
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE p0_working_memory
                SET version = ?, markdown = ?, token_count = ?, last_run_id = ?
                WHERE singleton = 1 AND version = ?
                """,
                (next_version, markdown, token_count, run_id, expected_version),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("working memory version changed during Stop Hook")
        return next_version

    def mark_unchanged(self, *, expected_version: int, run_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE p0_working_memory SET last_run_id = ?
                WHERE singleton = 1 AND version = ?
                """,
                (run_id, expected_version),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("working memory version changed during Stop Hook")

    def admit(
        self,
        *,
        user_id: str,
        user_message_id: str,
        idempotency_key: str,
        submission_ref: str,
        now_ms: int,
    ) -> _QueueReceipt:
        identity = f"{user_id}\0{user_message_id}\0{idempotency_key}"
        receipt_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"thine-p0-receipt:{identity}"))
        logical_run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"thine-p0-run:{identity}"))
        tick_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"thine-p0-tick:{identity}"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO p0_queue_receipts (
                    receipt_id, user_id, user_message_id, idempotency_key,
                    submission_ref, logical_run_id, tick_id, enqueued_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    user_id,
                    user_message_id,
                    idempotency_key,
                    submission_ref,
                    logical_run_id,
                    tick_id,
                    now_ms,
                ),
            )
            row = connection.execute(
                """
                SELECT receipt_id, user_id, user_message_id, idempotency_key,
                       submission_ref, logical_run_id, tick_id, enqueued_at_ms
                FROM p0_queue_receipts
                WHERE user_id = ? AND user_message_id = ? AND idempotency_key = ?
                """,
                (user_id, user_message_id, idempotency_key),
            ).fetchone()
        if row is None:
            raise RuntimeError(
                "durable P0 queue receipt was not readable after admission"
            )
        return _QueueReceipt(**dict(row))

    def load_receipt(self, receipt_id: str) -> _QueueReceipt:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT receipt_id, user_id, user_message_id, idempotency_key,
                       submission_ref, logical_run_id, tick_id, enqueued_at_ms
                FROM p0_queue_receipts WHERE receipt_id = ?
                """,
                (receipt_id,),
            ).fetchone()
        if row is None:
            raise KeyError(receipt_id)
        return _QueueReceipt(**dict(row))

    def persist_final(
        self,
        *,
        receipt: _QueueReceipt,
        assistant_message_id: str,
        text: str,
        terminal_sequence: int,
        now_ms: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO p0_final_reply_outbox (
                    outbox_id, queue_receipt_id, assistant_message_id,
                    user_message_id, idempotency_key, text, content_ref, status,
                    terminal_sequence, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending_backend_persistence', ?, ?, ?)
                """,
                (
                    f"outbox:{assistant_message_id}",
                    receipt.receipt_id,
                    assistant_message_id,
                    receipt.user_message_id,
                    f"assistant-message:{assistant_message_id}",
                    text,
                    f"assistant-content:{assistant_message_id}",
                    terminal_sequence,
                    now_ms,
                    now_ms,
                ),
            )
            connection.execute(
                "UPDATE p0_queue_receipts SET state = 'inference_complete' WHERE receipt_id = ?",
                (receipt.receipt_id,),
            )

    def record_failed_attempt(self, receipt_id: str) -> int:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE p0_queue_receipts
                SET attempt_count = attempt_count + 1
                WHERE receipt_id = ?
                """,
                (receipt_id,),
            )
            row = connection.execute(
                "SELECT attempt_count FROM p0_queue_receipts WHERE receipt_id = ?",
                (receipt_id,),
            ).fetchone()
        if row is None:
            raise KeyError(receipt_id)
        return int(row["attempt_count"])

    def recoverable_receipt_ids(self, *, max_attempts: int) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT receipt_id FROM p0_queue_receipts
                WHERE state != 'delivered' AND attempt_count < ?
                ORDER BY enqueued_at_ms, receipt_id
                """,
                (max_attempts,),
            ).fetchall()
        return [str(row["receipt_id"]) for row in rows]

    def pending_final(self, receipt_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM p0_final_reply_outbox WHERE queue_receipt_id = ?",
                (receipt_id,),
            ).fetchone()

    def queue_receipt_contract(self, *, receipt_ref: str, user_id: str) -> QueueReceipt:
        receipt_id = receipt_ref.removeprefix(_QUEUE_RECEIPT_REF_PREFIX)
        if not receipt_ref.startswith(_QUEUE_RECEIPT_REF_PREFIX) or not receipt_id:
            raise ValueError("invalid queue receipt reference")
        receipt = self.load_receipt(receipt_id)
        if receipt.user_id != user_id:
            raise PermissionError("queue receipt user mismatch")
        return QueueReceipt.from_dict({
            "schema_version": _VERSION,
            "receipt_id": receipt.receipt_id,
            "user_id": receipt.user_id,
            "user_message_id": receipt.user_message_id,
            "idempotency_key": receipt.idempotency_key,
            "logical_run_id": receipt.logical_run_id,
            "tick_id": receipt.tick_id,
            "resolution": "enqueued_now",
            "enqueued_at_ms": receipt.enqueued_at_ms,
            "extensions": {},
        })

    def final_outbox_contract(self, receipt_id: str) -> FinalReplyOutbox:
        outbox = self.pending_final(receipt_id)
        if outbox is None:
            raise KeyError(receipt_id)
        receipt = self.load_receipt(receipt_id)
        return FinalReplyOutbox.from_dict({
            "schema_version": _VERSION,
            "outbox_id": str(outbox["outbox_id"]),
            "assistant_message_id": str(outbox["assistant_message_id"]),
            "user_message_id": str(outbox["user_message_id"]),
            "idempotency_key": str(outbox["idempotency_key"]),
            "logical_run_id": receipt.logical_run_id,
            "content_ref": str(outbox["content_ref"]),
            "status": str(outbox["status"]),
            "backend_receipt_id": outbox["backend_receipt_id"],
            "created_at_ms": int(outbox["created_at_ms"]),
            "updated_at_ms": int(outbox["updated_at_ms"]),
            "extensions": {},
        })

    def resolve_assistant_content(self, *, user_id: str, content_ref: str) -> str:
        if not content_ref.startswith("assistant-content:"):
            raise ValueError("invalid assistant content reference")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT o.text, r.user_id
                FROM p0_final_reply_outbox o
                JOIN p0_queue_receipts r ON r.receipt_id = o.queue_receipt_id
                WHERE o.content_ref = ?
                """,
                (content_ref,),
            ).fetchone()
        if row is None:
            raise KeyError(content_ref)
        if row["user_id"] != user_id:
            raise PermissionError("assistant content user mismatch")
        return str(row["text"])

    def mark_final_persisted(
        self,
        *,
        receipt_id: str,
        backend_receipt: FinalReplyReceipt,
        now_ms: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE p0_final_reply_outbox
                SET status = 'persisted', backend_receipt_id = ?,
                    backend_message_id = ?, updated_at_ms = ?
                WHERE queue_receipt_id = ?
                """,
                (
                    backend_receipt.payload.receipt_id,
                    backend_receipt.payload.backend_message_id,
                    now_ms,
                    receipt_id,
                ),
            )
            connection.execute(
                """
                UPDATE p0_queue_receipts SET state = 'backend_persisted'
                WHERE receipt_id = ?
                """,
                (receipt_id,),
            )

    def mark_terminal_event_delivered(self, *, receipt_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE p0_queue_receipts SET state = 'delivered' WHERE receipt_id = ?",
                (receipt_id,),
            )


class P0ChatController:
    """Admit synchronously, then resolve and invoke one P0 turn at a time."""

    def __init__(
        self,
        *,
        store: P0ChatStore,
        backend: BackendPrivateChatPort,
        runtime: HermesInvocationRuntime | None = None,
        runtime_factory: Callable[[], HermesInvocationRuntime] | None = None,
        now_ms: Callable[[], int] | None = None,
        heartbeat_interval_seconds: float = 3.0,
        retry_delay_seconds: float = 1.0,
        max_attempts: int = 3,
    ) -> None:
        if (runtime is None) == (runtime_factory is None):
            raise ValueError("configure exactly one P0 runtime or runtime factory")
        self._store = store
        self._backend = backend
        self._runtime = runtime
        self._runtime_factory = runtime_factory
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._retry_delay_seconds = retry_delay_seconds
        self._max_attempts = max_attempts
        self._activations: queue.Queue[str | None] = queue.Queue()
        self._closed = threading.Event()
        self._worker = threading.Thread(
            target=self._run,
            name="thine-p0-chat",
            daemon=True,
        )
        self._worker.start()
        for receipt_id in self._store.recoverable_receipt_ids(
            max_attempts=self._max_attempts
        ):
            self._activations.put(receipt_id)

    def admit(
        self,
        request: HermesControlRequest,
        *,
        authenticated_user_id: str,
        transport_request_id: str,
    ) -> HermesControlResponse:
        payload = request.payload
        rejection = self._rejection_reason(
            request,
            authenticated_user_id=authenticated_user_id,
            transport_request_id=transport_request_id,
        )
        if rejection is not None:
            status, error_code = rejection
            return HermesControlResponse.from_dict({
                "schema_version": _VERSION,
                "request_id": payload.request_id,
                "operation": payload.operation,
                "idempotency_key": payload.idempotency_key,
                "deadline_at_ms": payload.deadline_at_ms,
                "timeout_ms": payload.timeout_ms,
                "status": status,
                "result_ref": None,
                "error_code": error_code,
                "responded_at_ms": self._now_ms(),
                "extensions": {},
            })
        idempotency_key = payload.idempotency_key
        user_message_id = idempotency_key.removeprefix(_USER_MESSAGE_KEY_PREFIX)
        receipt = self._store.admit(
            user_id=payload.user_id,
            user_message_id=user_message_id,
            idempotency_key=idempotency_key,
            submission_ref=payload.payload_ref,
            now_ms=self._now_ms(),
        )
        return HermesControlResponse.from_dict({
            "schema_version": _VERSION,
            "request_id": payload.request_id,
            "operation": "submit_p0",
            "idempotency_key": idempotency_key,
            "deadline_at_ms": payload.deadline_at_ms,
            "timeout_ms": payload.timeout_ms,
            "status": "succeeded",
            "result_ref": _QUEUE_RECEIPT_REF_PREFIX + receipt.receipt_id,
            "error_code": None,
            "responded_at_ms": self._now_ms(),
            "extensions": {},
        })

    def _rejection_reason(
        self,
        request: HermesControlRequest,
        *,
        authenticated_user_id: str,
        transport_request_id: str,
    ) -> tuple[str, str] | None:
        payload = request.payload
        if payload.request_id != transport_request_id:
            return "rejected", "request_id_mismatch"
        if payload.user_id != authenticated_user_id:
            return "rejected", "user_id_mismatch"
        if payload.deadline_at_ms <= self._now_ms():
            return "timed_out", "deadline_expired"
        if payload.operation != "submit_p0":
            return "rejected", "unsupported_operation"
        if not payload.payload_ref.startswith(_SUBMISSION_REF_PREFIX) or not (
            payload.payload_ref.removeprefix(_SUBMISSION_REF_PREFIX)
        ):
            return "rejected", "invalid_payload_ref"
        if not payload.idempotency_key.startswith(_USER_MESSAGE_KEY_PREFIX) or not (
            payload.idempotency_key.removeprefix(_USER_MESSAGE_KEY_PREFIX)
        ):
            return "rejected", "invalid_idempotency_key"
        return None

    def activate(self, receipt_ref: str) -> None:
        self._activations.put(receipt_ref.removeprefix(_QUEUE_RECEIPT_REF_PREFIX))

    def resolve_queue_receipt(self, *, user_id: str, result_ref: str) -> QueueReceipt:
        return self._store.queue_receipt_contract(
            receipt_ref=result_ref,
            user_id=user_id,
        )

    def resolve_assistant_content(self, *, user_id: str, content_ref: str) -> str:
        return self._store.resolve_assistant_content(
            user_id=user_id,
            content_ref=content_ref,
        )

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._activations.put(None)
        self._worker.join(timeout=2)
        close_backend = getattr(self._backend, "close", None)
        if callable(close_backend):
            close_backend()

    def _run(self) -> None:
        while not self._closed.is_set():
            receipt_id = self._activations.get()
            if receipt_id is None:
                return
            receipt = self._store.load_receipt(receipt_id)
            try:
                self._process(receipt)
            except Exception:
                attempt_count = self._store.record_failed_attempt(receipt_id)
                if attempt_count < self._max_attempts and not self._closed.wait(
                    self._retry_delay_seconds
                ):
                    self._activations.put(receipt_id)
                elif attempt_count >= self._max_attempts:
                    try:
                        self._publish_terminal_failure(receipt)
                    except Exception:
                        # Three attempts are the terminal boundary. A callback
                        # outage must not restart inference beyond that limit.
                        pass

    def _process(self, receipt: _QueueReceipt) -> None:
        existing_final = self._store.pending_final(receipt.receipt_id)
        if existing_final is not None:
            self._deliver_and_publish_final(receipt, existing_final)
            return

        submission = self._backend.resolve_submission(
            user_id=receipt.user_id,
            submission_ref=receipt.submission_ref,
        )
        if submission.user_message_id != receipt.user_message_id:
            raise RuntimeError("resolved submission user_message_id mismatch")
        self._backend.record_queue_receipt(
            self._store.queue_receipt_contract(
                receipt_ref=_QUEUE_RECEIPT_REF_PREFIX + receipt.receipt_id,
                user_id=receipt.user_id,
            )
        )
        sequence = 0
        sequence_lock = threading.Lock()

        def publish_fields(*, kind: str, phase: str, text: str) -> None:
            nonlocal sequence
            with sequence_lock:
                sequence += 1
                event_sequence = sequence
            mapped_kind = _chat_event_kind(kind, phase)
            self._backend.publish_event(
                ChatEvent.from_dict({
                    "schema_version": _VERSION,
                    "event_id": f"{receipt.receipt_id}:{event_sequence}",
                    "stream_id": f"stream:{receipt.receipt_id}",
                    "step_id": None,
                    "user_message_id": receipt.user_message_id,
                    "assistant_message_id": None,
                    "final_reply_receipt_id": None,
                    "kind": mapped_kind,
                    "phase": phase[:64] or "runtime",
                    "safe_display_text": text[:1000],
                    "ephemeral": mapped_kind not in {"failed", "interrupted"},
                    "origin": "user_initiated_chat",
                    "emitted_at_ms": self._now_ms(),
                    "heartbeat_max_silence_ms": 5000,
                    "extensions": {},
                })
            )

        def publish(event: InvocationEvent) -> None:
            if event.kind is InvocationEventKind.FINAL:
                return
            if event.kind is InvocationEventKind.FAILED:
                # Provider/runtime faults are retried as one logical P0 input.
                # Expose only the terminal third failure as recoverable.
                return
            publish_fields(
                kind=event.kind.value,
                phase=event.phase,
                text=event.text,
            )

        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(self._heartbeat_interval_seconds):
                publish_fields(
                    kind="heartbeat",
                    phase="runtime",
                    text="Still working",
                )

        heartbeat_worker = threading.Thread(
            target=heartbeat,
            name=f"thine-p0-heartbeat-{receipt.receipt_id}",
            daemon=True,
        )
        heartbeat_worker.start()
        current_memory = self._store.working_memory_snapshot()
        prompt = submission.text
        if current_memory.markdown:
            prompt = (
                "Working Memory from prior Thine ticks:\n"
                + current_memory.markdown
                + "\n\nCurrent user message:\n"
                + submission.text
            )
        try:
            runtime = self._get_runtime()
            result = runtime.invoke(
                InvocationRequest(
                    logical_run_id=receipt.logical_run_id,
                    kind=InvocationKind.USER_CHAT,
                    prompt=prompt,
                ),
                emit=publish,
            )
            if result.completed and not result.failed:
                hook_status: str | None = None
                finalize_memory = getattr(runtime, "finalize_working_memory", None)
                if callable(finalize_memory):
                    try:
                        hook_status = finalize_memory(
                            run_id=receipt.logical_run_id,
                            current=current_memory,
                            result=result,
                            store=self._store,
                        )
                    except WorkingMemoryTokenizerUnavailable:
                        hook_status = CONFIGURED_MODEL_TOKENIZER_LIMITATION
                    except Exception as exc:
                        hook_status = (
                            "working memory Stop Hook failed: " + type(exc).__name__
                        )
                if hook_status:
                    try:
                        publish_fields(
                            kind="progress",
                            phase="working_memory",
                            text=hook_status,
                        )
                    except Exception:
                        # Working Memory is a hook-only concern. It must not
                        # suppress an already-completed reply.
                        pass
        finally:
            heartbeat_stop.set()
            heartbeat_worker.join(timeout=1)
        if result.failed or not result.completed or result.final_output is None:
            raise RuntimeError(
                result.failure_reason or "Hermes P0 invocation did not complete"
            )
        assistant_message_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"thine-assistant:{receipt.receipt_id}")
        )
        self._store.persist_final(
            receipt=receipt,
            assistant_message_id=assistant_message_id,
            text=result.final_output,
            terminal_sequence=sequence + 1,
            now_ms=self._now_ms(),
        )
        outbox = self._store.pending_final(receipt.receipt_id)
        assert outbox is not None
        self._deliver_and_publish_final(receipt, outbox)

    def _deliver_and_publish_final(
        self, receipt: _QueueReceipt, outbox: sqlite3.Row
    ) -> None:
        if outbox["status"] != "persisted":
            self._deliver_final(receipt, outbox)
        persisted_outbox = self._store.pending_final(receipt.receipt_id)
        if persisted_outbox is None:
            raise RuntimeError("final outbox disappeared after backend persistence")
        outbox = persisted_outbox
        sequence = int(outbox["terminal_sequence"])
        backend_receipt_id = str(outbox["backend_receipt_id"])
        self._backend.publish_event(
            ChatEvent.from_dict({
                "schema_version": _VERSION,
                "event_id": f"{receipt.receipt_id}:{sequence}",
                "stream_id": f"stream:{receipt.receipt_id}",
                "step_id": None,
                "user_message_id": receipt.user_message_id,
                "assistant_message_id": str(outbox["assistant_message_id"]),
                "final_reply_receipt_id": backend_receipt_id,
                "kind": "final",
                "phase": "final",
                "safe_display_text": str(outbox["text"])[:1000],
                "ephemeral": False,
                "origin": "user_initiated_chat",
                "emitted_at_ms": self._now_ms(),
                "heartbeat_max_silence_ms": 5000,
                "extensions": {},
            })
        )
        self._store.mark_terminal_event_delivered(receipt_id=receipt.receipt_id)

    def _publish_terminal_failure(self, receipt: _QueueReceipt) -> None:
        self._backend.publish_event(
            ChatEvent.from_dict({
                "schema_version": _VERSION,
                "event_id": f"{receipt.receipt_id}:2147483647",
                "stream_id": f"stream:{receipt.receipt_id}",
                "step_id": None,
                "user_message_id": receipt.user_message_id,
                "assistant_message_id": None,
                "final_reply_receipt_id": None,
                "kind": "failed",
                "phase": "retry_exhausted",
                "safe_display_text": (
                    "Hermes could not complete this message after three attempts. "
                    "You can retry it from chat."
                ),
                "ephemeral": False,
                "origin": "user_initiated_chat",
                "emitted_at_ms": self._now_ms(),
                "heartbeat_max_silence_ms": 5000,
                "extensions": {},
            })
        )

    def _deliver_final(self, receipt: _QueueReceipt, outbox: sqlite3.Row) -> None:
        backend_receipt = self._backend.persist_final_reply(
            self._store.final_outbox_contract(receipt.receipt_id)
        )
        if (
            backend_receipt.payload.assistant_message_id
            != str(outbox["assistant_message_id"])
            or backend_receipt.payload.user_message_id != receipt.user_message_id
            or backend_receipt.payload.idempotency_key != str(outbox["idempotency_key"])
        ):
            raise RuntimeError(
                "backend final reply receipt does not match local outbox"
            )
        self._store.mark_final_persisted(
            receipt_id=receipt.receipt_id,
            backend_receipt=backend_receipt,
            now_ms=self._now_ms(),
        )

    def _get_runtime(self) -> HermesInvocationRuntime:
        if self._runtime is None:
            assert self._runtime_factory is not None
            self._runtime = self._runtime_factory()
        return self._runtime


__all__ = [
    "BackendPrivateChatClient",
    "BackendPrivateChatPort",
    "P0ChatController",
    "P0ChatStore",
    "ProductP0Runtime",
    "ResolvedSubmission",
    "build_p0_runtime",
]


def _chat_event_kind(kind: str, phase: str) -> str:
    if kind != InvocationEventKind.PROGRESS.value:
        return kind
    if phase == "assistant_delta":
        return "assistant_delta"
    if phase.startswith("tool"):
        return "tool_progress"
    return "safe_status"
