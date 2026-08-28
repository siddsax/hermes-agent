"""Fixed half-hour app-interaction input for the single-flight Harness.

The phone/backend own journal ingestion.  Hermes owns the wall-clock driver,
durable claim identity, model processing lifecycle, and acknowledgement suffix.
Interaction input is deliberately never attached to another Tick kind.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import threading
import time
from typing import Any, Callable, cast, Iterator, Mapping, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import uuid

import httpx

from .contracts import JSONValue
from .contracts.interactions import (
    InteractionBatch,
    InteractionCursorConsumptionReceipt,
)
from .contracts.recovery import InputGap, QuarantineRecord
from .contracts.runtime import InputReceipt, RunFinalization, RunReceipt, Tick
from .run_coordinator import (
    ActiveRunLease,
    FakeInvocationPort,
    InvocationContext,
    InvocationControl,
    InvocationOutcome,
    RunFinalizationResult,
    RunFinalizerPort,
    RunInputPort,
)
from .run_state import (
    DurableRunState,
    DurableStateError,
    PendingInteractionAck,
    PendingInteractionQuarantine,
    StoredInteractionClaim,
)
from .runtime import (
    HermesAIAgentSession,
    InvocationControl as ProviderInvocationControl,
    InvocationKind,
    InvocationRequest,
    RuntimeModelConfig,
)
from .working_memory import (
    CacheIdentity,
    HermesCachedStopHookContext,
    StopHookOutcomeKind,
    StopHookRunner,
    WorkingMemorySnapshot,
)


_VERSION = {"major": 1, "minor": 0}
_MAX_EVENTS = 500
_HALF_HOUR_MS = 30 * 60 * 1000
INTERACTION_AGENT_TOOL_NAME = "thine_run_inspect_interaction_batch"
INTERACTION_AGENT_TOOL_SCHEMA = {
    "name": INTERACTION_AGENT_TOOL_NAME,
    "description": (
        "Inspect the exact app-wide semantic interaction range claimed by the "
        "current Logical Run, including primary-input correlations."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


class InteractionClaimNotFound(LookupError):
    """The backend has not committed the requested idempotent claim."""


@dataclass(frozen=True)
class InteractionAvailability:
    available: bool
    next_cursor: int | None


@dataclass(frozen=True)
class InteractionClaimRequest:
    claim_request_id: str
    logical_run_id: str
    boundary_start_ms: int
    boundary_end_ms: int
    max_events: int = _MAX_EVENTS


@dataclass(frozen=True)
class InteractionQuarantineRequest:
    quarantine_id: str
    logical_run_id: str
    batch_id: str
    first_cursor: int
    last_cursor: int
    failure_code: str
    fault_attempts_total: int
    quarantined_at_ms: int


@dataclass(frozen=True)
class InteractionQuarantineResult:
    quarantine_id: str
    logical_run_id: str
    batch_id: str
    first_cursor: int
    last_cursor: int
    normal_cursor_advanced: bool
    input_retained: bool


@dataclass(frozen=True)
class InteractionRetryRequest:
    quarantine_id: str
    retry_run_id: str
    retry_request_id: str
    requested_at_ms: int


@dataclass(frozen=True)
class InteractionRetryResult:
    quarantine_id: str
    retry_run_id: str
    retry_request_id: str
    batch: InteractionBatch
    normal_cursor_rewound: bool
    quarantine_retained: bool


class InteractionSourcePort(Protocol):
    """Closed helper boundary; it does not expose arbitrary database access."""

    def availability(self, *, boundary_end_ms: int) -> InteractionAvailability: ...

    def claim(self, request: InteractionClaimRequest) -> InteractionBatch: ...

    def lookup_claim(self, claim_request_id: str) -> InteractionBatch: ...

    def consume(self, receipt: InteractionCursorConsumptionReceipt) -> None: ...

    def quarantine(
        self, request: InteractionQuarantineRequest
    ) -> InteractionQuarantineResult: ...

    def retry(self, request: InteractionRetryRequest) -> InteractionRetryResult: ...


class BackendInteractionClient:
    """Authenticated client for the backend's closed interaction helpers."""

    def __init__(
        self,
        *,
        origin: str,
        credential: str,
        firebase_uid: str,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not origin or not credential or not firebase_uid:
            raise ValueError("backend origin, credential, and UID are required")
        self._credential = credential
        self._firebase_uid = firebase_uid
        self._client = httpx.Client(
            base_url=origin.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def availability(self, *, boundary_end_ms: int) -> InteractionAvailability:
        value = self._post(
            "/v1/interactions/availability", {"boundary_end_ms": boundary_end_ms}
        )
        if set(value) != {"available", "next_cursor"}:
            raise ValueError("interaction availability has an open wire shape")
        available = value["available"]
        next_cursor = value["next_cursor"]
        if not isinstance(available, bool) or not (
            next_cursor is None
            or (
                isinstance(next_cursor, int)
                and not isinstance(next_cursor, bool)
                and next_cursor > 0
            )
        ):
            raise ValueError("interaction availability fields are invalid")
        return InteractionAvailability(available=available, next_cursor=next_cursor)

    def claim(self, request: InteractionClaimRequest) -> InteractionBatch:
        value = self._post(
            "/v1/interactions/claims",
            {
                "claim_request_id": request.claim_request_id,
                "logical_run_id": request.logical_run_id,
                "boundary_start_ms": request.boundary_start_ms,
                "boundary_end_ms": request.boundary_end_ms,
                "max_events": request.max_events,
            },
        )
        return self._decode_claim(
            value,
            claim_request_id=request.claim_request_id,
            logical_run_id=request.logical_run_id,
        )

    def lookup_claim(self, claim_request_id: str) -> InteractionBatch:
        value = self._post(
            "/v1/interactions/claims/lookup",
            {"claim_request_id": claim_request_id},
        )
        if value.get("claim") is None:
            raise InteractionClaimNotFound(claim_request_id)
        return self._decode_claim(value, claim_request_id=claim_request_id)

    def consume(self, receipt: InteractionCursorConsumptionReceipt) -> None:
        value = self._post("/v1/interactions/claims/ack", receipt.to_dict())
        if (
            InteractionCursorConsumptionReceipt.from_dict(value).to_json()
            != receipt.to_json()
        ):
            raise ValueError("interaction acknowledgement changed the receipt")

    def quarantine(
        self, request: InteractionQuarantineRequest
    ) -> InteractionQuarantineResult:
        value = self._post(
            "/v1/interactions/claims/quarantine",
            {
                "logical_run_id": request.logical_run_id,
                "batch_id": request.batch_id,
                "first_cursor": request.first_cursor,
                "last_cursor": request.last_cursor,
                "quarantine_id": request.quarantine_id,
                "failure_code": request.failure_code,
                "fault_attempts_total": request.fault_attempts_total,
                "quarantined_at_ms": request.quarantined_at_ms,
            },
        )
        expected = {
            "quarantine_id",
            "claim_id",
            "logical_run_id",
            "failure_code",
            "fault_attempts_total",
            "quarantined_at_ms",
            "normal_cursor_advanced",
            "input_retained",
            "batch",
        }
        if set(value) != expected or not isinstance(value["batch"], dict):
            raise ValueError("interaction quarantine has an open wire shape")
        batch = InteractionBatch.from_dict(cast(dict[str, JSONValue], value["batch"]))
        payload = batch.payload
        return InteractionQuarantineResult(
            quarantine_id=str(value["quarantine_id"]),
            logical_run_id=str(value["logical_run_id"]),
            batch_id=str(payload.batch_id),
            first_cursor=int(payload.first_cursor),
            last_cursor=int(payload.last_cursor),
            normal_cursor_advanced=value["normal_cursor_advanced"] is True,
            input_retained=value["input_retained"] is True,
        )

    def retry(self, request: InteractionRetryRequest) -> InteractionRetryResult:
        value = self._post(
            "/v1/interactions/quarantines/retry",
            {
                "quarantine_id": request.quarantine_id,
                "retry_run_id": request.retry_run_id,
                "retry_request_id": request.retry_request_id,
                "requested_at_ms": request.requested_at_ms,
            },
        )
        batch = self._decode_claim(
            value,
            claim_request_id=request.retry_request_id,
            logical_run_id=request.retry_run_id,
            quarantine_id=request.quarantine_id,
        )
        return InteractionRetryResult(
            quarantine_id=request.quarantine_id,
            retry_run_id=request.retry_run_id,
            retry_request_id=request.retry_request_id,
            batch=batch,
            normal_cursor_rewound=False,
            quarantine_retained=True,
        )

    def inspect_quarantine(self, quarantine_id: str) -> dict[str, JSONValue]:
        return self._post(
            "/v1/interactions/quarantines/inspect",
            {"quarantine_id": quarantine_id},
        )

    @staticmethod
    def _decode_claim(
        value: dict[str, JSONValue],
        *,
        claim_request_id: str,
        logical_run_id: str | None = None,
        quarantine_id: str | None = None,
    ) -> InteractionBatch:
        wrapper = value.get("claim", value)
        if not isinstance(wrapper, dict):
            raise InteractionClaimNotFound(claim_request_id)
        expected = {
            "claim_id",
            "claim_request_id",
            "logical_run_id",
            "claim_kind",
            "retry_of_quarantine_id",
            "batch",
        }
        if set(wrapper) != expected or not isinstance(wrapper["batch"], dict):
            raise ValueError("interaction claim has an open wire shape")
        if wrapper["claim_request_id"] != claim_request_id:
            raise ValueError("interaction claim request identity mismatch")
        if logical_run_id is not None and wrapper["logical_run_id"] != logical_run_id:
            raise ValueError("interaction claim run identity mismatch")
        expected_kind = "quarantine_retry" if quarantine_id is not None else "normal"
        if (
            wrapper["claim_kind"] != expected_kind
            or wrapper["retry_of_quarantine_id"] != quarantine_id
        ):
            raise ValueError("interaction claim recovery identity mismatch")
        return InteractionBatch.from_dict(cast(dict[str, JSONValue], wrapper["batch"]))

    def _post(self, path: str, body: dict[str, JSONValue]) -> dict[str, JSONValue]:
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
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("backend interaction response must be an object")
        return cast(dict[str, JSONValue], payload)


@dataclass(frozen=True)
class PreparedInteractionInput:
    batch: InteractionBatch
    boundary_end_ms: int
    explicit_retry: InteractionRetryRequest | None = None
    input_gaps: tuple[InputGap, ...] = ()


class InteractionBatchToolBinding:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: PreparedInteractionInput | None = None

    @contextmanager
    def activate(self, prepared: PreparedInteractionInput) -> Iterator[None]:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("another interaction range is already active")
            self._active = prepared
        try:
            yield
        finally:
            with self._lock:
                self._active = None

    def inspect(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if args:
            return json.dumps({"ok": False, "error_code": "unexpected_arguments"})
        with self._lock:
            prepared = self._active
        if prepared is None:
            return json.dumps({
                "ok": False,
                "error_code": "no_active_interaction_range",
            })
        return json.dumps(
            {
                "ok": True,
                "boundary_end_ms": prepared.boundary_end_ms,
                "batch": prepared.batch.to_dict(),
                "explicit_retry": (
                    None
                    if prepared.explicit_retry is None
                    else asdict(prepared.explicit_retry)
                ),
                "input_gaps": [gap.to_dict() for gap in prepared.input_gaps],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def register(self) -> None:
        from tools.registry import registry
        from .transcript_agent import TRANSCRIPT_AGENT_TOOLSET

        registry.register(
            name=INTERACTION_AGENT_TOOL_NAME,
            toolset=TRANSCRIPT_AGENT_TOOLSET,
            schema=INTERACTION_AGENT_TOOL_SCHEMA,
            handler=self.inspect,
            scope=registry.current_scope_key(),
        )


@dataclass(frozen=True)
class InteractionAgentArtifact:
    agent: Any
    result: Any
    current_memory: WorkingMemorySnapshot
    cache_identity: CacheIdentity
    provider: str
    model: str
    api_mode: str
    reasoning_effort: str
    tool_discoveries: tuple[str, ...]


class RealInteractionAgentRuntime:
    """Run interaction Ticks through the same long-lived GPT agent/session cache."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        agent: Any,
        binding: InteractionBatchToolBinding,
        config: RuntimeModelConfig | None = None,
    ) -> None:
        self._state = state
        self._agent = agent
        self._binding = binding
        self._config = config or RuntimeModelConfig.openai_gpt_5_6_sol_medium()
        self._session = HermesAIAgentSession(agent=agent, expected=self._config)

    def invoke(
        self,
        context: InvocationContext,
        *,
        tools: object,
        control: InvocationControl,
    ) -> InvocationOutcome:
        del tools
        if control.preemption_requested:
            return InvocationOutcome.preempted(
                remaining_work="interaction inference not started"
            )
        prepared = context.prepared_input
        if not isinstance(prepared, PreparedInteractionInput):
            raise ValueError("real interaction inference requires one claimed range")
        current = self._state.working_memory_snapshot(str(context.tick.payload.user_id))
        prompt = (
            "Process this app-wide semantic interaction Tick as historical context. "
            "It is separate from transcript and speaker inputs. Discover the "
            f"{INTERACTION_AGENT_TOOL_NAME} helper through tool_search if details "
            "are needed. Events with primary_correlation were already delivered and "
            "must not be treated as a second command. Choose tools or no tools; a "
            "no-action outcome is valid.\n\n<working_memory>\n"
            + current.markdown
            + "\n</working_memory>\n"
            + "<input_gaps>\n"
            + json.dumps(
                [gap.to_dict() for gap in prepared.input_gaps],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n</input_gaps>\n"
            + f"<logical_run_id>{context.tick.payload.logical_run_id}</logical_run_id>"
        )
        provider_control = ProviderInvocationControl()
        stopped = threading.Event()

        def relay() -> None:
            while not stopped.wait(0.02):
                if control.preemption_requested:
                    provider_control.cancel(control.reason or "p0_user_tick")
                    return

        thread = threading.Thread(
            target=relay, name="thine-interaction-preemption", daemon=True
        )
        thread.start()
        try:
            with self._binding.activate(prepared):
                result = self._session.invoke(
                    InvocationRequest(
                        logical_run_id=str(context.tick.payload.logical_run_id),
                        kind=InvocationKind.BACKGROUND,
                        prompt=prompt,
                        resume_token=str(context.tick.payload.logical_run_id),
                        original_input=prompt,
                    ),
                    emit=lambda _event: None,
                    control=provider_control,
                )
        finally:
            stopped.set()
            thread.join(timeout=0.1)
        if result.interrupted:
            return InvocationOutcome.preempted(
                remaining_work=result.remaining_work or "resume interaction inference"
            )
        if result.failed or not result.completed:
            return InvocationOutcome.fault(
                result.failure_reason or "real_model_incomplete"
            )
        from .transcript_agent import _cache_identity

        artifact = InteractionAgentArtifact(
            agent=self._agent,
            result=result,
            current_memory=current,
            cache_identity=_cache_identity(self._agent),
            provider=self._config.provider,
            model=self._config.model,
            api_mode=self._config.api_mode,
            reasoning_effort=self._config.reasoning_effort,
            tool_discoveries=tuple(
                str(message.get("tool_name") or message.get("name") or "")
                for message in result.context_messages
                if isinstance(message, dict) and message.get("role") == "tool"
            ),
        )
        return InvocationOutcome.no_action(finalization_context=artifact)


def latest_half_hour_boundary_ms(now_ms: int, timezone_name: str) -> int:
    """Return the latest passed :00/:30 local boundary as a UTC instant."""
    if now_ms < 0:
        raise ValueError("now_ms must be non-negative")
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("timezone_name must be a configured IANA timezone") from exc
    local = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).astimezone(zone)
    minute = 30 if local.minute >= 30 else 0
    boundary = local.replace(minute=minute, second=0, microsecond=0)
    return int(boundary.astimezone(timezone.utc).timestamp() * 1000)


def next_half_hour_boundary_ms(now_ms: int, timezone_name: str) -> int:
    """Return the next real UTC instant whose local clock is :00 or :30."""
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("timezone_name must be a configured IANA timezone") from exc
    candidate_ms = ((now_ms // 60_000) + 1) * 60_000
    # A timezone transition cannot remove every :00/:30 boundary for three hours.
    for _ in range(180):
        local = datetime.fromtimestamp(candidate_ms / 1000, tz=timezone.utc).astimezone(
            zone
        )
        if local.minute in (0, 30) and local.second == 0:
            return candidate_ms
        candidate_ms += 60_000
    raise RuntimeError("could not resolve the next local half-hour boundary")


class InteractionInputPump:
    """Scan fixed boundaries, enqueue a non-empty P1 Tick, and claim after lease."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        source: InteractionSourcePort,
        timezone_name: str = "Asia/Kolkata",
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        # Validate once so a typo cannot silently alter the clock contract.
        latest_half_hour_boundary_ms(0, timezone_name)
        self._state = state
        self._source = source
        self._timezone_name = timezone_name
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    @property
    def timezone_name(self) -> str:
        return self._timezone_name

    def scan_due(self, *, user_id: str) -> str | None:
        """Advance one observed boundary and append at most one non-empty Tick."""
        if not user_id:
            raise ValueError("user_id is required")
        now_ms = self._clock_ms()
        boundary_end_ms = latest_half_hour_boundary_ms(now_ms, self._timezone_name)
        with self._state._connect() as connection:
            clock = connection.execute(
                "SELECT * FROM interaction_clock_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if clock is not None:
                if str(clock["timezone_name"]) != self._timezone_name:
                    raise DurableStateError(
                        "interaction clock timezone cannot change implicitly"
                    )
                if boundary_end_ms <= int(clock["last_boundary_ms"]):
                    return None
            outstanding = connection.execute(
                """
                SELECT tick_id FROM queue_items
                WHERE user_id = ? AND kind = 'p1_interaction'
                  AND state IN ('queued', 'running', 'awaiting_interaction_ack',
                                'quarantine_pending')
                ORDER BY enqueue_sequence LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        if outstanding is not None:
            # Do not consume this boundary.  A later scan catches up through the
            # newest passed boundary after the outstanding range is finalized.
            return None
        availability = self._source.availability(boundary_end_ms=boundary_end_ms)
        identity = f"{user_id}\0{boundary_end_ms}"
        tick_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"thine-interaction-tick:{identity}")
        )
        logical_run_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"thine-interaction-run:{identity}")
        )
        source_id = f"interaction-window:{boundary_end_ms}"
        with self._state._transaction() as connection:
            existing_clock = connection.execute(
                "SELECT * FROM interaction_clock_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if existing_clock is not None and boundary_end_ms <= int(
                existing_clock["last_boundary_ms"]
            ):
                return None
            raced = connection.execute(
                """
                SELECT 1 FROM queue_items
                WHERE user_id = ? AND kind = 'p1_interaction'
                  AND state IN ('queued', 'running', 'awaiting_interaction_ack',
                                'quarantine_pending')
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if raced is not None:
                return None
            if availability.available:
                tick = Tick.from_dict({
                    "schema_version": _VERSION,
                    "tick_id": tick_id,
                    "user_id": user_id,
                    "logical_run_id": logical_run_id,
                    "kind": "p1_interaction",
                    "priority": "p1",
                    "occurred_at_ms": boundary_end_ms,
                    "received_at_ms": now_ms,
                    "queued_at_ms": now_ms,
                    "source_ref": {"kind": "interaction_window", "id": source_id},
                    "causation_id": None,
                    "correlation_id": tick_id,
                    "attempt_ordinal": 1,
                    "lease": None,
                    "communication_allowance_snapshot": None,
                    "payload": {
                        "payload_kind": "interaction_window",
                        "reference_id": source_id,
                    },
                    "extensions": {},
                })
                self._state._insert_tick_locked(connection, tick=tick, now_ms=now_ms)
            connection.execute(
                """
                INSERT INTO interaction_clock_state (
                    user_id, timezone_name, last_boundary_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    last_boundary_ms = excluded.last_boundary_ms,
                    updated_at_ms = excluded.updated_at_ms
                """,
                (user_id, self._timezone_name, boundary_end_ms, now_ms),
            )
        return tick_id if availability.available else None

    def prepare(
        self, context: InvocationContext, *, lease: ActiveRunLease
    ) -> PreparedInteractionInput | None:
        payload = context.tick.payload
        if payload.kind != "p1_interaction":
            return None
        source_id = str(payload.source_ref.id)
        try:
            boundary_end_ms = int(source_id.removeprefix("interaction-window:"))
        except ValueError as exc:
            raise DurableStateError("interaction Tick has an invalid boundary") from exc
        with self._state._connect() as connection:
            retry = connection.execute(
                """
                SELECT * FROM interaction_explicit_retries
                WHERE user_id = ? AND retry_run_id = ?
                """,
                (lease.user_id, lease.logical_run_id),
            ).fetchone()
            if retry is None:
                prior = connection.execute(
                    """
                    SELECT MAX(boundary_end_ms) AS boundary_start_ms
                    FROM interaction_claims WHERE user_id = ?
                    """,
                    (lease.user_id,),
                ).fetchone()
                boundary_start_ms = (
                    0
                    if prior is None or prior["boundary_start_ms"] is None
                    else int(prior["boundary_start_ms"])
                )
            else:
                original = connection.execute(
                    """
                    SELECT ic.boundary_start_ms
                    FROM interaction_quarantines iq
                    JOIN interaction_claims ic
                      ON ic.logical_run_id = iq.original_logical_run_id
                    WHERE iq.quarantine_id = ?
                    """,
                    (str(retry["quarantine_id"]),),
                ).fetchone()
                if original is None:
                    raise DurableStateError("interaction retry lost its original range")
                boundary_start_ms = int(original["boundary_start_ms"])
        claim_request_id = (
            str(retry["retry_request_id"])
            if retry is not None
            else str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"thine-interaction-claim:{lease.logical_run_id}",
                )
            )
        )
        now_ms = self._clock_ms()
        with self._state._transaction() as connection:
            item = self._state._require_active_owner(
                connection,
                user_id=lease.user_id,
                logical_run_id=lease.logical_run_id,
                owner=lease.owner,
                attempt_id=lease.attempt_id,
                lease_token=lease.lease_token,
                now_ms=now_ms,
            )
            if item["kind"] != "p1_interaction":
                raise DurableStateError("interaction claim belongs to another Tick")
            connection.execute(
                """
                INSERT OR IGNORE INTO interaction_claims (
                    user_id, logical_run_id, tick_id, claim_request_id,
                    boundary_start_ms, boundary_end_ms, state, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, 'requested', ?, ?)
                """,
                (
                    lease.user_id,
                    lease.logical_run_id,
                    str(payload.tick_id),
                    claim_request_id,
                    boundary_start_ms,
                    boundary_end_ms,
                    now_ms,
                    now_ms,
                ),
            )
            row = connection.execute(
                "SELECT * FROM interaction_claims WHERE logical_run_id = ?",
                (lease.logical_run_id,),
            ).fetchone()
            if row is None or str(row["claim_request_id"]) != claim_request_id:
                raise DurableStateError("interaction claim request identity changed")
            stored = self._stored_claim(row)
        if stored.batch is None:
            explicit_retry: InteractionRetryRequest | None = None
            if retry is not None:
                explicit_retry = InteractionRetryRequest(
                    quarantine_id=str(retry["quarantine_id"]),
                    retry_run_id=lease.logical_run_id,
                    retry_request_id=claim_request_id,
                    requested_at_ms=int(retry["created_at_ms"]),
                )
                retried = self._source.retry(explicit_retry)
                if (
                    retried.quarantine_id != explicit_retry.quarantine_id
                    or retried.retry_run_id != explicit_retry.retry_run_id
                    or retried.retry_request_id != explicit_retry.retry_request_id
                    or retried.normal_cursor_rewound
                    or not retried.quarantine_retained
                ):
                    raise DurableStateError(
                        "interaction explicit retry identity mismatch"
                    )
                batch = retried.batch
            else:
                try:
                    batch = self._source.lookup_claim(claim_request_id)
                except InteractionClaimNotFound:
                    batch = self._source.claim(
                        InteractionClaimRequest(
                            claim_request_id=claim_request_id,
                            logical_run_id=lease.logical_run_id,
                            boundary_start_ms=stored.boundary_start_ms,
                            boundary_end_ms=boundary_end_ms,
                        )
                    )
            if str(batch.payload.user_id) != lease.user_id:
                raise DurableStateError(
                    "interaction batch user does not match the lease"
                )
            if not batch.payload.events:
                raise DurableStateError(
                    "empty interaction batch cannot invoke the model"
                )
            if int(batch.payload.window_end_ms) > boundary_end_ms:
                raise DurableStateError("interaction batch crosses its fixed boundary")
            with self._state._transaction() as connection:
                self._state._require_active_owner(
                    connection,
                    user_id=lease.user_id,
                    logical_run_id=lease.logical_run_id,
                    owner=lease.owner,
                    attempt_id=lease.attempt_id,
                    lease_token=lease.lease_token,
                    now_ms=self._clock_ms(),
                )
                row = connection.execute(
                    "SELECT * FROM interaction_claims WHERE logical_run_id = ?",
                    (lease.logical_run_id,),
                ).fetchone()
                assert row is not None
                canonical = batch.to_json()
                if (
                    row["batch_json"] is not None
                    and str(row["batch_json"]) != canonical
                ):
                    raise DurableStateError(
                        "interaction claim changed its frozen range"
                    )
                connection.execute(
                    """
                    UPDATE interaction_claims
                    SET batch_id = ?, batch_json = ?, state = 'claimed', updated_at_ms = ?
                    WHERE logical_run_id = ?
                    """,
                    (
                        str(batch.payload.batch_id),
                        canonical,
                        self._clock_ms(),
                        lease.logical_run_id,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM interaction_claims WHERE logical_run_id = ?",
                    (lease.logical_run_id,),
                ).fetchone()
                assert row is not None
                stored = self._stored_claim(row)
        assert stored.batch is not None
        gaps: tuple[InputGap, ...] = ()
        if retry is None:
            with self._state._transaction() as connection:
                rows = connection.execute(
                    """
                    SELECT quarantine_id, input_gap_json
                    FROM interaction_quarantines
                    WHERE user_id = ? AND sync_state = 'synchronized'
                      AND gap_delivery_run_id IS NULL
                    ORDER BY quarantined_at_ms, quarantine_id
                    """,
                    (lease.user_id,),
                ).fetchall()
                gaps = tuple(
                    InputGap.from_json(str(row["input_gap_json"])) for row in rows
                )
                if rows:
                    connection.executemany(
                        """
                        UPDATE interaction_quarantines SET gap_delivery_run_id = ?
                        WHERE quarantine_id = ? AND gap_delivery_run_id IS NULL
                        """,
                        [
                            (lease.logical_run_id, str(row["quarantine_id"]))
                            for row in rows
                        ],
                    )
        return PreparedInteractionInput(
            batch=stored.batch,
            boundary_end_ms=stored.boundary_end_ms,
            explicit_retry=(
                None
                if retry is None
                else InteractionRetryRequest(
                    quarantine_id=str(retry["quarantine_id"]),
                    retry_run_id=lease.logical_run_id,
                    retry_request_id=claim_request_id,
                    requested_at_ms=int(retry["created_at_ms"]),
                )
            ),
            input_gaps=gaps,
        )

    def enqueue_explicit_retry(
        self,
        *,
        user_id: str,
        quarantine_id: str,
        retry_run_id: str,
        created_at_ms: int,
    ) -> str:
        """Create a separately identified retry without rewinding normal progress."""
        if not all((user_id, quarantine_id, retry_run_id)):
            raise ValueError("explicit retry identities are required")
        retry_request_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"thine-interaction-retry:{retry_run_id}")
        )
        tick_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL, f"thine-interaction-retry-tick:{retry_run_id}"
            )
        )
        with self._state._transaction() as connection:
            source = connection.execute(
                """
                SELECT iq.*, ic.boundary_end_ms
                FROM interaction_quarantines iq
                JOIN interaction_claims ic
                  ON ic.logical_run_id = iq.original_logical_run_id
                WHERE iq.user_id = ? AND iq.quarantine_id = ?
                  AND iq.sync_state = 'synchronized'
                """,
                (user_id, quarantine_id),
            ).fetchone()
            if source is None:
                raise KeyError(quarantine_id)
            boundary_end_ms = int(source["boundary_end_ms"])
            source_id = f"interaction-window:{boundary_end_ms}"
            tick = Tick.from_dict({
                "schema_version": _VERSION,
                "tick_id": tick_id,
                "user_id": user_id,
                "logical_run_id": retry_run_id,
                "kind": "p1_interaction",
                "priority": "p1",
                "occurred_at_ms": created_at_ms,
                "received_at_ms": created_at_ms,
                "queued_at_ms": created_at_ms,
                "source_ref": {"kind": "interaction_window", "id": source_id},
                "causation_id": quarantine_id,
                "correlation_id": tick_id,
                "attempt_ordinal": 1,
                "lease": None,
                "communication_allowance_snapshot": None,
                "payload": {
                    "payload_kind": "interaction_window",
                    "reference_id": source_id,
                },
                "extensions": {},
            })
            self._state._insert_tick_locked(connection, tick=tick, now_ms=created_at_ms)
            connection.execute(
                """
                INSERT OR IGNORE INTO interaction_explicit_retries (
                    retry_run_id, user_id, quarantine_id, retry_request_id,
                    state, created_at_ms
                ) VALUES (?, ?, ?, ?, 'queued', ?)
                """,
                (
                    retry_run_id,
                    user_id,
                    quarantine_id,
                    retry_request_id,
                    created_at_ms,
                ),
            )
        return tick_id

    def inspect(self, *, user_id: str) -> dict[str, object]:
        """Return safe clock/range/quarantine state for explicit helpers."""
        with self._state._connect() as connection:
            clock = connection.execute(
                "SELECT * FROM interaction_clock_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            claims = connection.execute(
                """
                SELECT logical_run_id, tick_id, claim_request_id, boundary_end_ms,
                       boundary_start_ms, batch_id, state, memory_version,
                       finalization_id
                FROM interaction_claims WHERE user_id = ?
                ORDER BY created_at_ms, logical_run_id
                """,
                (user_id,),
            ).fetchall()
            quarantines = connection.execute(
                """
                SELECT quarantine_id, original_logical_run_id, batch_id,
                       first_cursor, last_cursor, failure_code,
                       quarantined_at_ms, sync_state
                FROM interaction_quarantines WHERE user_id = ?
                ORDER BY quarantined_at_ms, quarantine_id
                """,
                (user_id,),
            ).fetchall()
        return {
            "clock": None if clock is None else dict(clock),
            "claims": [dict(row) for row in claims],
            "quarantines": [dict(row) for row in quarantines],
        }

    @staticmethod
    def _stored_claim(row: Any) -> StoredInteractionClaim:
        return StoredInteractionClaim(
            user_id=str(row["user_id"]),
            tick_id=str(row["tick_id"]),
            logical_run_id=str(row["logical_run_id"]),
            claim_request_id=str(row["claim_request_id"]),
            boundary_start_ms=int(row["boundary_start_ms"]),
            boundary_end_ms=int(row["boundary_end_ms"]),
            batch=(
                None
                if row["batch_json"] is None
                else InteractionBatch.from_json(str(row["batch_json"]))
            ),
            state=str(row["state"]),
        )


class FakeInteractionNoActionRuntime:
    """Deterministic seam for boundary/cursor recovery without a paid call."""

    def __init__(self, outcomes: list[InvocationOutcome] | None = None) -> None:
        self._outcomes = list(outcomes or [])
        self.invocations: list[InvocationContext] = []

    def invoke(
        self,
        context: InvocationContext,
        *,
        tools: object,
        control: InvocationControl,
    ) -> InvocationOutcome:
        del tools
        if control.preemption_requested:
            return InvocationOutcome.preempted(
                remaining_work="interaction inference not started"
            )
        if not isinstance(context.prepared_input, PreparedInteractionInput):
            raise ValueError("interaction inference requires one claimed range")
        self.invocations.append(context)
        return (
            self._outcomes.pop(0) if self._outcomes else InvocationOutcome.no_action()
        )


class _InteractionStagedMemory:
    def __init__(self) -> None:
        self.marked_unchanged = False
        self.markdown: str | None = None
        self.token_count: int | None = None

    def commit(
        self, *, expected_version: int, markdown: str, token_count: int, run_id: str
    ) -> int:
        del run_id
        self.markdown = markdown
        self.token_count = token_count
        return expected_version + 1

    def mark_unchanged(self, *, expected_version: int, run_id: str) -> None:
        del expected_version, run_id
        self.marked_unchanged = True


class InteractionRunFinalizer:
    """Commit memory, then retry only the interaction cursor suffix."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        source: InteractionSourcePort,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._state = state
        self._source = source
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def resume_pending(self, user_id: str) -> RunFinalizationResult | None:
        quarantined = self.finalize_quarantine(user_id)
        if quarantined is not None:
            return quarantined
        pending = self._next_pending_ack(user_id)
        return None if pending is None else self._ack_pending(pending)

    def finalize(
        self,
        context: InvocationContext,
        outcome: InvocationOutcome,
        *,
        lease: ActiveRunLease,
    ) -> RunFinalizationResult:
        if context.tick.payload.kind != "p1_interaction":
            raise ValueError("interaction finalizer received another Tick kind")
        if not isinstance(context.prepared_input, PreparedInteractionInput):
            raise ValueError("interaction finalizer requires its claimed range")
        if outcome.status != "completed":
            raise ValueError("only a completed interaction inference can finalize")
        pending = self._stage_memory(context, outcome, lease=lease)
        return self._ack_pending(pending)

    def _stage_memory(
        self,
        context: InvocationContext,
        outcome: InvocationOutcome,
        *,
        lease: ActiveRunLease,
    ) -> PendingInteractionAck:
        now_ms = self._clock_ms()
        prepared = cast(PreparedInteractionInput, context.prepared_input)
        artifact = outcome.finalization_context
        staged = _InteractionStagedMemory()
        hook_outcome = None
        if isinstance(artifact, InteractionAgentArtifact):
            hook_outcome = StopHookRunner().finalize(
                run_id=lease.logical_run_id,
                current=artifact.current_memory,
                context=HermesCachedStopHookContext(
                    agent=artifact.agent,
                    conversation_history=list(artifact.result.context_messages),
                    cache_identity=artifact.cache_identity,
                ),
                store=staged,
                interrupted=False,
            )
        else:
            staged.mark_unchanged(expected_version=0, run_id=lease.logical_run_id)
        with self._state._transaction() as connection:
            item = self._state._require_active_owner(
                connection,
                user_id=lease.user_id,
                logical_run_id=lease.logical_run_id,
                owner=lease.owner,
                attempt_id=lease.attempt_id,
                lease_token=lease.lease_token,
                now_ms=now_ms,
            )
            claim = connection.execute(
                "SELECT * FROM interaction_claims WHERE logical_run_id = ?",
                (lease.logical_run_id,),
            ).fetchone()
            if (
                item["kind"] != "p1_interaction"
                or claim is None
                or claim["state"] != "claimed"
                or claim["batch_json"] is None
            ):
                raise DurableStateError("interaction run has no durable claimed range")
            if str(claim["batch_json"]) != prepared.batch.to_json():
                raise DurableStateError("prepared interaction range changed")
            connection.execute(
                """
                INSERT OR IGNORE INTO working_memory_state (
                    user_id, version, markdown, token_count, last_run_id
                ) VALUES (?, 0, '', NULL, NULL)
                """,
                (lease.user_id,),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO working_memory_versions (
                    user_id, version, markdown, configured_model_token_count,
                    tokenizer_status, logical_run_id, committed_at_ms
                ) VALUES (?, 0, '', NULL, 'unresolved_fail_closed', NULL, 0)
                """,
                (lease.user_id,),
            )
            memory = connection.execute(
                "SELECT version, markdown, token_count FROM working_memory_state WHERE user_id = ?",
                (lease.user_id,),
            ).fetchone()
            assert memory is not None
            memory_version = int(memory["version"])
            if isinstance(artifact, InteractionAgentArtifact) and (
                memory_version != artifact.current_memory.version
            ):
                raise DurableStateError(
                    "working memory changed during interaction Stop Hook"
                )
            memory_outcome = "unchanged"
            if staged.markdown is not None:
                if (
                    staged.token_count is None
                    or hook_outcome is None
                    or (hook_outcome.kind is not StopHookOutcomeKind.COMMITTED)
                ):
                    raise DurableStateError(
                        "changed interaction memory requires exact configured-model tokens"
                    )
                if staged.token_count < 0 or staged.token_count > 16_000:
                    raise DurableStateError(
                        "changed interaction memory exceeds 16K tokens"
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
                        lease.user_id,
                        memory_version,
                        staged.markdown,
                        staged.token_count,
                        lease.logical_run_id,
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
                        staged.markdown,
                        staged.token_count,
                        lease.logical_run_id,
                        lease.user_id,
                        memory_version - 1,
                    ),
                )
                if updated.rowcount != 1:
                    raise DurableStateError(
                        "working memory changed during finalization"
                    )
                memory_outcome = "written"
            finalization_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"thine-interaction-finalization:{lease.logical_run_id}",
                )
            )
            if memory_outcome == "unchanged":
                connection.execute(
                    """
                    INSERT INTO working_memory_unchanged (
                        marker_id, user_id, logical_run_id, expected_version, recorded_at_ms
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        f"memory-unchanged:{lease.logical_run_id}",
                        lease.user_id,
                        lease.logical_run_id,
                        memory_version,
                        now_ms,
                    ),
                )
                connection.execute(
                    """
                    UPDATE working_memory_state SET last_run_id = ?
                    WHERE user_id = ? AND version = ?
                    """,
                    (lease.logical_run_id, lease.user_id, memory_version),
                )
            connection.execute(
                """
                INSERT INTO decision_outcomes (
                    decision_receipt_id, user_id, logical_run_id, outcome,
                    visible_action_intent_count, recorded_at_ms
                ) VALUES (?, ?, ?, 'no_action', 0, ?)
                """,
                (
                    f"decision-receipt:{lease.logical_run_id}",
                    lease.user_id,
                    lease.logical_run_id,
                    now_ms,
                ),
            )
            connection.execute(
                """
                UPDATE interaction_claims
                SET state = 'awaiting_ack', memory_version = ?,
                    working_memory_outcome = ?, finalization_id = ?, updated_at_ms = ?
                WHERE logical_run_id = ?
                """,
                (
                    memory_version,
                    memory_outcome,
                    finalization_id,
                    now_ms,
                    lease.logical_run_id,
                ),
            )
            if isinstance(artifact, InteractionAgentArtifact):
                assert hook_outcome is not None
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
                        lease.user_id,
                        lease.logical_run_id,
                        lease.attempt_id,
                        artifact.provider,
                        artifact.model,
                        artifact.api_mode,
                        artifact.reasoning_effort,
                        str(artifact.result.final_output or ""),
                        json.dumps(
                            list(artifact.tool_discoveries), separators=(",", ":")
                        ),
                        json.dumps(
                            dict(artifact.result.usage),
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        hook_outcome.kind.value,
                        json.dumps(
                            asdict(hook_outcome.cache_identity),
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        memory_version,
                        staged.token_count
                        if staged.markdown is not None
                        else memory["token_count"],
                        now_ms,
                    ),
                )
            connection.execute(
                """
                UPDATE attempts SET status = 'succeeded', finished_at_ms = ?
                WHERE attempt_id = ? AND status = 'running'
                """,
                (now_ms, lease.attempt_id),
            )
            connection.execute(
                """
                UPDATE queue_items
                SET state = 'awaiting_interaction_ack', lease_owner = NULL,
                    lease_token = NULL, lease_expires_at_ms = NULL, updated_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (now_ms, lease.user_id, lease.logical_run_id),
            )
        return PendingInteractionAck(
            user_id=lease.user_id,
            tick_id=str(context.tick.payload.tick_id),
            logical_run_id=lease.logical_run_id,
            attempt_ordinal=lease.attempt_ordinal,
            batch=prepared.batch,
            memory_version=memory_version,
            working_memory_outcome=memory_outcome,
            finalization_id=finalization_id,
        )

    def _next_pending_ack(self, user_id: str) -> PendingInteractionAck | None:
        with self._state._connect() as connection:
            row = connection.execute(
                """
                SELECT ic.*, q.tick_id, MAX(a.ordinal) AS attempt_ordinal
                FROM interaction_claims ic
                JOIN queue_items q ON q.logical_run_id = ic.logical_run_id
                JOIN attempts a ON a.logical_run_id = ic.logical_run_id
                WHERE ic.user_id = ? AND ic.state = 'awaiting_ack'
                GROUP BY ic.logical_run_id, q.tick_id
                ORDER BY ic.updated_at_ms, ic.logical_run_id LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return PendingInteractionAck(
            user_id=user_id,
            tick_id=str(row["tick_id"]),
            logical_run_id=str(row["logical_run_id"]),
            attempt_ordinal=int(row["attempt_ordinal"]),
            batch=InteractionBatch.from_json(str(row["batch_json"])),
            memory_version=int(row["memory_version"]),
            working_memory_outcome=str(row["working_memory_outcome"]),
            finalization_id=str(row["finalization_id"]),
        )

    def _ack_pending(self, pending: PendingInteractionAck) -> RunFinalizationResult:
        payload = pending.batch.payload
        consumed_at_ms = self._clock_ms()
        receipt = InteractionCursorConsumptionReceipt.from_dict({
            "schema_version": _VERSION,
            "consumption_receipt_id": f"interaction-consumption:{pending.logical_run_id}",
            "batch_id": str(payload.batch_id),
            "first_cursor": int(payload.first_cursor),
            "last_cursor": int(payload.last_cursor),
            "logical_run_id": pending.logical_run_id,
            "finalization_id": pending.finalization_id,
            "consumed_cursor": int(payload.last_cursor),
            "consumed_at_ms": consumed_at_ms,
            "extensions": {},
        })
        try:
            self._source.consume(receipt)
        except Exception:
            return RunFinalizationResult(
                tick_id=pending.tick_id,
                logical_run_id=pending.logical_run_id,
                attempt_ordinal=pending.attempt_ordinal,
                status="awaiting_interaction_ack",  # type: ignore[arg-type]
            )
        self._complete_ack(pending, receipt)
        return RunFinalizationResult(
            tick_id=pending.tick_id,
            logical_run_id=pending.logical_run_id,
            attempt_ordinal=pending.attempt_ordinal,
            status="completed",
        )

    def _complete_ack(
        self,
        pending: PendingInteractionAck,
        receipt: InteractionCursorConsumptionReceipt,
    ) -> None:
        now_ms = int(receipt.payload.consumed_at_ms)
        finalization = RunFinalization.from_dict({
            "schema_version": _VERSION,
            "finalization_id": pending.finalization_id,
            "logical_run_id": pending.logical_run_id,
            "tick_id": pending.tick_id,
            "tick_kind": "p1_interaction",
            "phase": "completed",
            "source_ack_id": str(receipt.payload.consumption_receipt_id),
            "final_reply_receipt_id": None,
            "recovery_mode": "ack_only",
            "inference_allowed": False,
            "restream_allowed": False,
            "working_memory_outcome": pending.working_memory_outcome,
            "finalized_at_ms": now_ms,
            "extensions": {},
        })
        input_receipt = InputReceipt.from_dict({
            "schema_version": _VERSION,
            "receipt_id": f"input-receipt:{receipt.payload.consumption_receipt_id}",
            "source_kind": "interaction",
            "source_identity": str(receipt.payload.batch_id),
            "logical_run_id": pending.logical_run_id,
            "ack_id": str(receipt.payload.consumption_receipt_id),
            "disposition": "acknowledged",
            "recorded_at_ms": now_ms,
            "extensions": {},
        })
        run_receipt = RunReceipt.from_dict({
            "schema_version": _VERSION,
            "receipt_id": f"run-receipt:{pending.logical_run_id}",
            "logical_run_id": pending.logical_run_id,
            "tick_id": pending.tick_id,
            "outcome": "completed",
            "attempts_total": pending.attempt_ordinal,
            "finalization_id": pending.finalization_id,
            "recorded_at_ms": now_ms,
            "extensions": {},
        })
        with self._state._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM interaction_claims WHERE logical_run_id = ?",
                (pending.logical_run_id,),
            ).fetchone()
            if row is None:
                raise DurableStateError("unknown interaction acknowledgement")
            if row["state"] == "completed":
                if str(row["consumption_receipt_json"]) != receipt.to_json():
                    raise DurableStateError(
                        "interaction acknowledgement replay changed"
                    )
                return
            if row["state"] != "awaiting_ack":
                raise DurableStateError("interaction acknowledgement is out of order")
            connection.execute(
                """
                INSERT INTO run_finalizations (
                    finalization_id, user_id, logical_run_id, tick_id, phase,
                    working_memory_outcome, source_ack_id, finalization_json, updated_at_ms
                ) VALUES (?, ?, ?, ?, 'completed', ?, ?, ?, ?)
                """,
                (
                    pending.finalization_id,
                    pending.user_id,
                    pending.logical_run_id,
                    pending.tick_id,
                    pending.working_memory_outcome,
                    receipt.payload.consumption_receipt_id,
                    finalization.to_json(),
                    now_ms,
                ),
            )
            connection.execute(
                """
                INSERT INTO input_receipts (
                    receipt_id, user_id, logical_run_id, ack_id, receipt_json, recorded_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    input_receipt.payload.receipt_id,
                    pending.user_id,
                    pending.logical_run_id,
                    input_receipt.payload.ack_id,
                    input_receipt.to_json(),
                    now_ms,
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
                    now_ms,
                ),
            )
            connection.execute(
                """
                UPDATE interaction_claims
                SET state = 'completed', consumption_receipt_json = ?, updated_at_ms = ?
                WHERE logical_run_id = ?
                """,
                (receipt.to_json(), now_ms, pending.logical_run_id),
            )
            connection.execute(
                """
                UPDATE queue_items SET state = 'completed', completed_at_ms = ?, updated_at_ms = ?
                WHERE user_id = ? AND logical_run_id = ?
                """,
                (now_ms, now_ms, pending.user_id, pending.logical_run_id),
            )
            connection.execute(
                """
                UPDATE interaction_explicit_retries
                SET state = 'completed', completed_at_ms = ?
                WHERE user_id = ? AND retry_run_id = ?
                """,
                (now_ms, pending.user_id, pending.logical_run_id),
            )

    def finalize_quarantine(self, user_id: str) -> RunFinalizationResult | None:
        pending = self._next_pending_quarantine(user_id)
        if pending is None:
            return None
        payload = pending.batch.payload
        try:
            result = self._source.quarantine(
                InteractionQuarantineRequest(
                    quarantine_id=pending.quarantine_id,
                    logical_run_id=pending.logical_run_id,
                    batch_id=str(payload.batch_id),
                    first_cursor=int(payload.first_cursor),
                    last_cursor=int(payload.last_cursor),
                    failure_code=pending.failure_code,
                    fault_attempts_total=3,
                    quarantined_at_ms=pending.quarantined_at_ms,
                )
            )
        except Exception:
            return RunFinalizationResult(
                tick_id=pending.tick_id,
                logical_run_id=pending.logical_run_id,
                attempt_ordinal=pending.attempt_ordinal,
                status="quarantine_pending",
            )
        if (
            result.quarantine_id != pending.quarantine_id
            or result.logical_run_id != pending.logical_run_id
            or result.batch_id != payload.batch_id
            or result.first_cursor != payload.first_cursor
            or result.last_cursor != payload.last_cursor
            or not result.normal_cursor_advanced
            or not result.input_retained
        ):
            raise DurableStateError("interaction quarantine identity mismatch")
        record = QuarantineRecord.from_dict({
            "schema_version": _VERSION,
            "quarantine_id": pending.quarantine_id,
            "source_kind": "interaction",
            "source_identity": f"interaction-range:{payload.first_cursor}-{payload.last_cursor}",
            "immutable_range": {
                "range_kind": "interaction_cursors",
                "first_cursor": int(payload.first_cursor),
                "last_cursor": int(payload.last_cursor),
            },
            "logical_run_id": pending.logical_run_id,
            "fault_attempts_total": 3,
            "normal_cursor_advanced": True,
            "created_at_ms": pending.quarantined_at_ms,
            "extensions": {},
        })
        gap = InputGap.from_dict({
            "schema_version": _VERSION,
            "gap_id": f"input-gap:{pending.quarantine_id}",
            "source_kind": "interaction",
            "source_identity": f"interaction-range:{payload.first_cursor}-{payload.last_cursor}",
            "quarantine_id": pending.quarantine_id,
            "normal_cursor_advanced": True,
            "reason": "attempts_exhausted",
            "recorded_at_ms": self._clock_ms(),
            "extensions": {},
        })
        with self._state._transaction() as connection:
            connection.execute(
                """
                UPDATE interaction_quarantines
                SET sync_state = 'synchronized', record_json = ?,
                    input_gap_json = ?, synchronized_at_ms = ?
                WHERE quarantine_id = ? AND sync_state = 'pending'
                """,
                (
                    record.to_json(),
                    gap.to_json(),
                    int(gap.payload.recorded_at_ms),
                    pending.quarantine_id,
                ),
            )
            connection.execute(
                """
                UPDATE interaction_claims SET state = 'quarantined', updated_at_ms = ?
                WHERE logical_run_id = ?
                """,
                (self._clock_ms(), pending.logical_run_id),
            )
        return RunFinalizationResult(
            tick_id=pending.tick_id,
            logical_run_id=pending.logical_run_id,
            attempt_ordinal=pending.attempt_ordinal,
            status="quarantined",
        )

    def _next_pending_quarantine(
        self, user_id: str
    ) -> PendingInteractionQuarantine | None:
        with self._state._connect() as connection:
            row = connection.execute(
                """
                SELECT iq.*, ic.batch_json, q.tick_id, MAX(a.ordinal) AS attempt_ordinal
                FROM interaction_quarantines iq
                JOIN interaction_claims ic
                  ON ic.logical_run_id = iq.original_logical_run_id
                JOIN queue_items q ON q.logical_run_id = iq.original_logical_run_id
                JOIN attempts a ON a.logical_run_id = iq.original_logical_run_id
                WHERE iq.user_id = ? AND iq.sync_state = 'pending'
                GROUP BY iq.quarantine_id, q.tick_id
                ORDER BY iq.quarantined_at_ms, iq.quarantine_id LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return PendingInteractionQuarantine(
            user_id=user_id,
            tick_id=str(row["tick_id"]),
            logical_run_id=str(row["original_logical_run_id"]),
            attempt_ordinal=int(row["attempt_ordinal"]),
            batch=InteractionBatch.from_json(str(row["batch_json"])),
            quarantine_id=str(row["quarantine_id"]),
            failure_code=str(row["failure_code"]),
            quarantined_at_ms=int(row["quarantined_at_ms"]),
        )


class BackgroundRuntimeRouter:
    """Route background kinds while retaining one coordinator/model boundary."""

    def __init__(self, routes: dict[str, FakeInvocationPort]) -> None:
        self._routes = dict(routes)

    def invoke(
        self, context: InvocationContext, *, tools: Any, control: Any
    ) -> InvocationOutcome:
        kind = str(context.tick.payload.kind)
        try:
            runtime = self._routes[kind]
        except KeyError as exc:
            raise RuntimeError(
                f"background runtime is not configured for {kind}"
            ) from exc
        return runtime.invoke(context, tools=tools, control=control)


class BackgroundInputRouter:
    def __init__(self, routes: dict[str, RunInputPort]) -> None:
        self._routes = dict(routes)

    def prepare(
        self, context: InvocationContext, *, lease: ActiveRunLease
    ) -> object | None:
        route = self._routes.get(str(context.tick.payload.kind))
        return None if route is None else route.prepare(context, lease=lease)


class BackgroundFinalizerRouter:
    """Resume durable suffixes in stable order and finalize by Tick kind."""

    def __init__(self, routes: dict[str, RunFinalizerPort]) -> None:
        self._routes = dict(routes)
        self._ordered = tuple(dict.fromkeys(routes.values()))

    def resume_pending(self, user_id: str) -> RunFinalizationResult | None:
        for finalizer in self._ordered:
            result = finalizer.resume_pending(user_id)
            if result is not None:
                return result
        return None

    def finalize(
        self,
        context: InvocationContext,
        outcome: InvocationOutcome,
        *,
        lease: ActiveRunLease,
    ) -> RunFinalizationResult:
        try:
            finalizer = self._routes[str(context.tick.payload.kind)]
        except KeyError as exc:
            raise RuntimeError("background finalizer is not configured") from exc
        return finalizer.finalize(context, outcome, lease=lease)

    def finalize_quarantine(self, user_id: str) -> RunFinalizationResult | None:
        for finalizer in self._ordered:
            callback = getattr(finalizer, "finalize_quarantine", None)
            if callable(callback):
                result = callback(user_id)
                if result is not None:
                    return result
        return None


class HalfHourInteractionDriver:
    """Wake the existing global coordinator only at fixed interaction boundaries."""

    def __init__(
        self,
        *,
        pump: InteractionInputPump,
        user_id: str,
        wake_coordinator: Callable[[], None],
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._pump = pump
        self._user_id = user_id
        self._wake_coordinator = wake_coordinator
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="thine-interaction-half-hour-driver",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=2)

    def scan_now(self) -> str | None:
        tick_id = self._pump.scan_due(user_id=self._user_id)
        if tick_id is not None:
            self._wake_coordinator()
        return tick_id

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                self.scan_now()
            except Exception:
                # Backend startup/restart is a transport condition, not an input
                # Attempt.  Keep the boundary unrecorded and retry quietly.
                if self._closed.wait(1.0):
                    return
                continue
            now_ms = self._clock_ms()
            next_boundary_ms = next_half_hour_boundary_ms(
                now_ms, self._pump.timezone_name
            )
            delay = max((next_boundary_ms - now_ms) / 1000, 0.01)
            if self._closed.wait(delay):
                return


__all__ = [
    "BackendInteractionClient",
    "BackgroundFinalizerRouter",
    "BackgroundInputRouter",
    "BackgroundRuntimeRouter",
    "FakeInteractionNoActionRuntime",
    "HalfHourInteractionDriver",
    "InteractionAvailability",
    "InteractionClaimNotFound",
    "InteractionClaimRequest",
    "InteractionInputPump",
    "InteractionRunFinalizer",
    "InteractionQuarantineRequest",
    "InteractionQuarantineResult",
    "InteractionRetryRequest",
    "InteractionRetryResult",
    "InteractionSourcePort",
    "InteractionBatchToolBinding",
    "PreparedInteractionInput",
    "RealInteractionAgentRuntime",
    "latest_half_hour_boundary_ms",
    "next_half_hour_boundary_ms",
]
