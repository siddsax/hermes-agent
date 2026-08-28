"""Durable ordered speaker-mapping ticks for the local Thine Harness."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import ipaddress
import json
import threading
import time
from typing import Any, Callable, cast, Iterator, Mapping, Protocol
from urllib.parse import urlparse
import uuid

import httpx

from .contracts import JSONValue
from .contracts.ports import (
    QuarantineId,
    SpeakerCursor,
    SpeakerEventId,
    SpeakerMappingPort,
)
from .contracts.recovery import ExplicitRetry
from .contracts.runtime import Tick
from .contracts.speakers import SpeakerCursorOutcome, SpeakerMappingEvent
from .run_coordinator import (
    ActiveRunLease,
    InvocationContext,
    InvocationControl,
    InvocationOutcome,
    RunCoordinator,
    RunFinalizationResult,
    RunFinalizerPort,
    RunInputPort,
)
from .run_state import (
    DurableRunState,
    PendingSpeakerAck,
    PendingSpeakerQuarantine,
)
from .runtime import (
    BackgroundCheckpointPayload,
    HermesAIAgentSession,
    InvocationControl as ProviderInvocationControl,
    InvocationEvent,
    RuntimeModelConfig,
    build_background_invocation_request,
)
from .transcript_agent import (
    TRANSCRIPT_AGENT_TOOLSET,
    _cache_identity,
    _StagedMemoryStore,
)
from .working_memory import (
    CacheIdentity,
    HermesCachedStopHookContext,
    StopHookOutcomeKind,
    StopHookRunner,
    WorkingMemorySnapshot,
)


_VERSION = {"major": 1, "minor": 0}
INSPECT_ACTIVE_MAPPING_TOOL_NAME = "thine_speakers_inspect_active_mapping"
INSPECT_MAPPING_HISTORY_TOOL_NAME = "thine_speakers_inspect_mapping_history"
INSPECT_ACTIVE_MAPPING_TOOL_SCHEMA = {
    "name": INSPECT_ACTIVE_MAPPING_TOOL_NAME,
    "description": (
        "Inspect the exact immutable speaker mapping retained for the current "
        "Logical Run. Null names mean Unknown and must not be inferred."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}
INSPECT_MAPPING_HISTORY_TOOL_SCHEMA = {
    "name": INSPECT_MAPPING_HISTORY_TOOL_NAME,
    "description": (
        "Inspect a retained speaker mapping by stable event ID or quarantine ID, "
        "including cursor state, immutable acknowledgement, and explicit retries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {"type": "string", "minLength": 1, "maxLength": 256},
            "quarantine_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            },
        },
        "additionalProperties": False,
    },
}


class OrderedSpeakerMappingPort(Protocol):
    """Backend-owned immutable mapping stream plus cursor transitions."""

    def next(
        self, user_id: str, after_cursor: int | None = None
    ) -> SpeakerMappingEvent | None: ...

    def acknowledge(
        self, event_id: SpeakerEventId, cursor: SpeakerCursor
    ) -> SpeakerCursorOutcome: ...

    def quarantine_and_advance(
        self,
        event_id: SpeakerEventId,
        cursor: SpeakerCursor,
        quarantine_id: QuarantineId,
    ) -> SpeakerCursorOutcome: ...


class BackendSpeakerMappingClient:
    """Authenticated client exposing only explicit speaker stream helpers."""

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
                "backend speaker origin must use a loopback IP literal"
            ) from exc
        if (
            parsed.scheme != "http"
            or not address.is_loopback
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("backend speaker origin must be loopback-only HTTP")
        if not credential or not firebase_uid:
            raise ValueError("backend speaker credential and Firebase UID are required")
        self._credential = credential
        self._firebase_uid = firebase_uid
        self._client = httpx.Client(
            base_url=origin.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def next(
        self, user_id: str, after_cursor: int | None = None
    ) -> SpeakerMappingEvent | None:
        if user_id != self._firebase_uid:
            raise ValueError("speaker fetch user does not match authenticated UID")
        body = self._post("/v1/speaker-mappings/next", {"after_cursor": after_cursor})
        return None if body is None else SpeakerMappingEvent.from_dict(body)

    def acknowledge(
        self, event_id: SpeakerEventId | str, cursor: SpeakerCursor | int
    ) -> SpeakerCursorOutcome:
        event = str(event_id)
        position = int(cursor)
        body = self._post(
            "/v1/speaker-mappings/ack",
            {
                "event_id": event,
                "cursor": position,
                "idempotency_key": f"speaker-ack:{event}:{position}",
            },
        )
        if body is None:
            raise ValueError("backend speaker acknowledgement is empty")
        return SpeakerCursorOutcome.from_dict(body)

    def quarantine_and_advance(
        self,
        event_id: SpeakerEventId | str,
        cursor: SpeakerCursor | int,
        quarantine_id: QuarantineId | str,
    ) -> SpeakerCursorOutcome:
        event = str(event_id)
        position = int(cursor)
        quarantine = str(quarantine_id)
        body = self._post(
            "/v1/speaker-mappings/quarantine",
            {
                "event_id": event,
                "cursor": position,
                "quarantine_id": quarantine,
                "idempotency_key": f"speaker-quarantine:{quarantine}",
            },
        )
        if body is None:
            raise ValueError("backend speaker quarantine acknowledgement is empty")
        return SpeakerCursorOutcome.from_dict(body)

    def _post(
        self, path: str, body: dict[str, JSONValue]
    ) -> dict[str, JSONValue] | None:
        response = self._client.post(
            path,
            headers={
                "Authorization": f"Bearer {self._credential}",
                "Content-Type": "application/json",
                "X-Thine-Firebase-UID": self._firebase_uid,
                "X-Request-ID": str(uuid.uuid4()),
            },
            json=body,
        )
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return None
        payload: object = response.json()
        if not isinstance(payload, dict):
            raise ValueError("backend speaker response must be a JSON object")
        return cast(dict[str, JSONValue], payload)


@dataclass(frozen=True)
class PreparedSpeakerMappingInput:
    event: SpeakerMappingEvent
    explicit_retry: ExplicitRetry | None = None


class SpeakerMappingToolBinding:
    """Expose only the exact mapping leased by the one active coordinator run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: PreparedSpeakerMappingInput | None = None

    @contextmanager
    def activate(self, prepared: PreparedSpeakerMappingInput) -> Iterator[None]:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("another speaker mapping is already active")
            self._active = prepared
        try:
            yield
        finally:
            with self._lock:
                self._active = None

    def inspect(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if args:
            return json.dumps(
                {"ok": False, "error_code": "unexpected_arguments"},
                separators=(",", ":"),
            )
        with self._lock:
            prepared = self._active
        if prepared is None:
            return json.dumps(
                {"ok": False, "error_code": "no_active_speaker_mapping"},
                separators=(",", ":"),
            )
        return json.dumps(
            {
                "ok": True,
                "event": prepared.event.to_dict(),
                "explicit_retry": (
                    None
                    if prepared.explicit_retry is None
                    else prepared.explicit_retry.to_dict()
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def register(self, *, registry_instance: Any | None = None) -> None:
        from tools.registry import registry

        active_registry = registry_instance or registry
        active_registry.register(
            name=INSPECT_ACTIVE_MAPPING_TOOL_NAME,
            toolset=TRANSCRIPT_AGENT_TOOLSET,
            schema=INSPECT_ACTIVE_MAPPING_TOOL_SCHEMA,
            handler=self.inspect,
            scope=active_registry.current_scope_key(),
        )


@dataclass(frozen=True)
class SpeakerMappingInspectionToolBinding:
    """Read retained mapping and quarantine history without mutating a cursor."""

    state: DurableRunState
    user_id: str

    def inspect(self, args: Mapping[str, object], **_kwargs: object) -> str:
        keys = set(args)
        if keys not in ({"event_id"}, {"quarantine_id"}):
            return json.dumps(
                {"ok": False, "error_code": "provide_one_stable_identifier"},
                separators=(",", ":"),
            )
        key = next(iter(keys))
        value = args.get(key)
        if not isinstance(value, str) or not value:
            return json.dumps(
                {"ok": False, "error_code": "invalid_stable_identifier"},
                separators=(",", ":"),
            )
        try:
            inspection = (
                self.state.inspect_speaker_mapping(user_id=self.user_id, event_id=value)
                if key == "event_id"
                else self.state.inspect_speaker_quarantine(
                    user_id=self.user_id, quarantine_id=value
                )
            )
        except KeyError:
            return json.dumps(
                {"ok": False, "error_code": "speaker_mapping_not_found"},
                separators=(",", ":"),
            )
        mapping = {
            "event": inspection.event.to_dict(),
            "state": inspection.state,
            "logical_run_id": inspection.logical_run_id,
            "normal_cursor": inspection.normal_cursor,
            "quarantine_id": inspection.quarantine_id,
            "quarantine_record": (
                None
                if inspection.quarantine_record is None
                else inspection.quarantine_record.to_dict()
            ),
            "acknowledgement": (
                None
                if inspection.acknowledgement is None
                else inspection.acknowledgement.to_dict()
            ),
            "retry_run_ids": list(inspection.retry_run_ids),
        }
        return json.dumps(
            {"ok": True, "mapping": mapping},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def register(self, *, registry_instance: Any | None = None) -> None:
        from tools.registry import registry

        active_registry = registry_instance or registry
        active_registry.register(
            name=INSPECT_MAPPING_HISTORY_TOOL_NAME,
            toolset=TRANSCRIPT_AGENT_TOOLSET,
            schema=INSPECT_MAPPING_HISTORY_TOOL_SCHEMA,
            handler=self.inspect,
            scope=active_registry.current_scope_key(),
        )


class SpeakerMappingInputPump:
    """Fetch one ordered mapping, retain it, and enqueue one P1 Tick."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        speaker_port: OrderedSpeakerMappingPort,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._state = state
        self._speaker_port = speaker_port
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def enqueue_next(self, user_id: str, *, coordinator: RunCoordinator) -> str | None:
        cursor = self._state.speaker_cursor(user_id)
        event = self._speaker_port.next(user_id, None if cursor == 0 else cursor)
        if event is None:
            return None
        mapping = event.payload
        if mapping.user_id != user_id:
            raise ValueError("speaker mapping user does not match requested user")
        tick_id = f"speaker-tick:{mapping.event_id}"
        logical_run_id = f"run:{mapping.event_id}"
        queued_at_ms = self._clock_ms()
        tick = Tick.from_dict({
            "schema_version": _VERSION,
            "tick_id": tick_id,
            "user_id": user_id,
            "logical_run_id": logical_run_id,
            "kind": "p1_speaker",
            "priority": "p1",
            "occurred_at_ms": int(mapping.changed_at_ms),
            "received_at_ms": queued_at_ms,
            "queued_at_ms": queued_at_ms,
            "source_ref": {
                "kind": "speaker_mapping",
                "id": mapping.event_id,
            },
            "causation_id": None,
            "correlation_id": tick_id,
            "attempt_ordinal": 1,
            "lease": None,
            "communication_allowance_snapshot": None,
            "payload": {
                "payload_kind": "speaker_mapping",
                "reference_id": mapping.event_id,
            },
            "extensions": {},
        })
        stored = self._state.enqueue_speaker_mapping(
            event=event, tick=tick, now_ms=queued_at_ms
        )
        coordinator.notify_enqueued(tick)
        return stored

    def prepare(
        self,
        context: InvocationContext,
        *,
        lease: ActiveRunLease,
    ) -> PreparedSpeakerMappingInput | None:
        if context.tick.payload.kind != "p1_speaker":
            return None
        stored = self._state.prepared_speaker_mapping(
            user_id=lease.user_id, logical_run_id=lease.logical_run_id
        )
        return PreparedSpeakerMappingInput(
            event=stored.event,
            explicit_retry=stored.explicit_retry,
        )

    def enqueue_explicit_retry(
        self,
        *,
        user_id: str,
        quarantine_id: str,
        coordinator: RunCoordinator,
    ) -> str:
        inspection = self._state.inspect_speaker_quarantine(
            user_id=user_id, quarantine_id=quarantine_id
        )
        ordinal = len(inspection.retry_run_ids) + 1
        tick_id = f"speaker-retry-tick:{quarantine_id}:{ordinal}"
        logical_run_id = f"speaker-retry-run:{quarantine_id}:{ordinal}"
        queued_at_ms = self._clock_ms()
        event_id = str(inspection.event.payload.event_id)
        tick = Tick.from_dict({
            "schema_version": _VERSION,
            "tick_id": tick_id,
            "user_id": user_id,
            "logical_run_id": logical_run_id,
            "kind": "p1_speaker",
            "priority": "p1",
            "occurred_at_ms": queued_at_ms,
            "received_at_ms": queued_at_ms,
            "queued_at_ms": queued_at_ms,
            "source_ref": {"kind": "speaker_mapping", "id": event_id},
            "causation_id": quarantine_id,
            "correlation_id": tick_id,
            "attempt_ordinal": 1,
            "lease": None,
            "communication_allowance_snapshot": None,
            "payload": {
                "payload_kind": "speaker_mapping",
                "reference_id": event_id,
            },
            "extensions": {},
        })
        stored = self._state.enqueue_speaker_retry(
            user_id=user_id,
            quarantine_id=quarantine_id,
            tick=tick,
            now_ms=queued_at_ms,
        )
        coordinator.notify_enqueued(tick)
        return stored


@dataclass(frozen=True)
class SpeakerMappingAgentArtifact:
    agent: Any
    result: Any
    current_memory: WorkingMemorySnapshot
    cache_identity: CacheIdentity
    provider: str
    model: str
    api_mode: str
    reasoning_effort: str
    tool_discoveries: tuple[str, ...]


class RealSpeakerMappingAgentRuntime:
    """Process one retained mapping with the transcript runtime's cached agent."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        agent: Any,
        binding: SpeakerMappingToolBinding,
        config: RuntimeModelConfig | None = None,
        communication_context: Callable[[str], Mapping[str, object]] | None = None,
    ) -> None:
        self._state = state
        self._agent = agent
        self._binding = binding
        self._config = config or RuntimeModelConfig.openai_gpt_5_6_sol_medium()
        self._communication_context = communication_context
        self._session = HermesAIAgentSession(agent=agent, expected=self._config)
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
                remaining_work="speaker mapping inference not started"
            )
        prepared = context.prepared_input
        if not isinstance(prepared, PreparedSpeakerMappingInput):
            raise ValueError("real speaker inference requires one retained mapping")
        payload = context.tick.payload
        current = self._state.working_memory_snapshot(str(payload.user_id))
        communication_context = (
            {}
            if self._communication_context is None
            else dict(self._communication_context(str(payload.user_id)))
        )
        prompt = (
            "Process the ordered speaker-mapping event for this Logical Run. "
            f"Discover and call {INSPECT_ACTIVE_MAPPING_TOOL_NAME} through the "
            "Thine namespace when exact event detail is needed. Preserve stable "
            "mapping and speaker IDs and the event order. A null old_name or "
            "new_name is Unknown; never infer an identity or name from transcript "
            "content, channel position, or prior guesses. Choose tools or no tools "
            "based on the event. A no-action outcome is valid.\n\n"
            "<working_memory>\n"
            + current.markdown
            + "\n</working_memory>\n"
            + "<mapping_summary>\n"
            + json.dumps(
                {
                    "event_id": prepared.event.payload.event_id,
                    "cursor": prepared.event.payload.cursor,
                    "kind": prepared.event.payload.kind,
                    "source_speaker_ids": list(
                        prepared.event.payload.source_speaker_ids
                    ),
                    "canonical_speaker_id": (
                        prepared.event.payload.canonical_speaker_id
                    ),
                    "old_name": prepared.event.payload.old_name,
                    "new_name": prepared.event.payload.new_name,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n</mapping_summary>\n"
            + "<explicit_retry>\n"
            + (
                "null"
                if prepared.explicit_retry is None
                else prepared.explicit_retry.to_json()
            )
            + "\n</explicit_retry>\n"
            + "<communication_context>\n"
            + json.dumps(
                communication_context,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n</communication_context>\n"
            f"<logical_run_id>{payload.logical_run_id}</logical_run_id>"
        )
        provider_control = ProviderInvocationControl()
        cancelled = threading.Event()

        def relay_preemption() -> None:
            while not cancelled.wait(0.02):
                if control.preemption_requested:
                    provider_control.cancel(control.reason or "p0_user_tick")
                    return

        relay = threading.Thread(
            target=relay_preemption,
            name="thine-speaker-preemption-relay",
            daemon=True,
        )
        relay.start()
        events: list[InvocationEvent] = []
        request = build_background_invocation_request(
            logical_run_id=str(payload.logical_run_id),
            initial_prompt=prompt,
            checkpoint=context.checkpoint,
            newest_working_memory=current.markdown,
            durable_action_receipts=tuple(
                asdict(receipt) for receipt in context.acknowledged_receipts
            ),
        )
        try:
            with self._binding.activate(prepared):
                result = self._session.invoke(
                    request,
                    emit=events.append,
                    control=provider_control,
                )
        finally:
            cancelled.set()
            relay.join(timeout=0.1)
        if result.interrupted:
            return InvocationOutcome.preempted(
                remaining_work=result.remaining_work or "resume speaker inference",
                checkpoint_payload=BackgroundCheckpointPayload.from_turn(
                    request, result
                ),
            )
        if result.failed or not result.completed:
            return InvocationOutcome.fault(
                result.failure_reason or "real_model_incomplete"
            )
        tool_discoveries = tuple(
            str(message.get("tool_name") or message.get("name") or "")
            for message in result.context_messages
            if isinstance(message, dict) and message.get("role") == "tool"
        )
        artifact = SpeakerMappingAgentArtifact(
            agent=self._agent,
            result=result,
            current_memory=current,
            cache_identity=_cache_identity(self._agent),
            provider=self._config.provider,
            model=self._config.model,
            api_mode=self._config.api_mode,
            reasoning_effort=self._config.reasoning_effort,
            tool_discoveries=tool_discoveries,
        )
        self.invocations.append(context)
        return InvocationOutcome.no_action(finalization_context=artifact)


class FakeSpeakerMappingNoActionRuntime:
    """Deterministic model-free runtime for the first mapping vertical slice."""

    def __init__(self) -> None:
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
                remaining_work="speaker mapping inference not started"
            )
        if not isinstance(context.prepared_input, PreparedSpeakerMappingInput):
            raise ValueError("speaker mapping runtime requires one retained event")
        self.invocations.append(context)
        return InvocationOutcome.no_action()


class BackgroundRuntimeDispatcher:
    """Route background kinds behind the one global coordinator lease."""

    def __init__(self, **runtimes: object) -> None:
        self._runtimes = dict(runtimes)

    def invoke(
        self,
        context: InvocationContext,
        *,
        tools: object,
        control: InvocationControl,
    ) -> InvocationOutcome:
        kind = str(context.tick.payload.kind)
        runtime = self._runtimes.get(kind)
        if runtime is None:
            raise RuntimeError(f"background runtime is not configured for {kind}")
        return runtime.invoke(context, tools=tools, control=control)  # type: ignore[attr-defined,no-any-return]


class BackgroundInputDispatcher:
    """Route leased background input to its kind-specific adapter."""

    def __init__(self, **inputs: RunInputPort) -> None:
        self._inputs = dict(inputs)

    def prepare(
        self,
        context: InvocationContext,
        *,
        lease: ActiveRunLease,
    ) -> object | None:
        adapter = self._inputs.get(str(context.tick.payload.kind))
        if adapter is None:
            raise RuntimeError(
                f"background input is not configured for {context.tick.payload.kind}"
            )
        return adapter.prepare(context, lease=lease)


class BackgroundFinalizerDispatcher:
    """Recover and finalize suffixes without creating another scheduler."""

    def __init__(self, **finalizers: RunFinalizerPort) -> None:
        self._finalizers = dict(finalizers)

    def resume_pending(self, user_id: str) -> RunFinalizationResult | None:
        for finalizer in self._finalizers.values():
            resumed = finalizer.resume_pending(user_id)
            if resumed is not None:
                return resumed
        return None

    def finalize(
        self,
        context: InvocationContext,
        outcome: InvocationOutcome,
        *,
        lease: ActiveRunLease,
    ) -> RunFinalizationResult:
        kind = str(context.tick.payload.kind)
        finalizer = self._finalizers.get(kind)
        if finalizer is None:
            raise RuntimeError(f"background finalizer is not configured for {kind}")
        return finalizer.finalize(context, outcome, lease=lease)

    def finalize_quarantine(self, user_id: str) -> RunFinalizationResult | None:
        for finalizer in self._finalizers.values():
            finalize = getattr(finalizer, "finalize_quarantine", None)
            if callable(finalize):
                result = finalize(user_id)
                if result is not None:
                    return result
        return None


class SpeakerMappingFinalizer:
    """Finalize Working Memory, then acknowledge or quarantine one mapping."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        speaker_port: SpeakerMappingPort,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._state = state
        self._speaker_port = speaker_port
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def resume_pending(self, user_id: str) -> RunFinalizationResult | None:
        quarantine = self.finalize_quarantine(user_id)
        if quarantine is not None:
            return quarantine
        pending = self._state.next_pending_speaker_ack(user_id)
        return None if pending is None else self._ack_pending(pending)

    def finalize(
        self,
        context: InvocationContext,
        outcome: InvocationOutcome,
        *,
        lease: ActiveRunLease,
    ) -> RunFinalizationResult:
        if context.tick.payload.kind != "p1_speaker":
            raise ValueError("speaker finalizer cannot finalize another Tick kind")
        if outcome.decision_outcome != "no_action":
            raise ValueError("speaker mapping finalization requires no_action")
        if not isinstance(context.prepared_input, PreparedSpeakerMappingInput):
            raise ValueError("speaker finalization requires its retained event")
        artifact = outcome.finalization_context
        expected_memory_version: int | None = None
        memory_markdown: str | None = None
        memory_token_count: int | None = None
        tokenizer_status = "unresolved_fail_closed"
        agent_inspection: dict[str, Any] | None = None
        if isinstance(artifact, SpeakerMappingAgentArtifact):
            hook_context = HermesCachedStopHookContext(
                agent=artifact.agent,
                conversation_history=list(artifact.result.context_messages),
                cache_identity=artifact.cache_identity,
            )
            staged = _StagedMemoryStore()
            hook = StopHookRunner().finalize(
                run_id=lease.logical_run_id,
                current=artifact.current_memory,
                context=hook_context,
                store=staged,
                interrupted=False,
            )
            expected_memory_version = artifact.current_memory.version
            memory_markdown = staged.markdown
            memory_token_count = staged.token_count
            tokenizer_status = (
                "exact"
                if hook.kind is StopHookOutcomeKind.COMMITTED
                else "unresolved_fail_closed"
            )
            agent_inspection = {
                "provider": artifact.provider,
                "model": artifact.model,
                "api_mode": artifact.api_mode,
                "reasoning_effort": artifact.reasoning_effort,
                "final_output": str(artifact.result.final_output or ""),
                "tool_discoveries": list(artifact.tool_discoveries),
                "usage": dict(artifact.result.usage),
                "stop_hook_outcome": hook.kind.value,
                "stop_hook_cache_identity": asdict(hook.cache_identity),
            }
        elif artifact is not None:
            raise ValueError("speaker finalization received another artifact type")
        pending = self._state.finalize_speaker_mapping_no_action(
            user_id=lease.user_id,
            logical_run_id=lease.logical_run_id,
            owner=lease.owner,
            attempt_id=lease.attempt_id,
            lease_token=lease.lease_token,
            now_ms=self._clock_ms(),
            expected_memory_version=expected_memory_version,
            memory_markdown=memory_markdown,
            memory_token_count=memory_token_count,
            tokenizer_status=tokenizer_status,
            agent_inspection=agent_inspection,
        )
        if pending is None:
            return RunFinalizationResult(
                tick_id=str(context.tick.payload.tick_id),
                logical_run_id=lease.logical_run_id,
                attempt_ordinal=lease.attempt_ordinal,
                status="completed",
            )
        return self._ack_pending(pending)

    def _ack_pending(self, pending: PendingSpeakerAck) -> RunFinalizationResult:
        try:
            acknowledgement = self._speaker_port.acknowledge(
                SpeakerEventId(pending.event_id),
                SpeakerCursor(pending.cursor),
            )
        except Exception:
            return RunFinalizationResult(
                tick_id=pending.tick_id,
                logical_run_id=pending.logical_run_id,
                attempt_ordinal=pending.attempt_ordinal,
                status="awaiting_speaker_cursor_ack",
            )
        self._state.complete_speaker_ack(
            pending=pending, acknowledgement=acknowledgement
        )
        return RunFinalizationResult(
            tick_id=pending.tick_id,
            logical_run_id=pending.logical_run_id,
            attempt_ordinal=pending.attempt_ordinal,
            status="completed",
        )

    def finalize_quarantine(self, user_id: str) -> RunFinalizationResult | None:
        pending = self._state.next_pending_speaker_quarantine(user_id)
        if pending is None:
            return None
        return self._quarantine_pending(pending)

    def _quarantine_pending(
        self, pending: PendingSpeakerQuarantine
    ) -> RunFinalizationResult:
        try:
            acknowledgement = self._speaker_port.quarantine_and_advance(
                SpeakerEventId(pending.event_id),
                SpeakerCursor(pending.cursor),
                QuarantineId(pending.quarantine_id),
            )
        except Exception:
            return RunFinalizationResult(
                tick_id=pending.tick_id,
                logical_run_id=pending.logical_run_id,
                attempt_ordinal=pending.attempt_ordinal,
                status="quarantine_pending",
            )
        self._state.complete_speaker_quarantine(
            pending=pending, acknowledgement=acknowledgement
        )
        return RunFinalizationResult(
            tick_id=pending.tick_id,
            logical_run_id=pending.logical_run_id,
            attempt_ordinal=pending.attempt_ordinal,
            status="quarantined",
        )


__all__ = [
    "BackendSpeakerMappingClient",
    "BackgroundFinalizerDispatcher",
    "BackgroundInputDispatcher",
    "BackgroundRuntimeDispatcher",
    "FakeSpeakerMappingNoActionRuntime",
    "INSPECT_ACTIVE_MAPPING_TOOL_NAME",
    "INSPECT_ACTIVE_MAPPING_TOOL_SCHEMA",
    "INSPECT_MAPPING_HISTORY_TOOL_NAME",
    "INSPECT_MAPPING_HISTORY_TOOL_SCHEMA",
    "OrderedSpeakerMappingPort",
    "PreparedSpeakerMappingInput",
    "RealSpeakerMappingAgentRuntime",
    "SpeakerMappingAgentArtifact",
    "SpeakerMappingFinalizer",
    "SpeakerMappingInputPump",
    "SpeakerMappingInspectionToolBinding",
    "SpeakerMappingToolBinding",
]
