"""Behavior-free typed ports at the Hermes contract boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .action import ActionIntent, ActionReceipt
from .chat import (
    ChatCurrentStatus,
    ChatEvent,
    ChatReconnect,
    FinalReplyOutbox,
    FinalReplyReceipt,
    QueueReceipt,
)
from .control import HermesControlRequest, HermesControlResponse
from .dashboard import DashboardReadModel, DashboardSnapshot
from .home import HomeActivation, HomeHistory, HomeRevision, HomeState
from .interactions import (
    InteractionBatch,
    InteractionCursorConsumptionReceipt,
    InteractionDeliveryAck,
)
from .notifications import NotificationIntent, NotificationOutcome, NotificationPermission
from .preferences import Preferences
from .recovery import ExplicitRetry, InputGap, QuarantineRecord
from .reset import ResetCommand, ResetResult
from .runtime import (
    Checkpoint,
    InvocationEvent,
    InvocationRequest,
    RunFinalization,
    RunReceipt,
    ToolResult,
)
from .schedule import Schedule
from .speakers import SpeakerCursorOutcome, SpeakerMappingEvent
from .topics import TopicLifecycle
from .transcripts import (
    TranscriptAck,
    TranscriptCanonicalLookup,
    TranscriptClaim,
    TranscriptClaimLookup,
    TranscriptClaimRequest,
    TranscriptLeaseRenewRequest,
    TranscriptLeaseRenewResult,
    TranscriptReclaimRequest,
    TranscriptReclaimResult,
    TranscriptRelease,
)
from .working_memory import StopHookDecision, WorkingMemorySnapshot


class InvocationPort(Protocol):
    def invoke(self, request: InvocationRequest) -> Iterable[InvocationEvent]: ...

    def cancel_or_yield(self, request: HermesControlRequest) -> HermesControlResponse: ...

    def checkpoint(self, checkpoint: Checkpoint) -> None: ...

    def record_tool_result(self, result: ToolResult) -> None: ...

    def finalize(self, finalization: RunFinalization) -> RunReceipt: ...


class WorkingMemoryPort(Protocol):
    def load(self, user_id: str) -> WorkingMemorySnapshot: ...

    def finalize(self, decision: StopHookDecision) -> WorkingMemorySnapshot: ...


class TranscriptPort(Protocol):
    def claim(self, request: TranscriptClaimRequest) -> TranscriptClaim: ...

    def lookup_claim(self, request: TranscriptClaimLookup) -> TranscriptClaim: ...

    def renew(self, request: TranscriptLeaseRenewRequest) -> TranscriptLeaseRenewResult: ...

    def reclaim(self, request: TranscriptReclaimRequest) -> TranscriptReclaimResult: ...

    def release(self, request: TranscriptRelease) -> TranscriptRelease: ...

    def acknowledge(self, request: TranscriptAck) -> TranscriptAck: ...

    def canonical_lookup(self, request: TranscriptCanonicalLookup) -> TranscriptCanonicalLookup: ...


class ActionPort(Protocol):
    def execute(self, intent: ActionIntent) -> ActionReceipt: ...

    def receipt(self, action_id: str) -> ActionReceipt | None: ...


class ChatPort(Protocol):
    def enqueue(self, user_message_id: str) -> QueueReceipt: ...

    def events(self, stream_id: str) -> Iterable[ChatEvent]: ...

    def current_status(self, stream_id: str) -> ChatCurrentStatus: ...

    def reconnect(self, request: ChatReconnect) -> ChatReconnect: ...

    def persist_reply(self, reply: FinalReplyOutbox) -> FinalReplyReceipt: ...


class HomeStatePort(Protocol):
    def current(self, user_id: str) -> HomeState: ...

    def history(self, user_id: str) -> HomeHistory: ...

    def publish(self, revision: HomeRevision) -> HomeRevision: ...

    def activate(self, activation: HomeActivation) -> HomeRevision: ...


class InteractionPort(Protocol):
    def append(self, batch: InteractionBatch) -> InteractionDeliveryAck: ...

    def consume(self, receipt: InteractionCursorConsumptionReceipt) -> None: ...


class NotificationPort(Protocol):
    def permission(self, user_id: str) -> NotificationPermission: ...

    def deliver(self, intent: NotificationIntent) -> NotificationOutcome: ...


class SchedulePort(Protocol):
    def save(self, schedule: Schedule) -> Schedule: ...

    def list(self, user_id: str) -> Iterable[Schedule]: ...


class SpeakerMappingPort(Protocol):
    def next(self, user_id: str) -> SpeakerMappingEvent | None: ...

    def acknowledge(self, outcome: SpeakerCursorOutcome) -> SpeakerCursorOutcome: ...


class PolicyPort(Protocol):
    def preferences(self, user_id: str) -> Preferences: ...

    def topic(self, topic_key: str) -> TopicLifecycle | None: ...


class RecoveryPort(Protocol):
    def quarantine(self, record: QuarantineRecord, gap: InputGap | None) -> None: ...

    def retry(self, retry: ExplicitRetry) -> QueueReceipt: ...

    def reset(self, command: ResetCommand) -> ResetResult: ...


class OperatorDashboardPort(Protocol):
    def read_model(self, user_id: str) -> DashboardReadModel: ...

    def snapshot(self, user_id: str) -> DashboardSnapshot: ...


class HermesControlPort(Protocol):
    def handle(self, request: HermesControlRequest) -> HermesControlResponse: ...


__all__ = [
    "ActionPort",
    "ChatPort",
    "HermesControlPort",
    "HomeStatePort",
    "InteractionPort",
    "InvocationPort",
    "NotificationPort",
    "OperatorDashboardPort",
    "PolicyPort",
    "RecoveryPort",
    "SchedulePort",
    "SpeakerMappingPort",
    "TranscriptPort",
    "WorkingMemoryPort",
]
