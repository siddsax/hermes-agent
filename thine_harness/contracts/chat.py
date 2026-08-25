"""Generated typed chat contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    MobileChatCurrentStatusView,
    MobileChatEventVariant1View,
    MobileChatEventVariant2View,
    MobileChatEventVariant3View,
    MobileChatReconnectView,
    MobileMobileChatOutboxVariant1View,
    MobileMobileChatOutboxVariant2View,
    MobileMobileChatOutboxVariant3View,
    RuntimeFinalReplyOutboxVariant1View,
    RuntimeFinalReplyOutboxVariant2View,
    RuntimeFinalReplyReceiptView,
    RuntimeP0SubmissionOutboxVariant1View,
    RuntimeP0SubmissionOutboxVariant2View,
    RuntimeP0SubmissionOutboxVariant3View,
    RuntimeQueueReceiptView,
)


@contract_type("chat_current_status")
class ChatCurrentStatus(ContractDTO[MobileChatCurrentStatusView]):
    """Immutable typed view of a validated chat_current_status payload."""

    __slots__ = ()


@contract_type("chat_event")
class ChatEvent(
    ContractDTO[
        MobileChatEventVariant1View
        | MobileChatEventVariant2View
        | MobileChatEventVariant3View
    ]
):
    """Immutable typed view of a validated chat_event payload."""

    __slots__ = ()


@contract_type("chat_reconnect")
class ChatReconnect(ContractDTO[MobileChatReconnectView]):
    """Immutable typed view of a validated chat_reconnect payload."""

    __slots__ = ()


@contract_type("final_reply_outbox")
class FinalReplyOutbox(
    ContractDTO[
        RuntimeFinalReplyOutboxVariant1View | RuntimeFinalReplyOutboxVariant2View
    ]
):
    """Immutable typed view of a validated final_reply_outbox payload."""

    __slots__ = ()


@contract_type("final_reply_receipt")
class FinalReplyReceipt(ContractDTO[RuntimeFinalReplyReceiptView]):
    """Immutable typed view of a validated final_reply_receipt payload."""

    __slots__ = ()


@contract_type("mobile_chat_outbox")
class MobileChatOutbox(
    ContractDTO[
        MobileMobileChatOutboxVariant1View
        | MobileMobileChatOutboxVariant2View
        | MobileMobileChatOutboxVariant3View
    ]
):
    """Immutable typed view of a validated mobile_chat_outbox payload."""

    __slots__ = ()


@contract_type("p0_submission_outbox")
class P0SubmissionOutbox(
    ContractDTO[
        RuntimeP0SubmissionOutboxVariant1View
        | RuntimeP0SubmissionOutboxVariant2View
        | RuntimeP0SubmissionOutboxVariant3View
    ]
):
    """Immutable typed view of a validated p0_submission_outbox payload."""

    __slots__ = ()


@contract_type("queue_receipt")
class QueueReceipt(ContractDTO[RuntimeQueueReceiptView]):
    """Immutable typed view of a validated queue_receipt payload."""

    __slots__ = ()


__all__ = [
    "ChatCurrentStatus",
    "ChatEvent",
    "ChatReconnect",
    "FinalReplyOutbox",
    "FinalReplyReceipt",
    "MobileChatOutbox",
    "P0SubmissionOutbox",
    "QueueReceipt",
]
