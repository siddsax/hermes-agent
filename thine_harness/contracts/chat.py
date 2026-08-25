"""Typed chat contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("chat_current_status")
class ChatCurrentStatus(ContractDTO):
    """Validated chat_current_status wire payload."""


@contract_type("chat_event")
class ChatEvent(ContractDTO):
    """Validated chat_event wire payload."""


@contract_type("chat_reconnect")
class ChatReconnect(ContractDTO):
    """Validated chat_reconnect wire payload."""


@contract_type("final_reply_outbox")
class FinalReplyOutbox(ContractDTO):
    """Validated final_reply_outbox wire payload."""


@contract_type("final_reply_receipt")
class FinalReplyReceipt(ContractDTO):
    """Validated final_reply_receipt wire payload."""


@contract_type("mobile_chat_outbox")
class MobileChatOutbox(ContractDTO):
    """Validated mobile_chat_outbox wire payload."""


@contract_type("p0_submission_outbox")
class P0SubmissionOutbox(ContractDTO):
    """Validated p0_submission_outbox wire payload."""


@contract_type("queue_receipt")
class QueueReceipt(ContractDTO):
    """Validated queue_receipt wire payload."""


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

