"""Typed interactions contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("interaction_batch")
class InteractionBatch(ContractDTO):
    """Validated interaction_batch wire payload."""


@contract_type("interaction_cursor_consumption_receipt")
class InteractionCursorConsumptionReceipt(ContractDTO):
    """Validated interaction_cursor_consumption_receipt wire payload."""


@contract_type("interaction_delivery_ack")
class InteractionDeliveryAck(ContractDTO):
    """Validated interaction_delivery_ack wire payload."""


@contract_type("interaction_event")
class InteractionEvent(ContractDTO):
    """Validated interaction_event wire payload."""


__all__ = [
    "InteractionBatch",
    "InteractionCursorConsumptionReceipt",
    "InteractionDeliveryAck",
    "InteractionEvent",
]

