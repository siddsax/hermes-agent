"""Generated typed interactions contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    MobileInteractionBatchView,
    MobileInteractionDeliveryAckView,
    MobileInteractionEventVariant10View,
    MobileInteractionEventVariant11View,
    MobileInteractionEventVariant12View,
    MobileInteractionEventVariant13View,
    MobileInteractionEventVariant14View,
    MobileInteractionEventVariant15View,
    MobileInteractionEventVariant1View,
    MobileInteractionEventVariant2View,
    MobileInteractionEventVariant3View,
    MobileInteractionEventVariant4View,
    MobileInteractionEventVariant5View,
    MobileInteractionEventVariant6View,
    MobileInteractionEventVariant7View,
    MobileInteractionEventVariant8View,
    MobileInteractionEventVariant9View,
    RuntimeInteractionCursorConsumptionReceiptView,
)


@contract_type("interaction_batch")
class InteractionBatch(ContractDTO[MobileInteractionBatchView]):
    """Immutable typed view of a validated interaction_batch payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("interaction_cursor_consumption_receipt")
class InteractionCursorConsumptionReceipt(
    ContractDTO[RuntimeInteractionCursorConsumptionReceiptView]
):
    """Immutable typed view of a validated interaction_cursor_consumption_receipt payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("interaction_delivery_ack")
class InteractionDeliveryAck(ContractDTO[MobileInteractionDeliveryAckView]):
    """Immutable typed view of a validated interaction_delivery_ack payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("interaction_event")
class InteractionEvent(
    ContractDTO[
        MobileInteractionEventVariant1View
        | MobileInteractionEventVariant2View
        | MobileInteractionEventVariant3View
        | MobileInteractionEventVariant4View
        | MobileInteractionEventVariant5View
        | MobileInteractionEventVariant6View
        | MobileInteractionEventVariant7View
        | MobileInteractionEventVariant8View
        | MobileInteractionEventVariant9View
        | MobileInteractionEventVariant10View
        | MobileInteractionEventVariant11View
        | MobileInteractionEventVariant12View
        | MobileInteractionEventVariant13View
        | MobileInteractionEventVariant14View
        | MobileInteractionEventVariant15View
    ]
):
    """Immutable typed view of a validated interaction_event payload."""

    __slots__ = ()
    _optional_fields = {}


__all__ = [
    "InteractionBatch",
    "InteractionCursorConsumptionReceipt",
    "InteractionDeliveryAck",
    "InteractionEvent",
]
