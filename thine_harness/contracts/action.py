"""Generated typed action contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    RuntimeActionIntentVariant1View,
    RuntimeActionIntentVariant2View,
    RuntimeActionIntentVariant3View,
    RuntimeActionIntentVariant4View,
    RuntimeActionIntentVariant5View,
    RuntimeActionReceiptView,
)


@contract_type("action_intent")
class ActionIntent(
    ContractDTO[
        RuntimeActionIntentVariant1View
        | RuntimeActionIntentVariant2View
        | RuntimeActionIntentVariant3View
        | RuntimeActionIntentVariant4View
        | RuntimeActionIntentVariant5View
    ]
):
    """Immutable typed view of a validated action_intent payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("action_receipt")
class ActionReceipt(ContractDTO[RuntimeActionReceiptView]):
    """Immutable typed view of a validated action_receipt payload."""

    __slots__ = ()
    _optional_fields = {}


__all__ = [
    "ActionIntent",
    "ActionReceipt",
]
