"""Typed action contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("action_intent")
class ActionIntent(ContractDTO):
    """Validated action_intent wire payload."""


@contract_type("action_receipt")
class ActionReceipt(ContractDTO):
    """Validated action_receipt wire payload."""


__all__ = [
    "ActionIntent",
    "ActionReceipt",
]

