"""Typed preferences contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("preferences")
class Preferences(ContractDTO):
    """Validated preferences wire payload."""


__all__ = [
    "Preferences",
]

