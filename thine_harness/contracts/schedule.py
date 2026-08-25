"""Typed schedule contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("schedule")
class Schedule(ContractDTO):
    """Validated schedule wire payload."""


__all__ = [
    "Schedule",
]

