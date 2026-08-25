"""Typed reset contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("reset_command")
class ResetCommand(ContractDTO):
    """Validated reset_command wire payload."""


@contract_type("reset_result")
class ResetResult(ContractDTO):
    """Validated reset_result wire payload."""


__all__ = [
    "ResetCommand",
    "ResetResult",
]

