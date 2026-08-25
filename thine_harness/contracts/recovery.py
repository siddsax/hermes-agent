"""Typed recovery contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("explicit_retry")
class ExplicitRetry(ContractDTO):
    """Validated explicit_retry wire payload."""


@contract_type("input_gap")
class InputGap(ContractDTO):
    """Validated input_gap wire payload."""


@contract_type("quarantine_record")
class QuarantineRecord(ContractDTO):
    """Validated quarantine_record wire payload."""


__all__ = [
    "ExplicitRetry",
    "InputGap",
    "QuarantineRecord",
]

