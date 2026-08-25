"""Typed control contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("hermes_control_request")
class HermesControlRequest(ContractDTO):
    """Validated hermes_control_request wire payload."""


@contract_type("hermes_control_response")
class HermesControlResponse(ContractDTO):
    """Validated hermes_control_response wire payload."""


__all__ = [
    "HermesControlRequest",
    "HermesControlResponse",
]

