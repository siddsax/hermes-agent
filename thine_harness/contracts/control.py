"""Generated typed control contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    RuntimeHermesControlRequestVariant1View,
    RuntimeHermesControlRequestVariant2View,
    RuntimeHermesControlResponseVariant1View,
    RuntimeHermesControlResponseVariant2View,
)


@contract_type("hermes_control_request")
class HermesControlRequest(
    ContractDTO[
        RuntimeHermesControlRequestVariant1View
        | RuntimeHermesControlRequestVariant2View
    ]
):
    """Immutable typed view of a validated hermes_control_request payload."""

    __slots__ = ()


@contract_type("hermes_control_response")
class HermesControlResponse(
    ContractDTO[
        RuntimeHermesControlResponseVariant1View
        | RuntimeHermesControlResponseVariant2View
    ]
):
    """Immutable typed view of a validated hermes_control_response payload."""

    __slots__ = ()


__all__ = [
    "HermesControlRequest",
    "HermesControlResponse",
]
