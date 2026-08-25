"""Generated typed reset contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    RuntimeResetCommandView,
    RuntimeResetResultVariant1View,
    RuntimeResetResultVariant2View,
    RuntimeResetResultVariant3View,
)


@contract_type("reset_command")
class ResetCommand(ContractDTO[RuntimeResetCommandView]):
    """Immutable typed view of a validated reset_command payload."""

    __slots__ = ()


@contract_type("reset_result")
class ResetResult(
    ContractDTO[
        RuntimeResetResultVariant1View
        | RuntimeResetResultVariant2View
        | RuntimeResetResultVariant3View
    ]
):
    """Immutable typed view of a validated reset_result payload."""

    __slots__ = ()


__all__ = [
    "ResetCommand",
    "ResetResult",
]
