"""Generated typed schedule contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    PolicyScheduleView,
)


@contract_type("schedule")
class Schedule(ContractDTO[PolicyScheduleView]):
    """Immutable typed view of a validated schedule payload."""

    __slots__ = ()
    _optional_fields = {}


__all__ = [
    "Schedule",
]
