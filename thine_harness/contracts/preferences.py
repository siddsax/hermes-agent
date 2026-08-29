"""Generated typed preferences contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    PolicyPreferencesView,
)


@contract_type("preferences")
class Preferences(ContractDTO[PolicyPreferencesView]):
    """Immutable typed view of a validated preferences payload."""

    __slots__ = ()
    _optional_fields = {}


__all__ = [
    "Preferences",
]
