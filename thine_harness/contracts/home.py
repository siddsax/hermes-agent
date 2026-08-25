"""Generated typed home contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    MobileHomeActivationView,
    MobileHomeHistoryView,
    MobileHomeRevisionView,
    MobileHomeStateView,
    MobileNavigationIntentView,
)


@contract_type("home_activation")
class HomeActivation(ContractDTO[MobileHomeActivationView]):
    """Immutable typed view of a validated home_activation payload."""

    __slots__ = ()


@contract_type("home_history")
class HomeHistory(ContractDTO[MobileHomeHistoryView]):
    """Immutable typed view of a validated home_history payload."""

    __slots__ = ()


@contract_type("home_revision")
class HomeRevision(ContractDTO[MobileHomeRevisionView]):
    """Immutable typed view of a validated home_revision payload."""

    __slots__ = ()


@contract_type("home_state")
class HomeState(ContractDTO[MobileHomeStateView]):
    """Immutable typed view of a validated home_state payload."""

    __slots__ = ()


@contract_type("navigation_intent")
class NavigationIntent(ContractDTO[MobileNavigationIntentView]):
    """Immutable typed view of a validated navigation_intent payload."""

    __slots__ = ()


__all__ = [
    "HomeActivation",
    "HomeHistory",
    "HomeRevision",
    "HomeState",
    "NavigationIntent",
]
