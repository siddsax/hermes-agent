"""Typed home contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("home_activation")
class HomeActivation(ContractDTO):
    """Validated home_activation wire payload."""


@contract_type("home_history")
class HomeHistory(ContractDTO):
    """Validated home_history wire payload."""


@contract_type("home_revision")
class HomeRevision(ContractDTO):
    """Validated home_revision wire payload."""


@contract_type("home_state")
class HomeState(ContractDTO):
    """Validated home_state wire payload."""


@contract_type("navigation_intent")
class NavigationIntent(ContractDTO):
    """Validated navigation_intent wire payload."""


__all__ = [
    "HomeActivation",
    "HomeHistory",
    "HomeRevision",
    "HomeState",
    "NavigationIntent",
]

