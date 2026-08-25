"""Typed notifications contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("notification_intent")
class NotificationIntent(ContractDTO):
    """Validated notification_intent wire payload."""


@contract_type("notification_outcome")
class NotificationOutcome(ContractDTO):
    """Validated notification_outcome wire payload."""


@contract_type("notification_permission")
class NotificationPermission(ContractDTO):
    """Validated notification_permission wire payload."""


__all__ = [
    "NotificationIntent",
    "NotificationOutcome",
    "NotificationPermission",
]

