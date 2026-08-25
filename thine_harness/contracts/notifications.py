"""Generated typed notifications contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    MobileNotificationIntentVariant1View,
    MobileNotificationIntentVariant2View,
    MobileNotificationOutcomeVariant1View,
    MobileNotificationOutcomeVariant2View,
    MobileNotificationOutcomeVariant3View,
    MobileNotificationPermissionView,
)


@contract_type("notification_intent")
class NotificationIntent(
    ContractDTO[
        MobileNotificationIntentVariant1View | MobileNotificationIntentVariant2View
    ]
):
    """Immutable typed view of a validated notification_intent payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("notification_outcome")
class NotificationOutcome(
    ContractDTO[
        MobileNotificationOutcomeVariant1View
        | MobileNotificationOutcomeVariant2View
        | MobileNotificationOutcomeVariant3View
    ]
):
    """Immutable typed view of a validated notification_outcome payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("notification_permission")
class NotificationPermission(ContractDTO[MobileNotificationPermissionView]):
    """Immutable typed view of a validated notification_permission payload."""

    __slots__ = ()
    _optional_fields = {}


__all__ = [
    "NotificationIntent",
    "NotificationOutcome",
    "NotificationPermission",
]
