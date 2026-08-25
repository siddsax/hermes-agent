"""Generated typed dashboard contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    RuntimeDashboardReadModelView,
    RuntimeDashboardSnapshotView,
)


@contract_type("dashboard_read_model")
class DashboardReadModel(ContractDTO[RuntimeDashboardReadModelView]):
    """Immutable typed view of a validated dashboard_read_model payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("dashboard_snapshot")
class DashboardSnapshot(ContractDTO[RuntimeDashboardSnapshotView]):
    """Immutable typed view of a validated dashboard_snapshot payload."""

    __slots__ = ()
    _optional_fields = {}


__all__ = [
    "DashboardReadModel",
    "DashboardSnapshot",
]
