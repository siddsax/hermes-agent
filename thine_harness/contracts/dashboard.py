"""Typed dashboard contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("dashboard_read_model")
class DashboardReadModel(ContractDTO):
    """Validated dashboard_read_model wire payload."""


@contract_type("dashboard_snapshot")
class DashboardSnapshot(ContractDTO):
    """Validated dashboard_snapshot wire payload."""


__all__ = [
    "DashboardReadModel",
    "DashboardSnapshot",
]

