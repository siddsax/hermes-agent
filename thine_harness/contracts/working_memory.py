"""Generated typed working_memory contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    RuntimeStopHookDecisionVariant1View,
    RuntimeStopHookDecisionVariant2View,
    RuntimeWorkingMemorySnapshotView,
)


@contract_type("stop_hook_decision")
class StopHookDecision(
    ContractDTO[
        RuntimeStopHookDecisionVariant1View | RuntimeStopHookDecisionVariant2View
    ]
):
    """Immutable typed view of a validated stop_hook_decision payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("working_memory_snapshot")
class WorkingMemorySnapshot(ContractDTO[RuntimeWorkingMemorySnapshotView]):
    """Immutable typed view of a validated working_memory_snapshot payload."""

    __slots__ = ()
    _optional_fields = {}


__all__ = [
    "StopHookDecision",
    "WorkingMemorySnapshot",
]
