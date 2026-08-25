"""Typed working memory contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("stop_hook_decision")
class StopHookDecision(ContractDTO):
    """Validated stop_hook_decision wire payload."""


@contract_type("working_memory_snapshot")
class WorkingMemorySnapshot(ContractDTO):
    """Validated working_memory_snapshot wire payload."""


__all__ = [
    "StopHookDecision",
    "WorkingMemorySnapshot",
]

