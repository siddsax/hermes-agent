"""Typed runtime contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("attempt")
class Attempt(ContractDTO):
    """Validated attempt wire payload."""


@contract_type("checkpoint")
class Checkpoint(ContractDTO):
    """Validated checkpoint wire payload."""


@contract_type("input_receipt")
class InputReceipt(ContractDTO):
    """Validated input_receipt wire payload."""


@contract_type("invocation_event")
class InvocationEvent(ContractDTO):
    """Validated invocation_event wire payload."""


@contract_type("invocation_request")
class InvocationRequest(ContractDTO):
    """Validated invocation_request wire payload."""


@contract_type("run_finalization")
class RunFinalization(ContractDTO):
    """Validated run_finalization wire payload."""


@contract_type("run_receipt")
class RunReceipt(ContractDTO):
    """Validated run_receipt wire payload."""


@contract_type("tick")
class Tick(ContractDTO):
    """Validated tick wire payload."""


@contract_type("tool_result")
class ToolResult(ContractDTO):
    """Validated tool_result wire payload."""


__all__ = [
    "Attempt",
    "Checkpoint",
    "InputReceipt",
    "InvocationEvent",
    "InvocationRequest",
    "RunFinalization",
    "RunReceipt",
    "Tick",
    "ToolResult",
]

