"""Generated typed runtime contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    RuntimeAttemptVariant1View,
    RuntimeAttemptVariant2View,
    RuntimeAttemptVariant3View,
    RuntimeAttemptVariant4View,
    RuntimeAttemptVariant5View,
    RuntimeAttemptVariant6View,
    RuntimeCheckpointView,
    RuntimeInputReceiptView,
    RuntimeInvocationEventVariant1View,
    RuntimeInvocationEventVariant2View,
    RuntimeInvocationEventVariant3View,
    RuntimeInvocationEventVariant4View,
    RuntimeInvocationRequestVariant10View,
    RuntimeInvocationRequestVariant1View,
    RuntimeInvocationRequestVariant2View,
    RuntimeInvocationRequestVariant3View,
    RuntimeInvocationRequestVariant4View,
    RuntimeInvocationRequestVariant5View,
    RuntimeInvocationRequestVariant6View,
    RuntimeInvocationRequestVariant7View,
    RuntimeInvocationRequestVariant8View,
    RuntimeInvocationRequestVariant9View,
    RuntimeRunFinalizationVariant1View,
    RuntimeRunFinalizationVariant2View,
    RuntimeRunFinalizationVariant3View,
    RuntimeRunFinalizationVariant4View,
    RuntimeRunFinalizationVariant5View,
    RuntimeRunFinalizationVariant6View,
    RuntimeRunFinalizationVariant7View,
    RuntimeRunFinalizationVariant8View,
    RuntimeRunReceiptVariant1View,
    RuntimeRunReceiptVariant2View,
    RuntimeTickVariant1View,
    RuntimeTickVariant2View,
    RuntimeTickVariant3View,
    RuntimeTickVariant4View,
    RuntimeTickVariant5View,
    RuntimeToolResultVariant1View,
    RuntimeToolResultVariant2View,
)


@contract_type("attempt")
class Attempt(
    ContractDTO[
        RuntimeAttemptVariant1View
        | RuntimeAttemptVariant2View
        | RuntimeAttemptVariant3View
        | RuntimeAttemptVariant4View
        | RuntimeAttemptVariant5View
        | RuntimeAttemptVariant6View
    ]
):
    """Immutable typed view of a validated attempt payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("checkpoint")
class Checkpoint(ContractDTO[RuntimeCheckpointView]):
    """Immutable typed view of a validated checkpoint payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("input_receipt")
class InputReceipt(ContractDTO[RuntimeInputReceiptView]):
    """Immutable typed view of a validated input_receipt payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("invocation_event")
class InvocationEvent(
    ContractDTO[
        RuntimeInvocationEventVariant1View
        | RuntimeInvocationEventVariant2View
        | RuntimeInvocationEventVariant3View
        | RuntimeInvocationEventVariant4View
    ]
):
    """Immutable typed view of a validated invocation_event payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("invocation_request")
class InvocationRequest(
    ContractDTO[
        RuntimeInvocationRequestVariant1View
        | RuntimeInvocationRequestVariant2View
        | RuntimeInvocationRequestVariant3View
        | RuntimeInvocationRequestVariant4View
        | RuntimeInvocationRequestVariant5View
        | RuntimeInvocationRequestVariant6View
        | RuntimeInvocationRequestVariant7View
        | RuntimeInvocationRequestVariant8View
        | RuntimeInvocationRequestVariant9View
        | RuntimeInvocationRequestVariant10View
    ]
):
    """Immutable typed view of a validated invocation_request payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("run_finalization")
class RunFinalization(
    ContractDTO[
        RuntimeRunFinalizationVariant1View
        | RuntimeRunFinalizationVariant2View
        | RuntimeRunFinalizationVariant3View
        | RuntimeRunFinalizationVariant4View
        | RuntimeRunFinalizationVariant5View
        | RuntimeRunFinalizationVariant6View
        | RuntimeRunFinalizationVariant7View
        | RuntimeRunFinalizationVariant8View
    ]
):
    """Immutable typed view of a validated run_finalization payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("run_receipt")
class RunReceipt(
    ContractDTO[RuntimeRunReceiptVariant1View | RuntimeRunReceiptVariant2View]
):
    """Immutable typed view of a validated run_receipt payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("tick")
class Tick(
    ContractDTO[
        RuntimeTickVariant1View
        | RuntimeTickVariant2View
        | RuntimeTickVariant3View
        | RuntimeTickVariant4View
        | RuntimeTickVariant5View
    ]
):
    """Immutable typed view of a validated tick payload."""

    __slots__ = ()
    _optional_fields = {}


@contract_type("tool_result")
class ToolResult(
    ContractDTO[RuntimeToolResultVariant1View | RuntimeToolResultVariant2View]
):
    """Immutable typed view of a validated tool_result payload."""

    __slots__ = ()
    _optional_fields = {}


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
