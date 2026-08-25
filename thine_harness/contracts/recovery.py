"""Generated typed recovery contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    DataplaneInputGapView,
    RuntimeExplicitRetryView,
    RuntimeQuarantineRecordVariant1View,
    RuntimeQuarantineRecordVariant2View,
    RuntimeQuarantineRecordVariant3View,
    RuntimeQuarantineRecordVariant4View,
)


@contract_type("explicit_retry")
class ExplicitRetry(ContractDTO[RuntimeExplicitRetryView]):
    """Immutable typed view of a validated explicit_retry payload."""

    __slots__ = ()


@contract_type("input_gap")
class InputGap(ContractDTO[DataplaneInputGapView]):
    """Immutable typed view of a validated input_gap payload."""

    __slots__ = ()


@contract_type("quarantine_record")
class QuarantineRecord(
    ContractDTO[
        RuntimeQuarantineRecordVariant1View
        | RuntimeQuarantineRecordVariant2View
        | RuntimeQuarantineRecordVariant3View
        | RuntimeQuarantineRecordVariant4View
    ]
):
    """Immutable typed view of a validated quarantine_record payload."""

    __slots__ = ()


__all__ = [
    "ExplicitRetry",
    "InputGap",
    "QuarantineRecord",
]
