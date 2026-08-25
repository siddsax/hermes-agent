"""Generated typed speakers contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    DataplaneSpeakerCursorOutcomeVariant1View,
    DataplaneSpeakerCursorOutcomeVariant2View,
    DataplaneSpeakerMappingEventView,
)


@contract_type("speaker_cursor_outcome")
class SpeakerCursorOutcome(
    ContractDTO[
        DataplaneSpeakerCursorOutcomeVariant1View
        | DataplaneSpeakerCursorOutcomeVariant2View
    ]
):
    """Immutable typed view of a validated speaker_cursor_outcome payload."""

    __slots__ = ()


@contract_type("speaker_mapping_event")
class SpeakerMappingEvent(ContractDTO[DataplaneSpeakerMappingEventView]):
    """Immutable typed view of a validated speaker_mapping_event payload."""

    __slots__ = ()


__all__ = [
    "SpeakerCursorOutcome",
    "SpeakerMappingEvent",
]
