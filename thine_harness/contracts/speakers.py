"""Typed speakers contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("speaker_cursor_outcome")
class SpeakerCursorOutcome(ContractDTO):
    """Validated speaker_cursor_outcome wire payload."""


@contract_type("speaker_mapping_event")
class SpeakerMappingEvent(ContractDTO):
    """Validated speaker_mapping_event wire payload."""


__all__ = [
    "SpeakerCursorOutcome",
    "SpeakerMappingEvent",
]

