"""Typed topics contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("topic_lifecycle")
class TopicLifecycle(ContractDTO):
    """Validated topic_lifecycle wire payload."""


__all__ = [
    "TopicLifecycle",
]

