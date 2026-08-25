"""Generated typed topics contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    RuntimeTopicLifecycleVariant1View,
    RuntimeTopicLifecycleVariant2View,
)


@contract_type("topic_lifecycle")
class TopicLifecycle(
    ContractDTO[RuntimeTopicLifecycleVariant1View | RuntimeTopicLifecycleVariant2View]
):
    """Immutable typed view of a validated topic_lifecycle payload."""

    __slots__ = ()
    _optional_fields = {}


__all__ = [
    "TopicLifecycle",
]
