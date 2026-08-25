"""Generated typed transcripts contract DTOs."""

from ._base import ContractDTO, contract_type
from ._views_generated import (
    DataplaneTranscriptAckView,
    DataplaneTranscriptCanonicalLookupView,
    DataplaneTranscriptClaimRequestView,
    DataplaneTranscriptClaimVariant1View,
    DataplaneTranscriptClaimVariant2View,
    DataplaneTranscriptClaimVariant3View,
    DataplaneTranscriptLeaseRenewRequestView,
    DataplaneTranscriptLeaseRenewResultView,
    DataplaneTranscriptReclaimRequestView,
    DataplaneTranscriptReclaimResultVariant1View,
    DataplaneTranscriptReclaimResultVariant2View,
    DataplaneTranscriptReleaseView,
)


@contract_type("transcript_ack")
class TranscriptAck(ContractDTO[DataplaneTranscriptAckView]):
    """Immutable typed view of a validated transcript_ack payload."""

    __slots__ = ()


@contract_type("transcript_canonical_lookup")
class TranscriptCanonicalLookup(ContractDTO[DataplaneTranscriptCanonicalLookupView]):
    """Immutable typed view of a validated transcript_canonical_lookup payload."""

    __slots__ = ()


@contract_type("transcript_claim")
class TranscriptClaim(
    ContractDTO[
        DataplaneTranscriptClaimVariant1View
        | DataplaneTranscriptClaimVariant2View
        | DataplaneTranscriptClaimVariant3View
    ]
):
    """Immutable typed view of a validated transcript_claim payload."""

    __slots__ = ()


@contract_type("transcript_claim_lookup")
class TranscriptClaimLookup(
    ContractDTO[
        DataplaneTranscriptClaimVariant1View
        | DataplaneTranscriptClaimVariant2View
        | DataplaneTranscriptClaimVariant3View
    ]
):
    """Immutable typed view of a validated transcript_claim_lookup payload."""

    __slots__ = ()


@contract_type("transcript_claim_request")
class TranscriptClaimRequest(ContractDTO[DataplaneTranscriptClaimRequestView]):
    """Immutable typed view of a validated transcript_claim_request payload."""

    __slots__ = ()


@contract_type("transcript_lease_renew_request")
class TranscriptLeaseRenewRequest(
    ContractDTO[DataplaneTranscriptLeaseRenewRequestView]
):
    """Immutable typed view of a validated transcript_lease_renew_request payload."""

    __slots__ = ()


@contract_type("transcript_lease_renew_result")
class TranscriptLeaseRenewResult(ContractDTO[DataplaneTranscriptLeaseRenewResultView]):
    """Immutable typed view of a validated transcript_lease_renew_result payload."""

    __slots__ = ()


@contract_type("transcript_reclaim_request")
class TranscriptReclaimRequest(ContractDTO[DataplaneTranscriptReclaimRequestView]):
    """Immutable typed view of a validated transcript_reclaim_request payload."""

    __slots__ = ()


@contract_type("transcript_reclaim_result")
class TranscriptReclaimResult(
    ContractDTO[
        DataplaneTranscriptReclaimResultVariant1View
        | DataplaneTranscriptReclaimResultVariant2View
    ]
):
    """Immutable typed view of a validated transcript_reclaim_result payload."""

    __slots__ = ()


@contract_type("transcript_release")
class TranscriptRelease(ContractDTO[DataplaneTranscriptReleaseView]):
    """Immutable typed view of a validated transcript_release payload."""

    __slots__ = ()


__all__ = [
    "TranscriptAck",
    "TranscriptCanonicalLookup",
    "TranscriptClaim",
    "TranscriptClaimLookup",
    "TranscriptClaimRequest",
    "TranscriptLeaseRenewRequest",
    "TranscriptLeaseRenewResult",
    "TranscriptReclaimRequest",
    "TranscriptReclaimResult",
    "TranscriptRelease",
]
