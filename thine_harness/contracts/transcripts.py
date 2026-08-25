"""Typed transcripts contract DTOs."""

from ._base import ContractDTO, contract_type


@contract_type("transcript_ack")
class TranscriptAck(ContractDTO):
    """Validated transcript_ack wire payload."""


@contract_type("transcript_canonical_lookup")
class TranscriptCanonicalLookup(ContractDTO):
    """Validated transcript_canonical_lookup wire payload."""


@contract_type("transcript_claim")
class TranscriptClaim(ContractDTO):
    """Validated transcript_claim wire payload."""


@contract_type("transcript_claim_lookup")
class TranscriptClaimLookup(ContractDTO):
    """Validated transcript_claim_lookup wire payload."""


@contract_type("transcript_claim_request")
class TranscriptClaimRequest(ContractDTO):
    """Validated transcript_claim_request wire payload."""


@contract_type("transcript_lease_renew_request")
class TranscriptLeaseRenewRequest(ContractDTO):
    """Validated transcript_lease_renew_request wire payload."""


@contract_type("transcript_lease_renew_result")
class TranscriptLeaseRenewResult(ContractDTO):
    """Validated transcript_lease_renew_result wire payload."""


@contract_type("transcript_reclaim_request")
class TranscriptReclaimRequest(ContractDTO):
    """Validated transcript_reclaim_request wire payload."""


@contract_type("transcript_reclaim_result")
class TranscriptReclaimResult(ContractDTO):
    """Validated transcript_reclaim_result wire payload."""


@contract_type("transcript_release")
class TranscriptRelease(ContractDTO):
    """Validated transcript_release wire payload."""


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

