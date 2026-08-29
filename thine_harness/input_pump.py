"""Transcript availability Input Pump and explicit backend helper adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ipaddress
import threading
import time
from typing import Callable, cast, Literal, Protocol
from urllib.parse import urlparse
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from .contracts import JSONValue
from .contracts.recovery import ExplicitRetry, InputGap
from .contracts.ports import (
    ClaimId,
    ClaimRequestId,
    MemoryVersion,
    RunId,
    SequenceNumber,
    TranscriptPort,
)
from .contracts.runtime import Tick
from .contracts.transcripts import (
    TranscriptAck,
    TranscriptCanonicalLookup,
    TranscriptClaim,
    TranscriptClaimLookup,
    TranscriptClaimRequest,
    TranscriptLeaseRenewRequest,
    TranscriptLeaseRenewResult,
    TranscriptReclaimRequest,
    TranscriptReclaimResult,
    TranscriptRelease,
)
from .run_coordinator import (
    ActiveRunLease,
    InvocationContext,
    InvocationControl,
    InvocationOutcome,
)
from .run_state import DurableRunState


_VERSION = {"major": 1, "minor": 0}
_CLAIM_LEASE_MS = 600_000
_TARGET_SOURCE_DURATION_MS = 600_000
_TARGET_TRANSCRIPT_TOKENS = 8_000
_ABSOLUTE_WINDOW_TOKENS = 200_000


def _require_exact_fields(
    value: dict[str, JSONValue], fields: set[str], *, label: str
) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} fields do not match the closed wire shape")


def _require_string(value: JSONValue, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_integer(value: JSONValue, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_positive_integer(value: JSONValue, *, label: str) -> int:
    result = _require_integer(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _require_boolean(value: JSONValue, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _require_list(value: JSONValue, *, label: str) -> list[JSONValue]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return cast(list[JSONValue], value)


def _require_adoption_kind(value: JSONValue) -> str:
    result = _require_string(value, label="adoption_kind")
    if result != "startup_existing_buffer":
        raise ValueError("adoption_kind is outside the closed recovery wire")
    return result


@dataclass(frozen=True)
class TranscriptQuarantineRequest:
    """Claim-scoped request for the backend-owned source advance."""

    claim_id: str
    logical_run_id: str
    quarantine_id: str
    failure_code: str
    fault_attempts_total: Literal[3]
    quarantined_at_ms: int

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "claim_id": self.claim_id,
            "logical_run_id": self.logical_run_id,
            "quarantine_id": self.quarantine_id,
            "failure_code": self.failure_code,
            "fault_attempts_total": self.fault_attempts_total,
            "quarantined_at_ms": self.quarantined_at_ms,
        }


@dataclass(frozen=True)
class TranscriptQuarantineResult:
    status: Literal["quarantined"]
    quarantine_id: str
    claim_id: str
    logical_run_id: str
    source_identity: str
    aggregation_buffer_ids: tuple[int, ...]
    sequence_numbers: tuple[int | None, ...]
    provenance: tuple[str, ...]
    adoption_kinds: tuple[str | None, ...]
    failure_code: str
    fault_attempts_total: Literal[3]
    quarantined_at_ms: int
    normal_cursor_advanced: bool
    input_retained: bool
    canonical_transcript_retained: bool
    input_gap: InputGap

    @classmethod
    def from_dict(cls, value: dict[str, JSONValue]) -> "TranscriptQuarantineResult":
        _require_exact_fields(
            value,
            {
                "status",
                "quarantine_id",
                "claim_id",
                "logical_run_id",
                "source_identity",
                "aggregation_buffer_ids",
                "sequence_numbers",
                "provenance",
                "adoption_kinds",
                "failure_code",
                "fault_attempts_total",
                "quarantined_at_ms",
                "normal_cursor_advanced",
                "input_retained",
                "canonical_transcript_retained",
                "input_gap",
            },
            label="transcript quarantine response",
        )
        if value["status"] != "quarantined" or value["fault_attempts_total"] != 3:
            raise ValueError("transcript quarantine response has invalid constants")
        if (
            value["normal_cursor_advanced"] is not True
            or value["input_retained"] is not True
            or value["canonical_transcript_retained"] is not True
        ):
            raise ValueError("transcript quarantine response has invalid invariants")
        buffer_ids = _require_list(
            value["aggregation_buffer_ids"], label="aggregation_buffer_ids"
        )
        sequences = _require_list(value["sequence_numbers"], label="sequence_numbers")
        provenance = _require_list(value["provenance"], label="provenance")
        adoption_kinds = _require_list(value["adoption_kinds"], label="adoption_kinds")
        if not (
            len(buffer_ids) == len(sequences) == len(provenance) == len(adoption_kinds)
        ):
            raise ValueError("transcript quarantine range arrays must align")
        return cls(
            status="quarantined",
            quarantine_id=_require_string(
                value["quarantine_id"], label="quarantine_id"
            ),
            claim_id=_require_string(value["claim_id"], label="claim_id"),
            logical_run_id=_require_string(
                value["logical_run_id"], label="logical_run_id"
            ),
            source_identity=_require_string(
                value["source_identity"], label="source_identity"
            ),
            aggregation_buffer_ids=tuple(
                _require_positive_integer(item, label="aggregation_buffer_id")
                for item in buffer_ids
            ),
            sequence_numbers=tuple(
                None
                if item is None
                else _require_positive_integer(item, label="sequence_number")
                for item in sequences
            ),
            provenance=tuple(
                _require_string(item, label="provenance") for item in provenance
            ),
            adoption_kinds=tuple(
                None if item is None else _require_adoption_kind(item)
                for item in adoption_kinds
            ),
            failure_code=_require_string(value["failure_code"], label="failure_code"),
            fault_attempts_total=3,
            quarantined_at_ms=_require_integer(
                value["quarantined_at_ms"], label="quarantined_at_ms"
            ),
            normal_cursor_advanced=_require_boolean(
                value["normal_cursor_advanced"], label="normal_cursor_advanced"
            ),
            input_retained=_require_boolean(
                value["input_retained"], label="input_retained"
            ),
            canonical_transcript_retained=_require_boolean(
                value["canonical_transcript_retained"],
                label="canonical_transcript_retained",
            ),
            input_gap=InputGap.from_dict(
                cast(dict[str, JSONValue], value["input_gap"])
            ),
        )


@dataclass(frozen=True)
class TranscriptRetryRequest:
    quarantine_id: str
    retry_run_id: str
    retry_request_id: str
    requested_at_ms: int
    lease_duration_ms: int

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "quarantine_id": self.quarantine_id,
            "retry_run_id": self.retry_run_id,
            "retry_request_id": self.retry_request_id,
            "requested_at_ms": self.requested_at_ms,
            "lease_duration_ms": self.lease_duration_ms,
        }


@dataclass(frozen=True)
class TranscriptRetryResult:
    quarantine_id: str
    retry_run_id: str
    retry_request_id: str
    requested_at_ms: int
    original_claim_id: str
    original_source_identity: str
    original_provenance: tuple[str, ...]
    normal_cursor_rewound: bool
    quarantine_retained: bool
    claim: TranscriptClaim

    @classmethod
    def from_dict(cls, value: dict[str, JSONValue]) -> "TranscriptRetryResult":
        _require_exact_fields(
            value,
            {
                "quarantine_id",
                "retry_run_id",
                "retry_request_id",
                "requested_at_ms",
                "original_claim_id",
                "original_source_identity",
                "original_provenance",
                "normal_cursor_rewound",
                "quarantine_retained",
                "claim",
            },
            label="transcript quarantine retry response",
        )
        original_provenance = _require_list(
            value["original_provenance"], label="original_provenance"
        )
        if (
            value["normal_cursor_rewound"] is not False
            or value["quarantine_retained"] is not True
        ):
            raise ValueError("transcript quarantine retry changed immutable state")
        return cls(
            quarantine_id=_require_string(
                value["quarantine_id"], label="quarantine_id"
            ),
            retry_run_id=_require_string(value["retry_run_id"], label="retry_run_id"),
            retry_request_id=_require_string(
                value["retry_request_id"], label="retry_request_id"
            ),
            requested_at_ms=_require_integer(
                value["requested_at_ms"], label="requested_at_ms"
            ),
            original_claim_id=_require_string(
                value["original_claim_id"], label="original_claim_id"
            ),
            original_source_identity=_require_string(
                value["original_source_identity"], label="original_source_identity"
            ),
            original_provenance=tuple(
                _require_string(item, label="original_provenance")
                for item in original_provenance
            ),
            normal_cursor_rewound=_require_boolean(
                value["normal_cursor_rewound"], label="normal_cursor_rewound"
            ),
            quarantine_retained=_require_boolean(
                value["quarantine_retained"], label="quarantine_retained"
            ),
            claim=TranscriptClaim.from_dict(cast(dict[str, JSONValue], value["claim"])),
        )


@dataclass(frozen=True)
class TranscriptRetryInspection:
    retry_request_id: str
    retry_run_id: str
    claim_id: str
    requested_at_ms: int
    claim_state: str


@dataclass(frozen=True)
class TranscriptQuarantineInspectionResult:
    quarantine: TranscriptQuarantineResult
    source_rows_present: bool
    retries: tuple[TranscriptRetryInspection, ...]

    @classmethod
    def from_dict(
        cls, value: dict[str, JSONValue]
    ) -> "TranscriptQuarantineInspectionResult":
        _require_exact_fields(
            value,
            {"quarantine", "source_rows_present", "retries"},
            label="transcript quarantine inspection response",
        )
        retry_values = _require_list(value["retries"], label="retries")
        retries: list[TranscriptRetryInspection] = []
        for item in retry_values:
            if not isinstance(item, dict):
                raise ValueError("retry inspection must be an object")
            retry_item = cast(dict[str, JSONValue], item)
            _require_exact_fields(
                retry_item,
                {
                    "retry_request_id",
                    "retry_run_id",
                    "claim_id",
                    "requested_at_ms",
                    "claim_state",
                },
                label="retry inspection",
            )
            retries.append(
                TranscriptRetryInspection(
                    retry_request_id=_require_string(
                        retry_item["retry_request_id"], label="retry_request_id"
                    ),
                    retry_run_id=_require_string(
                        retry_item["retry_run_id"], label="retry_run_id"
                    ),
                    claim_id=_require_string(retry_item["claim_id"], label="claim_id"),
                    requested_at_ms=_require_integer(
                        retry_item["requested_at_ms"], label="requested_at_ms"
                    ),
                    claim_state=_require_string(
                        retry_item["claim_state"], label="claim_state"
                    ),
                )
            )
        return cls(
            quarantine=TranscriptQuarantineResult.from_dict(
                cast(dict[str, JSONValue], value["quarantine"])
            ),
            source_rows_present=_require_boolean(
                value["source_rows_present"], label="source_rows_present"
            ),
            retries=tuple(retries),
        )


class TranscriptRecoveryPort(Protocol):
    def quarantine(
        self, request: TranscriptQuarantineRequest
    ) -> TranscriptQuarantineResult: ...

    def retry_quarantine(
        self, request: TranscriptRetryRequest
    ) -> TranscriptRetryResult: ...

    def inspect_quarantine(
        self, quarantine_id: str
    ) -> TranscriptQuarantineInspectionResult: ...


class TranscriptClaimNotFound(LookupError):
    """The backend has no durable claim for this request identity yet."""


@dataclass(frozen=True)
class TranscriptAvailability:
    """Content-free backend hint used only to enqueue an availability Tick."""

    available: bool
    source_hint: str | None
    occurred_at_ms: int | None

    @classmethod
    def from_dict(cls, value: dict[str, JSONValue]) -> "TranscriptAvailability":
        _require_exact_fields(
            value,
            {"available", "source_hint", "occurred_at_ms"},
            label="transcript availability response",
        )
        available = _require_boolean(value["available"], label="available")
        source_hint = value["source_hint"]
        occurred_at_ms = value["occurred_at_ms"]
        if available:
            return cls(
                available=True,
                source_hint=_require_string(source_hint, label="source_hint"),
                occurred_at_ms=_require_integer(
                    occurred_at_ms, label="occurred_at_ms"
                ),
            )
        if source_hint is not None or occurred_at_ms is not None:
            raise ValueError("unavailable transcript hint must carry null metadata")
        return cls(available=False, source_hint=None, occurred_at_ms=None)


class BackendTranscriptClient:
    """Typed loopback client exposing only frozen transcript helper operations."""

    def __init__(
        self,
        *,
        origin: str,
        credential: str,
        firebase_uid: str,
        timeout_seconds: float = 5.0,
        clock_ms: Callable[[], int] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlparse(origin)
        try:
            address = ipaddress.ip_address(parsed.hostname or "")
        except ValueError as exc:
            raise ValueError(
                "backend transcript origin must use a loopback IP literal"
            ) from exc
        if (
            parsed.scheme != "http"
            or not address.is_loopback
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("backend transcript origin must be loopback-only HTTP")
        if not credential or not firebase_uid:
            raise ValueError(
                "backend transcript credential and Firebase UID are required"
            )
        self._credential = credential
        self._firebase_uid = firebase_uid
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._client = httpx.Client(
            base_url=origin.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def availability(self) -> TranscriptAvailability:
        return TranscriptAvailability.from_dict(
            self._post("/v1/transcripts/availability", {})
        )

    def claim(self, request: TranscriptClaimRequest) -> TranscriptClaim:
        return TranscriptClaim.from_dict(
            self._post("/v1/transcripts/claims", request.to_dict())
        )

    def lookup_claim(self, claim_request_id: ClaimRequestId) -> TranscriptClaimLookup:
        return TranscriptClaimLookup.from_dict(
            self._post(
                "/v1/transcripts/claims/lookup",
                {"claim_request_id": str(claim_request_id)},
                not_found_claim_request=str(claim_request_id),
            )
        )

    def renew(self, request: TranscriptLeaseRenewRequest) -> TranscriptLeaseRenewResult:
        return TranscriptLeaseRenewResult.from_dict(
            self._post("/v1/transcripts/claims/renew", request.to_dict())
        )

    def reclaim(self, request: TranscriptReclaimRequest) -> TranscriptReclaimResult:
        return TranscriptReclaimResult.from_dict(
            self._post("/v1/transcripts/claims/reclaim", request.to_dict())
        )

    def release(self, claim_id: ClaimId, reason: str) -> TranscriptRelease:
        return TranscriptRelease.from_dict(
            self._post(
                "/v1/transcripts/claims/release",
                {
                    "claim_id": str(claim_id),
                    "reason": reason,
                    "released_at_ms": self._clock_ms(),
                },
            )
        )

    def acknowledge(
        self,
        claim_id: ClaimId,
        run_id: RunId,
        memory_version: MemoryVersion,
    ) -> TranscriptAck:
        return TranscriptAck.from_dict(
            self._post(
                "/v1/transcripts/claims/ack",
                {
                    "claim_id": str(claim_id),
                    "run_id": str(run_id),
                    "memory_version": str(memory_version),
                    "acknowledged_at_ms": self._clock_ms(),
                },
            )
        )

    def canonical_lookup(
        self, sequence_number: SequenceNumber
    ) -> TranscriptCanonicalLookup:
        return TranscriptCanonicalLookup.from_dict(
            self._post(
                "/v1/transcripts/canonical",
                {"sequence_number": int(sequence_number)},
            )
        )

    def quarantine(
        self, request: TranscriptQuarantineRequest
    ) -> TranscriptQuarantineResult:
        return TranscriptQuarantineResult.from_dict(
            self._post("/v1/transcripts/claims/quarantine", request.to_dict())
        )

    def retry_quarantine(
        self, request: TranscriptRetryRequest
    ) -> TranscriptRetryResult:
        return TranscriptRetryResult.from_dict(
            self._post("/v1/transcripts/quarantines/retry", request.to_dict())
        )

    def inspect_quarantine(
        self, quarantine_id: str
    ) -> TranscriptQuarantineInspectionResult:
        return TranscriptQuarantineInspectionResult.from_dict(
            self._post(
                "/v1/transcripts/quarantines/inspect",
                {"quarantine_id": quarantine_id},
            )
        )

    def _post(
        self,
        path: str,
        body: dict[str, JSONValue],
        *,
        not_found_claim_request: str | None = None,
    ) -> dict[str, JSONValue]:
        request_id = str(uuid.uuid4())
        response = self._client.post(
            path,
            headers={
                "Authorization": f"Bearer {self._credential}",
                "Content-Type": "application/json",
                "X-Thine-Firebase-UID": self._firebase_uid,
                "X-Request-ID": request_id,
            },
            json=body,
        )
        if response.status_code == 404 and not_found_claim_request is not None:
            raise TranscriptClaimNotFound(not_found_claim_request)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("backend transcript response must be a JSON object")
        return cast(dict[str, JSONValue], payload)


@dataclass(frozen=True)
class PreparedTranscriptInput:
    """One backend-owned claim attached to one leased P1 Logical Run."""

    claim: TranscriptClaim
    input_gaps: tuple[InputGap, ...] = ()
    explicit_retry: ExplicitRetry | None = None


def next_ten_minute_boundary_ms(now_ms: int, timezone_name: str) -> int:
    """Return the next UTC instant on a local :00/:10/.../:50 boundary."""
    if now_ms < 0:
        raise ValueError("now_ms must be non-negative")
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("timezone_name must be a configured IANA timezone") from exc
    candidate_ms = ((now_ms // 60_000) + 1) * 60_000
    # A timezone transition cannot remove every ten-minute boundary for three hours.
    for _ in range(180):
        local = datetime.fromtimestamp(candidate_ms / 1000, tz=timezone.utc).astimezone(
            zone
        )
        if local.minute % 10 == 0 and local.second == 0:
            return candidate_ms
        candidate_ms += 60_000
    raise RuntimeError("could not resolve the next local ten-minute boundary")


class TranscriptInputPump:
    """Persist availability, then claim only from inside a leased invocation."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        transcript_port: TranscriptPort,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._state = state
        self._transcript_port = transcript_port
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def scan_available(self, user_id: str) -> str | None:
        availability_reader = getattr(self._transcript_port, "availability", None)
        if not callable(availability_reader):
            raise TypeError("transcript port does not expose availability")
        availability = availability_reader()
        if not isinstance(availability, TranscriptAvailability):
            raise TypeError("transcript availability has an invalid type")
        if not availability.available:
            return None
        if availability.source_hint is None or availability.occurred_at_ms is None:
            raise ValueError("available transcript hint is incomplete")
        now_ms = self._clock_ms()
        return self.enqueue_availability(
            user_id=user_id,
            source_hint=availability.source_hint,
            occurred_at_ms=availability.occurred_at_ms,
            received_at_ms=now_ms,
        )

    def enqueue_availability(
        self,
        *,
        user_id: str,
        source_hint: str,
        occurred_at_ms: int,
        received_at_ms: int,
    ) -> str:
        if not user_id or not source_hint:
            raise ValueError("user_id and source_hint are required")
        identity = f"{user_id}\0{source_hint}"
        tick_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"thine-transcript-tick:{identity}")
        )
        logical_run_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"thine-transcript-run:{identity}")
        )
        reference_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"thine-transcript-source:{identity}")
        )
        queued_at_ms = self._clock_ms()
        tick = Tick.from_dict({
            "schema_version": _VERSION,
            "tick_id": tick_id,
            "user_id": user_id,
            "logical_run_id": logical_run_id,
            "kind": "p1_transcript",
            "priority": "p1",
            "occurred_at_ms": occurred_at_ms,
            "received_at_ms": received_at_ms,
            "queued_at_ms": queued_at_ms,
            "source_ref": {
                "kind": "transcript_availability",
                "id": reference_id,
            },
            "causation_id": None,
            "correlation_id": tick_id,
            "attempt_ordinal": 1,
            "lease": None,
            "communication_allowance_snapshot": None,
            "payload": {
                "payload_kind": "transcript_availability",
                "reference_id": reference_id,
            },
            "extensions": {},
        })
        return self._state.enqueue_transcript_availability(tick, now_ms=queued_at_ms)

    @staticmethod
    def claim_request(
        *,
        user_id: str,
        logical_run_id: str,
        claim_request_id: str,
        now_ms: int,
    ) -> TranscriptClaimRequest:
        del user_id  # user scope is carried by authenticated adapter credentials
        return TranscriptClaimRequest.from_dict({
            "schema_version": _VERSION,
            "claim_request_id": claim_request_id,
            "lease_owner": logical_run_id,
            "now_ms": now_ms,
            "lease_duration_ms": _CLAIM_LEASE_MS,
            "caps": {
                "target_source_duration_ms": _TARGET_SOURCE_DURATION_MS,
                "target_transcript_tokens": _TARGET_TRANSCRIPT_TOKENS,
                "absolute_window_tokens": _ABSOLUTE_WINDOW_TOKENS,
                "max_entries": None,
                "continuation_cursor": None,
            },
            "eligibility": {
                "source_type": "audio",
                "status_filter": None,
                "unknown_speaker_eligible": True,
                "excluded_source_types": ["audio_fast", "non_audio"],
            },
            "extensions": {},
        })

    def prepare(
        self,
        context: InvocationContext,
        *,
        lease: ActiveRunLease,
    ) -> PreparedTranscriptInput | None:
        payload = context.tick.payload
        if payload.kind != "p1_transcript":
            return None
        explicit_retry = self._state.explicit_transcript_retry(
            user_id=lease.user_id, retry_run_id=lease.logical_run_id
        )
        claim_request_id = (
            self._state.transcript_retry_request_id(
                user_id=lease.user_id, retry_run_id=lease.logical_run_id
            )
            if explicit_retry is not None
            else str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"thine-transcript-claim:{lease.logical_run_id}",
                )
            )
        )
        now_ms = self._clock_ms()
        stored = self._state.ensure_transcript_claim_request(
            user_id=lease.user_id,
            tick_id=str(payload.tick_id),
            logical_run_id=lease.logical_run_id,
            claim_request_id=claim_request_id,
            owner=lease.owner,
            attempt_id=lease.attempt_id,
            lease_token=lease.lease_token,
            now_ms=now_ms,
        )
        if stored.claim is None:
            if explicit_retry is not None:
                recovery_port = cast(TranscriptRecoveryPort, self._transcript_port)
                retried = recovery_port.retry_quarantine(
                    TranscriptRetryRequest(
                        quarantine_id=str(explicit_retry.payload.quarantine_id),
                        retry_run_id=lease.logical_run_id,
                        retry_request_id=claim_request_id,
                        requested_at_ms=int(explicit_retry.payload.created_at_ms),
                        lease_duration_ms=_CLAIM_LEASE_MS,
                    )
                )
                if (
                    retried.quarantine_id != explicit_retry.payload.quarantine_id
                    or retried.retry_run_id != lease.logical_run_id
                    or retried.retry_request_id != claim_request_id
                    or retried.requested_at_ms != explicit_retry.payload.created_at_ms
                    or retried.original_claim_id
                    != explicit_retry.payload.source_identity
                    or retried.original_source_identity
                    != explicit_retry.payload.source_identity
                    or not retried.quarantine_retained
                    or retried.normal_cursor_rewound
                ):
                    raise ValueError("backend transcript retry identity mismatch")
                claim = retried.claim
            else:
                try:
                    lookup = self._transcript_port.lookup_claim(
                        ClaimRequestId(claim_request_id)
                    )
                    claim = TranscriptClaim.from_dict(lookup.to_dict())
                except TranscriptClaimNotFound:
                    claim = self._transcript_port.claim(
                        self.claim_request(
                            user_id=lease.user_id,
                            logical_run_id=lease.logical_run_id,
                            claim_request_id=claim_request_id,
                            now_ms=now_ms,
                        )
                    )
            stored = self._state.record_transcript_claim(
                user_id=lease.user_id,
                logical_run_id=lease.logical_run_id,
                claim=claim,
                owner=lease.owner,
                attempt_id=lease.attempt_id,
                lease_token=lease.lease_token,
                now_ms=self._clock_ms(),
            )
        if stored.claim is None:
            raise RuntimeError("durable transcript claim disappeared")
        gaps = self._state.attach_pending_transcript_gaps(
            user_id=lease.user_id, logical_run_id=lease.logical_run_id
        )
        return PreparedTranscriptInput(
            stored.claim,
            input_gaps=gaps,
            explicit_retry=explicit_retry,
        )

    def enqueue_explicit_retry(
        self,
        *,
        user_id: str,
        quarantine_id: str,
        retry_run_id: str,
        created_at_ms: int,
    ) -> ExplicitRetry:
        return self._state.enqueue_transcript_retry(
            user_id=user_id,
            quarantine_id=quarantine_id,
            retry_run_id=retry_run_id,
            created_at_ms=created_at_ms,
        )


class TenMinuteTranscriptDriver:
    """Wake transcript processing only on fixed local ten-minute boundaries."""

    def __init__(
        self,
        *,
        pump: TranscriptInputPump,
        user_id: str,
        wake_coordinator: Callable[[], None],
        timezone_name: str = "Asia/Kolkata",
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._pump = pump
        self._user_id = user_id
        self._wake_coordinator = wake_coordinator
        self._timezone_name = timezone_name
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="thine-transcript-ten-minute-driver",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=2)

    def scan_now(self) -> str | None:
        tick_id = self._pump.scan_available(self._user_id)
        if tick_id is not None:
            self._wake_coordinator()
        return tick_id

    def _run(self) -> None:
        while not self._closed.is_set():
            now_ms = self._clock_ms()
            boundary_ms = next_ten_minute_boundary_ms(
                now_ms, self._timezone_name
            )
            delay = max((boundary_ms - now_ms) / 1000, 0.01)
            if self._closed.wait(delay):
                return
            while not self._closed.is_set():
                try:
                    self.scan_now()
                    break
                except Exception:
                    # A local backend restart must not permanently lose this
                    # boundary. Retry transport until it succeeds or closes.
                    if self._closed.wait(1.0):
                        return


class FakeTranscriptNoActionRuntime:
    """Deterministic decision Adapter for the model-free vertical slice."""

    def __init__(self) -> None:
        self.invocations: list[InvocationContext] = []

    def invoke(
        self,
        context: InvocationContext,
        *,
        tools: object,
        control: InvocationControl,
    ) -> InvocationOutcome:
        del tools
        if control.preemption_requested:
            return InvocationOutcome.preempted(
                remaining_work="transcript decision not started"
            )
        prepared = context.prepared_input
        if not isinstance(prepared, PreparedTranscriptInput):
            raise ValueError("fake transcript decision requires one prepared claim")
        if not prepared.claim.payload.entries:
            raise ValueError(
                "fake transcript decision cannot infer over an empty claim"
            )
        self.invocations.append(context)
        return InvocationOutcome.no_action()


__all__ = [
    "BackendTranscriptClient",
    "FakeTranscriptNoActionRuntime",
    "PreparedTranscriptInput",
    "TenMinuteTranscriptDriver",
    "TranscriptClaimNotFound",
    "TranscriptAvailability",
    "TranscriptInputPump",
    "TranscriptQuarantineRequest",
    "TranscriptQuarantineInspectionResult",
    "TranscriptQuarantineResult",
    "TranscriptRecoveryPort",
    "TranscriptRetryRequest",
    "TranscriptRetryInspection",
    "TranscriptRetryResult",
    "next_ten_minute_boundary_ms",
]
