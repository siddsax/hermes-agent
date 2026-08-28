"""Transcript availability Input Pump and explicit backend helper adapter."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import time
from typing import Callable, cast
from urllib.parse import urlparse
import uuid

import httpx

from .contracts import JSONValue
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


class TranscriptClaimNotFound(LookupError):
    """The backend has no durable claim for this request identity yet."""


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
            raise ValueError("backend transcript credential and Firebase UID are required")
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

    def claim(self, request: TranscriptClaimRequest) -> TranscriptClaim:
        return TranscriptClaim.from_dict(
            self._post("/v1/transcripts/claims", request.to_dict())
        )

    def lookup_claim(
        self, claim_request_id: ClaimRequestId
    ) -> TranscriptClaimLookup:
        return TranscriptClaimLookup.from_dict(
            self._post(
                "/v1/transcripts/claims/lookup",
                {"claim_request_id": str(claim_request_id)},
                not_found_claim_request=str(claim_request_id),
            )
        )

    def renew(
        self, request: TranscriptLeaseRenewRequest
    ) -> TranscriptLeaseRenewResult:
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
        tick_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"thine-transcript-tick:{identity}"))
        logical_run_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"thine-transcript-run:{identity}")
        )
        reference_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"thine-transcript-source:{identity}")
        )
        queued_at_ms = self._clock_ms()
        tick = Tick.from_dict(
            {
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
            }
        )
        return self._state.enqueue_transcript_availability(
            tick, now_ms=queued_at_ms
        )

    @staticmethod
    def claim_request(
        *,
        user_id: str,
        logical_run_id: str,
        claim_request_id: str,
        now_ms: int,
    ) -> TranscriptClaimRequest:
        del user_id  # user scope is carried by authenticated adapter credentials
        return TranscriptClaimRequest.from_dict(
            {
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
            }
        )

    def prepare(
        self,
        context: InvocationContext,
        *,
        lease: ActiveRunLease,
    ) -> PreparedTranscriptInput | None:
        payload = context.tick.payload
        if payload.kind != "p1_transcript":
            return None
        claim_request_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"thine-transcript-claim:{lease.logical_run_id}",
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
        if stored.claim is not None:
            return PreparedTranscriptInput(stored.claim)
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
        return PreparedTranscriptInput(stored.claim)


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
            raise ValueError("fake transcript decision cannot infer over an empty claim")
        self.invocations.append(context)
        return InvocationOutcome.no_action()


__all__ = [
    "BackendTranscriptClient",
    "FakeTranscriptNoActionRuntime",
    "PreparedTranscriptInput",
    "TranscriptClaimNotFound",
    "TranscriptInputPump",
]
