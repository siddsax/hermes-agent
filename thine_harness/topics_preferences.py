"""Durable Topics and explicit narrow preferences for the local Thine profile.

This is deliberately not a knowledge base. Topics retain only repetition-policy
state and evidence/action receipts; explicit corrections retain only bounded
user-authorized key/value facts that must outlive Working Memory compaction.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import re
import threading
import time
from typing import Any, Callable, Iterator, Literal, Mapping, cast
import uuid

from .contracts import JSONValue
from .contracts.preferences import Preferences
from .contracts.topics import TopicLifecycle
from .run_coordinator import InvocationContext
from .run_state import DurableRunState, DurableStateError


TOPIC_INSPECT_TOOL_NAME = "thine_topics_inspect"
TOPIC_UPDATE_TOOL_NAME = "thine_topics_update"
TOPIC_TOOLSET = "local-thine-transcripts"

DAY_MS = 24 * 60 * 60 * 1000
PERMISSION_CADENCE_MS = 7 * DAY_MS
MAX_TOPIC_ITEMS = 50
MAX_CORRECTIONS = 50
_VERSION = {"major": 1, "minor": 0}
_TOPIC_KEY = re.compile(r"^[a-z0-9_]{1,128}$")
_CORRECTION_KEY = re.compile(r"^[a-z0-9_]{1,128}$")
_PREFERENCE_KEYS = (
    "notifications_enabled",
    "speaker_tag_nudges_enabled",
)
_NON_DISABLEABLE = (
    "background_inference",
    "home_mutation",
    "schedule_creation",
    "proactive_chat",
)


TOPIC_INSPECT_TOOL_SCHEMA = {
    "name": TOPIC_INSPECT_TOOL_NAME,
    "description": (
        "Inspect durable topic repetition gates, relevant evidence/action receipts, "
        "explicit user corrections, and the two narrow preferences. Durable explicit "
        "state overrides Working Memory and inferred state. This never restores an old "
        "Working Memory snapshot."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic_key": {
                "type": "string",
                "pattern": "^[a-z0-9_]+$",
                "maxLength": 128,
            }
        },
        "additionalProperties": False,
    },
}

TOPIC_UPDATE_TOOL_SCHEMA = {
    "name": TOPIC_UPDATE_TOOL_NAME,
    "description": (
        "Update one durable Topic, record one receipted Topic action, or persist an "
        "explicit P0 user correction/preference. Preference changes are accepted only "
        "when the active user message explicitly requests that exact value. Only "
        "notifications_enabled and speaker_tag_nudges_enabled are switchable; "
        "background inference, Home mutation, scheduling, and proactive chat cannot "
        "be disabled."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "record_topic_action",
                    "set_topic_state",
                    "set_preference",
                    "record_correction",
                ],
            },
            "topic_key": {
                "type": "string",
                "pattern": "^[a-z0-9_]+$",
                "maxLength": 128,
            },
            "state": {
                "type": "string",
                "enum": [
                    "proposed",
                    "acknowledged",
                    "snoozed",
                    "resolved",
                    "expired",
                ],
            },
            "snooze_until_ms": {"type": ["integer", "null"], "minimum": 0},
            "last_action": {"type": "string", "minLength": 1, "maxLength": 500},
            "evidence_refs": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 256},
                "uniqueItems": True,
                "maxItems": 50,
            },
            "evidence_at_ms": {"type": "integer", "minimum": 0},
            "receipt_refs": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 256},
                "uniqueItems": True,
                "maxItems": 50,
            },
            "preference_key": {"type": "string", "maxLength": 128},
            "preference_value": {"type": "boolean"},
            "correction_key": {
                "type": "string",
                "pattern": "^[a-z0-9_]+$",
                "maxLength": 128,
            },
            "corrected_value": {"type": "string", "minLength": 1, "maxLength": 1000},
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}


class TopicRepeatBlocked(DurableStateError):
    def __init__(self, eligibility: "RepeatEligibility") -> None:
        super().__init__(eligibility.reason)
        self.eligibility = eligibility


@dataclass(frozen=True)
class RepeatEligibility:
    eligible: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {"eligible": self.eligible, "reason": self.reason}


@dataclass(frozen=True)
class TopicRecord:
    lifecycle: TopicLifecycle
    last_evidence_at_ms: int | None
    last_action: str | None
    receipt_refs: tuple[str, ...]

    @property
    def topic_key(self) -> str:
        return str(self.lifecycle.payload.topic_key)

    def to_dict(self) -> dict[str, object]:
        return {
            "lifecycle": self.lifecycle.to_dict(),
            "last_evidence_at_ms": self.last_evidence_at_ms,
            "last_action": self.last_action,
            "receipt_refs": list(self.receipt_refs),
        }


@dataclass(frozen=True)
class TopicUpdateReceipt:
    receipt_id: str
    operation: str
    subject_key: str
    created_at_ms: int

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "operation": self.operation,
            "subject_key": self.subject_key,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class TopicActionResult:
    topic: TopicRecord
    receipt_id: str


@dataclass(frozen=True)
class _ActiveInvocation:
    user_id: str
    logical_run_id: str
    kind: str
    user_message_id: str | None = None
    user_message_text: str | None = None


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _merge_refs(current: tuple[str, ...], incoming: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*current, *incoming]))[-MAX_TOPIC_ITEMS:]


class TopicPreferenceService:
    """Transactional policy owner for Topics, explicit preferences, and corrections."""

    def __init__(
        self,
        state: DurableRunState,
        *,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._state = state
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def repeat_eligibility(
        self,
        *,
        user_id: str,
        topic_key: str,
        proposed_evidence_refs: tuple[str, ...] = (),
        proposed_evidence_at_ms: int | None = None,
    ) -> RepeatEligibility:
        self._validate_topic_key(topic_key)
        now_ms = self._clock_ms()
        with self._state._connect() as connection:
            row = connection.execute(
                "SELECT * FROM durable_topics WHERE user_id = ? AND topic_key = ?",
                (user_id, topic_key),
            ).fetchone()
            memory = connection.execute(
                "SELECT markdown FROM working_memory_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return self._eligibility(
            row,
            topic_key=topic_key,
            working_memory="" if memory is None else str(memory["markdown"]),
            now_ms=now_ms,
            proposed_evidence_refs=proposed_evidence_refs,
            proposed_evidence_at_ms=proposed_evidence_at_ms,
        )

    def record_topic_action(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        topic_key: str,
        last_action: str,
        evidence_refs: tuple[str, ...],
        evidence_at_ms: int,
        receipt_refs: tuple[str, ...],
    ) -> TopicActionResult:
        self._validate_topic_key(topic_key)
        self._validate_refs(evidence_refs, "evidence_refs")
        self._validate_refs(receipt_refs, "receipt_refs")
        if not last_action.strip() or len(last_action) > 500:
            raise ValueError("last_action must contain 1 to 500 characters")
        if not evidence_refs:
            raise ValueError("record_topic_action requires evidence_refs")
        if evidence_at_ms < 0:
            raise ValueError("evidence_at_ms must be non-negative")
        now_ms = self._clock_ms()
        if evidence_at_ms > now_ms:
            raise ValueError("evidence_at_ms cannot be in the future")
        intent = {
            "operation": "record_topic_action",
            "topic_key": topic_key,
            "last_action": last_action,
            "evidence_refs": list(evidence_refs),
            "evidence_at_ms": evidence_at_ms,
            "receipt_refs": list(receipt_refs),
        }
        fingerprint = hashlib.sha256(_canonical(intent).encode()).hexdigest()
        with self._state._transaction() as connection:
            replay = self._receipt_row(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                fingerprint=fingerprint,
            )
            if replay is not None:
                result = json.loads(str(replay["result_json"]))
                return TopicActionResult(
                    topic=self._topic_from_dict(
                        cast(dict[str, object], result["topic"])
                    ),
                    receipt_id=str(replay["receipt_id"]),
                )
            row = connection.execute(
                "SELECT * FROM durable_topics WHERE user_id = ? AND topic_key = ?",
                (user_id, topic_key),
            ).fetchone()
            memory = connection.execute(
                "SELECT markdown FROM working_memory_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            eligibility = self._eligibility(
                row,
                topic_key=topic_key,
                working_memory="" if memory is None else str(memory["markdown"]),
                now_ms=now_ms,
                proposed_evidence_refs=evidence_refs,
                proposed_evidence_at_ms=evidence_at_ms,
            )
            if not eligibility.eligible:
                raise TopicRepeatBlocked(eligibility)
            current = None if row is None else self._topic_from_row(row)
            evidence = _merge_refs(
                ()
                if current is None
                else tuple(current.lifecycle.payload.evidence_refs),
                evidence_refs,
            )
            receipts = _merge_refs(
                () if current is None else current.receipt_refs,
                receipt_refs,
            )
            topic_id = (
                f"topic:{uuid.uuid5(uuid.NAMESPACE_URL, f'{user_id}:{topic_key}')}"
                if row is None
                else str(row["topic_id"])
            )
            next_eligible = now_ms + (
                PERMISSION_CADENCE_MS if topic_key == "enable_notifications" else DAY_MS
            )
            connection.execute(
                """
                INSERT INTO durable_topics (
                    topic_id, user_id, topic_key, state, last_asked_at_ms,
                    next_eligible_at_ms, evidence_refs_json,
                    last_evidence_at_ms, last_action, receipt_refs_json,
                    authorizing_message_id, do_not_ask, updated_at_ms
                ) VALUES (?, ?, ?, 'asked', ?, ?, ?, ?, ?, ?, NULL, 0, ?)
                ON CONFLICT(user_id, topic_key) DO UPDATE SET
                    state = 'asked',
                    last_asked_at_ms = excluded.last_asked_at_ms,
                    next_eligible_at_ms = excluded.next_eligible_at_ms,
                    evidence_refs_json = excluded.evidence_refs_json,
                    last_evidence_at_ms = excluded.last_evidence_at_ms,
                    last_action = excluded.last_action,
                    receipt_refs_json = excluded.receipt_refs_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                (
                    topic_id,
                    user_id,
                    topic_key,
                    now_ms,
                    next_eligible,
                    _canonical(list(evidence)),
                    evidence_at_ms,
                    last_action,
                    _canonical(list(receipts)),
                    now_ms,
                ),
            )
            topic = self._topic_from_row(
                connection.execute(
                    "SELECT * FROM durable_topics WHERE user_id = ? AND topic_key = ?",
                    (user_id, topic_key),
                ).fetchone()
            )
            receipt = self._store_receipt(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                operation="record_topic_action",
                subject_key=topic_key,
                fingerprint=fingerprint,
                result={"topic": topic.to_dict()},
                now_ms=now_ms,
            )
        return TopicActionResult(topic=topic, receipt_id=receipt.receipt_id)

    def set_topic_state(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        topic_key: str,
        state: Literal["proposed", "acknowledged", "snoozed", "resolved", "expired"],
        snooze_until_ms: int | None = None,
        authorizing_message_id: str | None = None,
    ) -> tuple[TopicRecord, TopicUpdateReceipt]:
        self._validate_topic_key(topic_key)
        if state == "snoozed" and (
            snooze_until_ms is None or snooze_until_ms <= self._clock_ms()
        ):
            raise ValueError("snoozed Topics require a future snooze_until_ms")
        if state != "snoozed" and snooze_until_ms is not None:
            raise ValueError("snooze_until_ms is only valid for snoozed Topics")
        now_ms = self._clock_ms()
        intent = {
            "operation": "set_topic_state",
            "topic_key": topic_key,
            "state": state,
            "snooze_until_ms": snooze_until_ms,
            "authorizing_message_id": authorizing_message_id,
        }
        fingerprint = hashlib.sha256(_canonical(intent).encode()).hexdigest()
        with self._state._transaction() as connection:
            replay = self._receipt_row(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                fingerprint=fingerprint,
            )
            if replay is not None:
                result = json.loads(str(replay["result_json"]))
                return (
                    self._topic_from_dict(cast(dict[str, object], result["topic"])),
                    self._receipt_from_row(replay),
                )
            existing = connection.execute(
                "SELECT * FROM durable_topics WHERE user_id = ? AND topic_key = ?",
                (user_id, topic_key),
            ).fetchone()
            topic_id = (
                f"topic:{uuid.uuid5(uuid.NAMESPACE_URL, f'{user_id}:{topic_key}')}"
                if existing is None
                else str(existing["topic_id"])
            )
            evidence_json = (
                "[]" if existing is None else str(existing["evidence_refs_json"])
            )
            receipt_json = (
                "[]" if existing is None else str(existing["receipt_refs_json"])
            )
            last_asked = None if existing is None else existing["last_asked_at_ms"]
            if state != "proposed" and last_asked is None:
                last_asked = now_ms
            last_evidence = (
                None if existing is None else existing["last_evidence_at_ms"]
            )
            last_action = None if existing is None else existing["last_action"]
            do_not_ask = 0 if existing is None else int(existing["do_not_ask"])
            connection.execute(
                """
                INSERT INTO durable_topics (
                    topic_id, user_id, topic_key, state, last_asked_at_ms,
                    next_eligible_at_ms, evidence_refs_json,
                    last_evidence_at_ms, last_action, receipt_refs_json,
                    authorizing_message_id, do_not_ask, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, topic_key) DO UPDATE SET
                    state = excluded.state,
                    last_asked_at_ms = excluded.last_asked_at_ms,
                    next_eligible_at_ms = excluded.next_eligible_at_ms,
                    authorizing_message_id = COALESCE(
                        excluded.authorizing_message_id,
                        durable_topics.authorizing_message_id
                    ),
                    updated_at_ms = excluded.updated_at_ms
                """,
                (
                    topic_id,
                    user_id,
                    topic_key,
                    state,
                    last_asked,
                    snooze_until_ms,
                    evidence_json,
                    last_evidence,
                    last_action,
                    receipt_json,
                    authorizing_message_id,
                    do_not_ask,
                    now_ms,
                ),
            )
            topic = self._topic_from_row(
                connection.execute(
                    "SELECT * FROM durable_topics WHERE user_id = ? AND topic_key = ?",
                    (user_id, topic_key),
                ).fetchone()
            )
            receipt = self._store_receipt(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                operation="set_topic_state",
                subject_key=topic_key,
                fingerprint=fingerprint,
                result={"topic": topic.to_dict()},
                now_ms=now_ms,
            )
        return topic, receipt

    def set_preference(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        preference_key: str,
        value: bool,
        authorizing_message_id: str,
    ) -> tuple[Preferences, TopicUpdateReceipt]:
        if preference_key not in _PREFERENCE_KEYS:
            raise ValueError("preference_not_switchable")
        if not authorizing_message_id:
            raise ValueError("authorizing_message_id is required")
        now_ms = self._clock_ms()
        intent = {
            "operation": "set_preference",
            "preference_key": preference_key,
            "value": value,
            "authorizing_message_id": authorizing_message_id,
        }
        fingerprint = hashlib.sha256(_canonical(intent).encode()).hexdigest()
        with self._state._transaction() as connection:
            replay = self._receipt_row(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                fingerprint=fingerprint,
            )
            if replay is not None:
                result = json.loads(str(replay["result_json"]))
                return (
                    Preferences.from_dict(
                        cast(dict[str, JSONValue], result["preferences"])
                    ),
                    self._receipt_from_row(replay),
                )
            revision_row = connection.execute(
                "SELECT MAX(revision) AS revision FROM explicit_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            revision = int(revision_row["revision"] or 0) + 1
            connection.execute(
                """
                INSERT INTO explicit_preferences (
                    user_id, preference_key, preference_value,
                    authorizing_message_id, updated_at_ms, revision
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, preference_key) DO UPDATE SET
                    preference_value = excluded.preference_value,
                    authorizing_message_id = excluded.authorizing_message_id,
                    updated_at_ms = excluded.updated_at_ms,
                    revision = excluded.revision
                """,
                (
                    user_id,
                    preference_key,
                    int(value),
                    authorizing_message_id,
                    now_ms,
                    revision,
                ),
            )
            topic_key = (
                "enable_notifications"
                if preference_key == "notifications_enabled"
                else "tag_unknown_speakers"
            )
            existing = connection.execute(
                "SELECT * FROM durable_topics WHERE user_id = ? AND topic_key = ?",
                (user_id, topic_key),
            ).fetchone()
            topic_id = (
                f"topic:{uuid.uuid5(uuid.NAMESPACE_URL, f'{user_id}:{topic_key}')}"
                if existing is None
                else str(existing["topic_id"])
            )
            state = "acknowledged" if value else "snoozed"
            last_asked_at_ms = (
                now_ms
                if existing is None or existing["last_asked_at_ms"] is None
                else int(existing["last_asked_at_ms"])
            )
            connection.execute(
                """
                INSERT INTO durable_topics (
                    topic_id, user_id, topic_key, state, last_asked_at_ms,
                    next_eligible_at_ms, evidence_refs_json,
                    last_evidence_at_ms, last_action, receipt_refs_json,
                    authorizing_message_id, do_not_ask, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, NULL, '[]', NULL, NULL, '[]', ?, ?, ?)
                ON CONFLICT(user_id, topic_key) DO UPDATE SET
                    state = excluded.state,
                    last_asked_at_ms = excluded.last_asked_at_ms,
                    next_eligible_at_ms = NULL,
                    authorizing_message_id = excluded.authorizing_message_id,
                    do_not_ask = excluded.do_not_ask,
                    updated_at_ms = excluded.updated_at_ms
                """,
                (
                    topic_id,
                    user_id,
                    topic_key,
                    state,
                    last_asked_at_ms,
                    authorizing_message_id,
                    int(not value),
                    now_ms,
                ),
            )
            preferences = self._preferences(connection, user_id)
            receipt = self._store_receipt(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                operation="set_preference",
                subject_key=preference_key,
                fingerprint=fingerprint,
                result={"preferences": preferences.to_dict()},
                now_ms=now_ms,
            )
        return preferences, receipt

    def record_correction(
        self,
        *,
        user_id: str,
        logical_run_id: str,
        correction_key: str,
        corrected_value: str,
        authorizing_message_id: str,
    ) -> tuple[dict[str, object], TopicUpdateReceipt]:
        if not _CORRECTION_KEY.fullmatch(correction_key):
            raise ValueError("invalid correction_key")
        if not corrected_value.strip() or len(corrected_value) > 1000:
            raise ValueError("corrected_value must contain 1 to 1000 characters")
        now_ms = self._clock_ms()
        intent = {
            "operation": "record_correction",
            "correction_key": correction_key,
            "corrected_value": corrected_value,
            "authorizing_message_id": authorizing_message_id,
        }
        fingerprint = hashlib.sha256(_canonical(intent).encode()).hexdigest()
        with self._state._transaction() as connection:
            replay = self._receipt_row(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                fingerprint=fingerprint,
            )
            if replay is not None:
                result = json.loads(str(replay["result_json"]))
                return cast(
                    dict[str, object], result["correction"]
                ), self._receipt_from_row(replay)
            correction = {
                "correction_key": correction_key,
                "corrected_value": corrected_value,
                "authorizing_message_id": authorizing_message_id,
                "updated_at_ms": now_ms,
            }
            connection.execute(
                """
                INSERT INTO explicit_corrections (
                    user_id, correction_key, corrected_value,
                    authorizing_message_id, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, correction_key) DO UPDATE SET
                    corrected_value = excluded.corrected_value,
                    authorizing_message_id = excluded.authorizing_message_id,
                    updated_at_ms = excluded.updated_at_ms
                """,
                (
                    user_id,
                    correction_key,
                    corrected_value,
                    authorizing_message_id,
                    now_ms,
                ),
            )
            receipt = self._store_receipt(
                connection,
                user_id=user_id,
                logical_run_id=logical_run_id,
                operation="record_correction",
                subject_key=correction_key,
                fingerprint=fingerprint,
                result={"correction": correction},
                now_ms=now_ms,
            )
        return correction, receipt

    def preference_value(self, user_id: str, key: str) -> bool | None:
        if key not in _PREFERENCE_KEYS:
            raise ValueError("preference_not_switchable")
        with self._state._connect() as connection:
            row = connection.execute(
                """
                SELECT preference_value FROM explicit_preferences
                WHERE user_id = ? AND preference_key = ?
                """,
                (user_id, key),
            ).fetchone()
        return None if row is None else bool(row["preference_value"])

    def prompt_context(self, user_id: str) -> dict[str, object]:
        now_ms = self._clock_ms()
        with self._state._connect() as connection:
            topic_rows = connection.execute(
                """
                SELECT * FROM durable_topics WHERE user_id = ?
                ORDER BY updated_at_ms DESC, topic_key LIMIT ?
                """,
                (user_id, MAX_TOPIC_ITEMS),
            ).fetchall()
            correction_rows = connection.execute(
                """
                SELECT correction_key, corrected_value, authorizing_message_id,
                       updated_at_ms
                FROM explicit_corrections WHERE user_id = ?
                ORDER BY updated_at_ms DESC, correction_key LIMIT ?
                """,
                (user_id, MAX_CORRECTIONS),
            ).fetchall()
            memory = connection.execute(
                "SELECT markdown FROM working_memory_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            preferences = self._preferences(connection, user_id)
        markdown = "" if memory is None else str(memory["markdown"])
        topics: list[dict[str, object]] = []
        for row in topic_rows:
            topic = self._topic_from_row(row)
            eligibility = self._eligibility(
                row,
                topic_key=topic.topic_key,
                working_memory=markdown,
                now_ms=now_ms,
                proposed_evidence_refs=(),
                proposed_evidence_at_ms=None,
            )
            topics.append({
                **topic.to_dict(),
                "recent_working_memory_mentions_topic": self._memory_mentions(
                    markdown, topic.topic_key
                ),
                "repeat_eligibility": eligibility.to_dict(),
            })
        return {
            "topics_preferences": {
                "topics": topics,
                "preferences": preferences.to_dict(),
                "explicit_corrections": [
                    {
                        "correction_key": str(row["correction_key"]),
                        "corrected_value": str(row["corrected_value"]),
                        "authorizing_message_id": str(row["authorizing_message_id"]),
                        "updated_at_ms": int(row["updated_at_ms"]),
                    }
                    for row in correction_rows
                ],
                "precedence": (
                    "explicit_preferences_and_corrections_override_inferred_state"
                ),
                "working_memory_role": "recent_operational_continuity_only",
                "memory_restore_available": False,
            }
        }

    def inspect(self, user_id: str, topic_key: str | None = None) -> dict[str, object]:
        context = self.prompt_context(user_id)["topics_preferences"]
        assert isinstance(context, dict)
        if topic_key is None:
            return context
        self._validate_topic_key(topic_key)
        return {
            **context,
            "topics": [
                topic
                for topic in cast(list[dict[str, object]], context["topics"])
                if cast(dict[str, object], topic["lifecycle"])["topic_key"] == topic_key
            ],
        }

    @staticmethod
    def _eligibility(
        row: Any | None,
        *,
        topic_key: str,
        working_memory: str,
        now_ms: int,
        proposed_evidence_refs: tuple[str, ...],
        proposed_evidence_at_ms: int | None,
    ) -> RepeatEligibility:
        if row is None:
            return RepeatEligibility(True, "new_topic")
        if bool(row["do_not_ask"]):
            return RepeatEligibility(False, "explicit_do_not_ask")
        state = str(row["state"])
        if state in {"resolved", "expired"}:
            return RepeatEligibility(False, f"topic_{state}")
        next_eligible = row["next_eligible_at_ms"]
        if (
            state == "snoozed"
            and next_eligible is not None
            and now_ms < int(next_eligible)
        ):
            return RepeatEligibility(False, "topic_snoozed")
        last_asked = row["last_asked_at_ms"]
        if last_asked is None:
            return RepeatEligibility(True, "not_previously_asked")
        last_asked_ms = int(last_asked)
        if topic_key == "enable_notifications":
            if now_ms < last_asked_ms + PERMISSION_CADENCE_MS:
                return RepeatEligibility(False, "permission_cadence")
            return RepeatEligibility(True, "permission_cadence_elapsed")
        current_refs = set(json.loads(str(row["evidence_refs_json"])))
        fresh_evidence = (
            proposed_evidence_at_ms is not None
            and last_asked_ms < proposed_evidence_at_ms <= now_ms
            and proposed_evidence_at_ms >= now_ms - DAY_MS
            and bool(set(proposed_evidence_refs) - current_refs)
        )
        if fresh_evidence:
            return RepeatEligibility(True, "fresh_evidence")
        if TopicPreferenceService._memory_mentions(working_memory, topic_key):
            return RepeatEligibility(False, "recent_working_memory")
        if now_ms < last_asked_ms + DAY_MS:
            return RepeatEligibility(False, "cooldown_requires_fresh_evidence")
        return RepeatEligibility(True, "cooldown_elapsed")

    @staticmethod
    def _memory_mentions(markdown: str, topic_key: str) -> bool:
        return bool(markdown) and topic_key.casefold() in markdown.casefold()

    @staticmethod
    def _validate_topic_key(topic_key: str) -> None:
        if not _TOPIC_KEY.fullmatch(topic_key):
            raise ValueError("invalid topic_key")

    @staticmethod
    def _validate_refs(refs: tuple[str, ...], name: str) -> None:
        if len(refs) > MAX_TOPIC_ITEMS or len(refs) != len(set(refs)):
            raise ValueError(f"{name} must contain at most 50 unique values")
        if any(not ref or len(ref) > 256 for ref in refs):
            raise ValueError(f"{name} values must contain 1 to 256 characters")

    @staticmethod
    def _receipt_row(
        connection: Any,
        *,
        user_id: str,
        logical_run_id: str,
        fingerprint: str,
    ) -> Any | None:
        return connection.execute(
            """
            SELECT * FROM topic_preference_receipts
            WHERE user_id = ? AND logical_run_id = ? AND intent_fingerprint = ?
            """,
            (user_id, logical_run_id, fingerprint),
        ).fetchone()

    @staticmethod
    def _store_receipt(
        connection: Any,
        *,
        user_id: str,
        logical_run_id: str,
        operation: str,
        subject_key: str,
        fingerprint: str,
        result: dict[str, object],
        now_ms: int,
    ) -> TopicUpdateReceipt:
        receipt_id = "topic-receipt:" + str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{user_id}:{logical_run_id}:{fingerprint}",
            )
        )
        connection.execute(
            """
            INSERT INTO topic_preference_receipts (
                receipt_id, user_id, logical_run_id, operation, subject_key,
                intent_fingerprint, result_json, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                user_id,
                logical_run_id,
                operation,
                subject_key,
                fingerprint,
                _canonical(result),
                now_ms,
            ),
        )
        return TopicUpdateReceipt(receipt_id, operation, subject_key, now_ms)

    @staticmethod
    def _receipt_from_row(row: Any) -> TopicUpdateReceipt:
        return TopicUpdateReceipt(
            receipt_id=str(row["receipt_id"]),
            operation=str(row["operation"]),
            subject_key=str(row["subject_key"]),
            created_at_ms=int(row["created_at_ms"]),
        )

    @staticmethod
    def _preferences(connection: Any, user_id: str) -> Preferences:
        rows = connection.execute(
            """
            SELECT * FROM explicit_preferences WHERE user_id = ?
            ORDER BY preference_key
            """,
            (user_id,),
        ).fetchall()
        revision = max((int(row["revision"]) for row in rows), default=1)
        return Preferences.from_dict({
            "schema_version": _VERSION,
            "user_id": user_id,
            "revision": revision,
            "narrow_preferences": [
                {
                    "key": str(row["preference_key"]),
                    "value": bool(row["preference_value"]),
                    "switchable_by_agent_on_explicit_user_request": True,
                    "authorizing_message_id": str(row["authorizing_message_id"]),
                    "updated_at_ms": int(row["updated_at_ms"]),
                }
                for row in rows
            ],
            "non_disableable_capabilities": list(_NON_DISABLEABLE),
            "extensions": {},
        })

    @staticmethod
    def _topic_from_row(row: Any) -> TopicRecord:
        if row is None:
            raise DurableStateError("durable Topic disappeared")
        lifecycle = TopicLifecycle.from_dict({
            "schema_version": _VERSION,
            "topic_id": str(row["topic_id"]),
            "user_id": str(row["user_id"]),
            "topic_key": str(row["topic_key"]),
            "state": str(row["state"]),
            "last_asked_at_ms": (
                None
                if row["last_asked_at_ms"] is None
                else int(row["last_asked_at_ms"])
            ),
            "next_eligible_at_ms": (
                None
                if row["next_eligible_at_ms"] is None
                else int(row["next_eligible_at_ms"])
            ),
            "evidence_refs": list(json.loads(str(row["evidence_refs_json"]))),
            "authorizing_message_id": (
                None
                if row["authorizing_message_id"] is None
                else str(row["authorizing_message_id"])
            ),
            "do_not_ask": bool(row["do_not_ask"]),
            "updated_at_ms": int(row["updated_at_ms"]),
            "extensions": {},
        })
        return TopicRecord(
            lifecycle=lifecycle,
            last_evidence_at_ms=(
                None
                if row["last_evidence_at_ms"] is None
                else int(row["last_evidence_at_ms"])
            ),
            last_action=(
                None if row["last_action"] is None else str(row["last_action"])
            ),
            receipt_refs=tuple(json.loads(str(row["receipt_refs_json"]))),
        )

    @staticmethod
    def _topic_from_dict(value: dict[str, object]) -> TopicRecord:
        return TopicRecord(
            lifecycle=TopicLifecycle.from_dict(
                cast(dict[str, JSONValue], value["lifecycle"])
            ),
            last_evidence_at_ms=cast(int | None, value["last_evidence_at_ms"]),
            last_action=cast(str | None, value["last_action"]),
            receipt_refs=tuple(cast(list[str], value["receipt_refs"])),
        )


class TopicPreferenceToolBinding:
    """Expose typed inspect/update helpers in one invocation-scoped binding."""

    def __init__(self, *, service: TopicPreferenceService) -> None:
        self._service = service
        self._lock = threading.Lock()
        self._active: _ActiveInvocation | None = None

    @contextmanager
    def activate(self, context: InvocationContext) -> Iterator[None]:
        payload = context.tick.payload
        with self._activation(
            _ActiveInvocation(
                user_id=str(payload.user_id),
                logical_run_id=str(payload.logical_run_id),
                kind=str(payload.kind),
            )
        ):
            yield

    @contextmanager
    def activate_p0(
        self,
        context: InvocationContext,
        *,
        user_message_id: str,
        user_message_text: str,
    ) -> Iterator[None]:
        payload = context.tick.payload
        if str(payload.kind) != "p0_user_chat":
            raise ValueError("P0 authorization requires a p0_user_chat Tick")
        with self._activation(
            _ActiveInvocation(
                user_id=str(payload.user_id),
                logical_run_id=str(payload.logical_run_id),
                kind=str(payload.kind),
                user_message_id=user_message_id,
                user_message_text=user_message_text,
            )
        ):
            yield

    @contextmanager
    def _activation(self, active: _ActiveInvocation) -> Iterator[None]:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("another Tick already owns Topic tools")
            self._active = active
        try:
            yield
        finally:
            with self._lock:
                self._active = None

    def register(self, *, registry_instance: Any | None = None) -> None:
        from tools.registry import registry

        active_registry = registry_instance or registry
        scope = active_registry.current_scope_key()
        active_registry.register(
            name=TOPIC_INSPECT_TOOL_NAME,
            toolset=TOPIC_TOOLSET,
            schema=TOPIC_INSPECT_TOOL_SCHEMA,
            handler=self.inspect,
            scope=scope,
        )
        active_registry.register(
            name=TOPIC_UPDATE_TOOL_NAME,
            toolset=TOPIC_TOOLSET,
            schema=TOPIC_UPDATE_TOOL_SCHEMA,
            handler=self.update,
            scope=scope,
        )

    def prompt_context(self, user_id: str) -> dict[str, object]:
        return self._service.prompt_context(user_id)

    def inspect(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if set(args) - {"topic_key"} or (
            "topic_key" in args and not isinstance(args["topic_key"], str)
        ):
            return self._json({"ok": False, "error_code": "invalid_arguments"})
        active = self._active_invocation()
        if active is None:
            return self._json({"ok": False, "error_code": "no_active_tick"})
        try:
            state = self._service.inspect(
                active.user_id,
                None if "topic_key" not in args else str(args["topic_key"]),
            )
        except ValueError as exc:
            return self._json({"ok": False, "error_code": str(exc)})
        return self._json({"ok": True, **state})

    def update(self, args: Mapping[str, object], **_kwargs: object) -> str:
        active = self._active_invocation()
        if active is None:
            return self._json({"ok": False, "error_code": "no_active_tick"})
        operation = args.get("operation")
        try:
            if operation == "record_topic_action":
                self._require_fields(
                    args,
                    required={
                        "operation",
                        "topic_key",
                        "last_action",
                        "evidence_refs",
                        "evidence_at_ms",
                        "receipt_refs",
                    },
                )
                result = self._service.record_topic_action(
                    user_id=active.user_id,
                    logical_run_id=active.logical_run_id,
                    topic_key=self._string(args, "topic_key"),
                    last_action=self._string(args, "last_action"),
                    evidence_refs=self._string_tuple(args, "evidence_refs"),
                    evidence_at_ms=self._integer(args, "evidence_at_ms"),
                    receipt_refs=self._string_tuple(args, "receipt_refs"),
                )
                return self._json({
                    "ok": True,
                    "topic": result.topic.to_dict(),
                    "receipt_id": result.receipt_id,
                })
            if operation == "set_topic_state":
                allowed = {
                    "operation",
                    "topic_key",
                    "state",
                    "snooze_until_ms",
                }
                if set(args) - allowed or not {
                    "operation",
                    "topic_key",
                    "state",
                } <= set(args):
                    raise ValueError("invalid_arguments")
                state = self._string(args, "state")
                if state not in {
                    "proposed",
                    "acknowledged",
                    "snoozed",
                    "resolved",
                    "expired",
                }:
                    raise ValueError("invalid_topic_state")
                snooze = args.get("snooze_until_ms")
                if snooze is not None and (
                    isinstance(snooze, bool) or not isinstance(snooze, int)
                ):
                    raise ValueError("invalid_snooze_until_ms")
                topic, receipt = self._service.set_topic_state(
                    user_id=active.user_id,
                    logical_run_id=active.logical_run_id,
                    topic_key=self._string(args, "topic_key"),
                    state=cast(Any, state),
                    snooze_until_ms=cast(int | None, snooze),
                    authorizing_message_id=active.user_message_id,
                )
                return self._json({
                    "ok": True,
                    "topic": topic.to_dict(),
                    "receipt": receipt.to_dict(),
                })
            if operation == "set_preference":
                self._require_fields(
                    args,
                    required={"operation", "preference_key", "preference_value"},
                )
                key = self._string(args, "preference_key")
                value = args["preference_value"]
                if not isinstance(value, bool):
                    raise ValueError("invalid_preference_value")
                authorization_error = self._preference_authorization(active, key, value)
                if authorization_error is not None:
                    return self._json({"ok": False, "error_code": authorization_error})
                assert active.user_message_id is not None
                preferences, receipt = self._service.set_preference(
                    user_id=active.user_id,
                    logical_run_id=active.logical_run_id,
                    preference_key=key,
                    value=value,
                    authorizing_message_id=active.user_message_id,
                )
                return self._json({
                    "ok": True,
                    "preferences": preferences.to_dict(),
                    "receipt": receipt.to_dict(),
                })
            if operation == "record_correction":
                self._require_fields(
                    args,
                    required={
                        "operation",
                        "correction_key",
                        "corrected_value",
                    },
                )
                if active.kind != "p0_user_chat" or not active.user_message_id:
                    return self._json({
                        "ok": False,
                        "error_code": "explicit_p0_authorization_required",
                    })
                corrected_value = self._string(args, "corrected_value")
                if not self._explicit_correction_matches(
                    active.user_message_text or "", corrected_value
                ):
                    return self._json({
                        "ok": False,
                        "error_code": "explicit_user_correction_not_found",
                    })
                correction, receipt = self._service.record_correction(
                    user_id=active.user_id,
                    logical_run_id=active.logical_run_id,
                    correction_key=self._string(args, "correction_key"),
                    corrected_value=corrected_value,
                    authorizing_message_id=active.user_message_id,
                )
                return self._json({
                    "ok": True,
                    "correction": correction,
                    "receipt": receipt.to_dict(),
                })
            return self._json({"ok": False, "error_code": "invalid_operation"})
        except TopicRepeatBlocked as exc:
            return self._json({
                "ok": False,
                "error_code": "topic_repeat_blocked",
                "repeat_eligibility": exc.eligibility.to_dict(),
            })
        except (DurableStateError, ValueError) as exc:
            return self._json({"ok": False, "error_code": str(exc)})

    @staticmethod
    def _preference_authorization(
        active: _ActiveInvocation, key: str, value: bool
    ) -> str | None:
        if key not in _PREFERENCE_KEYS:
            return "preference_not_switchable"
        if active.kind != "p0_user_chat" or not active.user_message_id:
            return "explicit_p0_authorization_required"
        text = TopicPreferenceToolBinding._direct_request_text(
            active.user_message_text or ""
        ).casefold()
        if key == "notifications_enabled":
            subject = "notification" in text
            positive = any(
                phrase in text
                for phrase in (
                    "turn on notification",
                    "enable notification",
                    "allow notification",
                    "resume notification",
                    "want notification",
                    "send me notification",
                )
            )
            negative = any(
                phrase in text
                for phrase in (
                    "turn off notification",
                    "disable notification",
                    "do not send notification",
                    "don't send notification",
                    "stop notification",
                    "stop sending notification",
                    "no notification",
                    "mute notification",
                )
            )
        else:
            subject = "speaker" in text and ("tag" in text or "name" in text)
            positive = any(
                phrase in text
                for phrase in (
                    "turn on speaker",
                    "enable speaker",
                    "resume speaker",
                    "start asking",
                    "ask me",
                )
            )
            negative = any(
                phrase in text
                for phrase in (
                    "turn off speaker",
                    "disable speaker",
                    "do not ask",
                    "don't ask",
                    "stop asking",
                )
            )
        if not subject or (positive if value else negative) is False:
            return "explicit_user_request_not_found"
        return None

    @staticmethod
    def _direct_request_text(message: str) -> str:
        """Exclude clearly quoted external data from P0 preference authority."""
        direct = re.sub(r"```.*?```", " ", message, flags=re.DOTALL)
        direct = re.sub(r"`[^`\n]*`", " ", direct)
        direct = re.sub(
            r"<(transcript|tool[_ -]?output|summary|interaction|speaker[_ -]?mapping|home[_ -]?content)\b[^>]*>.*?</\1\s*>",
            " ",
            direct,
            flags=re.DOTALL | re.IGNORECASE,
        )
        direct = re.sub(r"(?m)^\s*>.*$", " ", direct)
        direct = re.sub(r'"[^"\n]*"|“[^”\n]*”', " ", direct)
        direct = re.sub(r"(?<!\w)'[^'\n]*'(?!\w)|‘[^’\n]*’", " ", direct)

        external_markers = (
            "transcript",
            "recording",
            "tool output",
            "tool result",
            "summary",
            "email",
            "slack",
            "meeting notes",
            "prior chat",
            "chat content",
            "interaction evidence",
            "speaker mapping",
            "home content",
        )
        retained_lines: list[str] = []
        inside_external_block = False
        for line in direct.splitlines():
            folded = line.casefold()
            if any(marker in folded for marker in external_markers) and (
                line.rstrip().endswith(":")
            ):
                inside_external_block = True
                continue
            if inside_external_block:
                if not line.strip():
                    inside_external_block = False
                continue
            retained_lines.append(line)
        direct = "\n".join(retained_lines)
        units = re.split(r"(?<=[.!?])\s+|\n+", direct)
        return "\n".join(
            unit
            for unit in units
            if not any(marker in unit.casefold() for marker in external_markers)
        )

    @staticmethod
    def _explicit_correction_matches(message: str, corrected_value: str) -> bool:
        folded = message.casefold()
        return corrected_value.casefold() in folded and any(
            marker in folded
            for marker in (
                "actually",
                "correction",
                " is ",
                "i prefer",
                "please remember",
                "not ",
            )
        )

    @staticmethod
    def _require_fields(args: Mapping[str, object], *, required: set[str]) -> None:
        if set(args) != required:
            raise ValueError("invalid_arguments")

    @staticmethod
    def _string(args: Mapping[str, object], key: str) -> str:
        value = args.get(key)
        if not isinstance(value, str):
            raise ValueError(f"invalid_{key}")
        return value

    @staticmethod
    def _integer(args: Mapping[str, object], key: str) -> int:
        value = args.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"invalid_{key}")
        return value

    @staticmethod
    def _string_tuple(args: Mapping[str, object], key: str) -> tuple[str, ...]:
        value = args.get(key)
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError(f"invalid_{key}")
        return tuple(value)

    def _active_invocation(self) -> _ActiveInvocation | None:
        with self._lock:
            return self._active

    @staticmethod
    def _json(value: object) -> str:
        return _canonical(value)


__all__ = [
    "TOPIC_INSPECT_TOOL_NAME",
    "TOPIC_INSPECT_TOOL_SCHEMA",
    "TOPIC_UPDATE_TOOL_NAME",
    "TOPIC_UPDATE_TOOL_SCHEMA",
    "RepeatEligibility",
    "TopicActionResult",
    "TopicPreferenceService",
    "TopicPreferenceToolBinding",
    "TopicRecord",
    "TopicRepeatBlocked",
    "TopicUpdateReceipt",
]
