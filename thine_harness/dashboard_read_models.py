"""Non-authoritative dashboard projections over authoritative read helpers."""

from __future__ import annotations

import time
from typing import Callable, cast
import uuid

from .contracts.dashboard import DashboardReadModel, DashboardSnapshot
from .maintenance import AuthoritativeStateReader


_VERSION = {"major": 1, "minor": 0}
_MODELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("queue", ("hermes.queue_items", "hermes.attempts", "hermes.checkpoints")),
    ("current_run", ("hermes.queue_items", "hermes.attempts")),
    ("transcripts", ("hermes.transcript_claims", "thine.canonical_transcripts")),
    ("memory", ("hermes.working_memory_state", "hermes.working_memory_versions")),
    ("actions", ("hermes.tool_receipts", "hermes.communication_actions")),
    ("communications", ("hermes.communication_allowance_ledger",)),
    ("home", ("hermes.home_revisions", "hermes.home_current")),
    ("schedules", ("hermes.one_shot_schedules",)),
    ("failures", ("hermes.quarantines", "hermes.explicit_retries")),
    ("topics", ("hermes.durable_topics", "hermes.explicit_preferences")),
    ("interactions", ("hermes.interaction_clock_state", "hermes.interaction_claims")),
    ("debug_timeline", ("hermes.agent_run_inspections", "hermes.maintenance_events")),
)


class DashboardReadModelService:
    """Project counts/status only; mutation continues through owner services."""

    def __init__(
        self,
        reader: AuthoritativeStateReader,
        *,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._reader = reader
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def snapshot(self, user_id: str) -> DashboardSnapshot:
        now_ms = int(self._clock_ms())
        authoritative = self._reader.snapshot(user_id)
        counts = self._counts(authoritative)
        models = [
            DashboardReadModel.from_dict({
                "schema_version": _VERSION,
                "model_id": model_id,
                "generated_at_ms": now_ms,
                "source_state_paths": list(paths),
                "item_count": counts.get(model_id, 0),
                "status": "ok",
                "summary_ref": f"dashboard:{model_id}:{now_ms}",
                "authoritative": False,
                "redacted": True,
                "extensions": {},
            })
            for model_id, paths in _MODELS
        ]
        return DashboardSnapshot.from_dict({
            "schema_version": _VERSION,
            "snapshot_id": f"dashboard:{uuid.uuid4()}",
            "generated_at_ms": now_ms,
            "binding": "mac_loopback_only",
            "authoritative": False,
            "read_models": [model.to_dict() for model in models],
            "extensions": {},
        })

    @staticmethod
    def _counts(snapshot: dict[str, object]) -> dict[str, int]:
        queue_state = cast(dict[str, object], snapshot["queue_state"])
        assert isinstance(queue_state, dict)
        working_memory = cast(dict[str, object], snapshot["working_memory"])
        assert isinstance(working_memory, dict)
        topics = cast(dict[str, object], snapshot["topics_preferences"])
        assert isinstance(topics, dict)
        counts = cast(dict[str, object], snapshot["authoritative_counts"])
        assert isinstance(counts, dict)
        return {
            "queue": len(cast_list(queue_state["queue"])),
            "current_run": len(cast_list(queue_state["leases"])),
            "transcripts": cast(int, counts["transcript_claims"]),
            "memory": cast(int, counts["working_memory_versions"]),
            "actions": cast(int, counts["tool_receipts"])
            + cast(int, counts["communication_actions"]),
            "communications": cast(int, counts["communication_allowance"]),
            "home": 1,
            "schedules": len(cast_list(snapshot["schedules"])),
            "failures": len(cast_list(queue_state["quarantines"])),
            "topics": len(cast_list(topics["topics"])),
            "interactions": int(snapshot["interaction_clock"] is not None),
            "debug_timeline": cast(int, counts["agent_run_inspections"]),
        }


def cast_list(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


__all__ = ["DashboardReadModelService"]
