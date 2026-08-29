"""Reproducible latency, context-load, and backpressure diagnostics.

The default probe is hermetic: it uses a temporary Harness database, the real
queue/search implementations, a fake runtime, and no provider or backend.  It
records observations without imposing an SLA that the product has not chosen.
The existing ``thine_harness.integrated_probe`` remains the explicit opt-in
provider proof for cache-read and same-context Stop Hook telemetry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Iterable

from agent.model_metadata import estimate_request_tokens_rough

from .communications import (
    COMMUNICATION_SEND_TOOL_SCHEMA,
    COMMUNICATION_STATUS_TOOL_SCHEMA,
)
from .contracts.runtime import Tick
from .deferred_tools import DeferredNamespaceCatalog
from .envelope import RuntimeEnvelopeBudget
from .home_state import (
    GET_CURRENT_TOOL_SCHEMA,
    GET_HISTORY_TOOL_SCHEMA,
    REACTIVATE_REVISION_TOOL_SCHEMA,
    REPLACE_CURRENT_TOOL_SCHEMA,
)
from .interactions import (
    INTERACTION_AGENT_TOOL_SCHEMA,
    latest_half_hour_boundary_ms,
)
from .run_coordinator import (
    FakeFeatureAcknowledgement,
    InvocationOutcome,
    RunCoordinator,
)
from .run_state import DurableRunState
from .runtime import RuntimeModelConfig
from .schedules import (
    SCHEDULE_CANCEL_TOOL_SCHEMA,
    SCHEDULE_CREATE_TOOL_SCHEMA,
    SCHEDULE_EDIT_TOOL_SCHEMA,
    SCHEDULE_INSPECT_TOOL_SCHEMA,
    SCHEDULE_LIST_TOOL_SCHEMA,
    SCHEDULE_RUN_NOW_TOOL_SCHEMA,
)
from .speaker_mappings import (
    INSPECT_ACTIVE_MAPPING_TOOL_SCHEMA,
    INSPECT_MAPPING_HISTORY_TOOL_SCHEMA,
)
from .standalone_notifications import (
    STANDALONE_NOTIFICATION_SEND_TOOL_SCHEMA,
    STANDALONE_NOTIFICATION_STATUS_TOOL_SCHEMA,
)
from .topics_preferences import TOPIC_INSPECT_TOOL_SCHEMA, TOPIC_UPDATE_TOOL_SCHEMA
from .transcript_agent import INSPECT_CLAIM_TOOL_SCHEMA, INSPECT_RUN_TOOL_SCHEMA
from .working_memory import (
    CacheIdentity,
    COMPACTION_TARGET_TOKENS,
    MAX_WORKING_MEMORY_TOKENS,
    StopHookOutcomeKind,
    StopHookRequest,
    StopHookRunner,
    WorkingMemoryProposal,
    WorkingMemorySnapshot,
)


_VERSION = 1
_SQLITE_BUSY_TIMEOUT_MS = 5_000


@dataclass(frozen=True)
class ToolContextMeasurement:
    helper_count: int
    eager_schema_bytes: int
    deferred_bridge_bytes: int
    exact_bytes_saved: int
    eager_estimated_tokens: int
    deferred_estimated_tokens: int
    estimated_tokens_saved: int
    deferred_fraction: float
    estimator: str


@dataclass(frozen=True)
class QueuePressureMeasurement:
    transcript_burst: int
    total_ticks: int
    enqueue_elapsed_ms: float
    drain_elapsed_ms: float
    first_kind: str
    interaction_index: int
    promoted_schedule_index: int
    ordinary_schedule_index: int
    later_work_completed: bool
    completed_by_kind: dict[str, int]


@dataclass(frozen=True)
class SQLiteContentionMeasurement:
    configured_busy_timeout_ms: int
    held_write_lock_ms: float
    blocked_writer_elapsed_ms: float
    blocked_writer_completed: bool
    journal_mode: str


@dataclass(frozen=True)
class TimerDriftMeasurement:
    timezone_name: str
    expected_boundary_ms: int
    observed_scan_ms: int
    observed_drift_ms: int


@dataclass(frozen=True)
class WorkingMemoryMeasurement:
    exact_limit_tokens: int
    exact_limit_committed: bool
    oversized_candidate_tokens: int
    correction_target_tokens: int
    compacted_tokens: int
    compacted_committed: bool
    same_cache_identity: bool
    method: str


@dataclass(frozen=True)
class CacheEvidenceMeasurement:
    status: str
    same_prompt_cache_key: bool | None
    same_system_prompt_sha256: bool | None
    same_wire_tool_array: bool | None
    stop_hook_cache_read_tokens: int | None
    source: str


@dataclass(frozen=True)
class PerformanceReport:
    schema_version: int
    methodology: str
    model: dict[str, str | int]
    context_limits: dict[str, int | bool]
    tool_context: ToolContextMeasurement
    queue_pressure: QueuePressureMeasurement
    sqlite_contention: SQLiteContentionMeasurement
    timer_drift: TimerDriftMeasurement
    working_memory: WorkingMemoryMeasurement
    cache_evidence: CacheEvidenceMeasurement
    operating_limits: dict[str, int | str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def thine_tool_definitions() -> list[dict[str, Any]]:
    """Return the actual current Thine helper schemas in wire registration shape."""
    schemas = (
        INSPECT_CLAIM_TOOL_SCHEMA,
        INSPECT_RUN_TOOL_SCHEMA,
        INSPECT_ACTIVE_MAPPING_TOOL_SCHEMA,
        INSPECT_MAPPING_HISTORY_TOOL_SCHEMA,
        INTERACTION_AGENT_TOOL_SCHEMA,
        COMMUNICATION_STATUS_TOOL_SCHEMA,
        COMMUNICATION_SEND_TOOL_SCHEMA,
        STANDALONE_NOTIFICATION_STATUS_TOOL_SCHEMA,
        STANDALONE_NOTIFICATION_SEND_TOOL_SCHEMA,
        GET_CURRENT_TOOL_SCHEMA,
        GET_HISTORY_TOOL_SCHEMA,
        REPLACE_CURRENT_TOOL_SCHEMA,
        REACTIVATE_REVISION_TOOL_SCHEMA,
        SCHEDULE_CREATE_TOOL_SCHEMA,
        SCHEDULE_LIST_TOOL_SCHEMA,
        SCHEDULE_INSPECT_TOOL_SCHEMA,
        SCHEDULE_EDIT_TOOL_SCHEMA,
        SCHEDULE_CANCEL_TOOL_SCHEMA,
        SCHEDULE_RUN_NOW_TOOL_SCHEMA,
        TOPIC_INSPECT_TOOL_SCHEMA,
        TOPIC_UPDATE_TOOL_SCHEMA,
    )
    by_name = {str(schema["name"]): schema for schema in schemas}
    return [{"type": "function", "function": by_name[name]} for name in sorted(by_name)]


def measure_tool_context(
    tool_definitions: Iterable[dict[str, Any]],
) -> ToolContextMeasurement:
    tools = list(tool_definitions)
    from tools.registry import registry

    probe_registrations: list[str] = []
    for tool in tools:
        schema = dict(tool.get("function") or {})
        name = str(schema.get("name") or "")
        if not name or registry.get_entry(name) is not None:
            continue
        registry.register(
            name=name,
            toolset="mcp-thine-performance-probe",
            schema=schema,
            handler=lambda _args, **_kwargs: "{}",
        )
        probe_registrations.append(name)
    try:
        catalog = DeferredNamespaceCatalog(tools, context_length=272_000)
        bridge = catalog.model_tool_definitions()
    finally:
        for name in probe_registrations:
            registry.deregister(name)
    eager_bytes = _canonical_json_bytes(tools)
    deferred_bytes = _canonical_json_bytes(bridge)
    eager_tokens = estimate_request_tokens_rough([], tools=tools)
    deferred_tokens = estimate_request_tokens_rough([], tools=bridge)
    return ToolContextMeasurement(
        helper_count=len(tools),
        eager_schema_bytes=eager_bytes,
        deferred_bridge_bytes=deferred_bytes,
        exact_bytes_saved=max(eager_bytes - deferred_bytes, 0),
        eager_estimated_tokens=eager_tokens,
        deferred_estimated_tokens=deferred_tokens,
        estimated_tokens_saved=max(eager_tokens - deferred_tokens, 0),
        deferred_fraction=(deferred_bytes / eager_bytes if eager_bytes else 0.0),
        estimator="agent.model_metadata.estimate_request_tokens_rough",
    )


def measure_queue_pressure(
    database_path: Path,
    *,
    transcript_burst: int,
) -> QueuePressureMeasurement:
    if transcript_burst < 1:
        raise ValueError("transcript_burst must be positive")
    state = DurableRunState(database_path)
    runtime = _RecordingRuntime()
    coordinator = RunCoordinator(
        state,
        runtime=runtime,
        feature_port=_NoopFeature(),
        clock_ms=lambda: 1_000_000,
    )
    started = time.perf_counter_ns()
    for index in range(transcript_burst):
        coordinator.enqueue(_tick(f"transcript-{index:05d}", "p1_transcript"))
    coordinator.enqueue(_tick("interaction-boundary", "p1_interaction"))
    coordinator.enqueue(_tick("later-transcript", "p1_transcript"))
    coordinator.enqueue(
        _tick("promoted-overdue-schedule", "p2_scheduled", priority="p1")
    )
    coordinator.enqueue(_tick("ordinary-schedule", "p2_scheduled"))
    coordinator.enqueue(_tick("arriving-p0", "p0_user_chat"))
    enqueue_elapsed_ms = _elapsed_ms(started)

    started = time.perf_counter_ns()
    while coordinator.run_next("daily-user") is not None:
        pass
    drain_elapsed_ms = _elapsed_ms(started)
    order = runtime.order
    counts: dict[str, int] = {}
    for _, kind in order:
        counts[kind] = counts.get(kind, 0) + 1
    tick_ids = [tick_id for tick_id, _ in order]
    return QueuePressureMeasurement(
        transcript_burst=transcript_burst,
        total_ticks=len(order),
        enqueue_elapsed_ms=enqueue_elapsed_ms,
        drain_elapsed_ms=drain_elapsed_ms,
        first_kind=order[0][1],
        interaction_index=tick_ids.index("interaction-boundary"),
        promoted_schedule_index=tick_ids.index("promoted-overdue-schedule"),
        ordinary_schedule_index=tick_ids.index("ordinary-schedule"),
        later_work_completed=len(order) == transcript_burst + 5,
        completed_by_kind=counts,
    )


def measure_sqlite_contention(
    database_path: Path,
    *,
    hold_lock_seconds: float = 0.05,
) -> SQLiteContentionMeasurement:
    state = DurableRunState(database_path)
    lock_connection = state._connect()
    busy_timeout = int(lock_connection.execute("PRAGMA busy_timeout").fetchone()[0])
    journal_mode = str(lock_connection.execute("PRAGMA journal_mode").fetchone()[0])
    lock_connection.execute("BEGIN IMMEDIATE")
    writer_done = threading.Event()
    writer_errors: list[BaseException] = []
    writer_elapsed_ms = 0.0

    def blocked_writer() -> None:
        nonlocal writer_elapsed_ms
        started = time.perf_counter_ns()
        try:
            state.enqueue(_tick("contention-writer", "p1_transcript"), now_ms=1)
        except BaseException as exc:  # reported rather than hidden by the probe
            writer_errors.append(exc)
        finally:
            writer_elapsed_ms = _elapsed_ms(started)
            writer_done.set()

    writer = threading.Thread(target=blocked_writer, name="thine-sqlite-probe")
    writer.start()
    hold_started = time.perf_counter_ns()
    time.sleep(hold_lock_seconds)
    held_write_lock_ms = _elapsed_ms(hold_started)
    lock_connection.rollback()
    lock_connection.close()
    writer.join(timeout=busy_timeout / 1_000 + 1)
    completed = writer_done.is_set() and not writer_errors
    return SQLiteContentionMeasurement(
        configured_busy_timeout_ms=busy_timeout,
        held_write_lock_ms=held_write_lock_ms,
        blocked_writer_elapsed_ms=writer_elapsed_ms,
        blocked_writer_completed=completed,
        journal_mode=journal_mode,
    )


def measure_timer_drift(
    *,
    observed_scan_ms: int,
    timezone_name: str = "Asia/Kolkata",
) -> TimerDriftMeasurement:
    boundary = latest_half_hour_boundary_ms(observed_scan_ms, timezone_name)
    return TimerDriftMeasurement(
        timezone_name=timezone_name,
        expected_boundary_ms=boundary,
        observed_scan_ms=observed_scan_ms,
        observed_drift_ms=max(observed_scan_ms - boundary, 0),
    )


def measure_working_memory_compaction() -> WorkingMemoryMeasurement:
    """Exercise the exact 16K decision and one same-context correction."""
    identity = CacheIdentity("probe-session", "probe-cache", "probe-tools")

    def exact_test_counter(candidate: str) -> int:
        if candidate.startswith("limit:"):
            return MAX_WORKING_MEMORY_TOKENS
        if candidate.startswith("oversized:"):
            return MAX_WORKING_MEMORY_TOKENS + 1
        if candidate.startswith("compacted:"):
            return COMPACTION_TARGET_TOKENS
        raise AssertionError("unexpected controlled Working Memory candidate")

    limit_context = _WorkingMemoryProbeContext(
        identity=identity,
        proposals=(WorkingMemoryProposal.changed("limit:within-ceiling"),),
    )
    limit_outcome = StopHookRunner(token_counter=exact_test_counter).finalize(
        run_id="probe-limit",
        current=WorkingMemorySnapshot(1, "prior", 1),
        context=limit_context,
        store=_WorkingMemoryProbeStore(),
        interrupted=False,
    )
    compact_context = _WorkingMemoryProbeContext(
        identity=identity,
        proposals=(
            WorkingMemoryProposal.changed("oversized:needs-correction"),
            WorkingMemoryProposal.changed("compacted:at-target"),
        ),
    )
    compact_outcome = StopHookRunner(token_counter=exact_test_counter).finalize(
        run_id="probe-compaction",
        current=WorkingMemorySnapshot(1, "prior", 1),
        context=compact_context,
        store=_WorkingMemoryProbeStore(),
        interrupted=False,
    )
    return WorkingMemoryMeasurement(
        exact_limit_tokens=MAX_WORKING_MEMORY_TOKENS,
        exact_limit_committed=(
            limit_outcome.kind is StopHookOutcomeKind.COMMITTED
            and limit_outcome.token_count == MAX_WORKING_MEMORY_TOKENS
        ),
        oversized_candidate_tokens=MAX_WORKING_MEMORY_TOKENS + 1,
        correction_target_tokens=COMPACTION_TARGET_TOKENS,
        compacted_tokens=int(compact_outcome.token_count or 0),
        compacted_committed=(
            compact_outcome.kind is StopHookOutcomeKind.COMMITTED
            and compact_outcome.token_count == COMPACTION_TARGET_TOKENS
        ),
        same_cache_identity=(
            limit_outcome.cache_identity == identity
            and compact_outcome.cache_identity == identity
            and all(
                request.target_tokens in {16_000, 14_000}
                for request in compact_context.requests
            )
        ),
        method="controlled_exact_counter_real_stop_hook_runner",
    )


def summarize_cache_evidence(
    evidence: dict[str, Any] | None,
) -> CacheEvidenceMeasurement:
    if evidence is None:
        return CacheEvidenceMeasurement(
            status="not_run_offline",
            same_prompt_cache_key=None,
            same_system_prompt_sha256=None,
            same_wire_tool_array=None,
            stop_hook_cache_read_tokens=None,
            source="run python -m thine_harness.integrated_probe explicitly",
        )
    delta = evidence.get("stop_hook_usage_delta")
    return CacheEvidenceMeasurement(
        status=str(evidence.get("status") or "unknown"),
        same_prompt_cache_key=_optional_bool(evidence.get("same_prompt_cache_key")),
        same_system_prompt_sha256=_optional_bool(
            evidence.get("same_system_prompt_sha256")
        ),
        same_wire_tool_array=_optional_bool(evidence.get("same_wire_tool_array")),
        stop_hook_cache_read_tokens=(
            int(delta.get("cache_read_tokens") or 0)
            if isinstance(delta, dict)
            else None
        ),
        source="integrated provider probe evidence",
    )


def run_isolated_probe(
    *,
    transcript_burst: int = 100,
    observed_scan_ms: int = 1_787_644_800_137,
    cache_evidence: dict[str, Any] | None = None,
) -> PerformanceReport:
    with tempfile.TemporaryDirectory(prefix="thine-performance-") as directory:
        root = Path(directory)
        queue = measure_queue_pressure(
            root / "queue.sqlite3", transcript_burst=transcript_burst
        )
        contention = measure_sqlite_contention(root / "contention.sqlite3")
    budget = RuntimeEnvelopeBudget.pinned()
    model = RuntimeModelConfig.openai_gpt_5_6_sol_medium()
    return PerformanceReport(
        schema_version=_VERSION,
        methodology="isolated_state_real_queue_fake_runtime_no_provider_no_backend",
        model=model.__dict__,
        context_limits={
            **budget.as_dict(),
            "working_memory_exact_token_count_required": True,
            "working_memory_compaction_target_tokens": COMPACTION_TARGET_TOKENS,
            "working_memory_limit_matches_runtime": (
                budget.working_memory_tokens == MAX_WORKING_MEMORY_TOKENS
            ),
        },
        tool_context=measure_tool_context(thine_tool_definitions()),
        queue_pressure=queue,
        sqlite_contention=contention,
        timer_drift=measure_timer_drift(observed_scan_ms=observed_scan_ms),
        working_memory=measure_working_memory_compaction(),
        cache_evidence=summarize_cache_evidence(cache_evidence),
        operating_limits={
            "p0_safe_heartbeat_max_silence_ms": 5_000,
            "sqlite_busy_timeout_ms": _SQLITE_BUSY_TIMEOUT_MS,
            "interaction_boundary_period_ms": 30 * 60 * 1_000,
            "overdue_schedule_promotion_age_ms": 10 * 60 * 1_000,
            "logical_input_attempts_total": 3,
            "routine_transcript_target_tokens": budget.routine_batch_target_tokens,
            "absolute_transcript_window_tokens": budget.absolute_transcript_tokens,
            "working_memory_tokens": budget.working_memory_tokens,
            "working_memory_compaction_target_tokens": COMPACTION_TARGET_TOKENS,
            "sla_thresholds": "not_configured_measure_only",
        },
    )


class _RecordingRuntime:
    def __init__(self) -> None:
        self.order: list[tuple[str, str]] = []

    def invoke(self, context, *, tools, control):
        del tools, control
        self.order.append((
            str(context.tick.payload.tick_id),
            str(context.tick.payload.kind),
        ))
        return InvocationOutcome.completed()


class _NoopFeature:
    def apply(self, command):
        return FakeFeatureAcknowledgement(
            provider_reference=f"probe:{command.action_id}", result={}
        )


class _WorkingMemoryProbeContext:
    def __init__(
        self,
        *,
        identity: CacheIdentity,
        proposals: tuple[WorkingMemoryProposal, ...],
    ) -> None:
        self.identity = identity
        self._proposals = iter(proposals)
        self.requests: list[StopHookRequest] = []

    def continue_stop_hook(self, request: StopHookRequest) -> WorkingMemoryProposal:
        self.requests.append(request)
        return next(self._proposals)

    def count_candidate_tokens(self, candidate: str) -> int:
        raise AssertionError("the controlled counter must be used")


class _WorkingMemoryProbeStore:
    def commit(
        self,
        *,
        expected_version: int,
        markdown: str,
        token_count: int,
        run_id: str,
    ) -> int:
        del markdown, token_count, run_id
        return expected_version + 1

    def mark_unchanged(self, *, expected_version: int, run_id: str) -> None:
        del expected_version, run_id


def _tick(tick_id: str, kind: str, *, priority: str | None = None) -> Tick:
    priorities = {
        "p0_user_chat": "p0",
        "p1_transcript": "p1",
        "p1_interaction": "p1",
        "p2_scheduled": "p2",
    }
    source_kinds = {
        "p0_user_chat": "user_message",
        "p1_transcript": "transcript_availability",
        "p1_interaction": "interaction_window",
        "p2_scheduled": "schedule",
    }
    source_kind = source_kinds[kind]
    return Tick.from_dict({
        "schema_version": {"major": 1, "minor": 0},
        "tick_id": tick_id,
        "user_id": "daily-user",
        "logical_run_id": f"run:{tick_id}",
        "kind": kind,
        "priority": priority or priorities[kind],
        "occurred_at_ms": 1,
        "received_at_ms": 1,
        "queued_at_ms": 1,
        "source_ref": {"kind": source_kind, "id": tick_id},
        "causation_id": None,
        "correlation_id": f"correlation:{tick_id}",
        "attempt_ordinal": 1,
        "lease": None,
        "communication_allowance_snapshot": None,
        "payload": {"payload_kind": source_kind, "reference_id": tick_id},
        "extensions": {},
    })


def _canonical_json_bytes(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-burst", type=int, default=100)
    parser.add_argument(
        "--cache-evidence",
        type=Path,
        help="JSON output from an explicitly run integrated provider probe",
    )
    args = parser.parse_args()
    cache_evidence = None
    if args.cache_evidence is not None:
        cache_evidence = json.loads(args.cache_evidence.read_text(encoding="utf-8"))
    report = run_isolated_probe(
        transcript_burst=args.transcript_burst,
        cache_evidence=cache_evidence,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CacheEvidenceMeasurement",
    "PerformanceReport",
    "QueuePressureMeasurement",
    "SQLiteContentionMeasurement",
    "TimerDriftMeasurement",
    "ToolContextMeasurement",
    "WorkingMemoryMeasurement",
    "measure_queue_pressure",
    "measure_sqlite_contention",
    "measure_timer_drift",
    "measure_tool_context",
    "measure_working_memory_compaction",
    "run_isolated_probe",
    "summarize_cache_evidence",
    "thine_tool_definitions",
]
