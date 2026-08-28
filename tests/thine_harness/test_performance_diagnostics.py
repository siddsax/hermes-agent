from __future__ import annotations

from thine_harness.performance_diagnostics import (
    measure_queue_pressure,
    measure_sqlite_contention,
    measure_timer_drift,
    measure_tool_context,
    measure_working_memory_compaction,
    run_isolated_probe,
    summarize_cache_evidence,
    thine_tool_definitions,
)


def test_deferred_search_reports_exact_schema_bytes_and_estimated_context_savings():
    measurement = measure_tool_context(thine_tool_definitions())

    assert measurement.helper_count > 0
    assert measurement.deferred_bridge_bytes < measurement.eager_schema_bytes
    assert measurement.exact_bytes_saved == (
        measurement.eager_schema_bytes - measurement.deferred_bridge_bytes
    )
    assert measurement.deferred_estimated_tokens < measurement.eager_estimated_tokens
    assert measurement.estimated_tokens_saved == (
        measurement.eager_estimated_tokens - measurement.deferred_estimated_tokens
    )
    assert 0 < measurement.deferred_fraction < 1


def test_queue_pressure_keeps_priority_fifo_and_completes_later_work(tmp_path):
    measurement = measure_queue_pressure(
        tmp_path / "queue.sqlite3", transcript_burst=50
    )

    assert measurement.total_ticks == 55
    assert measurement.first_kind == "p0_user_chat"
    assert measurement.interaction_index == 51
    assert measurement.promoted_schedule_index == 53
    assert measurement.ordinary_schedule_index == 54
    assert measurement.later_work_completed is True
    assert measurement.completed_by_kind == {
        "p0_user_chat": 1,
        "p1_transcript": 51,
        "p1_interaction": 1,
        "p2_scheduled": 2,
    }


def test_sqlite_contention_measurement_uses_real_busy_timeout_and_recovers(tmp_path):
    measurement = measure_sqlite_contention(
        tmp_path / "contention.sqlite3", hold_lock_seconds=0.02
    )

    assert measurement.configured_busy_timeout_ms == 5_000
    assert measurement.held_write_lock_ms >= 0
    assert measurement.blocked_writer_elapsed_ms >= 0
    assert measurement.blocked_writer_completed is True
    assert measurement.journal_mode in {"delete", "wal"}


def test_timer_drift_is_measured_from_the_fixed_local_half_hour_boundary():
    measurement = measure_timer_drift(
        observed_scan_ms=1_787_644_800_137,
        timezone_name="Asia/Kolkata",
    )

    assert measurement.expected_boundary_ms == 1_787_644_800_000
    assert measurement.observed_drift_ms == 137


def test_working_memory_probe_exercises_exact_16k_and_compaction_to_14k():
    measurement = measure_working_memory_compaction()

    assert measurement.exact_limit_tokens == 16_000
    assert measurement.exact_limit_committed is True
    assert measurement.oversized_candidate_tokens == 16_001
    assert measurement.correction_target_tokens == 14_000
    assert measurement.compacted_tokens == 14_000
    assert measurement.compacted_committed is True
    assert measurement.same_cache_identity is True


def test_cache_evidence_reports_not_run_instead_of_claiming_an_offline_proof():
    offline = summarize_cache_evidence(None)
    proven = summarize_cache_evidence({
        "status": "ok",
        "same_prompt_cache_key": True,
        "same_system_prompt_sha256": True,
        "same_wire_tool_array": True,
        "stop_hook_usage_delta": {"cache_read_tokens": 4096},
    })

    assert offline.status == "not_run_offline"
    assert offline.stop_hook_cache_read_tokens is None
    assert proven.same_prompt_cache_key is True
    assert proven.same_system_prompt_sha256 is True
    assert proven.same_wire_tool_array is True
    assert proven.stop_hook_cache_read_tokens == 4096


def test_full_probe_is_hermetic_and_reports_limits_without_inventing_slas():
    report = run_isolated_probe(transcript_burst=10)

    assert report.methodology == (
        "isolated_state_real_queue_fake_runtime_no_provider_no_backend"
    )
    assert report.model == {
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "api_mode": "codex_responses",
        "reasoning_effort": "medium",
        "context_window_tokens": 272_000,
    }
    assert report.context_limits["working_memory_tokens"] == 16_000
    assert report.context_limits["working_memory_compaction_target_tokens"] == 14_000
    assert report.context_limits["working_memory_limit_matches_runtime"] is True
    assert report.working_memory.exact_limit_committed is True
    assert report.working_memory.compacted_committed is True
    assert report.operating_limits["sla_thresholds"] == ("not_configured_measure_only")
    assert report.cache_evidence.status == "not_run_offline"
