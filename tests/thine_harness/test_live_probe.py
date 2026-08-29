from __future__ import annotations

from types import SimpleNamespace

import pytest

import thine_harness.probe as probe
from thine_harness.probe import CodexCredentialUnavailable, build_live_runtime


class _RecordingAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.provider = kwargs["provider"]
        self.model = kwargs["model"]
        self.api_mode = kwargs["api_mode"]
        self.reasoning_config = kwargs["reasoning_config"]
        self.skip_memory = kwargs["skip_memory"]
        self.skip_background_review = kwargs["skip_background_review"]
        self.context_compressor = SimpleNamespace(context_length=272_000)


def test_live_probe_builds_exact_runtime_without_a_fallback_model():
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return _RecordingAgent(**kwargs)

    runtime = build_live_runtime(
        session_id="probe-session",
        token_loader=lambda: {"access_token": "not-printed"},
        agent_factory=factory,
    )

    assert runtime.diagnostics().as_dict() == {
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "api_mode": "codex_responses",
        "reasoning_effort": "medium",
        "context_window_tokens": 272_000,
    }
    assert captured["reasoning_config"] == {"enabled": True, "effort": "medium"}
    assert captured["fallback_model"] is None
    assert captured["session_id"] == "probe-session"
    assert captured["api_key"] == "not-printed"


def test_live_probe_fails_closed_when_codex_cli_credential_is_unavailable():
    with pytest.raises(CodexCredentialUnavailable, match="Codex CLI credential"):
        build_live_runtime(
            session_id="probe-session",
            token_loader=lambda: None,
            agent_factory=_RecordingAgent,
        )


def test_probe_main_does_not_reclassify_unexpected_implementation_faults(monkeypatch):
    def fail_unexpectedly():
        raise ValueError("implementation bug")

    monkeypatch.setattr(probe, "run_live_probe", fail_unexpectedly)

    with pytest.raises(ValueError, match="implementation bug"):
        probe.main()
