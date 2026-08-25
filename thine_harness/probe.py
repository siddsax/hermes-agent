"""Reproducible, credential-safe live proof for the pinned Harness runtime."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from typing import Any, Callable

from httpx import HTTPError
from openai import APIError

from thine_harness.runtime import (
    HermesAIAgentSession,
    HermesInvocationRuntime,
    InvocationEvent,
    InvocationKind,
    InvocationRequest,
    RuntimeModelConfig,
    RuntimeSelectionError,
)


CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
PROOF_MARKER = "HERMES_RUNTIME_OK"
logger = logging.getLogger(__name__)


class CodexCredentialUnavailable(RuntimeError):
    """The read-only Codex CLI credential source cannot supply a live token."""


def _load_codex_cli_token() -> dict[str, Any] | None:
    # Hermes intentionally keeps its auth store separate from Codex CLI.  This
    # proof reads the CLI store through Hermes' validator without persisting or
    # printing the credential.
    from hermes_cli.auth import _import_codex_cli_tokens

    return _import_codex_cli_tokens()


def _aiagent_factory(**kwargs: Any) -> Any:
    from run_agent import AIAgent

    return AIAgent(**kwargs)


def build_live_runtime(
    *,
    session_id: str,
    token_loader: Callable[[], dict[str, Any] | None] = _load_codex_cli_token,
    agent_factory: Callable[..., Any] = _aiagent_factory,
) -> HermesInvocationRuntime:
    """Build the exact live runtime, with no model/provider fallback."""
    credentials = token_loader()
    access_token = str((credentials or {}).get("access_token") or "")
    if not access_token:
        raise CodexCredentialUnavailable(
            "Codex CLI credential is missing, expired, or unavailable"
        )

    config = RuntimeModelConfig.openai_gpt_5_6_sol_medium()
    agent = agent_factory(
        base_url=CODEX_BASE_URL,
        api_key=access_token,
        provider=config.provider,
        requested_provider=config.provider,
        api_mode=config.api_mode,
        model=config.model,
        reasoning_config={"enabled": True, "effort": config.reasoning_effort},
        fallback_model=None,
        enabled_toolsets=["thine-proof-none"],
        quiet_mode=True,
        max_iterations=1,
        max_tokens=64,
        session_id=session_id,
        ephemeral_system_prompt="Return only the exact requested proof marker.",
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
    )
    return HermesInvocationRuntime(
        session=HermesAIAgentSession(agent=agent, expected=config),
        config=config,
    )


def run_live_probe() -> dict[str, Any]:
    """Execute one exact-model call and return only sanitized evidence."""
    runtime = build_live_runtime(session_id="thi3-41-live-proof")
    events: list[InvocationEvent] = []
    result = runtime.invoke(
        InvocationRequest(
            logical_run_id="thi3-41-live-proof",
            kind=InvocationKind.USER_CHAT,
            prompt=f"Return exactly {PROOF_MARKER}",
        ),
        emit=events.append,
    )
    return {
        "status": "ok",
        "diagnostics": runtime.diagnostics().as_dict(),
        "event_kinds": [event.kind.value for event in events],
        "final_marker": str(result.final_output or "").strip(),
        "usage": result.usage,
    }


def _sanitized_error(exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    message = str(exc)[:800]
    message = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", message, flags=re.I)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", message)
    return {
        "status": "blocked",
        "error_type": type(exc).__name__,
        "http_status": status,
        "provider_code": getattr(exc, "code", None),
        "message": message,
    }


def main() -> int:
    # Keep all Hermes-generated state out of the operator's normal profile.
    with tempfile.TemporaryDirectory(prefix="thi3-41-hermes-live-") as proof_home:
        os.environ["HERMES_HOME"] = proof_home
        try:
            evidence = run_live_probe()
        except (
            APIError,
            CodexCredentialUnavailable,
            HTTPError,
            RuntimeSelectionError,
        ) as exc:
            evidence = _sanitized_error(exc)
        except Exception:
            logger.exception("Unexpected THI3-41 live probe failure")
            raise
    print(json.dumps(evidence, sort_keys=True))
    return 0 if evidence["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CODEX_BASE_URL",
    "PROOF_MARKER",
    "CodexCredentialUnavailable",
    "build_live_runtime",
    "run_live_probe",
]
