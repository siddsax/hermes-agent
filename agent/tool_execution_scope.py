"""Invocation-local execution policy for model-emitted tool calls."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_TOOL_EXECUTION_DENY_REASON: ContextVar[str | None] = ContextVar(
    "hermes_tool_execution_deny_reason",
    default=None,
)


@contextmanager
def deny_tool_execution(reason: str) -> Iterator[None]:
    """Keep schemas visible while denying handler dispatch in this invocation."""
    token = _TOOL_EXECUTION_DENY_REASON.set(str(reason))
    try:
        yield
    finally:
        _TOOL_EXECUTION_DENY_REASON.reset(token)


def tool_execution_deny_reason() -> str | None:
    """Return the current invocation's deny reason, if execution is fenced."""
    return _TOOL_EXECUTION_DENY_REASON.get()


def tool_execution_denied_result(tool_name: str) -> str | None:
    """Return the canonical denied result for this invocation, if fenced."""
    reason = tool_execution_deny_reason()
    if reason is None:
        return None
    from tools.registry import tool_error

    return tool_error(
        f"Tool execution denied for this invocation: {reason}",
        error_type="invocation_tool_execution_denied",
        tool=tool_name,
    )


__all__ = [
    "deny_tool_execution",
    "tool_execution_denied_result",
    "tool_execution_deny_reason",
]
