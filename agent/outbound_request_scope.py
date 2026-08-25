"""Invocation-local observation at the provider SDK dispatch boundary."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator


OutboundRequestObserver = Callable[[str, dict[str, Any]], None]
_OUTBOUND_REQUEST_OBSERVER: ContextVar[OutboundRequestObserver | None] = ContextVar(
    "hermes_outbound_request_observer",
    default=None,
)


@contextmanager
def observe_outbound_requests(observer: OutboundRequestObserver) -> Iterator[None]:
    """Observe exact SDK kwargs for the current invocation only."""
    token = _OUTBOUND_REQUEST_OBSERVER.set(observer)
    try:
        yield
    finally:
        _OUTBOUND_REQUEST_OBSERVER.reset(token)


def notify_outbound_request(api_mode: str, kwargs: dict[str, Any]) -> None:
    """Publish the final kwargs immediately before provider SDK dispatch."""
    observer = _OUTBOUND_REQUEST_OBSERVER.get()
    if observer is not None:
        observer(api_mode, kwargs)


__all__ = ["notify_outbound_request", "observe_outbound_requests"]
