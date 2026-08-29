"""Concise discovery metadata for deferred local-Thine tool namespaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductToolNamespace:
    namespace: str
    description: str
    eager_tool_schemas: tuple[()] = ()


PRODUCT_TOOL_NAMESPACES = (
    ProductToolNamespace(
        "transcripts", "Inspect claimed audio and canonical transcript segments."
    ),
    ProductToolNamespace(
        "speakers", "Inspect speakers and immutable rename or merge history."
    ),
    ProductToolNamespace(
        "communications", "Inspect delivery state and request one user communication."
    ),
    ProductToolNamespace(
        "ui.state", "Read or atomically edit the agent-authored Home state."
    ),
    ProductToolNamespace(
        "schedules", "Create, inspect, edit, cancel, or run one-shot wakeups."
    ),
    ProductToolNamespace(
        "working_memory", "Read or finalize the bounded operational Working Memory."
    ),
    ProductToolNamespace(
        "topics", "Inspect topics, preferences, and prior user requests."
    ),
    ProductToolNamespace(
        "permissions", "Inspect notification preference, permission, and last ask."
    ),
    ProductToolNamespace(
        "run", "Inspect the current Tick, limits, receipts, and recovery state."
    ),
)


__all__ = ["PRODUCT_TOOL_NAMESPACES", "ProductToolNamespace"]
