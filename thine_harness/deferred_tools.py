"""Public deferred namespaced-helper search seam.

The implementation delegates to Hermes' existing progressive disclosure
machinery.  It keeps full helper schemas outside the model's eager tool array
while adding the logical namespace callers use in the cross-repository
contract.
"""

from __future__ import annotations

import json
from typing import Any

from tools.tool_search import (
    ToolSearchConfig,
    assemble_tool_defs,
    dispatch_tool_describe,
    dispatch_tool_search,
)


_WIRE_NAMESPACE_PREFIXES = (
    ("working_memory", "working_memory"),
    ("communications", "communications"),
    ("permissions", "permissions"),
    ("transcripts", "transcripts"),
    ("schedules", "schedules"),
    ("speakers", "speakers"),
    ("ui_state", "ui.state"),
    ("topics", "topics"),
    ("run", "run"),
)


def _logical_namespace(wire_name: str) -> str:
    for prefix, namespace in _WIRE_NAMESPACE_PREFIXES:
        if wire_name.startswith(f"thine_{prefix}_"):
            return namespace
    return "unknown"


class DeferredNamespaceCatalog:
    """Search and describe one session's already-registered helper catalog."""

    def __init__(self, tool_definitions: list[dict[str, Any]], *, context_length: int):
        self._tool_definitions = list(tool_definitions)
        self._context_length = context_length
        self._config = ToolSearchConfig.from_raw(
            {"enabled": "on", "listing": "off"}
        )

    def model_tool_definitions(self) -> list[dict[str, Any]]:
        return assemble_tool_defs(
            self._tool_definitions,
            context_length=self._context_length,
            config=self._config,
        ).tool_defs

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        payload = json.loads(
            dispatch_tool_search(
                {"query": query, "limit": limit},
                current_tool_defs=self._tool_definitions,
                config=self._config,
            )
        )
        matches = list(payload.get("matches") or [])
        return [
            {**match, "namespace": _logical_namespace(str(match.get("name") or ""))}
            for match in matches
        ]

    def describe(self, wire_name: str) -> dict[str, Any]:
        return json.loads(
            dispatch_tool_describe(
                {"name": wire_name},
                current_tool_defs=self._tool_definitions,
            )
        )


__all__ = ["DeferredNamespaceCatalog"]
