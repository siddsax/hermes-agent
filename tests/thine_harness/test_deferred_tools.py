from __future__ import annotations

import pytest

from thine_harness.deferred_tools import DeferredNamespaceCatalog
from tools.registry import registry
from tools.tool_search import BRIDGE_TOOL_NAMES


HELPER_NAME = "thine_transcripts_lookup"


@pytest.fixture
def registered_helper():
    schema = {
        "name": HELPER_NAME,
        "description": "Look up canonical transcript segments by sequence.",
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {"type": "integer"},
            },
            "required": ["sequence"],
        },
    }
    registry.register(
        name=HELPER_NAME,
        toolset="mcp-thine-proof",
        schema=schema,
        handler=lambda args, **kwargs: "{}",
    )
    try:
        yield {"type": "function", "function": schema}
    finally:
        registry.deregister(HELPER_NAME)


def test_namespaced_helper_is_searchable_without_eager_schema_loading(registered_helper):
    catalog = DeferredNamespaceCatalog([registered_helper], context_length=272_000)

    model_tools = catalog.model_tool_definitions()
    model_tool_names = {tool["function"]["name"] for tool in model_tools}

    assert HELPER_NAME not in model_tool_names
    assert model_tool_names == set(BRIDGE_TOOL_NAMES)
    assert "sequence" not in repr(model_tools)

    matches = catalog.search("canonical transcript sequence")
    assert [match["name"] for match in matches] == [HELPER_NAME]
    assert matches[0]["namespace"] == "transcripts"

    described = catalog.describe(HELPER_NAME)
    assert described["parameters"]["required"] == ["sequence"]
