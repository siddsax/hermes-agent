from __future__ import annotations

from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient

from thine_harness.deferred_tools import DeferredNamespaceCatalog
from thine_harness.home_state import (
    HomeActionConflict,
    HomeRevisionConflict,
    HomeStateProjector,
    HomeStateValidationError,
    HomeToolHandlers,
    GET_CURRENT_TOOL_NAME,
    GET_CURRENT_TOOL_SCHEMA,
    HOME_TOOLSET,
    REPLACE_CURRENT_TOOL_NAME,
    REPLACE_CURRENT_TOOL_SCHEMA,
    register_home_state_tools,
)
from thine_harness.private_service import create_private_service_app
from thine_harness.private_topology import load_private_service_config


def _replacement_nodes() -> list[dict[str, object]]:
    return [
        {
            "node_id": "node-people",
            "component_id": "home.card.people",
            "visible": False,
            "content": {
                "title": "People to tag",
                "body": "Three voices still need names.",
                "count": 3,
            },
        },
        {
            "node_id": "node-greeting",
            "component_id": "home.hero.greeting",
            "visible": True,
            "content": {
                "title": "Good evening",
                "body": "One thing is worth revisiting.",
                "count": None,
            },
        },
    ]


def test_current_home_exists_before_mutation_and_replace_is_durable(tmp_path: Path):
    database = tmp_path / "home-state.sqlite3"
    projector = HomeStateProjector(database, clock_ms=lambda: 100)

    initial = projector.current("daily-user")
    revision = projector.replace_current(
        user_id="daily-user",
        expected_revision=initial.payload.revision,
        nodes=_replacement_nodes(),
        reason="Put unresolved speaker tagging before the greeting.",
        originating_run_id="run-home-1",
        action_id="action-home-1",
    )

    assert initial.to_dict() == {
        "schema_version": {"major": 1, "minor": 0},
        "user_id": "daily-user",
        "revision": 1,
        "updated_at_ms": 100,
        "nodes": [],
        "app_owned_chrome": [
            "home.chrome.header",
            "home.banner.google-reconnect",
            "home.card.listening",
            "home.chrome.footer",
        ],
        "extensions": {},
    }
    assert revision.payload.revision == 2
    assert revision.payload.parent_revision == 1
    revision_state = revision.to_dict()["state"]
    assert isinstance(revision_state, dict)
    assert revision_state["nodes"] == [
        {
            "node_id": "node-people",
            "component_id": "home.card.people",
            "visible": False,
            "order": 0,
            "content": {
                "title": "People to tag",
                "body": "Three voices still need names.",
                "count": 3,
                "action_key": "open_speakers",
            },
            "navigation_template": "route.speakers-list",
        },
        {
            "node_id": "node-greeting",
            "component_id": "home.hero.greeting",
            "visible": True,
            "order": 1,
            "content": {
                "title": "Good evening",
                "body": "One thing is worth revisiting.",
                "count": None,
                "action_key": None,
            },
            "navigation_template": None,
        },
    ]

    restarted = HomeStateProjector(database, clock_ms=lambda: 999)
    assert restarted.current("daily-user").to_dict() == revision_state


def test_invalid_replacement_is_atomic_and_revision_conflict_requires_reread(
    tmp_path: Path,
):
    projector = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: 200)
    before = projector.current("daily-user")
    invalid = _replacement_nodes()
    invalid[0]["navigation_template"] = "route.profile"

    with pytest.raises(
        HomeStateValidationError, match="unsupported navigation_template"
    ):
        projector.replace_current(
            user_id="daily-user",
            expected_revision=1,
            nodes=invalid,
            reason="Attempt to force a route.",
            originating_run_id="run-invalid",
            action_id="action-invalid",
        )

    assert projector.current("daily-user").to_dict() == before.to_dict()
    projector.replace_current(
        user_id="daily-user",
        expected_revision=1,
        nodes=_replacement_nodes(),
        reason="Valid replacement.",
        originating_run_id="run-valid",
        action_id="action-valid",
    )
    with pytest.raises(HomeRevisionConflict) as conflict:
        projector.replace_current(
            user_id="daily-user",
            expected_revision=1,
            nodes=[],
            reason="Stale replacement.",
            originating_run_id="run-stale",
            action_id="action-stale",
        )

    assert conflict.value.current_revision == 2
    assert "call thine_ui_state_get_current and retry" in str(conflict.value)
    assert projector.current("daily-user").payload.revision == 2


def test_duplicate_action_is_idempotent_but_conflicting_reuse_is_rejected(
    tmp_path: Path,
):
    projector = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: 300)
    first = projector.replace_current(
        user_id="daily-user",
        expected_revision=1,
        nodes=_replacement_nodes(),
        reason="Stable replacement.",
        originating_run_id="run-home",
        action_id="action-home",
    )

    replay = projector.replace_current(
        user_id="daily-user",
        expected_revision=1,
        nodes=_replacement_nodes(),
        reason="Stable replacement.",
        originating_run_id="run-home",
        action_id="action-home",
    )

    assert replay.to_dict() == first.to_dict()
    with pytest.raises(HomeActionConflict):
        projector.replace_current(
            user_id="daily-user",
            expected_revision=2,
            nodes=[],
            reason="Different intent.",
            originating_run_id="run-home",
            action_id="action-home",
        )
    assert projector.current("daily-user").payload.revision == 2
    assert projector.current("other-user").payload.revision == 1


def test_tool_handlers_return_actionable_validation_without_screen_control(
    tmp_path: Path,
):
    projector = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: 400)
    handlers = HomeToolHandlers(projector=projector, user_id="daily-user")

    current = json.loads(handlers.get_current({}))
    invalid_nodes = _replacement_nodes()
    invalid_nodes[0]["force_current_screen"] = True
    invalid = json.loads(
        handlers.replace_current({
            "expected_revision": 1,
            "nodes": invalid_nodes,
            "reason": "Invalid forced-screen mutation.",
            "originating_run_id": "run-home",
            "action_id": "action-invalid",
        })
    )

    assert current["ok"] is True
    assert current["state"]["revision"] == 1
    assert invalid == {
        "ok": False,
        "error_code": "invalid_home_state",
        "message": "nodes[0] has invalid fields: unsupported force_current_screen",
    }
    assert projector.current("daily-user").payload.revision == 1
    schema_text = json.dumps(
        [GET_CURRENT_TOOL_SCHEMA, REPLACE_CURRENT_TOOL_SCHEMA], sort_keys=True
    )
    assert "navigation_template" not in schema_text
    assert "force_current_screen" not in schema_text


def test_home_tools_are_always_registered_in_the_active_local_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    from tools.registry import registry

    projector = HomeStateProjector(tmp_path / "home.sqlite3")
    register_home_state_tools(projector, user_id="daily-user")

    get_entry = registry.get_entry(GET_CURRENT_TOOL_NAME)
    replace_entry = registry.get_entry(REPLACE_CURRENT_TOOL_NAME)
    assert get_entry is not None and replace_entry is not None
    assert get_entry.toolset == replace_entry.toolset == HOME_TOOLSET
    assert get_entry.check_fn is None and replace_entry.check_fn is None
    assert get_entry.schema["name"] == GET_CURRENT_TOOL_NAME
    assert replace_entry.schema["name"] == REPLACE_CURRENT_TOOL_NAME

    definitions = [
        {"type": "function", "function": get_entry.schema},
        {"type": "function", "function": replace_entry.schema},
    ]
    catalog = DeferredNamespaceCatalog(definitions, context_length=272_000)
    eager_names = {
        tool["function"]["name"] for tool in catalog.model_tool_definitions()
    }
    matches = catalog.search("replace current Home state content and order")
    assert GET_CURRENT_TOOL_NAME not in eager_names
    assert REPLACE_CURRENT_TOOL_NAME not in eager_names
    assert any(
        match["name"] == REPLACE_CURRENT_TOOL_NAME and match["namespace"] == "ui.state"
        for match in matches
    )


def _private_config():
    return load_private_service_config(
        {
            "thine_harness": {
                "private_service": {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": 8789,
                    "firebase_uid": "daily-user",
                    "request_timeout_seconds": 5,
                    "credential": {
                        "env": "HERMES_CONTROL_TOKEN",
                        "file": "",
                    },
                }
            }
        },
        environ={"HERMES_CONTROL_TOKEN": "private-test-token"},
    )


def _headers(request_id: str) -> dict[str, str]:
    return {
        "Authorization": "Bearer private-test-token",
        "X-Thine-Firebase-UID": "daily-user",
        "X-Request-ID": request_id,
    }


def test_control_get_home_returns_a_resolvable_frozen_home_state(tmp_path: Path):
    projector = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: 500)
    app = create_private_service_app(_private_config(), home_state=projector)
    request = {
        "schema_version": {"major": 1, "minor": 0},
        "request_id": "home-request-1",
        "operation": "get_home",
        "user_id": "daily-user",
        "deadline_at_ms": 1_900_000_000_000,
        "timeout_ms": 5_000,
        "idempotency_key": "home-read:home-request-1",
        "payload_ref": None,
        "created_at_ms": 1_800_000_000_000,
        "extensions": {},
    }

    with TestClient(app) as client:
        response = client.post(
            "/v1/control", headers=_headers("home-request-1"), json=request
        )
        result_ref = response.json()["result_ref"]
        resolved = client.post(
            "/v1/home/state/resolve",
            headers=_headers("home-resolve-1"),
            json={"user_id": "daily-user", "result_ref": result_ref},
        )

    assert response.status_code == 200
    assert response.json()["operation"] == "get_home"
    assert response.json()["status"] == "succeeded"
    assert result_ref == "home-state:1"
    assert resolved.status_code == 200
    assert resolved.json() == projector.current("daily-user").to_dict()


def test_home_projection_resource_fails_closed_for_wrong_user_and_unknown_ref(
    tmp_path: Path,
):
    projector = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: 600)
    app = create_private_service_app(_private_config(), home_state=projector)

    with TestClient(app) as client:
        wrong_user = client.post(
            "/v1/home/state/resolve",
            headers=_headers("home-resolve-wrong-user"),
            json={"user_id": "someone-else", "result_ref": "home-state:1"},
        )
        unknown = client.post(
            "/v1/home/state/resolve",
            headers=_headers("home-resolve-unknown"),
            json={"user_id": "daily-user", "result_ref": "home-state:99"},
        )

    assert wrong_user.status_code == 403
    assert wrong_user.json()["error"] == "uid_mismatch"
    assert unknown.status_code == 404
    assert unknown.json()["error"] == "home_state_not_found"
