from __future__ import annotations

from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient

from thine_harness.deferred_tools import DeferredNamespaceCatalog
from thine_harness.home_state import (
    GET_HISTORY_TOOL_NAME,
    GET_HISTORY_TOOL_SCHEMA,
    HomeActionConflict,
    HomeRevisionConflict,
    HomeStateProjector,
    HomeStateValidationError,
    HomeToolHandlers,
    GET_CURRENT_TOOL_NAME,
    GET_CURRENT_TOOL_SCHEMA,
    HOME_TOOLSET,
    REACTIVATE_REVISION_TOOL_NAME,
    REACTIVATE_REVISION_TOOL_SCHEMA,
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


def _replace(
    projector: HomeStateProjector,
    *,
    expected_revision: int,
    action_id: str,
    nodes: object | None = None,
    reason: str = "Update the agent-composed Home.",
    tick_id: str | None = None,
):
    return projector.replace_current(
        user_id="daily-user",
        expected_revision=expected_revision,
        nodes=_replacement_nodes() if nodes is None else nodes,
        reason=reason,
        originating_run_id=f"run:{action_id}",
        source_tick_id=tick_id or f"tick:{action_id}",
        author="hermes_agent",
        action_id=action_id,
    )


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
        source_tick_id="tick-home-1",
        author="hermes_agent",
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
            source_tick_id="tick-invalid",
            author="hermes_agent",
            action_id="action-invalid",
        )

    assert projector.current("daily-user").to_dict() == before.to_dict()
    projector.replace_current(
        user_id="daily-user",
        expected_revision=1,
        nodes=_replacement_nodes(),
        reason="Valid replacement.",
        originating_run_id="run-valid",
        source_tick_id="tick-valid",
        author="hermes_agent",
        action_id="action-valid",
    )
    with pytest.raises(HomeRevisionConflict) as conflict:
        projector.replace_current(
            user_id="daily-user",
            expected_revision=1,
            nodes=[],
            reason="Stale replacement.",
            originating_run_id="run-stale",
            source_tick_id="tick-stale",
            author="hermes_agent",
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
        source_tick_id="tick-home",
        author="hermes_agent",
        action_id="action-home",
    )

    replay = projector.replace_current(
        user_id="daily-user",
        expected_revision=1,
        nodes=_replacement_nodes(),
        reason="Stable replacement.",
        originating_run_id="run-home",
        source_tick_id="tick-home",
        author="hermes_agent",
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
            source_tick_id="tick-home",
            author="hermes_agent",
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
            "source_tick_id": "tick-home",
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


def test_history_retains_latest_fifty_immutable_revisions_with_provenance(
    tmp_path: Path,
):
    clock = iter(range(1_000, 1_200))
    projector = HomeStateProjector(
        tmp_path / "home.sqlite3", clock_ms=lambda: next(clock)
    )
    assert projector.current("daily-user").payload.revision == 1

    for revision in range(2, 57):
        _replace(
            projector,
            expected_revision=revision - 1,
            action_id=f"action-{revision}",
            nodes=[],
            reason=f"Create revision {revision}.",
            tick_id=f"tick-{revision}",
        )

    history = projector.history("daily-user").to_dict()
    assert history["current_revision"] == 56
    assert history["retention_limit"] == 50
    assert [item["revision"] for item in history["revisions"]] == list(range(56, 6, -1))
    assert history["revisions"][0]["extensions"] == {
        "author": "hermes_agent",
        "source_tick_id": "tick-56",
        "mutation_kind": "replace",
    }
    assert history["revisions"][0]["reason"] == "Create revision 56."
    with pytest.raises(KeyError):
        projector.resolve_state_ref(user_id="daily-user", result_ref="home-state:6")

    restarted = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: 9_999)
    assert restarted.current("daily-user").payload.revision == 56
    assert len(restarted.history("daily-user").payload.revisions) == 50


def test_reactivation_creates_a_new_head_without_rewriting_history(tmp_path: Path):
    clock = iter((1_000, 2_000, 3_000, 4_000))
    projector = HomeStateProjector(
        tmp_path / "home.sqlite3", clock_ms=lambda: next(clock)
    )
    first = _replace(
        projector,
        expected_revision=1,
        action_id="action-first",
        reason="Publish the people-first Home.",
    )
    _replace(
        projector,
        expected_revision=2,
        action_id="action-second",
        nodes=[],
        reason="Temporarily clear the Home.",
    )

    activation = projector.reactivate_revision(
        user_id="daily-user",
        expected_revision=3,
        source_revision=2,
        reason="Return to the people-first Home.",
        originating_run_id="run-reactivate",
        source_tick_id="tick-reactivate",
        author="hermes_agent",
        action_id="action-reactivate",
    )

    assert activation.to_dict() == {
        "schema_version": {"major": 1, "minor": 0},
        "action_id": "action-reactivate",
        "source_revision": 2,
        "expected_current_revision": 3,
        "new_revision": 4,
        "revalidated": True,
        "navigation_changed": False,
        "created_at_ms": 4_000,
        "extensions": {},
    }
    current = projector.current("daily-user").to_dict()
    assert current["revision"] == 4
    assert current["nodes"] == first.to_dict()["state"]["nodes"]
    history = projector.history("daily-user").to_dict()["revisions"]
    by_revision = {item["revision"]: item for item in history}
    assert by_revision[2]["state"]["revision"] == 2
    assert by_revision[3]["state"]["nodes"] == []
    assert by_revision[4]["parent_revision"] == 3
    assert by_revision[4]["extensions"] == {
        "author": "hermes_agent",
        "source_tick_id": "tick-reactivate",
        "mutation_kind": "reactivation",
        "source_revision": 2,
    }

    replay = projector.reactivate_revision(
        user_id="daily-user",
        expected_revision=3,
        source_revision=2,
        reason="Return to the people-first Home.",
        originating_run_id="run-reactivate",
        source_tick_id="tick-reactivate",
        author="hermes_agent",
        action_id="action-reactivate",
    )
    assert replay.to_dict() == activation.to_dict()
    assert projector.current("daily-user").payload.revision == 4


def test_reactivation_rejects_stale_base_and_requires_current_reread(tmp_path: Path):
    projector = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: 1_000)
    _replace(projector, expected_revision=1, action_id="action-current")

    with pytest.raises(HomeRevisionConflict) as conflict:
        projector.reactivate_revision(
            user_id="daily-user",
            expected_revision=1,
            source_revision=1,
            reason="Stale reactivation.",
            originating_run_id="run-stale-reactivate",
            source_tick_id="tick-stale-reactivate",
            author="hermes_agent",
            action_id="action-stale-reactivate",
        )

    assert conflict.value.current_revision == 2
    assert "call thine_ui_state_get_current and retry" in str(conflict.value)
    assert projector.current("daily-user").payload.revision == 2


def test_reactivation_revalidates_historical_design_against_current_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import thine_harness.home_state as home_state_module

    projector = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: 1_000)
    _replace(projector, expected_revision=1, action_id="action-old-design")
    _replace(
        projector,
        expected_revision=2,
        action_id="action-current-design",
        nodes=[],
    )
    supported = dict(home_state_module._COMPONENTS)
    supported.pop("home.card.people")
    monkeypatch.setattr(home_state_module, "_COMPONENTS", supported)

    with pytest.raises(
        HomeStateValidationError, match="unsupported component_id 'home.card.people'"
    ):
        projector.reactivate_revision(
            user_id="daily-user",
            expected_revision=3,
            source_revision=2,
            reason="Try an obsolete component.",
            originating_run_id="run-obsolete",
            source_tick_id="tick-obsolete",
            author="hermes_agent",
            action_id="action-obsolete",
        )

    assert projector.current("daily-user").payload.revision == 3


def test_pruned_mutation_receipt_remains_idempotent(tmp_path: Path):
    clock = iter(range(1_000, 1_200))
    projector = HomeStateProjector(
        tmp_path / "home.sqlite3", clock_ms=lambda: next(clock)
    )
    projector.current("daily-user")
    original = _replace(
        projector,
        expected_revision=1,
        action_id="action-pruned",
        nodes=[],
    )
    for revision in range(3, 53):
        _replace(
            projector,
            expected_revision=revision - 1,
            action_id=f"action-{revision}",
            nodes=[],
        )

    with pytest.raises(KeyError):
        projector.resolve_state_ref(user_id="daily-user", result_ref="home-state:2")
    replay = _replace(
        projector,
        expected_revision=1,
        action_id="action-pruned",
        nodes=[],
    )
    assert replay.to_dict() == original.to_dict()
    assert projector.current("daily-user").payload.revision == 52


def test_home_history_and_reactivation_tools_are_namespaced_and_do_not_restore_memory(
    tmp_path: Path,
):
    projector = HomeStateProjector(tmp_path / "home.sqlite3", clock_ms=lambda: 1_000)
    handlers = HomeToolHandlers(projector=projector, user_id="daily-user")
    _replace(projector, expected_revision=1, action_id="action-source")

    history = json.loads(handlers.get_history({}))
    activated = json.loads(
        handlers.reactivate_revision({
            "expected_revision": 2,
            "source_revision": 1,
            "reason": "Return to the empty Home.",
            "originating_run_id": "run-reactivate-tool",
            "source_tick_id": "tick-reactivate-tool",
            "action_id": "action-reactivate-tool",
        })
    )

    assert history["ok"] is True
    assert history["history"]["current_revision"] == 2
    assert activated["ok"] is True
    assert activated["activation"]["new_revision"] == 3
    schemas = json.dumps(
        [GET_HISTORY_TOOL_SCHEMA, REACTIVATE_REVISION_TOOL_SCHEMA], sort_keys=True
    )
    assert "working_memory" not in schemas
    assert "restore" not in schemas.lower()

    from tools.registry import registry

    register_home_state_tools(projector, user_id="daily-user")
    history_entry = registry.get_entry(GET_HISTORY_TOOL_NAME)
    activation_entry = registry.get_entry(REACTIVATE_REVISION_TOOL_NAME)
    assert history_entry is not None and activation_entry is not None
    assert history_entry.toolset == activation_entry.toolset == HOME_TOOLSET
