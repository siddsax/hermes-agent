"""Authoritative, profile-scoped Home state for the local Thine Harness.

The projector owns validated Home data only.  iOS continues to own rendering
and navigation, so the model-facing replacement input deliberately contains no
navigation or screen-control command.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from hermes_constants import get_hermes_home

from .contracts.codec import ContractDecodeError
from .contracts.control import HermesControlRequest, HermesControlResponse
from .contracts.home import HomeRevision, HomeState


_VERSION = {"major": 1, "minor": 0}
_APP_OWNED_CHROME = [
    "home.chrome.header",
    "home.banner.google-reconnect",
    "home.card.listening",
    "home.chrome.footer",
]
_COMPONENTS: dict[str, tuple[str | None, str | None]] = {
    "home.hero.greeting": (None, None),
    "home.action.what-did-i-miss": ("route.chat", "open_chat"),
    "home.card.connectors": ("route.connectors", "open_connectors"),
    "home.card.people": ("route.speakers-list", "open_speakers"),
}
_COUNT_COMPONENTS = frozenset(("home.card.connectors", "home.card.people"))

HOME_TOOLSET = "local-thine"
GET_CURRENT_TOOL_NAME = "thine_ui_state_get_current"
REPLACE_CURRENT_TOOL_NAME = "thine_ui_state_replace_current"
_HOME_STATE_REF_PREFIX = "home-state:"


class HomeStateError(RuntimeError):
    """A Home operation could not preserve the accepted state contract."""

    code = "home_state_error"


class HomeStateValidationError(HomeStateError):
    code = "invalid_home_state"


class HomeRevisionConflict(HomeStateError):
    code = "revision_conflict"

    def __init__(self, *, expected_revision: int, current_revision: int) -> None:
        super().__init__(
            f"expected Home revision {expected_revision}, but current revision is "
            f"{current_revision}; call {GET_CURRENT_TOOL_NAME} and retry"
        )
        self.expected_revision = expected_revision
        self.current_revision = current_revision


class HomeActionConflict(HomeStateError):
    code = "action_conflict"


def default_database_path() -> Path:
    return get_hermes_home() / "thine-harness" / "home-state.sqlite3"


@dataclass(frozen=True)
class HomeToolHandlers:
    """JSON tool handlers bound to exactly one Local Thine profile."""

    projector: "HomeStateProjector"
    user_id: str

    def get_current(self, args: Mapping[str, object], **_kwargs: object) -> str:
        if args:
            return _tool_error(
                HomeStateValidationError(
                    f"{GET_CURRENT_TOOL_NAME} accepts no arguments"
                )
            )
        return _tool_success(state=self.projector.current(self.user_id).to_dict())

    def replace_current(self, args: Mapping[str, object], **_kwargs: object) -> str:
        required = {
            "expected_revision",
            "nodes",
            "reason",
            "originating_run_id",
            "action_id",
        }
        if set(args) != required:
            unknown = sorted(set(args) - required)
            missing = sorted(required - set(args))
            detail = []
            if missing:
                detail.append("missing: " + ", ".join(missing))
            if unknown:
                detail.append("unsupported: " + ", ".join(unknown))
            return _tool_error(
                HomeStateValidationError(
                    "replacement requires exactly the documented fields ("
                    + "; ".join(detail)
                    + ")"
                )
            )
        try:
            revision = self.projector.replace_current(
                user_id=self.user_id,
                expected_revision=args["expected_revision"],
                nodes=args["nodes"],
                reason=args["reason"],
                originating_run_id=args["originating_run_id"],
                action_id=args["action_id"],
            )
        except HomeStateError as exc:
            return _tool_error(exc)
        return _tool_success(revision=revision.to_dict())


class HomeProjectionControl:
    """Frozen HermesControlPort projection for backend/mobile Home reads."""

    def __init__(self, projector: "HomeStateProjector", *, clock_ms: Any | None = None):
        import time

        self._projector = projector
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))

    def handle(
        self,
        request: HermesControlRequest,
        *,
        authenticated_user_id: str,
        transport_request_id: str,
    ) -> HermesControlResponse:
        payload = request.payload
        status = "succeeded"
        error_code: str | None = None
        result_ref: str | None = None
        now_ms = int(self._clock_ms())
        if payload.request_id != transport_request_id:
            status, error_code = "rejected", "request_id_mismatch"
        elif payload.user_id != authenticated_user_id:
            status, error_code = "rejected", "user_id_mismatch"
        elif payload.deadline_at_ms <= now_ms:
            status, error_code = "timed_out", "deadline_expired"
        elif payload.operation != "get_home":
            status, error_code = "rejected", "unsupported_operation"
        elif payload.payload_ref is not None:
            status, error_code = "rejected", "invalid_payload_ref"
        else:
            state = self._projector.current(payload.user_id)
            result_ref = _HOME_STATE_REF_PREFIX + str(state.payload.revision)
        return HermesControlResponse.from_dict({
            "schema_version": _VERSION,
            "request_id": payload.request_id,
            "operation": payload.operation,
            "idempotency_key": payload.idempotency_key,
            "deadline_at_ms": payload.deadline_at_ms,
            "timeout_ms": payload.timeout_ms,
            "status": status,
            "result_ref": result_ref,
            "error_code": error_code,
            "responded_at_ms": now_ms,
            "extensions": {},
        })

    def resolve(self, *, user_id: str, result_ref: str) -> HomeState:
        return self._projector.resolve_state_ref(user_id=user_id, result_ref=result_ref)


class HomeStateProjector:
    """Transactional current-state projector with immutable revision rows."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        clock_ms: Any | None = None,
    ) -> None:
        import time

        self.path = Path(path) if path is not None else default_database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS home_revisions (
                    user_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    parent_revision INTEGER,
                    originating_run_id TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    intent_fingerprint TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    revision_json TEXT NOT NULL,
                    PRIMARY KEY (user_id, revision),
                    UNIQUE (user_id, action_id)
                );
                CREATE TABLE IF NOT EXISTS home_current (
                    user_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    FOREIGN KEY (user_id, revision)
                        REFERENCES home_revisions(user_id, revision)
                );
                COMMIT;
                """
            )
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def current(self, user_id: str) -> HomeState:
        normalized_user_id = _bounded_identity(user_id, field="user_id")
        with self._transaction() as connection:
            row = self._current_row(connection, normalized_user_id)
            if row is None:
                row = self._initialize_current(connection, normalized_user_id)
            return HomeState.from_json(row["state_json"])

    def replace_current(
        self,
        *,
        user_id: object,
        expected_revision: object,
        nodes: object,
        reason: object,
        originating_run_id: object,
        action_id: object,
    ) -> HomeRevision:
        normalized_user_id = _bounded_identity(user_id, field="user_id")
        normalized_run_id = _bounded_identity(
            originating_run_id, field="originating_run_id"
        )
        normalized_action_id = _bounded_identity(action_id, field="action_id")
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 1
        ):
            raise HomeStateValidationError(
                "expected_revision must be an integer greater than or equal to 1"
            )
        if not isinstance(reason, str) or not 1 <= len(reason) <= 1000:
            raise HomeStateValidationError(
                "reason must be a non-empty string of at most 1,000 characters"
            )
        normalized_nodes = _validated_tool_nodes(nodes)
        intent_fingerprint = _intent_fingerprint({
            "user_id": normalized_user_id,
            "expected_revision": expected_revision,
            "nodes": normalized_nodes,
            "reason": reason,
            "originating_run_id": normalized_run_id,
        })

        with self._transaction() as connection:
            duplicate = connection.execute(
                """
                SELECT intent_fingerprint, revision_json
                FROM home_revisions
                WHERE user_id = ? AND action_id = ?
                """,
                (normalized_user_id, normalized_action_id),
            ).fetchone()
            if duplicate is not None:
                if duplicate["intent_fingerprint"] != intent_fingerprint:
                    raise HomeActionConflict(
                        "action_id was already used for a different Home replacement"
                    )
                return HomeRevision.from_json(duplicate["revision_json"])

            current = self._current_row(connection, normalized_user_id)
            if current is None:
                current = self._initialize_current(connection, normalized_user_id)
            current_revision = int(current["revision"])
            if expected_revision != current_revision:
                raise HomeRevisionConflict(
                    expected_revision=expected_revision,
                    current_revision=current_revision,
                )

            created_at_ms = int(self._clock_ms())
            next_revision = current_revision + 1
            state = _build_home_state(
                user_id=normalized_user_id,
                revision=next_revision,
                updated_at_ms=created_at_ms,
                nodes=normalized_nodes,
            )
            revision = HomeRevision.from_dict({
                "schema_version": _VERSION,
                "revision": next_revision,
                "parent_revision": current_revision,
                "originating_run_id": normalized_run_id,
                "action_id": normalized_action_id,
                "reason": reason,
                "created_at_ms": created_at_ms,
                "state": state.to_dict(),
                "extensions": {},
            })
            connection.execute(
                """
                INSERT INTO home_revisions (
                    user_id, revision, parent_revision, originating_run_id,
                    action_id, intent_fingerprint, reason, created_at_ms,
                    state_json, revision_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_user_id,
                    next_revision,
                    current_revision,
                    normalized_run_id,
                    normalized_action_id,
                    intent_fingerprint,
                    reason,
                    created_at_ms,
                    state.to_json(),
                    revision.to_json(),
                ),
            )
            connection.execute(
                "UPDATE home_current SET revision = ? WHERE user_id = ?",
                (next_revision, normalized_user_id),
            )
            return revision

    def resolve_state_ref(self, *, user_id: str, result_ref: str) -> HomeState:
        normalized_user_id = _bounded_identity(user_id, field="user_id")
        if not result_ref.startswith(_HOME_STATE_REF_PREFIX):
            raise KeyError(result_ref)
        revision_text = result_ref.removeprefix(_HOME_STATE_REF_PREFIX)
        if not revision_text.isascii() or not revision_text.isdecimal():
            raise KeyError(result_ref)
        revision = int(revision_text)
        if revision < 1 or str(revision) != revision_text:
            raise KeyError(result_ref)
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT state_json FROM home_revisions
                WHERE user_id = ? AND revision = ?
                """,
                (normalized_user_id, revision),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise KeyError(result_ref)
        return HomeState.from_json(row["state_json"])

    @staticmethod
    def _current_row(
        connection: sqlite3.Connection, user_id: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT revision, state_json
            FROM home_revisions
            WHERE user_id = ? AND revision = (
                SELECT revision FROM home_current WHERE user_id = ?
            )
            """,
            (user_id, user_id),
        ).fetchone()

    def _initialize_current(
        self, connection: sqlite3.Connection, user_id: str
    ) -> sqlite3.Row:
        created_at_ms = int(self._clock_ms())
        state = _build_home_state(
            user_id=user_id,
            revision=1,
            updated_at_ms=created_at_ms,
            nodes=[],
        )
        revision = HomeRevision.from_dict({
            "schema_version": _VERSION,
            "revision": 1,
            "parent_revision": None,
            "originating_run_id": "system:home-bootstrap",
            "action_id": "system:home-bootstrap",
            "reason": "Create the default agent-composed Home state.",
            "created_at_ms": created_at_ms,
            "state": state.to_dict(),
            "extensions": {},
        })
        connection.execute(
            """
            INSERT INTO home_revisions (
                user_id, revision, parent_revision, originating_run_id,
                action_id, intent_fingerprint, reason, created_at_ms,
                state_json, revision_json
            ) VALUES (?, 1, NULL, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                "system:home-bootstrap",
                "system:home-bootstrap",
                _intent_fingerprint({"user_id": user_id, "bootstrap": True}),
                "Create the default agent-composed Home state.",
                created_at_ms,
                state.to_json(),
                revision.to_json(),
            ),
        )
        connection.execute(
            "INSERT INTO home_current (user_id, revision) VALUES (?, 1)",
            (user_id,),
        )
        row = self._current_row(connection, user_id)
        if row is None:  # pragma: no cover - guarded by the transaction above
            raise HomeStateError("default Home state was not persisted")
        return row


def _bounded_identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise HomeStateValidationError(
            f"{field} must be a non-empty string of at most 128 characters"
        )
    return value


def _validated_tool_nodes(nodes: object) -> list[dict[str, object]]:
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes, bytearray)):
        raise HomeStateValidationError("nodes must be an ordered array")
    result: list[dict[str, object]] = []
    node_ids: set[str] = set()
    component_ids: set[str] = set()
    for index, raw_node in enumerate(nodes):
        if not isinstance(raw_node, Mapping):
            raise HomeStateValidationError(f"nodes[{index}] must be an object")
        required = {"node_id", "component_id", "visible", "content"}
        if set(raw_node) != required:
            unsupported = sorted(set(raw_node) - required)
            missing = sorted(required - set(raw_node))
            parts = []
            if missing:
                parts.append("missing " + ", ".join(missing))
            if unsupported:
                parts.append("unsupported " + ", ".join(unsupported))
            raise HomeStateValidationError(
                f"nodes[{index}] has invalid fields: " + "; ".join(parts)
            )
        node_id = _bounded_identity(
            raw_node["node_id"], field=f"nodes[{index}].node_id"
        )
        if node_id in node_ids:
            raise HomeStateValidationError(f"duplicate node_id {node_id!r}")
        node_ids.add(node_id)
        component_id = raw_node["component_id"]
        if not isinstance(component_id, str) or component_id not in _COMPONENTS:
            supported = ", ".join(sorted(_COMPONENTS))
            raise HomeStateValidationError(
                f"unsupported component_id {component_id!r}; supported: {supported}"
            )
        if component_id in component_ids:
            raise HomeStateValidationError(
                f"component_id {component_id!r} is a singleton and cannot repeat"
            )
        component_ids.add(component_id)
        visible = raw_node["visible"]
        if not isinstance(visible, bool):
            raise HomeStateValidationError(f"nodes[{index}].visible must be boolean")
        content = raw_node["content"]
        if not isinstance(content, Mapping) or set(content) != {
            "title",
            "body",
            "count",
        }:
            raise HomeStateValidationError(
                f"nodes[{index}].content requires exactly title, body, and count; "
                "navigation/action fields are app-owned"
            )
        count = content["count"]
        if component_id not in _COUNT_COMPONENTS and count is not None:
            raise HomeStateValidationError(
                f"nodes[{index}].content.count must be null for {component_id}"
            )
        if component_id in _COUNT_COMPONENTS and (
            isinstance(count, bool) or not (count is None or isinstance(count, int))
        ):
            raise HomeStateValidationError(
                f"nodes[{index}].content.count must be a non-negative integer or null"
            )
        if isinstance(count, int) and count < 0:
            raise HomeStateValidationError(
                f"nodes[{index}].content.count must not be negative"
            )
        navigation_template, action_key = _COMPONENTS[component_id]
        result.append({
            "node_id": node_id,
            "component_id": component_id,
            "visible": visible,
            "order": index,
            "content": {
                "title": content["title"],
                "body": content["body"],
                "count": count,
                "action_key": action_key,
            },
            "navigation_template": navigation_template,
        })
    return result


def _build_home_state(
    *, user_id: str, revision: int, updated_at_ms: int, nodes: list[dict[str, object]]
) -> HomeState:
    try:
        return HomeState.from_dict({
            "schema_version": _VERSION,
            "user_id": user_id,
            "revision": revision,
            "updated_at_ms": updated_at_ms,
            "nodes": nodes,
            "app_owned_chrome": _APP_OWNED_CHROME,
            "extensions": {},
        })
    except ContractDecodeError as exc:
        raise HomeStateValidationError(str(exc)) from exc


def _intent_fingerprint(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _tool_success(**payload: object) -> str:
    return json.dumps(
        {"ok": True, **payload}, ensure_ascii=False, separators=(",", ":")
    )


def _tool_error(exc: HomeStateError) -> str:
    result: dict[str, object] = {
        "ok": False,
        "error_code": exc.code,
        "message": str(exc),
    }
    if isinstance(exc, HomeRevisionConflict):
        result["current_revision"] = exc.current_revision
        result["retry_with"] = GET_CURRENT_TOOL_NAME
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def register_home_state_tools(
    projector: HomeStateProjector,
    *,
    user_id: str,
    registry_instance: Any | None = None,
) -> HomeToolHandlers:
    """Register both helpers in the active Local Thine profile scope.

    The registration has no ``check_fn`` or preference gate: Home mutation is
    a core capability of this maintained-fork profile.  The caller still
    chooses the ``local-thine`` toolset when constructing the profile agent.
    """
    from tools.registry import registry

    active_registry = registry_instance or registry
    handlers = HomeToolHandlers(
        projector=projector, user_id=_bounded_identity(user_id, field="user_id")
    )
    scope = active_registry.current_scope_key()
    active_registry.register(
        name=GET_CURRENT_TOOL_NAME,
        toolset=HOME_TOOLSET,
        schema=GET_CURRENT_TOOL_SCHEMA,
        handler=handlers.get_current,
        scope=scope,
    )
    active_registry.register(
        name=REPLACE_CURRENT_TOOL_NAME,
        toolset=HOME_TOOLSET,
        schema=REPLACE_CURRENT_TOOL_SCHEMA,
        handler=handlers.replace_current,
        scope=scope,
    )
    return handlers


GET_CURRENT_TOOL_SCHEMA = {
    "name": GET_CURRENT_TOOL_NAME,
    "description": (
        "Read the authoritative current agent-composed Thine Home state and revision. "
        "Call this before proposing a replacement."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

REPLACE_CURRENT_TOOL_SCHEMA = {
    "name": REPLACE_CURRENT_TOOL_NAME,
    "description": (
        "Atomically replace the current Thine Home nodes after reading its revision. "
        "Array position defines order. Navigation and screen control remain app-owned."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expected_revision": {"type": "integer", "minimum": 1},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "minLength": 1, "maxLength": 128},
                        "component_id": {"type": "string", "enum": sorted(_COMPONENTS)},
                        "visible": {"type": "boolean"},
                        "content": {
                            "type": "object",
                            "properties": {
                                "title": {"type": ["string", "null"], "maxLength": 120},
                                "body": {"type": ["string", "null"], "maxLength": 500},
                                "count": {"type": ["integer", "null"], "minimum": 0},
                            },
                            "required": ["title", "body", "count"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["node_id", "component_id", "visible", "content"],
                    "additionalProperties": False,
                },
            },
            "reason": {"type": "string", "minLength": 1, "maxLength": 1000},
            "originating_run_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "action_id": {"type": "string", "minLength": 1, "maxLength": 128},
        },
        "required": [
            "expected_revision",
            "nodes",
            "reason",
            "originating_run_id",
            "action_id",
        ],
        "additionalProperties": False,
    },
}


__all__ = [
    "GET_CURRENT_TOOL_NAME",
    "GET_CURRENT_TOOL_SCHEMA",
    "HOME_TOOLSET",
    "HomeActionConflict",
    "HomeRevisionConflict",
    "HomeStateError",
    "HomeStateProjector",
    "HomeStateValidationError",
    "HomeProjectionControl",
    "HomeToolHandlers",
    "REPLACE_CURRENT_TOOL_NAME",
    "REPLACE_CURRENT_TOOL_SCHEMA",
    "default_database_path",
    "register_home_state_tools",
]
