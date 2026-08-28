"""Authenticated loopback HTTP boundary for the local Thine backend."""

from __future__ import annotations

import asyncio
import hmac
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .contracts.codec import ContractDecodeError
from .contracts.control import HermesControlRequest
from .private_topology import PrivateServiceConfig

if TYPE_CHECKING:
    from .p0_chat import P0ChatController


@dataclass(frozen=True)
class PrivateRequestScope:
    """Identity established for one private backend request."""

    request_id: str
    firebase_uid: str


class _PrivateRequestRejected(Exception):
    def __init__(self, *, status_code: int, error: str, request_id: str | None) -> None:
        super().__init__(error)
        self.status_code = status_code
        self.error = error
        self.request_id = request_id


class _RequestDeadlineMiddleware:
    def __init__(self, app: ASGIApp, timeout_seconds: float) -> None:
        self.app = app
        self.timeout_seconds = timeout_seconds

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def track_response(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            async with asyncio.timeout(self.timeout_seconds):
                await self.app(scope, receive, track_response)
        except TimeoutError:
            if response_started:
                raise
            request_id = Request(scope).headers.get("X-Request-ID", "")
            content: dict[str, str] = {"error": "request_timed_out"}
            if (
                request_id
                and request_id == request_id.strip()
                and len(request_id) <= 128
            ):
                content["request_id"] = request_id
            response = JSONResponse(status_code=504, content=content)
            await response(scope, receive, send)


def _validated_request_id(request: Request) -> str:
    request_id = request.headers.get("X-Request-ID", "")
    if not request_id or request_id != request_id.strip() or len(request_id) > 128:
        raise _PrivateRequestRejected(
            status_code=400,
            error="invalid_request_id",
            request_id=None,
        )
    return request_id


def _authenticate_request(
    request: Request,
    *,
    config: PrivateServiceConfig,
) -> PrivateRequestScope:
    request_id = _validated_request_id(request)
    authorization = request.headers.get("Authorization", "")
    scheme, separator, bearer = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(
            bearer.encode("utf-8"),
            config.credential.encode("utf-8"),
        )
    ):
        raise _PrivateRequestRejected(
            status_code=401,
            error="unauthorized",
            request_id=request_id,
        )

    firebase_uid = request.headers.get("X-Thine-Firebase-UID", "")
    if not hmac.compare_digest(
        firebase_uid.encode("utf-8"),
        config.firebase_uid.encode("utf-8"),
    ):
        raise _PrivateRequestRejected(
            status_code=403,
            error="uid_mismatch",
            request_id=request_id,
        )
    return PrivateRequestScope(request_id=request_id, firebase_uid=firebase_uid)


def create_private_service_app(
    config: PrivateServiceConfig,
    *,
    process_instance_id: str | None = None,
    started_at_ms: int | None = None,
    p0_control: "P0ChatController | None" = None,
) -> FastAPI:
    """Build the deliberately small private HTTP surface.

    The surface is intentionally closed: authenticated health, typed P0
    control, and two explicit semantic-resource resolvers. It has no generic
    dispatch or arbitrary backend RPC behavior.
    """

    if not config.enabled or not config.credential or not config.firebase_uid:
        raise ValueError("private service configuration is not enabled and ready")
    instance_id = process_instance_id or str(uuid.uuid4())
    if not instance_id:
        raise ValueError("process_instance_id must not be empty")
    start_ms = int(time.time() * 1000) if started_at_ms is None else started_at_ms

    app = FastAPI(
        title="Hermes private control boundary",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        # Starlette accepts callable ASGI middleware classes here; ty's
        # variadic middleware-factory protocol does not recognize the class.
        _RequestDeadlineMiddleware,  # ty: ignore[invalid-argument-type]
        config.request_timeout_seconds,
    )

    @app.exception_handler(_PrivateRequestRejected)
    async def rejected_request(
        _request: Request, exc: _PrivateRequestRejected
    ) -> JSONResponse:
        body: dict[str, str] = {"error": exc.error}
        if exc.request_id is not None:
            body["request_id"] = exc.request_id
        return JSONResponse(status_code=exc.status_code, content=body)

    def request_scope(request: Request) -> PrivateRequestScope:
        return _authenticate_request(request, config=config)

    @app.get("/health")
    async def health(
        scope: PrivateRequestScope = Depends(request_scope),
    ) -> dict[str, object]:
        return {
            "schema_version": {"major": 1, "minor": 0},
            "service": "hermes_control",
            "status": "ready",
            "process_instance_id": instance_id,
            "started_at_ms": start_ms,
            "request_id": scope.request_id,
        }

    @app.post("/v1/control")
    async def control(
        request: Request,
        background_tasks: BackgroundTasks,
        scope: PrivateRequestScope = Depends(request_scope),
    ) -> JSONResponse:
        if p0_control is not None:
            try:
                decoded = HermesControlRequest.from_json(await request.body())
            except ContractDecodeError as exc:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "invalid_control_request",
                        "request_id": scope.request_id,
                        "detail": str(exc),
                    },
                )
            response = p0_control.admit(
                decoded,
                authenticated_user_id=scope.firebase_uid,
                transport_request_id=scope.request_id,
            )
            if response.payload.result_ref is not None:
                background_tasks.add_task(
                    p0_control.activate, response.payload.result_ref
                )
            return JSONResponse(status_code=200, content=response.to_dict())
        return JSONResponse(
            status_code=501,
            content={
                "error": "control_not_implemented",
                "request_id": scope.request_id,
            },
        )

    @app.post("/v1/chat/queue-receipts/resolve")
    async def resolve_queue_receipt(
        request: Request,
        scope: PrivateRequestScope = Depends(request_scope),
    ) -> JSONResponse:
        if p0_control is None:
            return JSONResponse(
                status_code=501,
                content={
                    "error": "control_not_implemented",
                    "request_id": scope.request_id,
                },
            )
        body = await _strict_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        if set(body) != {"user_id", "result_ref"} or not all(
            isinstance(body.get(key), str) and body[key]
            for key in ("user_id", "result_ref")
        ):
            return _invalid_resource_request(scope.request_id)
        if body["user_id"] != scope.firebase_uid:
            return JSONResponse(
                status_code=403,
                content={"error": "uid_mismatch", "request_id": scope.request_id},
            )
        try:
            receipt = p0_control.resolve_queue_receipt(
                user_id=body["user_id"],
                result_ref=body["result_ref"],
            )
        except (KeyError, ValueError):
            return JSONResponse(
                status_code=404,
                content={
                    "error": "queue_receipt_not_found",
                    "request_id": scope.request_id,
                },
            )
        return JSONResponse(status_code=200, content=receipt.to_dict())

    @app.post("/v1/chat/assistant-content/resolve")
    async def resolve_assistant_content(
        request: Request,
        scope: PrivateRequestScope = Depends(request_scope),
    ) -> JSONResponse:
        if p0_control is None:
            return JSONResponse(
                status_code=501,
                content={
                    "error": "control_not_implemented",
                    "request_id": scope.request_id,
                },
            )
        body = await _strict_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        if set(body) != {"user_id", "content_ref"} or not all(
            isinstance(body.get(key), str) and body[key]
            for key in ("user_id", "content_ref")
        ):
            return _invalid_resource_request(scope.request_id)
        if body["user_id"] != scope.firebase_uid:
            return JSONResponse(
                status_code=403,
                content={"error": "uid_mismatch", "request_id": scope.request_id},
            )
        try:
            text = p0_control.resolve_assistant_content(
                user_id=body["user_id"],
                content_ref=body["content_ref"],
            )
        except (KeyError, ValueError):
            return JSONResponse(
                status_code=404,
                content={
                    "error": "assistant_content_not_found",
                    "request_id": scope.request_id,
                },
            )
        return JSONResponse(
            status_code=200,
            content={"content_ref": body["content_ref"], "text": text},
        )

    return app


async def _strict_json_object(request: Request) -> dict[str, object] | JSONResponse:
    try:
        body = await request.json()
    except ValueError:
        return _invalid_resource_request(request.headers.get("X-Request-ID", ""))
    if not isinstance(body, dict):
        return _invalid_resource_request(request.headers.get("X-Request-ID", ""))
    return body


def _invalid_resource_request(request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_resource_request", "request_id": request_id},
    )


__all__ = [
    "PrivateRequestScope",
    "create_private_service_app",
]
