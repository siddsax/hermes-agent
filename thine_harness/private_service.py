"""Authenticated loopback HTTP boundary for the local Thine backend."""

from __future__ import annotations

import asyncio
import hmac
import time
import uuid
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .private_topology import PrivateServiceConfig


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
) -> FastAPI:
    """Build the deliberately small private HTTP surface.

    The only implemented operation is authenticated health. ``/v1/control``
    is reserved for the typed ``HermesControlPort`` integration ticket and
    intentionally has no generic dispatch behavior.
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
    async def reserved_control(
        scope: PrivateRequestScope = Depends(request_scope),
    ) -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content={
                "error": "control_not_implemented",
                "request_id": scope.request_id,
            },
        )

    return app


__all__ = [
    "PrivateRequestScope",
    "create_private_service_app",
]
