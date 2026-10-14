"""Consistent error envelope: {"error": {"code", "message", "details"}}."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.errors import (
    ConflictError,
    DomainError,
    ExternalServiceError,
    NotFoundError,
    RateLimited,
    ServiceNotConfigured,
    Unauthorized,
    ValidationFailed,
)

log = logging.getLogger(__name__)

_STATUS = {
    Unauthorized: 401,
    NotFoundError: 404,
    ValidationFailed: 422,
    ConflictError: 409,
    RateLimited: 429,
    ServiceNotConfigured: 503,
    ExternalServiceError: 502,
}


def _body(code: str, message: str, details: dict | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def domain_error(_: Request, exc: DomainError) -> JSONResponse:
        status = next((s for t, s in _STATUS.items() if isinstance(exc, t)), 400)
        headers = {}
        if isinstance(exc, RateLimited) and "retry_after_seconds" in exc.details:
            headers["Retry-After"] = str(exc.details["retry_after_seconds"])
        return JSONResponse(_body(exc.code, exc.message, exc.details), status_code=status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        first = errors[0]["msg"] if errors else "Invalid request"
        return JSONResponse(_body("validation_error", first, {"errors": errors}), status_code=422)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse(_body("internal_error", "Internal server error."), status_code=500)
