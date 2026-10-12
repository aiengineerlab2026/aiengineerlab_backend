from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request

from app.container import Infra, Services
from app.domain.errors import RateLimited
from app.infrastructure.db.session import transaction


def get_infra(request: Request) -> Infra:
    return request.app.state.infra


async def get_services(request: Request) -> AsyncIterator[Services]:
    """One transaction per request; committed right after the handler returns (before the response is sent)."""
    infra: Infra = request.app.state.infra
    async with transaction(infra.session_factory) as session:
        yield Services(infra, session)


# scope="function": run the dependency's exit (commit) as soon as the endpoint returns.
ServicesDep = Annotated[Services, Depends(get_services, scope="function")]
InfraDep = Annotated[Infra, Depends(get_infra)]


def client_ip(request: Request, trust_proxy_headers: bool) -> str:
    if trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


def enforce_chat_rate_limit(request: Request) -> None:
    """Dependency for endpoints that trigger LLM work: at most N messages per client IP per window."""
    infra: Infra = request.app.state.infra
    s = infra.settings
    result = infra.chat_limiter.hit(client_ip(request, s.trust_proxy_headers))
    if not result.allowed:
        raise RateLimited(
            f"Message limit reached: you can send up to {s.chat_rate_limit_max} messages in total. "
            "This demo runs on the free version of Gemini.",
            details={
                "limit": s.chat_rate_limit_max,
                "window_seconds": s.chat_rate_limit_window_seconds,
                "retry_after_seconds": result.retry_after_seconds,
            },
        )
