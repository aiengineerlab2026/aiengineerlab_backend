from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.errors import RateLimited
from app.myresume.container import MyResumeInfra
from app.myresume.presentation.schemas import LimitUsage
from app.rate_limit import SlidingWindowLimiter
from app.restaurant.presentation.deps import client_ip


def get_infra(request: Request) -> MyResumeInfra:
    return request.app.state.myresume


InfraDep = Annotated[MyResumeInfra, Depends(get_infra)]


def client_key(request: Request) -> str:
    infra: MyResumeInfra = request.app.state.myresume
    return client_ip(request, infra.settings.trust_proxy_headers)


# The demo-capped actions, each with its own per-client budget (in the infra, see container.py).
LIMITERS = {"upload": "upload_limiter", "vector": "vector_limiter"}
WINDOWS = {
    "upload": "myresume_upload_rate_limit_window_seconds",
    "vector": "myresume_vector_rate_limit_window_seconds",
}


def limiter(request: Request, kind: str) -> SlidingWindowLimiter:
    return getattr(request.app.state.myresume, LIMITERS[kind])


def usage(request: Request, kind: str) -> LimitUsage:
    u = limiter(request, kind).peek(client_key(request))
    return LimitUsage(
        limit=u.limit, used=u.used, remaining=u.remaining, reset_in_seconds=u.reset_in_seconds,
        window_seconds=getattr(request.app.state.myresume.settings, WINDOWS[kind]),
    )


def count_use(request: Request, kind: str, what: str) -> None:
    """Count one use of ``kind`` for this client IP, or raise ``RateLimited`` when its budget is used up."""
    lim = limiter(request, kind)
    result = lim.hit(client_key(request))
    if not result.allowed:
        raise RateLimited(
            f"Limit reached: {what} can be used {lim.max_events} times. This demo runs on free model tiers.",
            details={
                "limit": lim.max_events, "window_seconds": usage(request, kind).window_seconds,
                "retry_after_seconds": result.retry_after_seconds,
            },
        )


def refund_use(request: Request, kind: str) -> None:
    """Give back a counted use whose action did not happen (an unreadable file, a failed search)."""
    limiter(request, kind).refund(client_key(request))
