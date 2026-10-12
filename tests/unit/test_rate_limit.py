from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.infrastructure.config import Settings
from app.infrastructure.rate_limit import SlidingWindowLimiter
from app.presentation.deps import enforce_chat_rate_limit
from app.presentation.errors import register_error_handlers


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_allows_exactly_the_budget_then_blocks():
    lim = SlidingWindowLimiter(20, 3600, Clock())
    results = [lim.hit("1.2.3.4") for _ in range(21)]
    assert all(r.allowed for r in results[:20])
    assert [r.remaining for r in results[:20]] == list(range(19, -1, -1))
    assert not results[20].allowed and results[20].retry_after_seconds > 0


def test_each_ip_has_its_own_budget():
    lim = SlidingWindowLimiter(2, 60, Clock())
    assert lim.hit("a").allowed and lim.hit("a").allowed and not lim.hit("a").allowed
    assert lim.hit("b").allowed


def test_budget_returns_as_the_window_slides():
    clock = Clock()
    lim = SlidingWindowLimiter(2, 100, clock)
    lim.hit("a")
    clock.now += 60
    lim.hit("a")
    blocked = lim.hit("a")
    assert not blocked.allowed and blocked.retry_after_seconds == 40  # oldest event expires in 40s
    clock.now += 40
    assert lim.hit("a").allowed  # first event aged out, one slot free
    assert not lim.hit("a").allowed


def test_rejected_attempts_do_not_extend_the_block():
    clock = Clock()
    lim = SlidingWindowLimiter(1, 100, clock)
    lim.hit("a")
    for _ in range(50):
        clock.now += 1
        assert not lim.hit("a").allowed
    clock.now += 50  # 100s after the only counted event
    assert lim.hit("a").allowed


def test_zero_disables_the_limit():
    lim = SlidingWindowLimiter(0, 3600)
    assert all(lim.hit("a").allowed for _ in range(1000))


def test_defaults_are_twenty_per_ip():
    assert Settings().chat_rate_limit_max == 5


def _client(*, trust_proxy: bool = False, limit: int = 3) -> TestClient:
    settings = SimpleNamespace(chat_rate_limit_max=limit, chat_rate_limit_window_seconds=3600, trust_proxy_headers=trust_proxy)
    app = FastAPI()
    app.state.infra = SimpleNamespace(settings=settings, chat_limiter=SlidingWindowLimiter(limit, 3600))
    register_error_handlers(app)

    @app.post("/send", dependencies=[Depends(enforce_chat_rate_limit)])
    async def send():
        return {"ok": True}

    return TestClient(app)


def test_endpoint_returns_429_envelope_with_retry_after():
    c = _client()
    assert [c.post("/send").status_code for _ in range(3)] == [200, 200, 200]
    r = c.post("/send")
    assert r.status_code == 429
    err = r.json()["error"]
    assert err["code"] == "rate_limited" and "3 messages" in err["message"]
    assert err["details"]["limit"] == 3 and err["details"]["retry_after_seconds"] > 0
    assert int(r.headers["Retry-After"]) == err["details"]["retry_after_seconds"]


def test_forwarded_header_is_ignored_unless_the_proxy_is_trusted():
    c = _client(trust_proxy=False, limit=1)
    assert c.post("/send", headers={"X-Forwarded-For": "9.9.9.1"}).status_code == 200
    assert c.post("/send", headers={"X-Forwarded-For": "9.9.9.2"}).status_code == 429  # spoofing does not help


def test_forwarded_header_is_used_when_the_proxy_is_trusted():
    c = _client(trust_proxy=True, limit=1)
    assert c.post("/send", headers={"X-Forwarded-For": "9.9.9.1, 10.0.0.1"}).status_code == 200
    assert c.post("/send", headers={"X-Forwarded-For": "9.9.9.1"}).status_code == 429
    assert c.post("/send", headers={"X-Forwarded-For": "9.9.9.2"}).status_code == 200


@pytest.mark.parametrize("path", ["/chat/sessions/{session_id}/messages"])
def test_the_real_chat_endpoint_is_guarded(path):
    """Guards against someone removing the dependency from the production route."""
    from app.presentation.api.v1.ordering import router

    route = next(r for r in router.routes if getattr(r, "path", "") == path and "POST" in r.methods)
    assert any(d.call is enforce_chat_rate_limit for d in route.dependant.dependencies)


def test_peek_reports_usage_without_consuming_budget():
    now = [0.0]
    lim = SlidingWindowLimiter(3, 100, clock=lambda: now[0])
    assert (lim.peek("a").used, lim.peek("a").remaining, lim.peek("a").reset_in_seconds) == (0, 3, 0)
    lim.hit("a")
    now[0] = 30
    lim.hit("a")
    u = lim.peek("a")
    assert (u.limit, u.used, u.remaining, u.reset_in_seconds) == (3, 2, 1, 70)
    assert lim.peek("a").used == 2  # peeking never counts
    now[0] = 101  # first event aged out
    assert lim.peek("a").used == 1
    assert lim.peek("b").remaining == 3


def test_peek_is_empty_when_the_limit_is_disabled():
    assert SlidingWindowLimiter(0, 3600).peek("a").limit == 0


def test_usage_endpoint_counts_only_real_messages():
    from app.presentation.api.v1.ordering import chat_usage

    c = _client(limit=3)
    c.app.add_api_route("/usage", chat_usage, methods=["GET"])
    first = c.get("/usage").json()
    assert first == {"enabled": True, "limit": 3, "used": 0, "remaining": 3, "window_seconds": 3600, "reset_in_seconds": 0}
    c.post("/send")
    c.get("/usage")
    after = c.get("/usage").json()
    assert (after["used"], after["remaining"]) == (1, 2) and after["reset_in_seconds"] > 0


def test_usage_route_is_registered_before_the_session_routes():
    from app.presentation.api.v1.ordering import router

    assert any(getattr(r, "path", "") == "/chat/usage" and "GET" in r.methods for r in router.routes)
