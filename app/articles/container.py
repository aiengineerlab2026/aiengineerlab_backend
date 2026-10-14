from __future__ import annotations

import secrets
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.articles.store import AdminStore, SqlAdminStore
from app.config import Settings
from app.db import transaction
from app.rate_limit import SlidingWindowLimiter

StoreFactory = Callable[[], AbstractAsyncContextManager[AdminStore]]


def sql_store_factory(session_factory: async_sessionmaker[AsyncSession]) -> StoreFactory:
    @asynccontextmanager
    async def open_store():
        async with transaction(session_factory) as session:  # one transaction per request
            yield SqlAdminStore(session)

    return open_store


class AdminInfra:
    """Process-wide state of the articles feature: where data lives, the token secret and the login brute-force guard."""

    def __init__(self, settings: Settings, store_factory: StoreFactory):
        self.settings = settings
        self.store_factory = store_factory
        self.secret = settings.admin_token_secret or secrets.token_urlsafe(32)
        self.login_limiter = SlidingWindowLimiter(settings.admin_login_max_failures, settings.admin_login_window_seconds)
