from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import Settings

log = logging.getLogger("app.db")


class Base(DeclarativeBase):
    """Shared declarative base: every app's tables register on this metadata (Alembic reads it)."""


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.sqlalchemy_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=5,
        pool_pre_ping=True,  # test a connection before handing it out: a dead one is replaced instead of failing the request
        pool_recycle=settings.db_pool_recycle_seconds,
        pool_timeout=settings.db_pool_timeout_seconds,
        pool_use_lifo=True,  # reuse the freshest connection, so idle ones can age out and be recycled
        connect_args=settings.connect_args,
    )


async def warm_pool(engine: AsyncEngine, connections: int) -> None:
    """Open ``connections`` real connections at once and run ``SELECT 1`` on each (raises if the database is unreachable).

    The pool connects lazily, and a connection is slow to set up (DNS, TCP, TLS, auth, often through a pooler). Doing it
    here, before the app serves traffic, keeps that cost off the first user requests.
    """
    t0 = time.perf_counter()

    async def ping() -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            await asyncio.sleep(0.05)  # hold it briefly so the others cannot reuse it: each task opens its own connection

    await asyncio.gather(*(ping() for _ in range(max(1, connections))))
    log.info("database pool warmed", extra={"connections": max(1, connections), "ms": round((time.perf_counter() - t0) * 1000)})


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@asynccontextmanager
async def transaction(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """One unit of work: commit on success, roll back on error."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
