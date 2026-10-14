"""Connection pool settings and the start-up warm-up (no real database: the engine is a stand-in)."""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.db import create_engine, warm_pool


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, database_url="postgresql://u:p@db.example.com:5432/app", **kw)


def test_engine_pings_recycles_and_times_out_connections():
    s = _settings()
    engine = create_engine(s)
    pool = engine.pool
    assert pool._pre_ping is True and pool._recycle == 1800 and pool._timeout == s.db_pool_timeout_seconds
    assert s.connect_args["timeout"] == s.db_connect_timeout_seconds  # a connection attempt must not hang for 60 s


class _Conn:
    def __init__(self, engine: _Engine):
        self.engine = engine

    async def __aenter__(self):
        self.engine.open += 1
        self.engine.peak = max(self.engine.peak, self.engine.open)
        return self

    async def __aexit__(self, *exc):
        self.engine.open -= 1

    async def execute(self, statement):
        self.engine.statements.append(str(statement))


class _Engine:
    open = peak = 0

    def __init__(self):
        self.statements: list[str] = []

    def connect(self) -> _Conn:
        return _Conn(self)


def test_warm_up_opens_the_requested_connections_at_once_and_pings_each():
    engine = _Engine()
    asyncio.run(warm_pool(engine, 3))  # type: ignore[arg-type]
    assert engine.peak == 3 and engine.statements == ["SELECT 1"] * 3
