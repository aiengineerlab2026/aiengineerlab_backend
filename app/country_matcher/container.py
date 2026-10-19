"""Composition root of Country Matcher: the shared pieces every request uses.

The two encoders live here, loaded lazily and only once per process (or at start-up with COUNTRY_MATCHER_WARM_MODELS),
never per request. The database comes from the application's existing engine.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.country_matcher.application.evaluation import EvaluationService
from app.country_matcher.application.indexing import IndexingService
from app.country_matcher.application.ports import ImageStorage
from app.country_matcher.application.prediction import PredictionService
from app.country_matcher.infrastructure.embedders import EmbedderRegistry
from app.country_matcher.infrastructure.repository import SqlCountryStore
from app.country_matcher.infrastructure.storage import build_storage
from app.db import transaction
from app.errors import ServiceNotConfigured
from app.rate_limit import SlidingWindowLimiter

log = logging.getLogger("country_matcher")


def sql_store_factory(session_factory: async_sessionmaker[AsyncSession]):
    @asynccontextmanager
    async def open_store():
        try:
            async with transaction(session_factory) as session:
                yield SqlCountryStore(session)
        except ProgrammingError as exc:
            # A missing table means the migration has not been applied: say so instead of a bare 500.
            if "country_matcher_" in str(exc.orig) and "does not exist" in str(exc.orig):
                raise ServiceNotConfigured("Country Matcher's database tables are missing. Run `alembic upgrade head`.") from exc
            raise

    return open_store


class CountryMatcherInfra:
    def __init__(self, settings: Settings, session_factory: async_sessionmaker[AsyncSession]):
        self.settings = settings
        self.open_store = sql_store_factory(session_factory)
        self.embedders = EmbedderRegistry(settings)
        self.storage: ImageStorage = build_storage(settings)
        self.predictor = PredictionService(
            self.open_store, self.embedders.get, max_concurrent=settings.country_matcher_max_concurrent_inference
        )
        self.indexing = IndexingService(self.open_store, self.storage, self.embedders.get)
        self.evaluation = EvaluationService(self.open_store, self.storage, self.predictor, self.embedders.get)
        self.predict_limiter = SlidingWindowLimiter(
            settings.country_matcher_predict_rate_limit_max, settings.country_matcher_predict_rate_limit_window_seconds
        )

    async def warm_up(self) -> None:
        """Load both encoders off the event loop. Non-fatal: on failure they load on first use instead."""
        try:
            await asyncio.to_thread(self.embedders.load_all)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("Country Matcher model warm-up failed (will load lazily): %s", exc)

