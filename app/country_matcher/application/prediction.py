"""Predicting a country: embed the query once, retrieve its nearest labelled images, aggregate their labels.

Everything heavy is done outside the database transaction: the connection is only held for the vector search.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from PIL import Image

from app.country_matcher.application.ports import CountryStore, ImageEmbedder
from app.country_matcher.domain.aggregation import aggregate, explain
from app.country_matcher.domain.models import Aggregation, ModelName, Neighbor, Prediction, Timings
from app.errors import ServiceNotConfigured

log = logging.getLogger("country_matcher.predict")

StoreFactory = Callable[[], AbstractAsyncContextManager[CountryStore]]
EmbedderLookup = Callable[[ModelName], ImageEmbedder]


class IndexNotReady(ServiceNotConfigured):
    code = "index_not_ready"


class PredictionService:
    def __init__(self, open_store: StoreFactory, embedders: EmbedderLookup, *, max_concurrent: int = 2):
        self.open_store, self.embedders = open_store, embedders
        self._inference = asyncio.Semaphore(max(1, max_concurrent))  # the models are CPU/memory bound

    async def predict(
        self, image: Image.Image, model: ModelName, *, k: int = 5, aggregation: Aggregation = Aggregation.WEIGHTED
    ) -> Prediction:
        embedder = self.embedders(model)
        await asyncio.to_thread(embedder.load)  # a one-off weights load must not count as this request's embedding time
        t0 = time.perf_counter()
        async with self._inference:
            vector = (await asyncio.to_thread(embedder.embed, [image]))[0]
        t1 = time.perf_counter()
        info = embedder.info
        async with self.open_store() as store:
            hits = await store.search(info, vector, k)
        t2 = time.perf_counter()
        if not hits:
            raise IndexNotReady(
                f"The {model.value} reference index is empty. Ingest the dataset and run the indexing command first.",
                details={"model": model.value},
            )
        neighbors = [Neighbor(h.image, h.similarity) for h in hits]
        ranking = aggregate(neighbors, aggregation)
        timings = Timings(
            embed_ms=(t1 - t0) * 1000, search_ms=(t2 - t1) * 1000, total_ms=(time.perf_counter() - t0) * 1000
        )
        log.info(
            "prediction",
            extra={
                "model": model.value, "k": k, "aggregation": aggregation.value, "country": ranking[0].country_code,
                **{name: round(value, 1) for name, value in vars(timings).items()},
            },
        )
        return Prediction(
            model=model, aggregation=aggregation, k=k, country=ranking[0], ranking=ranking, neighbors=neighbors,
            explanation=explain(ranking, aggregation), timings=timings, embedding=info,
        )

    async def compare(
        self, image: Image.Image, *, k: int = 5, aggregation: Aggregation = Aggregation.WEIGHTED
    ) -> dict[ModelName, Prediction]:
        # One after the other, not in parallel: concurrent runs would compete for the CPU and distort the timings shown.
        return {m: await self.predict(image, m, k=k, aggregation=aggregation) for m in ModelName}
