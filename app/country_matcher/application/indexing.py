"""Computing and storing the reference embeddings (offline, admin-only: ``cli index``).

Each stored image is embedded once per model *version* and the vector is kept in Postgres. Users switching models later
only embed their own query image; the dataset is never re-embedded. Running it again only embeds what is missing, so it
resumes after a failure and picks up newly ingested images.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field

from app.country_matcher.application.ports import CountryStore, ImageEmbedder, ImageStorage
from app.country_matcher.domain.models import ModelName, Split
from app.country_matcher.infrastructure.imaging import decode_image
from app.errors import DomainError

log = logging.getLogger("country_matcher.index")

StoreFactory = Callable[[], AbstractAsyncContextManager[CountryStore]]
EmbedderLookup = Callable[[ModelName], ImageEmbedder]


@dataclass
class IndexReport:
    model_name: str = ""
    model_version: str = ""
    embedded: int = 0
    already_indexed: int = 0
    failed: list[str] = field(default_factory=list)


class IndexingService:
    def __init__(self, open_store: StoreFactory, storage: ImageStorage, embedders: EmbedderLookup, *, batch_size: int = 16):
        self.open_store, self.storage, self.embedders, self.batch_size = open_store, storage, embedders, batch_size

    async def index(self, models: Sequence[ModelName]) -> dict[ModelName, IndexReport]:
        return {m: await self._index_model(m) for m in models}

    async def _index_model(self, model: ModelName) -> IndexReport:
        embedder = self.embedders(model)
        info = await asyncio.to_thread(lambda: embedder.info)  # loads the weights off the event loop
        async with self.open_store() as store:
            missing = await store.images_missing_embedding(info)
            total = await store.count_images(split=Split.INDEX)
        report = IndexReport(info.model_name, info.model_version, already_indexed=total - len(missing))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            loaded = await asyncio.gather(*(self._load(r) for r in batch))
            ok = [(r, img) for r, img in zip(batch, loaded, strict=True) if img is not None]
            report.failed += [f"{r.id}" for r, img in zip(batch, loaded, strict=True) if img is None]
            if not ok:
                continue
            vectors = await asyncio.to_thread(embedder.embed, [img for _, img in ok])
            async with self.open_store() as store:
                await store.save_embeddings(info, [(r.id, v) for (r, _), v in zip(ok, vectors, strict=True)])
            report.embedded += len(ok)
            log.info("indexed", extra={"model": model.value, "done": report.embedded, "of": len(missing)})
        return report

    async def _load(self, record):
        try:
            return decode_image(await self.storage.get(record.storage_path), min_side=1)
        except (DomainError, OSError) as exc:
            log.warning("could not load stored image", extra={"image": str(record.id), "reason": str(exc)})
            return None
