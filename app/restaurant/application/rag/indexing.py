"""Index lifecycle: fingerprint -> reuse an existing index or build a new one."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from app.errors import ConflictError
from app.restaurant.domain.canonical import (
    CANONICAL_FORMAT_VERSION,
    MENU_DATASET_VERSION,
    CanonicalCorpus,
    build_corpus,
)
from app.restaurant.domain.rag.entities import IndexStatus, RagIndex
from app.restaurant.domain.rag.ports import Chunker, EmbeddingModel, IndexRepository, VectorStore
from app.restaurant.domain.rag.value_objects import (
    ChunkingConfig,
    ChunkingStrategy,
    EmbeddingSpec,
    compute_fingerprint,
)
from app.restaurant.domain.repositories import MenuRepository

log = logging.getLogger(__name__)

ProgressFn = Callable[[dict], Awaitable[None]]
ChunkerResolver = Callable[[ChunkingStrategy], Chunker]
EMBED_BATCH = 64


class IndexingService:
    """Owns index fingerprints, cache reuse and the (slow) build procedure."""

    def __init__(
        self, indexes: IndexRepository, store: VectorStore, menu: MenuRepository,
        embedder: EmbeddingModel, chunkers: ChunkerResolver,
        embedder_for: Callable[[EmbeddingSpec], EmbeddingModel] | None = None,
    ):
        self._indexes, self._store, self._menu = indexes, store, menu
        self._embedder, self._chunkers = embedder, chunkers
        # An index is always built with *its own* embedding model, which need not be the live one.
        self._embedder_for = embedder_for or (lambda _spec: embedder)

    async def corpus(self) -> CanonicalCorpus:
        return build_corpus(await self._menu.list_dishes())

    async def fingerprint(
        self, chunking: ChunkingConfig, corpus: CanonicalCorpus | None = None, spec: EmbeddingSpec | None = None
    ) -> str:
        corpus = corpus or await self.corpus()
        return compute_fingerprint(
            dataset_version=MENU_DATASET_VERSION,
            dataset_hash=corpus.content_hash,
            chunking=chunking,
            embedding=spec or self._embedder.spec,
            canonical_format_version=CANONICAL_FORMAT_VERSION,
        )

    async def request_index(
        self, chunking: ChunkingConfig, spec: EmbeddingSpec | None = None
    ) -> tuple[RagIndex, bool]:
        """Return the index for this configuration, creating a PENDING one if absent.

        Returns ``(index, reused)``. A ready/building/pending index is reused as-is (no re-embedding);
        a FAILED index is re-queued.
        """
        spec = spec or self._embedder.spec
        corpus = await self.corpus()
        fp = await self.fingerprint(chunking, corpus, spec)
        existing = await self._indexes.get_by_fingerprint(fp)
        if existing is not None:
            if existing.status == IndexStatus.FAILED:
                await self._indexes.update_status(existing.id, IndexStatus.PENDING, error=None, progress={})
                existing.status = IndexStatus.PENDING
                return existing, False
            return existing, True
        await self._indexes.add(
            RagIndex(
                id=str(uuid.uuid4()), fingerprint=fp, dataset_version=MENU_DATASET_VERSION,
                chunking=chunking, embedding=spec,
            )
        )
        created = await self._indexes.get_by_fingerprint(fp)
        assert created is not None
        return created, False

    async def get_by_fingerprint(self, fingerprint: str) -> RagIndex | None:
        return await self._indexes.get_by_fingerprint(fingerprint)

    async def latest_ready(self, model: str, dimensions: int) -> RagIndex | None:
        return await self._indexes.latest_ready(model, dimensions)

    async def build(self, index: RagIndex, report: ProgressFn) -> None:
        """Chunk the canonical corpus, embed every chunk, and persist. Idempotent (replaces chunks)."""
        t0 = time.perf_counter()
        await report({"stage": "chunking", "done": 0, "total": 0})
        corpus = await self.corpus()
        if not corpus.dishes:
            raise ConflictError("Menu is empty. Run `python -m app.cli seed` first.")
        drafts = await self._chunkers(index.chunking.strategy).chunk(corpus, index.chunking)
        if not drafts:
            raise ConflictError("Chunking produced no chunks; check the chunking configuration.")
        embedder = self._embedder_for(index.embedding)
        total = len(drafts)
        await report({"stage": "embedding", "done": 0, "total": total})
        embeddings: list[list[float]] = []
        for start in range(0, total, EMBED_BATCH):
            batch = drafts[start : start + EMBED_BATCH]
            embeddings.extend(await embedder.embed_documents([d.content for d in batch]))
            await report({"stage": "embedding", "done": len(embeddings), "total": total})
        await report({"stage": "storing", "done": total, "total": total})
        stored = await self._store.replace_chunks(index.id, list(zip(drafts, embeddings, strict=True)))
        log.info(
            "index built",
            extra={"index_id": index.id, "chunks": stored, "seconds": round(time.perf_counter() - t0, 2),
                   "strategy": index.chunking.strategy.value},
        )
        await self._indexes.update_status(
            index.id, IndexStatus.READY, chunk_count=stored, progress={"stage": "ready", "done": total, "total": total}
        )

    async def mark_failed(self, index_id: str, error: str) -> None:
        await self._indexes.update_status(index_id, IndexStatus.FAILED, error=error[:2000])

    async def claim_pending(self) -> RagIndex | None:
        return await self._indexes.claim_pending()

    async def requeue_interrupted(self) -> int:
        return await self._indexes.requeue_interrupted()

    async def set_progress(self, index_id: str, progress: dict) -> None:
        idx = await self._indexes.get(index_id)
        if idx is not None:
            await self._indexes.update_status(index_id, idx.status, progress=progress, error=idx.error)
