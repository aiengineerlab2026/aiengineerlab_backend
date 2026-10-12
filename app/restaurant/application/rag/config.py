"""The one live RAG configuration.

There are no named experiments: a single configuration is edited in place, and every edit applies to the very
next question. Retrieval settings take effect immediately. A chunking change needs a different index; it is
reused when one already exists, otherwise built in the background - and until it is ready the chat keeps answering
from the most recent ready index (the trace says so), so the app never goes dark while you tune.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.restaurant.application.rag.indexing import IndexingService
from app.restaurant.domain.canonical import MENU_DATASET_VERSION
from app.restaurant.domain.rag.entities import CONFIG_ID, IndexStatus, RagConfig, RagIndex
from app.restaurant.domain.rag.ports import ConfigRepository, EmbeddingModel
from app.restaurant.domain.rag.value_objects import (
    ChunkingConfig,
    ChunkingStrategy,
    EmbeddingSpec,
    RetrievalConfig,
    RetrievalStrategy,
)

DEFAULT_CHUNKING = ChunkingConfig(strategy=ChunkingStrategy.ENTITY)
DEFAULT_RETRIEVAL = RetrievalConfig(strategy=RetrievalStrategy.VECTOR, top_k=5)


@dataclass(frozen=True)
class Serving:
    """What a question is actually answered with right now."""

    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    index_fingerprint: str
    warning: str | None = None  # set while a newer index is still building
    embedding: EmbeddingSpec | None = None  # the embedding model whose index this is


class ConfigService:
    def __init__(
        self, repo: ConfigRepository, indexing: IndexingService, embedder: EmbeddingModel, llm_model: str,
        prebuilt: Sequence[EmbeddingSpec] = (),
    ):
        self._repo, self._indexing, self._embedder, self._llm_model = repo, indexing, embedder, llm_model
        self._prebuilt = list(prebuilt)  # embedding models whose index is kept ready ahead of time

    async def get(self) -> RagConfig:
        """The live configuration (created from the defaults on first use)."""
        cfg = await self._repo.get()
        if cfg is None:
            cfg, _ = await self.update(DEFAULT_CHUNKING, DEFAULT_RETRIEVAL)
        return cfg

    async def update(
        self, chunking: ChunkingConfig, retrieval: RetrievalConfig, models: dict | None = None
    ) -> tuple[RagConfig, RagIndex]:
        """Replace the live configuration. ``models`` is the saved model choice; None keeps the one already stored."""
        # Only chunking (and embedding/dataset) can change the fingerprint; retrieval never does.
        index, _ = await self._indexing.request_index(chunking)
        for spec in self._prebuilt:  # every embedding model gets its index built ahead of time; switching just points to it
            if spec != self._embedder.spec:
                await self._indexing.request_index(chunking, spec)
        if models is None:
            previous = await self._repo.get()
            models = previous.models if previous else {}
        cfg = RagConfig(
            id=CONFIG_ID, chunking=chunking, retrieval=retrieval, embedding=self._embedder.spec,
            llm_model=self._llm_model, dataset_version=MENU_DATASET_VERSION, index_fingerprint=index.fingerprint,
            models=models,
        )
        await self._repo.save(cfg)
        saved = await self._repo.get()
        assert saved is not None
        return saved, index

    async def ensure_index(self) -> RagIndex:
        """The index for the current chunking: reused if it exists, queued for building otherwise."""
        cfg = await self.get()
        index, _ = await self._indexing.request_index(cfg.chunking)
        if index.fingerprint != cfg.index_fingerprint or cfg.embedding != self._embedder.spec:
            await self.update(cfg.chunking, cfg.retrieval)  # menu or embedding model changed since it was saved
        return index

    async def index_status(self) -> RagIndex:
        cfg = await self.get()
        index = await self._indexing.get_by_fingerprint(cfg.index_fingerprint)
        if index is None or cfg.embedding != self._embedder.spec:
            index = await self.ensure_index()
        return index

    async def serving(self, spec: EmbeddingSpec | None = None) -> Serving:
        """The configuration + index to answer with, for one embedding model (``spec``; default: the app default).

        Each embedding model has its own pre-built index; this just points to it, nothing is re-embedded. While that
        index is still building, the last ready index of the same model is used instead.
        """
        cfg = await self.get()
        spec = spec or self._embedder.spec
        if spec == self._embedder.spec:
            index = await self.index_status()
        else:
            index, _ = await self._indexing.request_index(cfg.chunking, spec)  # queues it if it does not exist yet
        if index.status == IndexStatus.READY:
            return Serving(cfg.chunking, cfg.retrieval, index.fingerprint, embedding=spec)
        previous = await self._indexing.latest_ready(spec.model, spec.dimensions)
        if previous is None:  # nothing to fall back to: retrieval reports "index not ready"
            return Serving(cfg.chunking, cfg.retrieval, index.fingerprint, embedding=spec)
        note = (
            f"The index for the new chunking settings is still {index.status.value}; this answer used the previous "
            f"index ({previous.chunking.strategy.value} chunking)."
        )
        return Serving(previous.chunking, cfg.retrieval, previous.fingerprint, warning=note, embedding=spec)
