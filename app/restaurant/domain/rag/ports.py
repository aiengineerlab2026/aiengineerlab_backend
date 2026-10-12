"""Ports (interfaces) implemented by the infrastructure layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from app.restaurant.domain.canonical import CanonicalCorpus
from app.restaurant.domain.rag.entities import (
    Chunk,
    ChunkDraft,
    IndexStatus,
    RagConfig,
    RagIndex,
    ScoredChunk,
)
from app.restaurant.domain.rag.value_objects import ChunkingConfig, EmbeddingSpec

T = TypeVar("T", bound=BaseModel)


class EmbeddingModel(ABC):
    @property
    @abstractmethod
    def spec(self) -> EmbeddingSpec: ...

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...


class Reranker(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance score per document (cross-encoder logits; only meaningful within one call)."""


class LLMClient(ABC):
    """Provider-independent LLM interface."""

    @property
    @abstractmethod
    def provider(self) -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    def is_configured(self) -> bool:
        return True

    @abstractmethod
    async def generate(self, system: str, user: str) -> str: ...

    @abstractmethod
    async def generate_structured(self, system: str, user: str, schema: type[T]) -> T: ...


class Chunker(ABC):
    """Strategy interface for chunking the canonical corpus."""

    @abstractmethod
    async def chunk(self, corpus: CanonicalCorpus, config: ChunkingConfig) -> list[ChunkDraft]: ...


class ConfigRepository(ABC):
    """Storage for the single live RAG configuration."""

    @abstractmethod
    async def get(self) -> RagConfig | None: ...

    @abstractmethod
    async def save(self, config: RagConfig) -> None:
        """Insert or replace the configuration."""


class IndexRepository(ABC):
    @abstractmethod
    async def get(self, index_id: str) -> RagIndex | None: ...

    @abstractmethod
    async def get_by_fingerprint(self, fingerprint: str) -> RagIndex | None: ...

    @abstractmethod
    async def add(self, index: RagIndex) -> None: ...

    @abstractmethod
    async def update_status(
        self,
        index_id: str,
        status: IndexStatus,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
        progress: dict | None = None,
    ) -> None: ...

    @abstractmethod
    async def latest_ready(self, embedding_model: str, dimensions: int) -> RagIndex | None:
        """Most recently completed READY index made with this embedding model (used while a new one builds)."""

    @abstractmethod
    async def claim_pending(self) -> RagIndex | None:
        """Atomically move one pending index to BUILDING and return it (SKIP LOCKED)."""

    @abstractmethod
    async def requeue_interrupted(self) -> int:
        """Reset BUILDING indexes (left over from a crashed process) to PENDING."""


class VectorStore(ABC):
    @abstractmethod
    async def replace_chunks(
        self, index_id: str, chunks: Sequence[tuple[ChunkDraft, list[float]]]
    ) -> int: ...

    @abstractmethod
    async def count_chunks(self, index_id: str) -> int: ...

    @abstractmethod
    async def vector_search(
        self,
        index_id: str,
        embedding: Sequence[float],
        limit: int,
        allowed_dish_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """Cosine-distance search. ``ScoredChunk.score`` is the cosine *distance* (lower = closer)."""

    @abstractmethod
    async def lexical_search(
        self,
        index_id: str,
        query: str,
        limit: int,
        allowed_dish_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """PostgreSQL full-text search (ts_rank_cd). NOT BM25."""

    @abstractmethod
    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]: ...

    @abstractmethod
    async def chunks_covering(self, index_id: str, dish_ids: Sequence[str]) -> list[Chunk]:
        """Every chunk of the index that covers at least one of ``dish_ids``, in index order."""
