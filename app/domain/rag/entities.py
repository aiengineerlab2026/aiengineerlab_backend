"""RAG domain entities, retrieval trace models and job status."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.domain.rag.value_objects import ChunkingConfig, EmbeddingSpec, RetrievalConfig


def _now() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    AWAITING = "awaiting"  # live evaluation only: recorded, waiting for the customer to press "Evaluate"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexStatus(StrEnum):
    """Index lifecycle exposed to the UI: pending -> building -> ready | failed."""

    PENDING = "pending"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


@dataclass
class ChunkDraft:
    """A chunk produced by a chunking strategy, before embedding."""

    content: str
    dish_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_type: str = "dish"


@dataclass
class Chunk:
    id: str
    index_id: str
    position: int
    source_type: str
    source_id: str
    dish_ids: list[str]
    content: str
    metadata: dict[str, Any]


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # raw score of the retriever that produced it (see score_type)
    score_type: str


@dataclass
class RagIndex:
    id: str
    fingerprint: str
    dataset_version: str
    chunking: ChunkingConfig
    embedding: EmbeddingSpec
    status: IndexStatus = IndexStatus.PENDING
    chunk_count: int = 0
    progress: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    completed_at: datetime | None = None


CONFIG_ID = "00000000-0000-0000-0000-000000000001"  # the one live configuration has a fixed id


@dataclass
class RagConfig:
    """The one live RAG configuration. Every edit replaces it and applies to the next question."""

    id: str
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    embedding: EmbeddingSpec
    llm_model: str
    dataset_version: str
    index_fingerprint: str
    models: dict[str, Any] = field(default_factory=dict)  # the model choice made in the UI ({} = .env defaults)
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


# ------------------------------- retrieval trace -------------------------------------------------


class TraceChunk(BaseModel):
    """A retrieved chunk with *labelled* scores. Score types are never comparable across keys."""

    rank: int
    chunk_id: str
    content: str
    dish_ids: list[str]
    dish_names: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceStage(BaseModel):
    name: str  # e.g. "vector[q1]", "lexical", "rrf", "rerank"
    description: str
    score_type: str
    query: str | None = None
    items: list[TraceChunk] = Field(default_factory=list)
    latency_ms: float = 0.0


class RetrievalTrace(BaseModel):
    query: str
    query_variants: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    index_fingerprint: str
    embedding_model: str
    stages: list[TraceStage] = Field(default_factory=list)
    chunks: list[TraceChunk] = Field(default_factory=list)  # final ranking
    context: str = ""  # exact text handed to the LLM
    final_score_type: str = ""
    latency_ms: float = 0.0
    latency_breakdown_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
