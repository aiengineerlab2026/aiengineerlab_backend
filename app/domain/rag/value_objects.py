"""RAG configuration value objects.

``ChunkingConfig`` + embedding spec + dataset version determine the index *fingerprint*.
``RetrievalConfig`` is deliberately NOT part of the fingerprint: changing retrieval settings must
never trigger re-embedding.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    ENTITY = "entity"


class RetrievalStrategy(StrEnum):
    VECTOR = "vector"
    MULTI_QUERY = "multi_query"
    MULTI_QUERY_RERANK = "multi_query_rerank"
    HYBRID_RRF = "hybrid_rrf"


class LengthUnit(StrEnum):
    CHARACTERS = "characters"
    TOKENS = "tokens"


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)

    strategy: ChunkingStrategy = ChunkingStrategy.ENTITY
    chunk_size: int = Field(400, ge=50, le=4000)
    chunk_overlap: int = Field(40, ge=0, le=1000)
    length_unit: LengthUnit = LengthUnit.CHARACTERS
    separators: tuple[str, ...] = DEFAULT_SEPARATORS
    breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile", "gradient"] = (
        "percentile"
    )
    breakpoint_threshold_amount: float = Field(90.0, ge=0, le=100)

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> ChunkingConfig:
        if self.strategy in (ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE):
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def fingerprint_params(self) -> dict[str, Any]:
        """Only the parameters that actually influence the produced chunks."""
        s = self.strategy
        if s == ChunkingStrategy.ENTITY:
            return {"strategy": s.value}
        if s == ChunkingStrategy.FIXED:
            return {
                "strategy": s.value,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "length_unit": self.length_unit.value,
            }
        if s == ChunkingStrategy.RECURSIVE:
            return {
                "strategy": s.value,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "length_unit": self.length_unit.value,
                "separators": list(self.separators),
            }
        return {
            "strategy": s.value,
            "breakpoint_threshold_type": self.breakpoint_threshold_type,
            "breakpoint_threshold_amount": self.breakpoint_threshold_amount,
        }


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: RetrievalStrategy = RetrievalStrategy.VECTOR
    top_k: int = Field(5, ge=1, le=20)
    num_query_variants: int = Field(3, ge=1, le=8)
    candidate_count: int = Field(20, ge=1, le=100, description="Chunks fetched per query variant / retriever branch")
    rerank_enabled: bool = False
    reranker_model: str = DEFAULT_RERANKER_MODEL
    rrf_k: int = Field(60, ge=1, le=1000)

    @model_validator(mode="after")
    def _normalize(self) -> RetrievalConfig:
        # Reranking is intrinsic to multi_query_rerank, optional for hybrid, meaningless otherwise.
        if self.strategy == RetrievalStrategy.MULTI_QUERY_RERANK:
            object.__setattr__(self, "rerank_enabled", True)
        elif self.strategy != RetrievalStrategy.HYBRID_RRF:
            object.__setattr__(self, "rerank_enabled", False)
        if self.candidate_count < self.top_k:
            object.__setattr__(self, "candidate_count", self.top_k)
        return self


class EmbeddingSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    dimensions: int
    version: str = "1"


def compute_fingerprint(
    *,
    dataset_version: str,
    dataset_hash: str,
    chunking: ChunkingConfig,
    embedding: EmbeddingSpec,
    canonical_format_version: str,
) -> str:
    payload = {
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "canonical_format": canonical_format_version,
        "chunking": chunking.fingerprint_params(),
        "embedding": embedding.model_dump(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class ModelRef(BaseModel):
    """One chosen model: which provider serves it and its id there."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str


class ModelSelection(BaseModel):
    """The two language models the live configuration runs on.

    The embedding model is not part of it: every request names its own (see ``SendMessageIn.embedding``).
    """

    llm: ModelRef  # writes the chat answers and rewrites queries
    judge: ModelRef  # the evaluation (RAGAS / relevance) judge
