"""Retrieval configuration of the My Resume lab: which method runs and how it is tuned."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Method(StrEnum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


METHOD_NAMES = {
    Method.BM25: "BM25 keyword search",
    Method.VECTOR: "Vector similarity",
    Method.HYBRID: "Hybrid (BM25 + Vector, RRF)",
}
METHOD_DESCRIPTIONS = {
    Method.BM25: "spaCy preprocesses the resume and the jobs, then BM25 ranks jobs by keyword overlap (term frequency, rarity, length).",
    Method.VECTOR: "The resume and the jobs are embedded; jobs are ranked by cosine similarity, so different words with the same meaning still match.",
    Method.HYBRID: "BM25 and vector search each retrieve their own Top candidates; Reciprocal Rank Fusion merges the two rankings.",
}


class Bm25Settings(BaseModel):
    """spaCy preprocessing applied to both the jobs and the resume before BM25 scores them."""

    model_config = ConfigDict(frozen=True)

    lowercase: bool = True
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    lemmatize: bool = True  # "building" -> "build"; needs spaCy's en_core_web_sm model
    min_token_length: int = Field(2, ge=1, le=10)
    k1: float = Field(1.5, ge=0.0, le=3.0)  # term-frequency saturation
    b: float = Field(0.75, ge=0.0, le=1.0)  # document-length normalisation


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Method = Method.HYBRID
    top_k: int = Field(5, ge=1, le=20)  # jobs returned
    candidate_count: int = Field(20, ge=1, le=40)  # hybrid: Top N each retriever contributes to the fusion
    vector_threshold: float = Field(0.0, ge=0.0, le=1.0)  # vector candidates below this cosine similarity are dropped
    rrf_k: int = Field(60, ge=1, le=1000)  # RRF constant
    bm25: Bm25Settings = Field(default_factory=Bm25Settings)


class ScoredJob(BaseModel):
    """One ranked job with every score that produced its position.

    ``bm25_score`` / ``vector_score`` are the raw scores of each retriever (shown for every method that computed them);
    the ranks are positions in that retriever's candidate list (None when the job is not in it).
    """

    job_id: str
    rank: int
    score: float  # the score the final ranking is sorted by (BM25, cosine, or RRF depending on the method)
    bm25_score: float | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    match_percent: int = 0  # 0-100 reading of ``score`` (see ``match_percent`` in domain.ranking)


class MatchResult(BaseModel):
    config: RetrievalConfig
    results: list[ScoredJob]
    latency_ms: float  # whole retrieval, including embedding the resume (index already built)
    embed_ms: float | None = None  # the part spent embedding the resume (vector / hybrid)
    query_terms: int | None = None  # tokens left after spaCy preprocessing (bm25 / hybrid)
    notices: list[str] = Field(default_factory=list)
