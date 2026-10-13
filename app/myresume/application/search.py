"""Job retrieval for a resume: BM25, vector similarity, or both fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence

import numpy as np

from app.errors import ValidationFailed
from app.myresume.domain.models import Job
from app.myresume.domain.ranking import match_percent, rrf_fuse
from app.myresume.domain.retrieval import MatchResult, Method, RetrievalConfig, ScoredJob
from app.myresume.infrastructure.bm25 import Bm25Registry
from app.myresume.infrastructure.vector_index import VectorIndexRegistry
from app.restaurant.domain.rag.value_objects import EmbeddingSpec
from app.restaurant.infrastructure.rag.embeddings import LangChainEmbedder

MIN_RESUME_CHARS = 40


def _ranked(scores: np.ndarray, job_ids: Sequence[str], *, floor: float, limit: int) -> list[tuple[str, float]]:
    """Job ids by descending score, keeping only scores above ``floor`` (stable: ties keep corpus order)."""
    order = np.argsort(-scores, kind="stable")
    return [(job_ids[i], float(scores[i])) for i in order if scores[i] > floor][:limit]


class JobSearchService:
    def __init__(
        self,
        jobs: Sequence[Job],
        bm25: Bm25Registry,
        vectors: VectorIndexRegistry,
        embedder_for: Callable[[EmbeddingSpec], LangChainEmbedder],
    ):
        self._job_ids = [j.id for j in jobs]
        self._bm25, self._vectors, self._embedder_for = bm25, vectors, embedder_for

    async def prepare(self, config: RetrievalConfig, spec: EmbeddingSpec) -> None:
        """Build whatever index the method needs, so ``match`` times retrieval only (not one-off index builds)."""
        if config.method in (Method.BM25, Method.HYBRID):
            await asyncio.to_thread(self._bm25.get, config.bm25)
        if config.method in (Method.VECTOR, Method.HYBRID):
            await self._vectors.get(spec)

    async def match(self, resume_text: str, config: RetrievalConfig, spec: EmbeddingSpec) -> MatchResult:
        text = resume_text.strip()
        if len(text) < MIN_RESUME_CHARS:
            raise ValidationFailed(f"The resume is too short to match: write at least {MIN_RESUME_CHARS} characters.")
        await self.prepare(config, spec)

        started = time.perf_counter()
        use_bm25 = config.method in (Method.BM25, Method.HYBRID)
        use_vec = config.method in (Method.VECTOR, Method.HYBRID)
        # Hybrid asks each retriever for its own Top N (at least Top K, so the final list can always be filled).
        n = max(config.candidate_count, config.top_k) if config.method == Method.HYBRID else config.top_k

        async def run_bm25():
            index = self._bm25.get(config.bm25)
            scores, terms = await asyncio.to_thread(index.score, text)
            return index, scores, terms

        async def run_vector():
            index = await self._vectors.get(spec)
            t = time.perf_counter()
            query_vec = await self._embedder_for(spec).embed_query(text)
            return index.similarities(query_vec), (time.perf_counter() - t) * 1000

        # "Independently": the two retrievers share nothing, so hybrid runs them concurrently.
        bm25_out, vec_out = await asyncio.gather(
            run_bm25() if use_bm25 else _none(), run_vector() if use_vec else _none()
        )

        notices: list[str] = []
        bm25_scores = vec_scores = None
        terms = embed_ms = None
        bm25_cands: list[tuple[str, float]] = []
        vec_cands: list[tuple[str, float]] = []
        if bm25_out:
            index, bm25_scores, terms = bm25_out
            bm25_cands = _ranked(bm25_scores, self._job_ids, floor=0.0, limit=n)
            if index.lemmatizer_missing:
                notices.append("spaCy's English model is not installed, so lemmatisation was skipped.")
            if terms == 0:
                notices.append("No usable keywords were left after preprocessing; BM25 found nothing.")
        if vec_out:
            vec_scores, embed_ms = vec_out
            vec_cands = _ranked(vec_scores, self._job_ids, floor=config.vector_threshold - 1e-12, limit=n)  # keep sim >= threshold
            if not vec_cands:
                notices.append(f"No job reached the vector similarity threshold of {config.vector_threshold:.2f}.")

        bm25_rank = {j: r for r, (j, _) in enumerate(bm25_cands, 1)}
        vec_rank = {j: r for r, (j, _) in enumerate(vec_cands, 1)}
        idx = {j: i for i, j in enumerate(self._job_ids)}

        def scored(job_id: str, rank: int, score: float, rrf: float | None = None) -> ScoredJob:
            i = idx[job_id]
            return ScoredJob(
                job_id=job_id, rank=rank, score=score, rrf_score=rrf,
                bm25_score=float(bm25_scores[i]) if bm25_scores is not None else None, bm25_rank=bm25_rank.get(job_id),
                vector_score=float(vec_scores[i]) if vec_scores is not None else None, vector_rank=vec_rank.get(job_id),
            )

        if config.method == Method.BM25:
            results = [scored(j, r, s) for r, (j, s) in enumerate(bm25_cands[: config.top_k], 1)]
        elif config.method == Method.VECTOR:
            results = [scored(j, r, s) for r, (j, s) in enumerate(vec_cands[: config.top_k], 1)]
        else:
            fused = rrf_fuse([[j for j, _ in bm25_cands], [j for j, _ in vec_cands]], k=config.rrf_k)
            results = [scored(j, r, s, rrf=s) for r, (j, s) in enumerate(fused[: config.top_k], 1)]

        top = results[0].score if results else 0.0
        low = results[-1].score if results else 0.0  # hybrid normalises the fused scores between the best and the last job returned
        results = [
            r.model_copy(update={"match_percent": match_percent(config.method.value, r.score, top, low)}) for r in results
        ]
        return MatchResult(
            config=config, results=results, latency_ms=(time.perf_counter() - started) * 1000, embed_ms=embed_ms,
            query_terms=terms, notices=notices,
        )


async def _none():
    return None
