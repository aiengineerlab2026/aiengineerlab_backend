"""Configurable retrieval strategies with a full, labelled trace.

Score types are never mixed: every stage and every final chunk carries named scores
(``cosine_distance``/``cosine_similarity``, ``ts_rank_cd``, ``rrf_score``, ``cross_encoder_score``).
Lexical retrieval is PostgreSQL full-text search (``ts_rank_cd``) - NOT BM25.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from app.application import prompts
from app.application.rag.indexing import IndexingService
from app.domain.errors import ConflictError, ServiceNotConfigured
from app.domain.rag.entities import Chunk, IndexStatus, RetrievalTrace, ScoredChunk, TraceChunk, TraceStage
from app.domain.rag.ports import EmbeddingModel, LLMClient, Reranker, VectorStore
from app.domain.rag.ranking import merge_best_score, rrf_fuse
from app.domain.rag.value_objects import ChunkingConfig, EmbeddingSpec, RetrievalConfig, RetrievalStrategy
from app.domain.restaurant.filters import DishFilter
from app.domain.restaurant.repositories import MenuRepository


class QueryVariants(BaseModel):
    queries: list[str]


def build_context(chunks: Sequence[TraceChunk]) -> str:
    """The exact context string handed to the LLM (also shown verbatim in the Retrieval Inspector)."""
    parts = []
    for c in chunks:
        parts.append(f"[{c.rank}] (dishes: {', '.join(c.dish_ids) or 'n/a'})\n{c.content}")
    return "\n\n".join(parts)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def clean_variants(original: str, candidates: Sequence[str], n: int) -> list[str]:
    seen = {original.strip().lower()}
    out: list[str] = []
    for q in candidates:
        q = q.strip().strip('"').strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
        if len(out) >= n:
            break
    return out


@dataclass
class _Run:
    """Mutable state of one retrieval call."""

    index_id: str
    query: str
    chunking: ChunkingConfig
    cfg: RetrievalConfig
    allowed: list[str] | None
    embedder: EmbeddingModel
    stages: list[TraceStage] = field(default_factory=list)
    chunks: dict[str, Chunk] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    variants: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_time(self, key: str, ms: float) -> None:
        self.timings[key] = round(self.timings.get(key, 0.0) + ms, 2)


def _trace_chunk(rank: int, chunk: Chunk, scores: dict[str, float], extra: dict | None = None) -> TraceChunk:
    meta = dict(chunk.metadata)
    meta.update(extra or {})
    return TraceChunk(
        rank=rank, chunk_id=chunk.id, content=chunk.content, dish_ids=chunk.dish_ids,
        dish_names=list(chunk.metadata.get("dish_names", [])), scores=scores, metadata=meta,
    )


class RetrievalService:
    def __init__(
        self, indexing: IndexingService, store: VectorStore, embedder: EmbeddingModel,
        rerankers: Callable[[str], Reranker], llm: LLMClient, menu: MenuRepository,
        embedder_for: Callable[[EmbeddingSpec], EmbeddingModel] | None = None,
    ):
        self._indexing, self._store, self._embedder = indexing, store, embedder
        # The query is embedded with the model of the index that is searched, so both always match.
        self._embedder_for = embedder_for or (lambda _spec: embedder)
        self._rerankers, self._llm, self._menu = rerankers, llm, menu

    # ------------------------------------------------------------------ public

    async def retrieve(
        self, query: str, *, chunking: ChunkingConfig, retrieval: RetrievalConfig,
        index_fingerprint: str, dish_filter: DishFilter | None = None, notice: str | None = None,
    ) -> RetrievalTrace:
        t_total = time.perf_counter()
        index = await self._indexing.get_by_fingerprint(index_fingerprint)
        if index is None or index.status != IndexStatus.READY:
            state = index.status.value if index else "missing"
            raise ConflictError(
                f"The vector index for this configuration is not ready (status: {state}). "
                "It builds automatically after a chunking change; try again in a moment.",
                details={"index_status": state},
            )
        allowed: list[str] | None = None
        if dish_filter is not None and not dish_filter.is_empty():
            allowed = [d.id for d in await self._menu.list_dishes() if dish_filter.matches(d)]
        run = _Run(index.id, query, chunking, retrieval, allowed, self._embedder_for(index.embedding))
        if notice:
            run.warnings.append(notice)
        filters = dish_filter.to_dict() if dish_filter else {}
        if allowed is not None and not allowed:
            run.warnings.append("No menu dish satisfies the hard filters; nothing to retrieve.")
            final: list[TraceChunk] = []
            score_type = ""
        else:
            handler = {
                RetrievalStrategy.VECTOR: self._vector,
                RetrievalStrategy.MULTI_QUERY: self._multi_query,
                RetrievalStrategy.MULTI_QUERY_RERANK: self._multi_query,
                RetrievalStrategy.HYBRID_RRF: self._hybrid,
            }[retrieval.strategy]
            final, score_type = await handler(run)
        if allowed is not None:
            filters = {**filters, "allowed_dish_count": len(allowed)}
        return RetrievalTrace(
            query=query, query_variants=run.variants, filters=filters, chunking=chunking, retrieval=retrieval,
            index_fingerprint=index_fingerprint, embedding_model=index.embedding.model, stages=run.stages,
            chunks=final, context=build_context(final), final_score_type=score_type, latency_ms=_ms(t_total),
            latency_breakdown_ms=run.timings, warnings=run.warnings,
        )

    async def reference_contexts(self, index_fingerprint: str, dish_ids: Sequence[str]) -> list[str]:
        """Text of every chunk in this index that covers one of ``dish_ids``: the ground-truth contexts that the
        LLM-free RAGAS metrics compare retrieved chunks against (valid for any chunking strategy)."""
        index = await self._indexing.get_by_fingerprint(index_fingerprint)
        if index is None or not dish_ids:
            return []
        chunks = await self._store.chunks_covering(index.id, dish_ids)
        return list(dict.fromkeys(c.content for c in chunks))

    # ------------------------------------------------------------------ building blocks

    async def _vector_stage(self, run: _Run, text: str, emb: list[float], name: str, limit: int) -> list[ScoredChunk]:
        t0 = time.perf_counter()
        hits = await self._store.vector_search(run.index_id, emb, limit, run.allowed)
        ms = _ms(t0)
        run.add_time("vector_search_ms", ms)
        items = []
        for rank, h in enumerate(hits, start=1):
            run.chunks[h.chunk.id] = h.chunk
            items.append(_trace_chunk(rank, h.chunk, {"cosine_distance": round(h.score, 5), "cosine_similarity": round(1 - h.score, 5)}))
        run.stages.append(
            TraceStage(name=name, description=f"pgvector cosine search (limit {limit})", score_type="cosine_distance",
                       query=text, items=items, latency_ms=ms)
        )
        return hits

    async def _lexical_stage(self, run: _Run, limit: int) -> list[ScoredChunk]:
        t0 = time.perf_counter()
        hits = await self._store.lexical_search(run.index_id, run.query, limit, run.allowed)
        ms = _ms(t0)
        run.add_time("lexical_search_ms", ms)
        items = []
        for rank, h in enumerate(hits, start=1):
            run.chunks[h.chunk.id] = h.chunk
            items.append(_trace_chunk(rank, h.chunk, {"ts_rank_cd": round(h.score, 5)}))
        run.stages.append(
            TraceStage(
                name="lexical", score_type="ts_rank_cd", query=run.query, items=items, latency_ms=ms,
                description=f"PostgreSQL full-text search (to_tsquery OR + ts_rank_cd; not BM25), limit {limit}",
            )
        )
        return hits

    async def _rerank(self, run: _Run, ordered: list[tuple[str, dict[str, float]]]) -> list[tuple[str, dict[str, float]]]:
        reranker = self._rerankers(run.cfg.reranker_model)
        t0 = time.perf_counter()
        scores = await reranker.score(run.query, [run.chunks[cid].content for cid, _ in ordered])
        ms = _ms(t0)
        run.add_time("rerank_ms", ms)
        rescored = [(cid, {**s, "cross_encoder_score": round(sc, 5)}) for (cid, s), sc in zip(ordered, scores, strict=True)]
        rescored.sort(key=lambda x: -x[1]["cross_encoder_score"])
        run.stages.append(
            TraceStage(
                name="rerank", score_type="cross_encoder_score", query=run.query, latency_ms=ms,
                description=f"Cross-encoder '{reranker.model_name}' re-scores {len(ordered)} candidates against the original query",
                items=[_trace_chunk(i, run.chunks[cid], s) for i, (cid, s) in enumerate(rescored, start=1)],
            )
        )
        return rescored

    def _finalize(self, run: _Run, ordered: list[tuple[str, dict[str, float]]], extra: dict[str, dict] | None = None) -> list[TraceChunk]:
        return [
            _trace_chunk(i, run.chunks[cid], s, (extra or {}).get(cid))
            for i, (cid, s) in enumerate(ordered[: run.cfg.top_k], start=1)
        ]

    # ------------------------------------------------------------------ strategies

    async def _vector(self, run: _Run) -> tuple[list[TraceChunk], str]:
        t0 = time.perf_counter()
        emb = await run.embedder.embed_query(run.query)
        run.add_time("embedding_ms", _ms(t0))
        hits = await self._vector_stage(run, run.query, emb, "vector", run.cfg.top_k)
        ordered = [(h.chunk.id, {"cosine_distance": round(h.score, 5), "cosine_similarity": round(1 - h.score, 5)}) for h in hits]
        return self._finalize(run, ordered), "cosine_similarity"

    async def _generate_variants(self, run: _Run) -> list[str]:
        if not self._llm.is_configured:
            raise ServiceNotConfigured(
                "Multi-query retrieval needs an LLM to generate query variants. Set LLM_API_KEY in backend/.env "
                "or pick the 'vector' / 'hybrid' retrieval strategy."
            )
        t0 = time.perf_counter()
        n = run.cfg.num_query_variants
        out = await self._llm.generate_structured(prompts.MULTI_QUERY.system.format(n=n), run.query, QueryVariants)
        run.add_time("query_generation_ms", _ms(t0))
        return clean_variants(run.query, out.queries, n)

    async def _multi_query(self, run: _Run) -> tuple[list[TraceChunk], str]:
        run.variants = await self._generate_variants(run)
        queries = [run.query, *run.variants]  # the original query is always searched too
        t0 = time.perf_counter()
        embs = await run.embedder.embed_documents(queries)
        run.add_time("embedding_ms", _ms(t0))
        per_query: list[list[tuple[str, float]]] = []
        for i, (q, e) in enumerate(zip(queries, embs, strict=True)):
            hits = await self._vector_stage(run, q, e, "vector[original]" if i == 0 else f"vector[variant {i}]", run.cfg.candidate_count)
            per_query.append([(h.chunk.id, 1 - h.score) for h in hits])
        t0 = time.perf_counter()
        merged = merge_best_score(per_query)  # dedupe by chunk id, keep best similarity
        run.add_time("merge_ms", _ms(t0))
        run.stages.append(
            TraceStage(
                name="merge", score_type="best_cosine_similarity", query=run.query, latency_ms=_ms(t0),
                description=f"Deduplicated {sum(len(p) for p in per_query)} hits from {len(queries)} queries into {len(merged)} unique chunks (best similarity; ties by variant hits)",
                items=[_trace_chunk(i, run.chunks[c], {"best_cosine_similarity": round(s, 5), "variant_hits": float(h)}) for i, (c, s, h) in enumerate(merged, start=1)],
            )
        )
        ordered = [(c, {"best_cosine_similarity": round(s, 5), "variant_hits": float(h)}) for c, s, h in merged]
        score_type = "best_cosine_similarity"
        if run.cfg.rerank_enabled:
            ordered = await self._rerank(run, ordered[: run.cfg.candidate_count])
            score_type = "cross_encoder_score"
        return self._finalize(run, ordered), score_type

    async def _hybrid(self, run: _Run) -> tuple[list[TraceChunk], str]:
        t0 = time.perf_counter()
        emb = await run.embedder.embed_query(run.query)
        run.add_time("embedding_ms", _ms(t0))
        vec = await self._vector_stage(run, run.query, emb, "vector", run.cfg.candidate_count)
        lex = await self._lexical_stage(run, run.cfg.candidate_count)
        t0 = time.perf_counter()
        v_rank = {h.chunk.id: i for i, h in enumerate(vec, start=1)}
        l_rank = {h.chunk.id: i for i, h in enumerate(lex, start=1)}
        fused = rrf_fuse([[h.chunk.id for h in vec], [h.chunk.id for h in lex]], k=run.cfg.rrf_k)
        run.add_time("fusion_ms", _ms(t0))
        ranks = {c: {"vector_rank": v_rank.get(c), "lexical_rank": l_rank.get(c)} for c, _ in fused}
        run.stages.append(
            TraceStage(
                name="rrf", score_type="rrf_score", query=run.query, latency_ms=_ms(t0),
                description=f"Reciprocal Rank Fusion (k={run.cfg.rrf_k}) over vector and lexical ranks; raw scores are ignored",
                items=[_trace_chunk(i, run.chunks[c], {"rrf_score": round(s, 6)}, ranks[c]) for i, (c, s) in enumerate(fused, start=1)],
            )
        )
        ordered = [(c, {"rrf_score": round(s, 6)}) for c, s in fused]
        score_type = "rrf_score"
        if run.cfg.rerank_enabled:
            ordered = await self._rerank(run, ordered[: run.cfg.candidate_count])
            score_type = "cross_encoder_score"
        return self._finalize(run, ordered, ranks), score_type
