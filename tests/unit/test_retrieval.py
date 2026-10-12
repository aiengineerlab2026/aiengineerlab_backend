from __future__ import annotations

import pytest
from conftest import ScriptedLLM, World  # type: ignore[import-not-found]

from app.errors import ConflictError, ServiceNotConfigured
from app.restaurant.application.rag.retrieval import QueryVariants, build_context, clean_variants
from app.restaurant.domain.filters import DishFilter
from app.restaurant.domain.rag.value_objects import (
    ChunkingConfig,
    ChunkingStrategy,
    RetrievalConfig,
    RetrievalStrategy,
)

ENTITY = ChunkingConfig(strategy=ChunkingStrategy.ENTITY)


async def _ready(llm=None) -> tuple[World, str]:
    w = World(llm)
    index = await w.build_index(ENTITY)
    return w, index.fingerprint


async def _retrieve(w, fp, query, **cfg):
    return await w.retrieval.retrieve(query, chunking=ENTITY, retrieval=RetrievalConfig(**cfg), index_fingerprint=fp)


async def test_vector_retrieval_trace_is_complete_and_labelled():
    w, fp = await _ready()
    trace = await _retrieve(w, fp, "spicy jalapeno burger with chipotle mayo", top_k=3)
    assert trace.chunks[0].dish_names == ["Spicy Jalapeno Burger"]
    assert [c.rank for c in trace.chunks] == [1, 2, 3]
    assert trace.final_score_type == "cosine_similarity"
    for c in trace.chunks:
        assert c.scores["cosine_similarity"] == pytest.approx(1 - c.scores["cosine_distance"], abs=1e-4)
    sims = [c.scores["cosine_similarity"] for c in trace.chunks]
    assert sims == sorted(sims, reverse=True)
    assert [s.name for s in trace.stages] == ["vector"]
    assert trace.latency_ms >= 0 and "embedding_ms" in trace.latency_breakdown_ms
    assert trace.index_fingerprint == fp and trace.embedding_model == "bow-256"


async def test_context_is_exactly_what_the_llm_receives():
    w, fp = await _ready()
    trace = await _retrieve(w, fp, "veggie falafel burger", top_k=2)
    assert trace.context == build_context(trace.chunks)
    assert trace.chunks[0].content in trace.context and "[1]" in trace.context and "[2]" in trace.context


async def test_hybrid_uses_rrf_over_vector_and_lexical_ranks():
    w, fp = await _ready()
    trace = await _retrieve(w, fp, "spicy chicken burger", strategy="hybrid_rrf", top_k=5, rrf_k=60)
    assert [s.name for s in trace.stages] == ["vector", "lexical", "rrf"]
    assert trace.final_score_type == "rrf_score"
    vec = {c.chunk_id: i for i, c in enumerate(trace.stages[0].items, 1)}
    lex = {c.chunk_id: i for i, c in enumerate(trace.stages[1].items, 1)}
    for c in trace.chunks:
        expected = sum(1 / (60 + r[c.chunk_id]) for r in (vec, lex) if c.chunk_id in r)
        assert c.scores["rrf_score"] == pytest.approx(expected, abs=1e-5)
        assert c.metadata["vector_rank"] == vec.get(c.chunk_id)
        assert c.metadata["lexical_rank"] == lex.get(c.chunk_id)
    scores = [c.scores["rrf_score"] for c in trace.chunks]
    assert scores == sorted(scores, reverse=True)
    assert trace.stages[1].score_type == "ts_rank_cd"


async def test_hybrid_with_rerank_adds_a_cross_encoder_stage():
    w, fp = await _ready()
    trace = await _retrieve(w, fp, "vegan falafel", strategy="hybrid_rrf", rerank_enabled=True, top_k=3, candidate_count=10)
    assert trace.stages[-1].name == "rerank" and trace.final_score_type == "cross_encoder_score"
    assert w.reranker.calls == 1
    assert all("cross_encoder_score" in c.scores and "rrf_score" in c.scores for c in trace.chunks)


async def test_multi_query_requires_a_configured_llm():
    w, fp = await _ready(ScriptedLLM(configured=False))
    with pytest.raises(ServiceNotConfigured):
        await _retrieve(w, fp, "something", strategy="multi_query")


async def test_multi_query_dedupes_chunks_across_variants():
    llm = ScriptedLLM(structured={"QueryVariants": QueryVariants(queries=["falafel patty tahini", "plant based burger", "falafel"])})
    w, fp = await _ready(llm)
    trace = await _retrieve(w, fp, "vegetarian burger", strategy="multi_query", top_k=5, candidate_count=8, num_query_variants=3)
    assert trace.query_variants == ["falafel patty tahini", "plant based burger", "falafel"]
    ids = [c.chunk_id for c in trace.chunks]
    assert len(ids) == len(set(ids))
    names = [s.name for s in trace.stages]
    assert names[:4] == ["vector[original]", "vector[variant 1]", "vector[variant 2]", "vector[variant 3]"] and names[-1] == "merge"
    merge = trace.stages[-1]
    assert len({i.chunk_id for i in merge.items}) == len(merge.items)
    assert sum(len(s.items) for s in trace.stages[:4]) > len(merge.items), "overlapping hits must have been merged"
    assert trace.final_score_type == "best_cosine_similarity"
    assert not trace.stages[-1].name == "rerank"


async def test_multi_query_rerank_reorders_by_cross_encoder_score():
    llm = ScriptedLLM(structured={"QueryVariants": QueryVariants(queries=["pizza mozzarella", "basil"])})
    w, fp = await _ready(llm)
    trace = await _retrieve(w, fp, "margherita pizza", strategy="multi_query_rerank", top_k=3, candidate_count=10, num_query_variants=2)
    assert trace.stages[-1].name == "rerank" and w.reranker.calls == 1
    scores = [c.scores["cross_encoder_score"] for c in trace.chunks]
    assert scores == sorted(scores, reverse=True)
    assert trace.final_score_type == "cross_encoder_score"
    assert "Margherita" in trace.chunks[0].dish_names[0]


def test_variant_cleaning_drops_duplicates_blanks_and_the_original():
    assert clean_variants("Spicy Burger", ["spicy burger", " ", '"hot burger"', "Hot Burger", "bbq", "extra"], 2) == ["hot burger", "bbq"]


async def test_hard_filters_are_applied_inside_the_search():
    w, fp = await _ready()
    flt = DishFilter(exclude_allergens=("dairy",))
    trace = await w.retrieval.retrieve(
        "cheese pizza", chunking=ENTITY, retrieval=RetrievalConfig(top_k=10), index_fingerprint=fp, dish_filter=flt
    )
    by_id = {d.id: d for d in w.dishes}
    assert trace.chunks and all("dairy" not in by_id[i].allergens for c in trace.chunks for i in c.dish_ids)
    assert trace.filters["exclude_allergens"] == ["dairy"] and trace.filters["allowed_dish_count"] > 0


async def test_impossible_filters_return_no_chunks_and_a_warning():
    w, fp = await _ready()
    trace = await w.retrieval.retrieve(
        "anything", chunking=ENTITY, retrieval=RetrievalConfig(), index_fingerprint=fp, dish_filter=DishFilter(max_price=1)
    )
    assert trace.chunks == [] and trace.warnings and trace.context == ""


async def test_retrieval_fails_clearly_when_the_index_is_not_ready():
    w = World()
    index, _ = await w.indexing.request_index(ENTITY)  # pending, never built
    with pytest.raises(ConflictError) as e:
        await _retrieve(w, index.fingerprint, "x")
    assert e.value.details["index_status"] == "pending"
    with pytest.raises(ConflictError) as e:
        await _retrieve(w, "does-not-exist", "x")
    assert e.value.details["index_status"] == "missing"


async def test_same_index_serves_every_non_llm_strategy_without_reembedding():
    w, fp = await _ready()
    calls = w.embedder.document_calls
    for strategy in (RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID_RRF):
        await _retrieve(w, fp, "chicken", strategy=strategy.value, top_k=3)
    assert w.embedder.document_calls == calls
