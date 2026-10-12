from __future__ import annotations

import pytest

from app.domain.evaluation.metrics import (
    mean_ignoring_none,
    precision_at_k,
    recall_at_k,
    retrieval_metrics,
)
from app.domain.rag.ranking import merge_best_score, rank_dishes, rrf_fuse
from app.domain.rag.value_objects import (
    ChunkingConfig,
    ChunkingStrategy,
    EmbeddingSpec,
    RetrievalConfig,
    RetrievalStrategy,
    compute_fingerprint,
)

# ------------------------------------------------------------------------------ RRF


def test_rrf_matches_hand_computation():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert fused["d"] == pytest.approx(1 / 63)
    assert [d for d, _ in rrf_fuse([["a", "b", "c"], ["b", "c", "d"]])] == ["b", "c", "a", "d"]


def test_rrf_ignores_duplicates_and_breaks_ties_by_first_appearance():
    assert rrf_fuse([["a", "a", "a"]]) == [("a", pytest.approx(1 / 61))]
    assert [d for d, _ in rrf_fuse([["x"], ["y"]])] == ["x", "y"]
    assert rrf_fuse([]) == []


def test_rrf_rejects_bad_k():
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k=0)


# ------------------------------------------------------------------------------ multi-query merge


def test_merge_keeps_best_similarity_and_counts_hits():
    merged = merge_best_score([[("a", 0.9), ("b", 0.5)], [("b", 0.8), ("c", 0.4)], [("b", 0.1), ("b", 0.99)]])
    by_id = {cid: (sim, hits) for cid, sim, hits in merged}
    assert by_id["a"] == (0.9, 1)
    assert by_id["b"] == (0.8, 3)  # duplicate inside one list is a single hit and the first score wins
    assert by_id["c"] == (0.4, 1)
    assert [cid for cid, _, _ in merged] == ["a", "b", "c"]  # no duplicates, best similarity first


def test_merge_ties_prefer_more_variant_hits():
    merged = merge_best_score([[("a", 0.5), ("b", 0.5)], [("b", 0.5)]])
    assert [c for c, _, _ in merged] == ["b", "a"]


# ------------------------------------------------------------------------------ dish attribution


def test_rank_dishes_credits_every_dish_in_a_chunk_once():
    chunks = [["d1", "d2"], ["d2", "d3"], ["d1"], ["d4"]]
    assert rank_dishes(chunks) == ["d1", "d2", "d3", "d4"]
    assert rank_dishes(chunks, limit=2) == ["d1", "d2"]
    assert rank_dishes([]) == []


# ------------------------------------------------------------------------------ Precision / Recall


def test_precision_and_recall_known_values():
    retrieved = ["a", "b", "c", "d", "e", "f"]
    relevant = ["a", "c", "z"]
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)
    assert precision_at_k(retrieved, relevant, 1) == 1.0
    assert recall_at_k(retrieved, relevant, 1) == pytest.approx(1 / 3)


def test_precision_uses_the_actual_number_retrieved_when_fewer_than_k():
    assert precision_at_k(["a", "x"], ["a"], 10) == 0.5


def test_metrics_dedupe_retrieved_dishes():
    assert precision_at_k(["a", "a", "a", "b"], ["a"], 5) == 0.5
    assert recall_at_k(["a", "a", "a"], ["a", "b"], 5) == 0.5


def test_metrics_for_questions_without_relevant_dishes_are_undefined_not_zero():
    assert precision_at_k(["a"], [], 5) is None
    assert recall_at_k(["a"], [], 5) is None
    assert precision_at_k([], ["a"], 5) == 0.0
    assert recall_at_k([], ["a"], 5) == 0.0


def test_retrieval_metrics_bundle_and_mean():
    m = retrieval_metrics(["a", "b"], ["a"])
    assert set(m) == {"precision_at_5", "recall_at_5", "precision_at_10", "recall_at_10"}
    assert mean_ignoring_none([1.0, None, 0.0]) == 0.5
    assert mean_ignoring_none([None]) is None


# ------------------------------------------------------------------------------ fingerprint


def _fp(chunking: ChunkingConfig, **kw) -> str:
    return compute_fingerprint(
        dataset_version=kw.get("dataset_version", "menu-v1"), dataset_hash=kw.get("dataset_hash", "h"),
        chunking=chunking, embedding=kw.get("embedding", EmbeddingSpec(provider="p", model="m", dimensions=4)),
        canonical_format_version="canon-v1",
    )


def test_fingerprint_is_stable_and_sensitive_to_indexing_inputs():
    base = ChunkingConfig(strategy=ChunkingStrategy.RECURSIVE, chunk_size=400, chunk_overlap=40)
    assert _fp(base) == _fp(base.model_copy())
    assert _fp(base) != _fp(base.model_copy(update={"chunk_size": 500}))
    assert _fp(base) != _fp(base.model_copy(update={"chunk_overlap": 41}))
    assert _fp(base) != _fp(base, dataset_hash="other")
    assert _fp(base) != _fp(base, embedding=EmbeddingSpec(provider="p", model="m2", dimensions=4))


def test_fingerprint_ignores_parameters_the_strategy_does_not_use():
    # entity chunking has no size/overlap -> changing them must not invalidate the index
    a = ChunkingConfig(strategy=ChunkingStrategy.ENTITY, chunk_size=300)
    b = ChunkingConfig(strategy=ChunkingStrategy.ENTITY, chunk_size=900, chunk_overlap=100)
    assert _fp(a) == _fp(b)
    # ... and semantic-only parameters do not affect fixed chunking
    f1 = ChunkingConfig(strategy=ChunkingStrategy.FIXED, breakpoint_threshold_amount=50)
    f2 = ChunkingConfig(strategy=ChunkingStrategy.FIXED, breakpoint_threshold_amount=95)
    assert _fp(f1) == _fp(f2)


def test_fingerprint_does_not_depend_on_retrieval_settings():
    # RetrievalConfig is not an input of compute_fingerprint at all: this guards the signature.
    import inspect

    assert "retrieval" not in inspect.signature(compute_fingerprint).parameters


# ------------------------------------------------------------------------------ config validation


def test_overlap_must_be_smaller_than_size_for_size_based_strategies():
    with pytest.raises(ValueError):
        ChunkingConfig(strategy=ChunkingStrategy.FIXED, chunk_size=100, chunk_overlap=100)
    ChunkingConfig(strategy=ChunkingStrategy.ENTITY, chunk_size=100, chunk_overlap=100)  # irrelevant for entity


def test_retrieval_config_normalisation():
    assert RetrievalConfig(strategy=RetrievalStrategy.MULTI_QUERY_RERANK).rerank_enabled is True
    assert RetrievalConfig(strategy=RetrievalStrategy.VECTOR, rerank_enabled=True).rerank_enabled is False
    assert RetrievalConfig(strategy=RetrievalStrategy.HYBRID_RRF, rerank_enabled=True).rerank_enabled is True
    assert RetrievalConfig(top_k=10, candidate_count=3).candidate_count == 10
