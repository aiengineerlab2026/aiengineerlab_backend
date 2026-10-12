from __future__ import annotations

import pytest
from conftest import FakeEmbedder, World, _SemanticEmbeddings  # type: ignore[import-not-found]

from app.domain.errors import ValidationFailed
from app.domain.rag.entities import IndexStatus
from app.domain.rag.value_objects import (
    ChunkingConfig,
    ChunkingStrategy,
    LengthUnit,
    RetrievalConfig,
    RetrievalStrategy,
)
from app.domain.restaurant.canonical import build_corpus, normalize_ws, render_dish
from app.infrastructure.rag.chunking.strategies import ChunkerFactory


@pytest.fixture
def corpus(dishes):
    return build_corpus(dishes)


@pytest.fixture
def factory():
    return ChunkerFactory(lambda: _SemanticEmbeddings(FakeEmbedder()))


def test_corpus_is_deterministic_and_spans_cover_every_dish(dishes, corpus):
    assert build_corpus(list(reversed(dishes))).content_hash == corpus.content_hash  # order independent
    assert len(corpus.spans) == len(dishes)
    for span, dish in zip(corpus.spans, corpus.dishes, strict=True):
        assert corpus.normalized_text[span.start:span.end] == normalize_ws(render_dish(dish))


async def test_entity_chunking_is_one_chunk_per_dish(factory, corpus):
    drafts = await factory.create(ChunkingStrategy.ENTITY).chunk(corpus, ChunkingConfig(strategy=ChunkingStrategy.ENTITY))
    assert len(drafts) == len(corpus.dishes)
    assert all(len(d.dish_ids) == 1 for d in drafts)
    assert [d.dish_ids[0] for d in drafts] == [d.id for d in corpus.dishes]
    assert all(d.metadata["multi_dish"] is False for d in drafts)


@pytest.mark.parametrize(
    "strategy", [ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE, ChunkingStrategy.SEMANTIC],
)
async def test_text_chunkers_keep_provenance(factory, corpus, strategy):
    cfg = ChunkingConfig(strategy=strategy, chunk_size=400, chunk_overlap=40)
    drafts = await factory.create(strategy).chunk(corpus, cfg)
    assert drafts
    known = {d.id for d in corpus.dishes}
    covered: set[str] = set()
    for d in drafts:
        assert d.dish_ids and set(d.dish_ids) <= known
        assert d.metadata["dish_names"] and d.metadata["strategy"] == strategy.value
        assert d.metadata["multi_dish"] == (len(d.dish_ids) > 1)
        covered |= set(d.dish_ids)
    assert covered == known, "every dish must appear in at least one chunk"


async def test_chunk_size_is_respected(factory, corpus):
    cfg = ChunkingConfig(strategy=ChunkingStrategy.RECURSIVE, chunk_size=300, chunk_overlap=30)
    drafts = await factory.create(ChunkingStrategy.RECURSIVE).chunk(corpus, cfg)
    assert max(len(d.content) for d in drafts) <= 300
    fixed = await factory.create(ChunkingStrategy.FIXED).chunk(corpus, cfg.model_copy(update={"strategy": ChunkingStrategy.FIXED}))
    assert max(len(d.content) for d in fixed) <= 300
    # smaller chunks -> more of them
    big = await factory.create(ChunkingStrategy.RECURSIVE).chunk(corpus, cfg.model_copy(update={"chunk_size": 900}))
    assert len(drafts) > len(big)


async def test_token_length_unit(factory, corpus):
    pytest.importorskip("tiktoken")
    cfg = ChunkingConfig(strategy=ChunkingStrategy.RECURSIVE, chunk_size=120, chunk_overlap=10, length_unit=LengthUnit.TOKENS)
    try:
        drafts = await factory.create(ChunkingStrategy.RECURSIVE).chunk(corpus, cfg)
    except ValidationFailed as exc:  # tiktoken needs to download its vocabulary on first use
        pytest.skip(f"tiktoken encoding unavailable offline: {exc}")
    assert drafts


# ------------------------------------------------------------------------------ index lifecycle


async def test_identical_chunking_reuses_the_index_without_reembedding(world: World):
    cfg = ChunkingConfig(strategy=ChunkingStrategy.ENTITY)
    first = await world.build_index(cfg)
    embedded = world.embedder.document_calls
    assert first.status == IndexStatus.READY and first.chunk_count == len(world.dishes)

    again, reused = await world.indexing.request_index(cfg)
    assert reused and again.id == first.id
    assert world.embedder.document_calls == embedded
    assert world.store.replace_calls == 1


async def test_different_chunking_gets_its_own_index(world: World):
    a = await world.build_index(ChunkingConfig(strategy=ChunkingStrategy.ENTITY))
    b = await world.build_index(ChunkingConfig(strategy=ChunkingStrategy.RECURSIVE, chunk_size=300, chunk_overlap=30))
    assert a.fingerprint != b.fingerprint and a.id != b.id
    assert b.chunk_count > 0


async def test_changing_only_retrieval_never_changes_the_fingerprint(world: World):
    chunking = ChunkingConfig(strategy=ChunkingStrategy.ENTITY)
    cfg, index = await world.config.update(chunking, RetrievalConfig(strategy=RetrievalStrategy.VECTOR))
    original_fp = cfg.index_fingerprint
    await world.build_index(chunking)
    calls_before = world.embedder.document_calls

    updated, index2 = await world.config.update(
        chunking, RetrievalConfig(strategy=RetrievalStrategy.HYBRID_RRF, top_k=8, rrf_k=30)
    )
    assert updated.index_fingerprint == original_fp and index2.id == index.id
    assert index2.status == IndexStatus.READY
    assert world.embedder.document_calls == calls_before

    changed, index3 = await world.config.update(
        ChunkingConfig(strategy=ChunkingStrategy.RECURSIVE, chunk_size=300, chunk_overlap=30), updated.retrieval
    )
    assert changed.index_fingerprint != original_fp
    assert index3.status == IndexStatus.PENDING  # must be built before it can be used


async def test_there_is_only_one_live_configuration_and_it_is_replaced_in_place(world: World):
    first = await world.config.get()  # created from the defaults on first use
    assert first.chunking.strategy == ChunkingStrategy.ENTITY and first.retrieval.strategy == RetrievalStrategy.VECTOR
    second, _ = await world.config.update(first.chunking, RetrievalConfig(strategy=RetrievalStrategy.HYBRID_RRF, top_k=7))
    assert second.id == first.id and (await world.config.get()).retrieval.top_k == 7


async def test_chat_keeps_answering_from_the_previous_index_while_a_new_one_builds(world: World):
    entity = ChunkingConfig(strategy=ChunkingStrategy.ENTITY)
    await world.config.update(entity, RetrievalConfig(top_k=5))
    ready = await world.build_index(entity)
    assert (await world.config.serving()).warning is None

    recursive = ChunkingConfig(strategy=ChunkingStrategy.RECURSIVE, chunk_size=300, chunk_overlap=30)
    await world.config.update(recursive, RetrievalConfig(top_k=3))  # new index is only PENDING
    serving = await world.config.serving()
    assert serving.index_fingerprint == ready.fingerprint and serving.chunking == entity  # old index, honestly labelled
    assert serving.retrieval.top_k == 3  # retrieval settings are live immediately
    assert serving.warning and "still pending" in serving.warning

    await world.build_index(recursive)
    now = await world.config.serving()
    assert now.chunking == recursive and now.warning is None


async def test_failed_index_is_requeued(world: World):
    cfg = ChunkingConfig(strategy=ChunkingStrategy.ENTITY)
    index, _ = await world.indexing.request_index(cfg)
    await world.indexing.mark_failed(index.id, "boom")
    again, reused = await world.indexing.request_index(cfg)
    assert not reused and again.status == IndexStatus.PENDING and again.id == index.id


async def test_every_embedding_model_gets_its_own_index_built_with_its_own_embedder(world: World):
    from app.application.rag.config import DEFAULT_CHUNKING, DEFAULT_RETRIEVAL, ConfigService
    from app.application.rag.indexing import IndexingService
    from app.domain.rag.value_objects import EmbeddingSpec
    from tests.conftest import FakeEmbedder

    other_spec = EmbeddingSpec(provider="fake", model="wide-1024", dimensions=1024)

    class Other(FakeEmbedder):
        @property
        def spec(self):
            return other_spec

    other = Other()
    indexing = IndexingService(
        world.indexes, world.store, world.menu, world.embedder, world.chunkers.create,
        embedder_for=lambda s: other if s == other_spec else world.embedder,
    )
    config = ConfigService(world.config_repo, indexing, world.embedder, "llm", prebuilt=[world.embedder.spec, other_spec])
    cfg, live_index = await config.update(DEFAULT_CHUNKING, DEFAULT_RETRIEVAL)

    pre, reused = await indexing.request_index(DEFAULT_CHUNKING, other_spec)  # already queued by update()
    assert reused and pre.fingerprint != live_index.fingerprint and pre.embedding == other_spec

    async def _noop(_p):
        return None

    await indexing.build(pre, _noop)
    assert other.document_calls > 0 and world.embedder.document_calls == 0  # built with *its* model, not the live one
    assert (await indexing.get_by_fingerprint(pre.fingerprint)).status.value == "ready"
    assert cfg.index_fingerprint == live_index.fingerprint


async def test_each_request_searches_the_index_of_the_embedding_model_it_names(world: World):
    """Per request, not global: two embedding models are served side by side, each from its own pre-built index."""
    from app.application.rag.config import DEFAULT_CHUNKING, DEFAULT_RETRIEVAL, ConfigService
    from app.application.rag.indexing import IndexingService
    from app.application.rag.retrieval import RetrievalService
    from app.domain.rag.value_objects import EmbeddingSpec
    from tests.conftest import FakeEmbedder

    other_spec = EmbeddingSpec(provider="fake", model="other-model", dimensions=256)

    class Other(FakeEmbedder):
        @property
        def spec(self):
            return other_spec

    other = Other()
    pick = lambda s: other if s == other_spec else world.embedder  # noqa: E731
    indexing = IndexingService(world.indexes, world.store, world.menu, world.embedder, world.chunkers.create, embedder_for=pick)
    config = ConfigService(world.config_repo, indexing, world.embedder, "llm", prebuilt=[world.embedder.spec, other_spec])
    retrieval = RetrievalService(indexing, world.store, world.embedder, lambda _m: world.reranker, world.llm, world.menu, pick)
    await config.update(DEFAULT_CHUNKING, DEFAULT_RETRIEVAL)

    async def _noop(_p):
        return None

    for spec in (world.embedder.spec, other_spec):
        index, _ = await indexing.request_index(DEFAULT_CHUNKING, spec)
        await indexing.build(index, _noop)
    world.embedder.document_calls = other.document_calls = 0

    served = {}
    for spec in (world.embedder.spec, other_spec):
        live = await config.serving(spec)
        trace = await retrieval.retrieve(
            "spicy chicken burger", chunking=live.chunking, retrieval=live.retrieval, index_fingerprint=live.index_fingerprint
        )
        served[spec.model] = (live.index_fingerprint, trace.embedding_model, live.embedding)
    assert served["bow-256"][0] != served["other-model"][0]  # different pre-built indexes
    assert served["bow-256"][1] == "bow-256" and served["other-model"][1] == "other-model"
    assert served["other-model"][2] == other_spec
    assert world.embedder.document_calls == 0 and other.document_calls == 0  # nothing was re-embedded
