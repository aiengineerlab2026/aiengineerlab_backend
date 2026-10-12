from __future__ import annotations

from fastapi import APIRouter, status

from app.restaurant.application.chat.parser import to_dish_filter
from app.restaurant.domain.rag.ranking import rank_dishes
from app.restaurant.domain.rag.value_objects import (
    DEFAULT_RERANKER_MODEL,
    ChunkingConfig,
    ModelSelection,
    RetrievalConfig,
)
from app.restaurant.infrastructure.llm import catalog
from app.restaurant.presentation.deps import InfraDep, ServicesDep
from app.restaurant.presentation.schemas import (
    ConfigIn,
    ConfigOut,
    IndexOut,
    ModelOptionOut,
    ModelsOut,
    ProviderOut,
    RetrieveIn,
    RetrieveOut,
    StrategiesOut,
    StrategyInfo,
)

router = APIRouter(prefix="/rag", tags=["restaurant: rag"])

CHUNKING = [
    StrategyInfo(
        id="fixed", name="Fixed-size",
        description="LangChain CharacterTextSplitter: length-based windows with overlap. Ignores structure, so chunks can cut through a dish.",
        uses=["chunk_size", "chunk_overlap", "length_unit"],
    ),
    StrategyInfo(
        id="recursive", name="Recursive",
        description="RecursiveCharacterTextSplitter: splits on the first separator that fits (paragraph, line, sentence, word).",
        uses=["chunk_size", "chunk_overlap", "length_unit", "separators"],
    ),
    StrategyInfo(
        id="semantic", name="Semantic",
        description="Embedding-based breakpoints (LangChain SemanticChunker): starts a new chunk where adjacent sentences diverge in meaning.",
        uses=["breakpoint_threshold_type", "breakpoint_threshold_amount"],
    ),
    StrategyInfo(
        id="entity", name="Entity-based",
        description="One chunk per dish (name, description, ingredients, allergens, price, nutrition, category). Preserves dish identity.",
        uses=[],
    ),
]
RETRIEVAL = [
    StrategyInfo(
        id="vector", name="Vector similarity",
        description="Embed the query and take the Top-K nearest chunks with pgvector (cosine distance).",
        uses=["top_k"],
    ),
    StrategyInfo(
        id="multi_query", name="Multi-query",
        description="An LLM rewrites the query into several variants; each is searched, then candidates are deduplicated and ranked by best similarity.",
        uses=["top_k", "num_query_variants", "candidate_count"], requires_llm=True,
    ),
    StrategyInfo(
        id="multi_query_rerank", name="Multi-query + rerank",
        description="Multi-query candidate generation followed by a cross-encoder that re-scores each (query, chunk) pair.",
        uses=["top_k", "num_query_variants", "candidate_count", "rerank_enabled", "reranker_model"], requires_llm=True,
    ),
    StrategyInfo(
        id="hybrid_rrf", name="Hybrid (vector + full-text) + RRF",
        description="Vector search and PostgreSQL full-text search (ts_rank_cd, not BM25) fused with Reciprocal Rank Fusion; optional cross-encoder rerank.",
        uses=["top_k", "candidate_count", "rrf_k", "rerank_enabled", "reranker_model"],
    ),
]


@router.get("/strategies", response_model=StrategiesOut)
async def strategies(infra: InfraDep) -> StrategiesOut:
    return StrategiesOut(
        chunking=CHUNKING,
        retrieval=RETRIEVAL,
        rerankers=[DEFAULT_RERANKER_MODEL, "cross-encoder/ms-marco-MiniLM-L-12-v2"],
        defaults={
            "chunking": ChunkingConfig().model_dump(mode="json"),
            "retrieval": RetrievalConfig().model_dump(mode="json"),
        },
        embedding=infra.embedder.spec,
        llm={
            "provider": infra.llm.provider,
            "model": infra.llm.model_name,
            "configured": infra.llm.is_configured,
            "judge_model": infra.judge_llm.model_name,
        },
    )


@router.get("/config", response_model=ConfigOut, summary="The live RAG configuration (one for the whole app)")
async def get_config(svc: ServicesDep) -> ConfigOut:
    cfg = await svc.config.get()
    return ConfigOut.of(cfg, await svc.config.index_status())


@router.put(
    "/config", response_model=ConfigOut,
    summary="Replace the live configuration. Applies to the next question; a new index is reused or built in the background",
)
async def put_config(body: ConfigIn, svc: ServicesDep) -> ConfigOut:
    cfg, index = await svc.config.update(body.chunking, body.retrieval)
    return ConfigOut.of(cfg, index)


def _options(items: list[catalog.ModelOption], infra) -> list[ModelOptionOut]:
    return [
        ModelOptionOut(
            provider=o.provider, model=o.model, label=o.label, note=o.note, dimensions=o.dimensions,
            quota_exhausted_for_seconds=infra.quota.remaining(o.provider, o.model),
        )
        for o in items
    ]


def _models_out(infra) -> ModelsOut:
    s = infra.settings
    return ModelsOut(
        providers=[ProviderOut(**p) for p in catalog.providers(s)],
        chat=_options(catalog.chat_models(s), infra),
        embedding=_options(catalog.embedding_models(s), infra),
        selection=infra.selection,
        default_embedding=infra.default_embedding,
    )


@router.get("/models", response_model=ModelsOut, summary="Models that can be chosen (chat, evaluation, embeddings) and the current choice")
async def get_models(infra: InfraDep) -> ModelsOut:
    return _models_out(infra)


@router.put(
    "/config/models", response_model=ConfigOut,
    summary="Choose the chat, evaluation and embedding models. Applies to the next question; a new embedding model re-indexes the menu in the background",
)
async def put_models(body: ModelSelection, infra: InfraDep, svc: ServicesDep) -> ConfigOut:
    """The answer and evaluation models. The embedding model is chosen per request, not here."""
    infra.apply_models(body)  # before `svc.config` is first used, so it picks up the new model name
    current = await svc.config.get()
    cfg, index = await svc.config.update(current.chunking, current.retrieval, models=body.model_dump(mode="json"))
    return ConfigOut.of(cfg, index)


@router.post("/config/index", response_model=IndexOut, status_code=status.HTTP_202_ACCEPTED,
             summary="Ensure the live configuration's index exists (reuses a ready index; otherwise queues a build)")
async def build_index(svc: ServicesDep) -> IndexOut:
    return IndexOut.of(await svc.config.ensure_index())


@router.get("/config/index-status", response_model=IndexOut)
async def index_status(svc: ServicesDep) -> IndexOut:
    return IndexOut.of(await svc.config.index_status())


@router.post("/retrieve", response_model=RetrieveOut, summary="Run retrieval with the live configuration")
async def retrieve(body: RetrieveIn, svc: ServicesDep, infra: InfraDep) -> RetrieveOut:
    live = await svc.config.serving(infra.resolve_embedding(body.embedding))
    flt = None
    if body.filters is not None:
        categories = [c.name for c in await svc.menu_repo.list_categories()]
        flt = to_dish_filter(body.filters, categories)
    trace = await svc.retrieval.retrieve(
        body.query, chunking=live.chunking, retrieval=live.retrieval, index_fingerprint=live.index_fingerprint,
        dish_filter=flt, notice=live.warning,
    )
    return RetrieveOut(trace=trace, dish_ids=rank_dishes([c.dish_ids for c in trace.chunks]))
