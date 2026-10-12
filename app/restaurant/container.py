"""Composition root: wires infrastructure singletons and per-transaction services (manual DI)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import cached_property

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import Settings, get_settings
from app.db import create_engine, create_session_factory, transaction
from app.rate_limit import SlidingWindowLimiter
from app.restaurant.application.chat.parser import TurnParser
from app.restaurant.application.chat.service import ChatService
from app.restaurant.application.evaluation.live import LiveEvaluator
from app.restaurant.application.evaluation.runner import EvaluationRunner
from app.restaurant.application.evaluation.service import EvaluationService
from app.restaurant.application.ordering.services import CartService, OrderService
from app.restaurant.application.rag.config import ConfigService
from app.restaurant.application.rag.indexing import IndexingService
from app.restaurant.application.rag.retrieval import RetrievalService
from app.restaurant.application.service import MenuService
from app.restaurant.domain.rag.value_objects import EmbeddingSpec, ModelRef, ModelSelection
from app.restaurant.infrastructure.evaluation.ragas_evaluator import RagasFrameworkEvaluator
from app.restaurant.infrastructure.llm import catalog
from app.restaurant.infrastructure.llm.chat_models import LangChainLLM, build_langchain_chat_model
from app.restaurant.infrastructure.llm.quota import QuotaTracker
from app.restaurant.infrastructure.rag.chunking.strategies import ChunkerFactory
from app.restaurant.infrastructure.rag.embeddings import LangChainEmbedder
from app.restaurant.infrastructure.rag.reranker import RerankerRegistry
from app.restaurant.infrastructure.repositories.chat import SqlChatRepository
from app.restaurant.infrastructure.repositories.evaluation import SqlEvaluationRepository
from app.restaurant.infrastructure.repositories.ordering import SqlCartRepository, SqlOrderRepository
from app.restaurant.infrastructure.repositories.rag import (
    PgVectorStore,
    SqlConfigRepository,
    SqlIndexRepository,
)
from app.restaurant.infrastructure.repositories.restaurant import SqlMenuRepository
from app.restaurant.infrastructure.seed.evaluation_data import EVAL_DATASET_NAME

log = logging.getLogger(__name__)


class Infra:
    """Process-wide singletons (engine, models). Created once at startup."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.engine: AsyncEngine = create_engine(self.settings)
        self.session_factory: async_sessionmaker[AsyncSession] = create_session_factory(self.engine)
        self.quota = QuotaTracker(self.settings.quota_cooldown_seconds)
        self.embedder = LangChainEmbedder(self.settings, self.quota)
        self._embedders: dict[EmbeddingSpec, LangChainEmbedder] = {}
        self.rerankers = RerankerRegistry()
        self.llm = LangChainLLM(self.settings, quota=self.quota)
        self.chat_limiter = SlidingWindowLimiter(
            self.settings.chat_rate_limit_max, self.settings.chat_rate_limit_window_seconds
        )
        self.judge_llm = LangChainLLM(
            self.settings, model=self.settings.effective_eval_model, provider=self.settings.effective_eval_provider,
            quota=self.quota,
        )
        self.ragas = RagasFrameworkEvaluator(
            # a fresh chat model per call: RAGAS uses its own event loop, so a cached async client would be bound to a dead one
            llm_factory=lambda: build_langchain_chat_model(self.settings, self.judge_llm.model_name, self.judge_llm.provider),
            embeddings_factory=lambda: self.embedder.langchain_embeddings,
        )
        self.eval_limiter = SlidingWindowLimiter(
            self.settings.eval_rate_limit_max, self.settings.eval_rate_limit_window_seconds
        )
        self.chunkers = ChunkerFactory(lambda: self.embedder.langchain_embeddings)

    def embedder_for(self, spec: EmbeddingSpec) -> LangChainEmbedder:
        """The embedder for ``spec``: the live one when it matches, otherwise a cached one of its own."""
        if spec == self.embedder.spec:
            return self.embedder
        if spec not in self._embedders:
            self._embedders[spec] = LangChainEmbedder(self.settings, self.quota, spec=spec)
        return self._embedders[spec]

    @property
    def selection(self) -> ModelSelection:
        """The language models in use right now (answers, evaluation judge)."""
        return ModelSelection(
            llm=ModelRef(provider=self.llm.provider, model=self.llm.model_name),
            judge=ModelRef(provider=self.judge_llm.provider, model=self.judge_llm.model_name),
        )

    @property
    def default_embedding(self) -> ModelRef:
        """The embedding model used when a request does not name one."""
        e = self.embedder.spec
        return ModelRef(provider=e.provider, model=e.model)

    def resolve_embedding(self, ref: ModelRef | None) -> EmbeddingSpec:
        """The embedding model a request asked for (default when it did not). Raises ``ValidationFailed`` if unusable."""
        if ref is None or ref == self.default_embedding:
            return self.embedder.spec
        catalog.check_slot(self.settings, "embedding", ref)
        return EmbeddingSpec(
            provider=ref.provider, model=ref.model, version=self.settings.embedding_version,
            dimensions=catalog.embedding_dimensions(self.settings, ref.provider, ref.model),
        )

    def apply_models(self, sel: ModelSelection) -> None:
        """Switch the running app to these language models (raises ``ValidationFailed`` for unknown or unconfigured ones)."""
        catalog.validate(self.settings, sel)
        self.llm.select(sel.llm.provider, sel.llm.model)
        self.judge_llm.select(sel.judge.provider, sel.judge.model)

    def restore_models(self, saved: dict) -> None:
        """Re-apply the choice saved with the live configuration; anything no longer valid keeps the .env default."""
        sel = self.selection
        for slot in ("llm", "judge"):
            try:  # one stale entry (e.g. a removed provider) must not discard the other
                ref = ModelRef.model_validate((saved or {}).get(slot))
                catalog.check_slot(self.settings, slot, ref)
                sel = sel.model_copy(update={slot: ref})
            except Exception as exc:  # noqa: BLE001 - stale or partial data must never stop startup
                if saved and saved.get(slot):
                    log.warning("saved %s model ignored: %s", slot, exc)
        self.apply_models(sel)

    def uow(self):
        """Unit of work yielding a fresh ``Services`` bound to one transaction."""

        @asynccontextmanager
        async def _ctx() -> AsyncIterator[Services]:
            async with transaction(self.session_factory) as session:
                yield Services(self, session)

        return _ctx()

    async def dispose(self) -> None:
        await self.engine.dispose()


class Services:
    """Repositories + use cases bound to a single DB session/transaction."""

    def __init__(self, infra: Infra, session: AsyncSession):
        self.infra, self.session = infra, session

    # repositories
    @cached_property
    def menu_repo(self): return SqlMenuRepository(self.session)
    @cached_property
    def chat_repo(self): return SqlChatRepository(self.session)
    @cached_property
    def cart_repo(self): return SqlCartRepository(self.session)
    @cached_property
    def order_repo(self): return SqlOrderRepository(self.session)
    @cached_property
    def config_repo(self): return SqlConfigRepository(self.session)
    @cached_property
    def index_repo(self): return SqlIndexRepository(self.session)
    @cached_property
    def vector_store(self): return PgVectorStore(self.session)
    @cached_property
    def eval_repo(self): return SqlEvaluationRepository(self.session)

    # use cases
    @cached_property
    def menu(self): return MenuService(self.menu_repo)

    @cached_property
    def carts(self): return CartService(self.cart_repo, self.menu_repo, self.chat_repo)

    @cached_property
    def orders(self): return OrderService(self.cart_repo, self.order_repo, self.menu_repo, self.chat_repo)

    @cached_property
    def indexing(self):
        return IndexingService(self.index_repo, self.vector_store, self.menu_repo, self.infra.embedder, self.infra.chunkers.create, self.infra.embedder_for)

    @cached_property
    def config(self):
        return ConfigService(
            self.config_repo, self.indexing, self.infra.embedder, self.infra.llm.model_name,
            prebuilt=catalog.embedding_specs(self.infra.settings),
        )

    @cached_property
    def retrieval(self):
        return RetrievalService(
            self.indexing, self.vector_store, self.infra.embedder, self.infra.rerankers.get, self.infra.llm, self.menu_repo,
            self.infra.embedder_for,
        )

    @cached_property
    def chat(self):
        return ChatService(
            self.chat_repo, self.menu_repo, self.carts, self.orders, self.config, self.retrieval,
            self.infra.llm, TurnParser(self.infra.llm),
            live_evals=self.eval_repo if self.infra.settings.live_evaluation_enabled else None,
        )

    def live_evaluator(self) -> LiveEvaluator:
        return LiveEvaluator(self.infra.uow, self.infra.judge_llm, self.infra.ragas)

    @cached_property
    def evaluation(self):
        return EvaluationService(self.eval_repo, self.config, self.infra.llm, self.infra.ragas, EVAL_DATASET_NAME)

    def evaluation_runner(self) -> EvaluationRunner:
        return EvaluationRunner(self.infra.uow, self.infra.llm, self.infra.ragas)
