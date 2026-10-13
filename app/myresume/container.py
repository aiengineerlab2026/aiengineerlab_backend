"""Composition root of the My Resume lab.

Stateless on purpose: the jobs are a fixed corpus held in memory and every request carries its own settings (retrieval
method, models), so visitors never affect each other and no database is involved. The embedding model comes from the
same Gemini / Cloudflare Workers AI catalogue as the restaurant lab and is selectable per request. Retrieval and its
evaluation (ranx) use no language model.
"""

from __future__ import annotations

from app.config import Settings
from app.myresume.application.evaluation import RankingEvaluationService
from app.myresume.application.search import JobSearchService
from app.myresume.infrastructure.bm25 import Bm25Registry
from app.myresume.infrastructure.data.jobs import JOBS
from app.myresume.infrastructure.vector_index import VectorIndexRegistry
from app.rate_limit import SlidingWindowLimiter
from app.restaurant.domain.rag.value_objects import EmbeddingSpec, ModelRef
from app.restaurant.infrastructure.llm import catalog
from app.restaurant.infrastructure.llm.quota import QuotaTracker
from app.restaurant.infrastructure.rag.embeddings import LangChainEmbedder


class MyResumeInfra:
    def __init__(self, settings: Settings, quota: QuotaTracker):
        self.settings, self.quota = settings, quota
        self._embedders: dict[EmbeddingSpec, LangChainEmbedder] = {}
        self.upload_limiter = SlidingWindowLimiter(
            settings.myresume_upload_rate_limit_max, settings.myresume_upload_rate_limit_window_seconds
        )
        self.vector_limiter = SlidingWindowLimiter(
            settings.myresume_vector_rate_limit_max, settings.myresume_vector_rate_limit_window_seconds
        )
        self.bm25 = Bm25Registry(JOBS)
        self.vectors = VectorIndexRegistry(JOBS, self.embedder_for)
        self.search = JobSearchService(JOBS, self.bm25, self.vectors, self.embedder_for)
        self.evaluation = RankingEvaluationService(JOBS)

    # ------------------------------------------------------------------ models
    @property
    def default_embedding(self) -> ModelRef:
        return ModelRef(provider=self.settings.embedding_provider, model=self.settings.embedding_model)

    def resolve_embedding(self, ref: ModelRef | None) -> EmbeddingSpec:
        """The embedding model a request asked for (the default when it did not); raises ``ValidationFailed`` if unusable."""
        ref = ref or self.default_embedding
        if ref != self.default_embedding:
            catalog.check_slot(self.settings, "embedding", ref)
        s = self.settings
        return EmbeddingSpec(
            provider=ref.provider, model=ref.model, version=s.embedding_version,
            dimensions=catalog.embedding_dimensions(s, ref.provider, ref.model),
        )

    def embedder_for(self, spec: EmbeddingSpec) -> LangChainEmbedder:
        if spec not in self._embedders:
            self._embedders[spec] = LangChainEmbedder(self.settings, self.quota, spec=spec)
        return self._embedders[spec]
