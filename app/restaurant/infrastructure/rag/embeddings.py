"""Embedding adapters (provider-independent EmbeddingModel port over LangChain Embeddings)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.errors import ExternalServiceError, LLMQuotaExceeded, ServiceNotConfigured
from app.restaurant.domain.rag.ports import EmbeddingModel
from app.restaurant.domain.rag.value_objects import EmbeddingSpec
from app.restaurant.infrastructure.llm import catalog
from app.restaurant.infrastructure.llm.quota import QuotaTracker, is_quota_error

log = logging.getLogger(__name__)


def _l2(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else vec


class GeminiEmbeddings(Embeddings):
    """Gemini embeddings through LangChain, truncated (Matryoshka) to ``dimensions`` and L2-normalised.

    ``gemini-embedding-001`` natively returns 3072-d vectors but supports a smaller ``output_dimensionality``;
    we request the size of the pgvector column so no schema change is needed. Documents use the
    RETRIEVAL_DOCUMENT task type and queries RETRIEVAL_QUERY (asymmetric retrieval), which is LangChain's default.
    """

    def __init__(self, model: str, api_key: str, dimensions: int):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._dims = dimensions
        self._inner = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_l2(v) for v in self._inner.embed_documents(texts, output_dimensionality=self._dims)]

    def embed_query(self, text: str) -> list[float]:
        return _l2(self._inner.embed_query(text, output_dimensionality=self._dims))


class NormalizedEmbeddings(Embeddings):
    """Wraps any LangChain embeddings and L2-normalises the vectors (cosine search expects unit length)."""

    def __init__(self, inner: Embeddings):
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_l2(v) for v in self._inner.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return _l2(self._inner.embed_query(text))


def build_langchain_embeddings(settings: Settings, provider: str | None = None, model: str | None = None) -> Embeddings:
    p = provider or settings.embedding_provider
    model = model or settings.embedding_model
    api_key = settings.embedding_api_key or settings.llm_api_key
    if p == "cloudflare":
        # Workers AI exposes the OpenAI embeddings protocol; LangChain's OpenAIEmbeddings talks to it directly.
        from langchain_openai import OpenAIEmbeddings

        if not settings.provider_configured("cloudflare"):
            raise ServiceNotConfigured("Cloudflare embeddings need CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID in backend/.env.")
        return NormalizedEmbeddings(
            OpenAIEmbeddings(
                model=model,
                api_key=settings.cloudflare_api_token,
                base_url=settings.cloudflare_base_url,
                check_embedding_ctx_length=False,  # send the text itself, not OpenAI token ids
                chunk_size=50,  # texts per request
            )
        )
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not api_key:
            raise ServiceNotConfigured("EMBEDDING_PROVIDER=openai needs EMBEDDING_API_KEY or LLM_API_KEY.")
        return OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=settings.embedding_base_url or settings.llm_base_url or None,
            dimensions=settings.embedding_dimensions if "text-embedding-3" in model else None,
            check_embedding_ctx_length=False,
        )
    if p == "google":
        if not api_key:
            raise ServiceNotConfigured("EMBEDDING_PROVIDER=google needs EMBEDDING_API_KEY or LLM_API_KEY.")
        return GeminiEmbeddings(model, api_key, settings.embedding_dimensions)
    raise ServiceNotConfigured(f"Unsupported EMBEDDING_PROVIDER '{p}'.")


class LangChainEmbedder(EmbeddingModel):
    def __init__(self, settings: Settings, quota: QuotaTracker | None = None, spec: EmbeddingSpec | None = None):
        self._settings = settings
        self._quota = quota
        self._lc: Embeddings | None = None
        self._spec = spec or EmbeddingSpec(
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            dimensions=catalog.embedding_dimensions(settings, settings.embedding_provider, settings.embedding_model),
            version=settings.embedding_version,
        )
        self._checked = False

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    @property
    def langchain_embeddings(self) -> Embeddings:
        if self._lc is None:
            self._lc = build_langchain_embeddings(self._settings, self._spec.provider, self._spec.model)
        return self._lc

    def _check_dims(self, vec: Sequence[float]) -> None:
        if not self._checked:
            if len(vec) != self._spec.dimensions:
                raise ExternalServiceError(
                    f"Embedding model '{self._spec.model}' produced {len(vec)}-d vectors but "
                    f"EMBEDDING_DIMENSIONS={self._spec.dimensions}. The pgvector column is created with "
                    "that dimension; fix the setting (and re-run migrations on a fresh schema)."
                )
            self._checked = True

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            out = await asyncio.to_thread(self.langchain_embeddings.embed_documents, list(texts))
        except ServiceNotConfigured:
            raise
        except Exception as exc:
            if is_quota_error(exc):
                if self._quota:
                    self._quota.mark(self._spec.provider, self._spec.model)
                raise LLMQuotaExceeded(provider=self._spec.provider) from exc
            raise ExternalServiceError(f"Embedding request failed: {str(exc)[:300]}") from exc
        self._check_dims(out[0])
        return out

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
