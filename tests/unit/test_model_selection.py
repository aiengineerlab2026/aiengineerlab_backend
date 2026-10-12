"""Choosing chat / evaluation / embedding models (Google, Cloudflare Workers AI, local) in the Configuration tab."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from app.domain.errors import ExternalServiceError, ValidationFailed
from app.domain.rag.value_objects import ModelRef, ModelSelection
from app.infrastructure.config import Settings
from app.infrastructure.llm import catalog
from app.infrastructure.llm.chat_models import LangChainLLM, build_langchain_chat_model
from app.infrastructure.rag.embeddings import (
    LangChainEmbedder,
    NormalizedEmbeddings,
    build_langchain_embeddings,
)

CF = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
BGE = "@cf/baai/bge-m3"


def settings(**kw) -> Settings:
    base = dict(llm_provider="google", llm_model="gemini-2.5-flash-lite", llm_api_key="g-key",
                embedding_provider="google", embedding_model="gemini-embedding-001")
    return Settings(_env_file=None, **{**base, **kw})  # never read the developer's real .env


def cloudflare(**kw) -> Settings:
    return settings(cloudflare_api_token="cf-token", cloudflare_account_id="abc123", **kw)


def selection(llm=("google", "gemini-2.5-flash-lite"), judge=("google", "gemini-2.5-flash-lite")):
    return ModelSelection(llm=ModelRef(provider=llm[0], model=llm[1]), judge=ModelRef(provider=judge[0], model=judge[1]))


def test_cloudflare_needs_both_token_and_account_id():
    assert not settings().provider_configured("cloudflare")
    assert not settings(cloudflare_api_token="t").provider_configured("cloudflare")
    assert cloudflare().provider_configured("cloudflare")
    assert cloudflare().cloudflare_base_url == "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1"


def test_catalog_lists_cloudflare_for_chat_judge_and_embeddings():
    s = cloudflare()
    assert any(o.provider == "cloudflare" and o.model == CF for o in catalog.chat_models(s))
    assert any(o.provider == "cloudflare" and o.model == BGE for o in catalog.embedding_models(s))
    cf = next(p for p in catalog.providers(s) if p["id"] == "cloudflare")
    assert cf["configured"] and cf["hint"] is None
    assert not next(p for p in catalog.providers(settings()) if p["id"] == "cloudflare")["configured"]


def test_only_gemini_and_bge_m3_can_embed_and_each_has_its_own_width():
    options = {o.model: o.dimensions for o in catalog.embedding_models(cloudflare())}
    assert options == {"gemini-embedding-001": 384, BGE: 1024}
    assert catalog.embedding_dimensions(cloudflare(embedding_dimensions=768), "google", "gemini-embedding-001") == 768


def test_specs_to_pre_index_cover_every_usable_embedding_model():
    assert [s.model for s in catalog.embedding_specs(cloudflare())] == ["gemini-embedding-001", BGE]
    assert [s.model for s in catalog.embedding_specs(settings())] == ["gemini-embedding-001"]  # no Cloudflare creds


def test_env_model_is_always_selectable():
    s = settings(llm_model="gemini-9-custom")
    assert any(o.model == "gemini-9-custom" for o in catalog.chat_models(s))


def test_validate_accepts_cloudflare_when_configured():
    catalog.validate(cloudflare(), selection(llm=("cloudflare", CF), judge=("cloudflare", CF)))


def test_validate_rejects_unconfigured_provider_and_unknown_model():
    with pytest.raises(ValidationFailed, match="CLOUDFLARE_ACCOUNT_ID"):
        catalog.validate(settings(), selection(llm=("cloudflare", CF)))
    with pytest.raises(ValidationFailed, match="not available"):
        catalog.validate(cloudflare(), selection(llm=("cloudflare", "@cf/nope/model")))
    with pytest.raises(ValidationFailed):  # a chat model is not an embedding model
        catalog.check_slot(cloudflare(), "embedding", ModelRef(provider="cloudflare", model=CF))


def test_cloudflare_chat_model_is_a_langchain_client_for_workers_ai():
    lc = build_langchain_chat_model(cloudflare(), CF, "cloudflare")
    assert type(lc).__name__ == "ChatOpenAI"
    assert lc.model_name == CF and str(lc.openai_api_base).endswith("/accounts/abc123/ai/v1")


def test_cloudflare_embeddings_are_a_normalised_langchain_embeddings():
    emb = build_langchain_embeddings(cloudflare(), "cloudflare", BGE)
    assert isinstance(emb, NormalizedEmbeddings)


def test_llm_select_switches_provider_and_model():
    llm = LangChainLLM(cloudflare())
    assert (llm.provider, llm.model_name) == ("google", "gemini-2.5-flash-lite")
    llm.select("cloudflare", CF)
    assert (llm.provider, llm.model_name, llm.is_configured) == ("cloudflare", CF, True)
    assert not LangChainLLM(settings(), provider="cloudflare", model=CF).is_configured


def test_each_embedder_is_fixed_to_its_own_model_and_width():
    from app.domain.rag.value_objects import EmbeddingSpec

    spec = EmbeddingSpec(provider="cloudflare", model=BGE, dimensions=1024)
    e = LangChainEmbedder(cloudflare(), spec=spec)
    assert e.spec == spec
    assert LangChainEmbedder(cloudflare()).spec.model == "gemini-embedding-001"  # the default


class Answer(BaseModel):
    city: str


def test_cloudflare_structured_output_parses_json_from_plain_text():
    class Fake(LangChainLLM):
        async def generate(self, system, user):
            assert "JSON" in system
            return 'Sure!\n```json\n{"city": "Cairo"}\n```'

    llm = Fake(cloudflare(), provider="cloudflare", model=CF)
    assert asyncio.run(llm.generate_structured("s", "u", Answer)) == Answer(city="Cairo")

    class Bad(LangChainLLM):
        async def generate(self, system, user):
            return "no json here"

    with pytest.raises(ExternalServiceError):
        asyncio.run(Bad(cloudflare(), provider="cloudflare", model=CF).generate_structured("s", "u", Answer))


# ---------------------------------------------------------------- free quota used up -> model is dimmed
from app.domain.errors import LLMQuotaExceeded  # noqa: E402
from app.infrastructure.llm.quota import QuotaTracker  # noqa: E402


def test_quota_tracker_dims_a_model_for_the_cooldown_then_releases_it():
    now = [0.0]
    q = QuotaTracker(cooldown_seconds=100, clock=lambda: now[0])
    assert q.remaining("google", "gemini-2.5-flash") == 0
    q.mark("google", "gemini-2.5-flash")
    assert q.remaining("google", "gemini-2.5-flash") > 0
    assert q.remaining("google", "gemini-2.5-flash-lite") == 0  # Gemini limits are per model
    now[0] = 101
    assert q.remaining("google", "gemini-2.5-flash") == 0


def test_cloudflare_quota_is_account_wide():
    q = QuotaTracker()
    q.mark("cloudflare", CF)
    assert q.remaining("cloudflare", "@cf/meta/llama-3.1-8b-instruct") > 0
    assert q.remaining("cloudflare", BGE) > 0  # embeddings draw on the same allowance
    assert q.remaining("google", "gemini-2.5-flash") == 0


def test_llm_quota_error_marks_the_model_and_names_the_provider():
    class Boom:
        async def ainvoke(self, _):
            raise RuntimeError("429 You exceeded your current quota")

    q = QuotaTracker()
    llm = LangChainLLM(cloudflare(llm_max_retries=1), provider="cloudflare", model=CF, quota=q)
    with pytest.raises(LLMQuotaExceeded, match="Cloudflare"):
        asyncio.run(llm._invoke(Boom(), []))
    assert q.remaining("cloudflare", CF) > 0


def test_embedding_quota_error_marks_the_model():
    class Boom:
        def embed_documents(self, _):
            raise RuntimeError("429 Too Many Requests: quota")

    q = QuotaTracker()
    from app.domain.rag.value_objects import EmbeddingSpec

    e = LangChainEmbedder(cloudflare(), quota=q, spec=EmbeddingSpec(provider="cloudflare", model=BGE, dimensions=1024))
    e._lc = Boom()
    with pytest.raises(LLMQuotaExceeded):
        asyncio.run(e.embed_query("hi"))
    assert q.remaining("cloudflare", BGE) > 0
