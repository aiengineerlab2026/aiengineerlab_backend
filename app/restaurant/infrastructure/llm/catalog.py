"""The models that can be picked in the Configuration tab, per provider.

Only models that fit the app are listed: chat models must follow instructions and write JSON, and embedding
models are the two the menu is indexed with (Gemini at ``EMBEDDING_DIMENSIONS`` wide, bge-m3 at 1024).
The model configured in ``backend/.env`` is also offered when it comes from one of these providers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.config import Settings
from app.errors import ValidationFailed
from app.restaurant.domain.rag.value_objects import EmbeddingSpec, ModelRef, ModelSelection

# Only hosted providers can be picked in the UI: no local models.
PROVIDER_NAMES = {
    "google": "Google Gemini",
    "cloudflare": "Cloudflare Workers AI",
}
PROVIDER_HINTS = {
    "google": "Set LLM_API_KEY in backend/.env.",
    "cloudflare": "Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID in backend/.env.",
}


@dataclass(frozen=True)
class ModelOption:
    provider: str
    model: str
    label: str
    note: str = ""
    dimensions: int = 0  # embedding models: width of the vectors


_CHAT = [
    ModelOption("google", "gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", "Fastest, cheapest"),
    ModelOption("cloudflare", "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "Llama 3.3 70B (fast)", "Best quality on Workers AI"),
    ModelOption("cloudflare", "@cf/meta/llama-3.1-8b-instruct", "Llama 3.1 8B", "Small and quick"),
    ModelOption("cloudflare", "@cf/meta/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B"),
    ModelOption("cloudflare", "@cf/mistralai/mistral-small-3.1-24b-instruct", "Mistral Small 3.1 24B"),
    ModelOption("cloudflare", "@cf/google/gemma-3-12b-it", "Gemma 3 12B"),
    ModelOption("cloudflare", "@cf/openai/gpt-oss-20b", "GPT-OSS 20B"),
    ModelOption("cloudflare", "@cf/openai/gpt-oss-120b", "GPT-OSS 120B"),
    ModelOption("cloudflare", "@cf/qwen/qwen3-30b-a3b-fp8", "Qwen3 30B-A3B"),
]
# The two embedding models of the app. The menu is indexed with both ahead of time, so switching is instant.
_EMBEDDING = [
    ModelOption("google", "gemini-embedding-001", "Gemini Embedding 001", "Vectors cut to the standard index width"),
    ModelOption("cloudflare", "@cf/baai/bge-m3", "BGE-M3", "1024-d, multilingual", dimensions=1024),
]


def _with_env_default(options: list[ModelOption], provider: str, model: str) -> list[ModelOption]:
    if provider not in PROVIDER_NAMES or not model or any(o.provider == provider and o.model == model for o in options):
        return options
    return [ModelOption(provider, model, model, "From backend/.env"), *options]


def chat_models(settings: Settings) -> list[ModelOption]:
    out = _with_env_default(list(_CHAT), settings.llm_provider, settings.llm_model)
    return _with_env_default(out, settings.effective_eval_provider, settings.effective_eval_model)


def embedding_models(settings: Settings) -> list[ModelOption]:
    options = _with_env_default(list(_EMBEDDING), settings.embedding_provider, settings.embedding_model)
    # Models without a native width of their own produce EMBEDDING_DIMENSIONS-wide vectors (Gemini is cut to it).
    return [o if o.dimensions else replace(o, dimensions=settings.embedding_dimensions) for o in options]


def embedding_dimensions(settings: Settings, provider: str, model: str) -> int:
    for o in embedding_models(settings):
        if o.provider == provider and o.model == model:
            return o.dimensions
    return settings.embedding_dimensions


def embedding_specs(settings: Settings) -> list[EmbeddingSpec]:
    """Every embedding model that can be used right now (credentials present): the menu is pre-indexed with each."""
    return [
        EmbeddingSpec(provider=o.provider, model=o.model, dimensions=o.dimensions, version=settings.embedding_version)
        for o in embedding_models(settings)
        if settings.provider_configured(o.provider)
    ]


def providers(settings: Settings) -> list[dict]:
    used = {o.provider for o in [*chat_models(settings), *embedding_models(settings)]}
    return [
        {
            "id": p, "name": PROVIDER_NAMES.get(p, p), "configured": settings.provider_configured(p),
            "hint": None if settings.provider_configured(p) else PROVIDER_HINTS.get(p),
        }
        for p in PROVIDER_NAMES
        if p in used
    ]


def _check(kind: str, ref: ModelRef, options: list[ModelOption], settings: Settings) -> None:
    if not any(o.provider == ref.provider and o.model == ref.model for o in options):
        raise ValidationFailed(f"{kind} model '{ref.model}' is not available for provider '{ref.provider}'.")
    if not settings.provider_configured(ref.provider):
        raise ValidationFailed(
            f"{PROVIDER_NAMES.get(ref.provider, ref.provider)} is not set up. "
            f"{PROVIDER_HINTS.get(ref.provider, '')}".strip()
        )


def check_slot(settings: Settings, slot: str, ref: ModelRef) -> None:
    """Raise ``ValidationFailed`` unless ``ref`` is a selectable, configured model for ``slot`` (llm/judge/embedding)."""
    if slot == "embedding":
        _check("Embedding", ref, embedding_models(settings), settings)
    else:
        _check("Evaluation" if slot == "judge" else "Chat", ref, chat_models(settings), settings)


def validate(settings: Settings, sel: ModelSelection) -> None:
    for slot in ("llm", "judge"):
        check_slot(settings, slot, getattr(sel, slot))
