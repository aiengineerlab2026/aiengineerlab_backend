"""Provider-independent LLM adapter built on LangChain chat models."""

from __future__ import annotations

import logging
from typing import Any

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.domain.errors import ExternalServiceError, LLMQuotaExceeded, ServiceNotConfigured
from app.domain.rag.ports import LLMClient, T
from app.infrastructure.config import Settings
from app.infrastructure.llm.quota import QuotaTracker, is_quota_error

log = logging.getLogger(__name__)


def build_langchain_chat_model(settings: Settings, model: str | None = None, provider: str | None = None) -> Any:
    """Create the LangChain chat model for ``provider`` (default: the configured one; lazy provider imports)."""
    name = model or settings.llm_model
    provider = provider or settings.llm_provider
    if provider == "cloudflare":
        # Workers AI speaks the OpenAI chat-completions protocol, so LangChain's ChatOpenAI talks to it directly.
        from langchain_openai import ChatOpenAI

        if not settings.provider_configured("cloudflare"):
            raise ServiceNotConfigured("Cloudflare Workers AI needs CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID in backend/.env.")
        return ChatOpenAI(
            model=name,
            api_key=settings.cloudflare_api_token,
            base_url=settings.cloudflare_base_url,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=name,
            api_key=settings.llm_api_key or "not-needed",
            base_url=settings.llm_base_url or None,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # retries are handled by our own backoff policy
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=name,
            google_api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
    raise ServiceNotConfigured(f"Unsupported LLM provider '{provider}'.")


def _retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    # Authentication / bad-request style errors will not fix themselves.
    return not any(s in msg for s in ("401", "403", "invalid api key", "incorrect api key", "api key not valid", "400"))


class LangChainLLM(LLMClient):
    def __init__(
        self, settings: Settings, model: str | None = None, provider: str | None = None, quota: QuotaTracker | None = None
    ):
        self._settings = settings
        self._quota = quota
        self._provider = provider or settings.llm_provider
        self._model_name = model or settings.llm_model
        self._lc: Any | None = None

    def select(self, provider: str, model: str) -> None:
        """Switch to another model; the next call builds a fresh LangChain client for it."""
        self._provider, self._model_name, self._lc = provider, model, None

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_configured(self) -> bool:
        return self._settings.provider_configured(self._provider)

    @property
    def langchain_model(self) -> Any:
        if not self.is_configured:
            raise ServiceNotConfigured(
                f"The {self._provider} LLM is not configured. Set its credentials (LLM_API_KEY, or the CLOUDFLARE_* "
                "variables) in backend/.env."
            )
        if self._lc is None:
            self._lc = build_langchain_chat_model(self._settings, self._model_name, self._provider)
        return self._lc

    async def _invoke(self, runnable: Any, messages: list[tuple[str, str]]) -> Any:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._settings.llm_max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=20),
                retry=retry_if_exception(_retryable),
                reraise=True,
            ):
                with attempt:
                    return await runnable.ainvoke(messages)
        except ServiceNotConfigured:
            raise
        except Exception as exc:
            log.warning("LLM call failed", extra={"model": self._model_name, "error": str(exc)[:300]})
            if is_quota_error(exc):
                if self._quota:
                    self._quota.mark(self._provider, self._model_name)
                raise LLMQuotaExceeded(provider=self._provider) from exc
            raise ExternalServiceError(f"LLM request failed: {str(exc)[:300]}") from exc

    async def generate(self, system: str, user: str) -> str:
        msg = await self._invoke(self.langchain_model, [("system", system), ("human", user)])
        content = msg.content
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return str(content).strip()

    async def generate_structured(self, system: str, user: str, schema: type[T]) -> T:
        if self.provider == "cloudflare":
            return await self._structured_from_json(system, user, schema)
        kwargs = {"method": "function_calling"} if self.provider == "openai" else {}
        runnable = self.langchain_model.with_structured_output(schema, **kwargs)
        result = await self._invoke(runnable, [("system", system), ("human", user)])
        if isinstance(result, dict):
            result = schema.model_validate(result)
        if result is None:
            raise ExternalServiceError("LLM returned no structured output.")
        return result

    async def _structured_from_json(self, system: str, user: str, schema: type[T]) -> T:
        """Structured output for models without reliable tool calling: ask for JSON, parse it with LangChain.

        Works with every Workers AI chat model, because it needs nothing but plain text generation.
        """
        from langchain_core.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=schema)
        prompt = f"{system}\n\n{parser.get_format_instructions()}\nReply with the JSON object only, no other text."
        text = await self.generate(prompt, user)
        try:
            return parser.parse(text)
        except Exception as exc:  # malformed JSON from the model
            raise ExternalServiceError(f"The model did not return valid JSON: {str(exc)[:200]}") from exc
