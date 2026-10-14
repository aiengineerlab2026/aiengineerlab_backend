"""Application settings (12-factor, loaded from environment / backend/.env)."""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["local", "test", "production"] = "local"
    log_level: str = "INFO"
    cors_origins: str = "*"  # comma-separated origins, or "*" for all (no credentials are used)

    # --- database -------------------------------------------------------------------------
    # postgresql://user:pass@host:5432/db  (password may be percent-encoded).
    # For networks without IPv6 use the Supabase *session pooler* URL instead.
    database_url: str = ""
    db_ssl: Literal["auto", "require", "disable"] = "auto"
    db_pool_size: int = 5
    db_echo: bool = False

    # --- LLM ---------------------------------------------------------------------------------
    llm_provider: Literal["openai", "google", "cloudflare"] = "openai"  # "openai" = any OpenAI-compatible endpoint
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""  # e.g. https://api.groq.com/openai/v1, http://localhost:11434/v1
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    # --- embeddings --------------------------------------------------------------------------
    embedding_provider: Literal["openai", "google", "cloudflare"] = "google"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 384  # must match the pgvector column created by the migration
    embedding_version: str = "1"
    embedding_api_key: str = ""  # falls back to LLM_API_KEY
    embedding_base_url: str = ""  # falls back to LLM_BASE_URL

    # --- Cloudflare Workers AI (used through LangChain's OpenAI-compatible client) -------------------
    # Secret. Keep it only in the git-ignored backend/.env. The account id is in the dashboard URL.
    cloudflare_api_token: str = ""
    cloudflare_account_id: str = ""

    # How long a model stays dimmed in the Configuration tab after its provider reports the quota is used up.
    quota_cooldown_seconds: float = 900.0

    # --- evaluation (judge) ----------------------------------------------------------------
    eval_llm_provider: str = ""  # defaults to LLM_PROVIDER
    eval_llm_model: str = ""  # defaults to LLM_MODEL
    eval_max_workers: int = 4
    eval_timeout_seconds: int = 300

    # --- abuse protection ----------------------------------------------------------------------
    # Chat messages (each one may call the LLM) allowed per client within the window. 0 disables the limit.
    chat_rate_limit_max: int = 5
    chat_rate_limit_window_seconds: int = 2_592_000  # 30 days: effectively a total cap per IP
    # Evaluations (the "Evaluate" button: judge LLM + LLM-based RAGAS metrics for every new question) one client may
    # run within the window. Chat keeps working once it is used up. 0 disables the limit.
    eval_rate_limit_max: int = 2
    eval_rate_limit_window_seconds: int = 86400
    # Behind a reverse proxy, read the client IP from X-Forwarded-For. Only enable if you trust the proxy,
    # otherwise clients can spoof the header and dodge the limit.
    trust_proxy_headers: bool = False

    # --- My Resume lab ---------------------------------------------------------------------------
    myresume_upload_rate_limit_max: int = 3  # resumes one client may upload in total (the lab is a demo); 0 disables
    myresume_upload_rate_limit_window_seconds: int = 315_360_000
    myresume_vector_rate_limit_max: int = 3  # vector searches (they call an embedding model) one client may run in total; 0 disables
    myresume_vector_rate_limit_window_seconds: int = 315_360_000

    # --- admin (articles) ------------------------------------------------------------------------
    # The login user. The password starts as ADMIN_DEFAULT_PASSWORD and is changed from the admin area, after which
    # only its hash is stored in the database and this value is no longer used.
    admin_username: str = "admin"
    admin_default_password: str = "admin"
    # Signs the login tokens. Empty = a random secret per process (everyone is logged out on restart).
    admin_token_secret: str = ""
    admin_token_ttl_seconds: int = 43_200  # 12 hours
    admin_login_max_failures: int = 8  # wrong passwords one client IP may try within the window; 0 disables
    admin_login_window_seconds: int = 900

    # --- background worker ---------------------------------------------------------------------
    # Score every real customer search question automatically (extra judge-LLM calls per question).
    live_evaluation_enabled: bool = True
    run_worker: bool = True
    worker_poll_seconds: float = 2.0

    @field_validator("database_url")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def effective_eval_provider(self) -> str:
        return self.eval_llm_provider or self.llm_provider

    @property
    def effective_eval_model(self) -> str:
        return self.eval_llm_model or self.llm_model

    @property
    def cloudflare_base_url(self) -> str:
        """Cloudflare's OpenAI-compatible endpoint for Workers AI (chat completions and embeddings)."""
        return f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/v1"

    def provider_configured(self, provider: str) -> bool:
        """Whether the credentials for ``provider`` (chat or embeddings) are present."""
        if provider == "cloudflare":
            return bool(self.cloudflare_api_token and self.cloudflare_account_id)
        if provider == "openai" and self.llm_base_url and "localhost" in self.llm_base_url:
            return True  # local servers such as Ollama accept any key
        return bool(self.llm_api_key)

    @property
    def llm_configured(self) -> bool:
        return self.provider_configured(self.llm_provider)

    @property
    def sqlalchemy_url(self) -> URL:
        return build_async_url(self.database_url)

    @property
    def connect_args(self) -> dict:
        url = self.sqlalchemy_url
        args: dict = {}
        host = url.host or ""
        is_local = host in ("localhost", "127.0.0.1", "::1", "postgres", "db")
        if self.db_ssl == "require" or (self.db_ssl == "auto" and not is_local):
            args["ssl"] = "require"
        # Supavisor transaction mode (port 6543) does not support prepared statements.
        if url.port == 6543:
            args["statement_cache_size"] = 0
            args["prepared_statement_cache_size"] = 0
            args["prepared_statement_name_func"] = lambda: f"__asyncpg_{uuid.uuid4()}__"
        return args


def build_async_url(raw: str) -> URL:
    """postgresql://... -> postgresql+asyncpg://... (``make_url`` percent-decodes the password)."""
    if not raw:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy backend/.env.example to backend/.env and fill it in."
        )
    url = make_url(raw)
    if url.drivername in ("postgres", "postgresql", "postgresql+psycopg2", "postgresql+psycopg"):
        url = url.set(drivername="postgresql+asyncpg")
    # asyncpg does not understand libpq's sslmode query parameter.
    query = {k: v for k, v in url.query.items() if k not in ("sslmode", "pgbouncer", "supa")}
    return url.set(query=query)


@lru_cache
def get_settings() -> Settings:
    return Settings()
