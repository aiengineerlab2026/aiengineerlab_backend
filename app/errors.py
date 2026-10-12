"""Domain-level errors. Presentation maps these to HTTP status codes."""


class DomainError(Exception):
    code = "domain_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(DomainError):
    code = "not_found"


class ValidationFailed(DomainError):
    code = "validation_failed"


class ConflictError(DomainError):
    code = "conflict"


class RateLimited(DomainError):
    """Too many requests from one client. ``details['retry_after_seconds']`` tells when to retry."""

    code = "rate_limited"


class ExternalServiceError(DomainError):
    """An external dependency (LLM, embedding model, ...) is missing or failed."""

    code = "external_service_error"


class LLMQuotaExceeded(ExternalServiceError):
    """The LLM provider refused the call because its (free-tier) quota or rate limit is used up."""

    code = "llm_quota_exceeded"

    def __init__(self, message: str | None = None, *, details: dict | None = None, provider: str = "google"):
        free = "the free version of Cloudflare Workers AI" if provider == "cloudflare" else "the free version of Gemini"
        super().__init__(
            message or f"We're using {free} and its usage limit has been reached for now. Please try again in a little while.",
            details=details,
        )


class ServiceNotConfigured(ExternalServiceError):
    code = "service_not_configured"
