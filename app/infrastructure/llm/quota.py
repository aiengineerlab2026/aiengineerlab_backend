"""Remembers which models just reported that their (free-tier) quota is used up.

The provider tells us in its response (HTTP 429 / RESOURCE_EXHAUSTED / "quota exceeded"). The model is then
dimmed in the Configuration tab for a cooldown, after which it can be picked (and tried) again.
Cloudflare Workers AI meters one daily allowance for the whole account, so all its models are marked together.
"""

from __future__ import annotations

import time
from collections.abc import Callable

ACCOUNT_WIDE = {"cloudflare"}


def is_quota_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in ("429", "resource_exhausted", "resource exhausted", "quota", "rate limit", "rate_limit"))


class QuotaTracker:
    def __init__(self, cooldown_seconds: float = 900, clock: Callable[[], float] = time.monotonic):
        self._cooldown, self._clock = cooldown_seconds, clock
        self._until: dict[tuple[str, str], float] = {}

    @staticmethod
    def _key(provider: str, model: str) -> tuple[str, str]:
        return (provider, "*" if provider in ACCOUNT_WIDE else model)

    def mark(self, provider: str, model: str) -> None:
        self._until[self._key(provider, model)] = self._clock() + self._cooldown

    def remaining(self, provider: str, model: str) -> int:
        """Seconds until the model may be tried again; 0 when it is usable."""
        left = self._until.get(self._key(provider, model), 0) - self._clock()
        return max(0, int(left) + 1) if left > 0 else 0
