"""In-memory sliding-window rate limiter (per key, e.g. client IP).

Process-local by design: it needs no extra infrastructure, but counters reset on restart and are not shared
between several API processes. Swap the implementation (same ``hit`` contract) for Redis if you scale out.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LimitResult:
    allowed: bool
    remaining: int
    retry_after_seconds: int  # 0 when allowed


@dataclass(frozen=True)
class Usage:
    limit: int
    used: int
    remaining: int
    reset_in_seconds: int  # until the oldest counted event expires (0 when nothing is counted)


class SlidingWindowLimiter:
    def __init__(self, max_events: int, window_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.max_events, self.window, self._clock = max_events, window_seconds, clock
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_events > 0 and self.window > 0

    def peek(self, key: str) -> Usage:
        """Current usage for ``key`` without recording an event."""
        if not self.enabled:
            return Usage(0, 0, 0, 0)
        now = self._clock()
        with self._lock:
            q = self._events.get(key)
            while q and now - q[0] >= self.window:
                q.popleft()
            used = len(q) if q else 0
            reset = max(1, math.ceil(self.window - (now - q[0]))) if q else 0
        return Usage(self.max_events, used, max(0, self.max_events - used), reset)

    def hit(self, key: str) -> LimitResult:
        """Record one event for ``key`` if the budget allows it. Rejected attempts are not counted."""
        if not self.enabled:
            return LimitResult(True, 10**9, 0)
        now = self._clock()
        with self._lock:
            q = self._events.setdefault(key, deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.max_events:
                return LimitResult(False, 0, max(1, math.ceil(self.window - (now - q[0]))))
            q.append(now)
            if len(self._events) > 10_000:  # opportunistic cleanup so abandoned IPs do not accumulate
                for k in [k for k, v in self._events.items() if not v or now - v[-1] >= self.window]:
                    del self._events[k]
            return LimitResult(True, self.max_events - len(q), 0)
