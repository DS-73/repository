"""Very small in-process rate limiter (sliding window).

Good enough to blunt accidental hammering / a naive abuse script against a
single replica.  In a multi-replica deployment this belongs in Redis or at the
API gateway - see ``docs/architecture.md``.  The limiter fails open on internal
errors: availability of the underwriting API matters more than perfect policing.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable


class SlidingWindowRateLimiter:
    """Allow at most ``limit`` requests per ``window_seconds`` per key."""

    def __init__(
        self,
        *,
        limit: int,
        window_seconds: float,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = max(limit, 1)
        self._window = window_seconds
        self._max_keys = max_keys
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)``."""
        now = self._clock()
        with self._lock:
            if len(self._hits) > self._max_keys:
                self._evict(now)
            hits = self._hits[key]
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                retry_after = self._window - (now - hits[0])
                return False, max(retry_after, 0.0)
            hits.append(now)
            if not hits:
                self._hits.pop(key, None)
            return True, 0.0

    def _evict(self, now: float) -> None:
        stale = [
            key
            for key, hits in self._hits.items()
            if not hits or now - hits[-1] >= self._window
        ]
        for key in stale:
            self._hits.pop(key, None)
