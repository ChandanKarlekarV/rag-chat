"""Tiny in-memory sliding-window rate limiter (single process; fine for one container)."""

import threading
import time
from collections import defaultdict, deque

from app.errors import TooMany


class RateLimiter:
    """Allow at most `limit` hits per `window_s` seconds for each key."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, message: str) -> None:
        """Record a hit or raise `TooMany` with the given friendly message."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                raise TooMany("rate_limited", message)
            q.append(now)
