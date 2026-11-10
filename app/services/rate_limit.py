"""Sliding-window rate limiting: in-memory for one process, Redis when several servers share load."""

import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Protocol

from app.errors import TooMany


class RateLimiter(Protocol):
    """Allow at most `limit` hits per window for each key."""

    def hit(self, key: str, message: str) -> None:
        """Record a hit or raise `TooMany` with the given friendly message."""
        ...


class MemoryRateLimiter:
    """Per-process limiter (single worker deployments and tests)."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, message: str) -> None:
        """Record a hit or raise `TooMany`."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                raise TooMany("rate_limited", message)
            q.append(now)


class RedisRateLimiter:
    """Limiter shared by every worker/server: one Redis sorted set of hit timestamps per key."""

    def __init__(self, redis, limit: int, window_s: float = 60.0, prefix: str = "rl:") -> None:
        self.redis = redis
        self.limit = limit
        self.window_s = window_s
        self.prefix = prefix

    def hit(self, key: str, message: str) -> None:
        """Atomically drop old hits, count, and add this one; undo the add if over the limit."""
        k = self.prefix + key
        now = time.time()
        member = f"{now}:{uuid.uuid4().hex[:8]}"
        pipe = self.redis.pipeline(transaction=True)
        pipe.zremrangebyscore(k, 0, now - self.window_s)
        pipe.zadd(k, {member: now})
        pipe.zcard(k)
        pipe.expire(k, int(self.window_s) + 1)
        _, _, count, _ = pipe.execute()
        if count > self.limit:
            self.redis.zrem(k, member)
            raise TooMany("rate_limited", message)
