"""State shared between workers: ingest progress and per-user write locks.

In-memory implementations serve a single process; the Redis ones let several Uvicorn workers or
containers see the same progress and serialise index rebuilds for the same user.
"""

import json
import threading
from contextlib import AbstractContextManager
from typing import Protocol

PROGRESS_TTL_S = 3600
LOCK_TIMEOUT_S = 120


class ProgressStore(Protocol):
    """Stage and percent of documents being ingested."""

    def set(self, doc_id: str, stage: str, percent: int) -> None: ...
    def get(self, doc_id: str) -> tuple[str, int] | None: ...
    def delete(self, doc_id: str) -> None: ...


class LockProvider(Protocol):
    """Mutual exclusion per key (used per user around index writes)."""

    def lock(self, key: str) -> AbstractContextManager[object]: ...


class MemoryProgress:
    """Progress kept in this process."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, int]] = {}

    def set(self, doc_id: str, stage: str, percent: int) -> None:
        """Record a stage."""
        self._data[doc_id] = (stage, percent)

    def get(self, doc_id: str) -> tuple[str, int] | None:
        """Latest stage, if known."""
        return self._data.get(doc_id)

    def delete(self, doc_id: str) -> None:
        """Forget a document."""
        self._data.pop(doc_id, None)


class RedisProgress:
    """Progress in Redis, visible to every worker; entries expire after an hour."""

    def __init__(self, redis, prefix: str = "progress:") -> None:
        self.redis = redis
        self.prefix = prefix

    def set(self, doc_id: str, stage: str, percent: int) -> None:
        """Record a stage."""
        self.redis.set(self.prefix + doc_id, json.dumps([stage, percent]), ex=PROGRESS_TTL_S)

    def get(self, doc_id: str) -> tuple[str, int] | None:
        """Latest stage, if known."""
        raw = self.redis.get(self.prefix + doc_id)
        if raw is None:
            return None
        stage, percent = json.loads(raw)
        return str(stage), int(percent)

    def delete(self, doc_id: str) -> None:
        """Forget a document."""
        self.redis.delete(self.prefix + doc_id)


class MemoryLocks:
    """Thread locks, one per key."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def lock(self, key: str) -> threading.Lock:
        """The lock for `key` (use as a context manager)."""
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())


class RedisLocks:
    """Redis locks shared across processes (auto-expire so a crashed worker can't block forever)."""

    def __init__(self, redis, prefix: str = "lock:") -> None:
        self.redis = redis
        self.prefix = prefix

    def lock(self, key: str) -> AbstractContextManager[object]:
        """A blocking Redis lock for `key`."""
        return self.redis.lock(self.prefix + key, timeout=LOCK_TIMEOUT_S, blocking_timeout=LOCK_TIMEOUT_S)
