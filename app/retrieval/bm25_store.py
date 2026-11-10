"""Per-user BM25 keyword indexes, persisted as one pickle per user and cached in memory."""

import pickle
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Plus

from app.schemas import Chunk

_WORD = re.compile(r"[a-z0-9]+")
_SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]+$")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "what when where which who why how with does do did".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics and drop common stopwords."""
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class _Index:
    chunk_ids: list[str]
    doc_ids: list[str]
    vocab: list[frozenset[str]]
    bm25: BM25Plus | None

    @classmethod
    def build(cls, chunk_ids: list[str], doc_ids: list[str], tokens: list[list[str]]) -> "_Index":
        return cls(chunk_ids, doc_ids, [frozenset(t) for t in tokens], BM25Plus(tokens) if tokens else None)


class BM25Store:
    """One BM25+ index per user. Rebuilt from that user's chunks whenever they change.

    BM25+ rather than Okapi: Okapi's IDF goes negative for terms in most documents, which breaks
    tiny per-user corpora (a one-chunk document scored below zero). rank_bm25's BM25+ adds the
    same positive offset per query term to every document, so ranking is unaffected but chunks
    with no query term would still score above zero; `search` therefore only returns chunks that
    contain at least one query term.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        directory.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, _Index] = {}
        self._lock = threading.Lock()

    def _path(self, user_id: str) -> Path:
        if not _SAFE_ID.match(user_id):
            raise ValueError("invalid user id")
        return self._dir / f"{user_id}.pkl"

    def _index(self, user_id: str) -> _Index:
        with self._lock:
            if user_id not in self._cache:
                path = self._path(user_id)
                if path.exists():
                    with path.open("rb") as fh:
                        state = pickle.load(fh)
                    self._cache[user_id] = _Index.build(state["chunk_ids"], state["doc_ids"], state["corpus_tokens"])
                else:
                    self._cache[user_id] = _Index.build([], [], [])
            return self._cache[user_id]

    def rebuild(self, user_id: str, chunks: list[Chunk]) -> None:
        """Rebuild a user's index from scratch and persist it."""
        ids, docs, tokens = [c.chunk_id for c in chunks], [c.doc_id for c in chunks], [tokenize(c.text) for c in chunks]
        with self._path(user_id).open("wb") as fh:
            pickle.dump({"chunk_ids": ids, "doc_ids": docs, "corpus_tokens": tokens}, fh)
        with self._lock:
            self._cache[user_id] = _Index.build(ids, docs, tokens)

    def delete_user(self, user_id: str) -> None:
        """Forget a user's index."""
        with self._lock:
            self._cache.pop(user_id, None)
        self._path(user_id).unlink(missing_ok=True)

    def search(
        self, user_id: str, query: str, k: int, doc_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Up to `k` (chunk_id, score) pairs from the user's own index, best first."""
        terms = tokenize(query)
        idx = self._index(user_id)
        if idx.bm25 is None or not terms or doc_ids == []:
            return []
        allowed = set(doc_ids) if doc_ids else None
        wanted = set(terms)
        scores = idx.bm25.get_scores(terms)
        hits = [
            (cid, float(s))
            for cid, did, vocab, s in zip(idx.chunk_ids, idx.doc_ids, idx.vocab, scores)
            if wanted & vocab and (allowed is None or did in allowed)
        ]
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits[:k]
