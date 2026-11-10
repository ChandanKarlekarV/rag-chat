"""BM25 keyword index, persisted as a pickle and rebuilt whenever the corpus changes."""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.schemas import Chunk

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "what when where which who why how with does do did".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics and drop common stopwords."""
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


class BM25Store:
    """Okapi BM25 over all chunk texts, with optional per-document filtering."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self.chunk_ids: list[str] = []
        self.doc_ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        if path.exists():
            with path.open("rb") as fh:
                state = pickle.load(fh)
            self._load(state["chunk_ids"], state["doc_ids"], state["corpus_tokens"])

    def _load(self, chunk_ids: list[str], doc_ids: list[str], tokens: list[list[str]]) -> None:
        self.chunk_ids, self.doc_ids, self.corpus_tokens = chunk_ids, doc_ids, tokens
        self._bm25 = BM25Okapi(tokens) if tokens else None

    def rebuild(self, chunks: list[Chunk]) -> None:
        """Rebuild the index from scratch and persist it."""
        self._load([c.chunk_id for c in chunks], [c.doc_id for c in chunks], [tokenize(c.text) for c in chunks])
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as fh:
            pickle.dump(
                {"chunk_ids": self.chunk_ids, "doc_ids": self.doc_ids, "corpus_tokens": self.corpus_tokens}, fh
            )

    def search(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[tuple[str, float]]:
        """Return up to `k` (chunk_id, score) pairs with a positive score, best first."""
        terms = tokenize(query)
        if self._bm25 is None or not terms or doc_ids == []:
            return []
        allowed = set(doc_ids) if doc_ids else None
        scores = self._bm25.get_scores(terms)
        hits = [
            (cid, float(s))
            for cid, did, s in zip(self.chunk_ids, self.doc_ids, scores)
            if s > 0 and (allowed is None or did in allowed)
        ]
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits[:k]
