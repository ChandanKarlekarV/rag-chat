"""Lightweight model doubles so tests run without downloading models. (`FakeLLM` lives in the app.)"""

import hashlib
import math

from app.retrieval.bm25_store import tokenize
from app.schemas import RetrievedChunk

DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words embeddings: deterministic and good enough to rank lexical matches."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeReader:
    """Extractive-reader double: the sentence with the most question words is the 'answer span'."""

    def best_spans(self, question: str, sources: list[RetrievedChunk]):
        from app.generation.reader import Span

        q = set(tokenize(question))
        spans = []
        for i, s in enumerate(sources, start=1):
            text = s.chunk.text
            best, pos = (0, 0, len(text)), 0
            for sentence in text.split(". "):
                overlap = len(q & set(tokenize(sentence)))
                if overlap > best[0]:
                    best = (overlap, pos, pos + len(sentence))
                pos += len(sentence) + 2
            spans.append(Span(i, best[1], best[2], float(best[0]) - 1.0))
        return sorted(spans, key=lambda sp: sp.margin, reverse=True)


class FakeReranker:
    """Scores by query-term overlap, shifted so zero overlap is clearly negative."""

    def score(self, question: str, passages: list[str]) -> list[float]:
        q = set(tokenize(question))
        return [float(len(q & set(tokenize(p)))) - 1.0 for p in passages]

    def rerank(self, question: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        q = set(tokenize(question))
        scored = [
            c.model_copy(update={"rerank_score": float(len(q & set(tokenize(c.chunk.text)))) - 1.0}) for c in candidates
        ]
        scored.sort(key=lambda c: c.rerank_score, reverse=True)
        return scored[:top_k]
