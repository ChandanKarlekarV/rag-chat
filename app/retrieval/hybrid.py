"""Hybrid retrieval: vector + BM25 merged with Reciprocal Rank Fusion, then cross-encoder rerank."""

from typing import Literal, Protocol

from app.retrieval.bm25_store import BM25Store
from app.retrieval.vector_store import VectorStore
from app.schemas import RetrievedChunk

Mode = Literal["vector", "hybrid", "rerank"]


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class ChunkReranker(Protocol):
    def rerank(self, question: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank), rank starting at 1.

    Ties are broken by first appearance so the result is deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    order = {item: i for i, item in enumerate(dict.fromkeys(i for r in rankings for i in r))}
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


class HybridRetriever:
    """Runs the full retrieval pipeline and keeps every intermediate score."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        vectors: VectorStore,
        bm25: BM25Store,
        reranker: ChunkReranker,
        candidates: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.vectors = vectors
        self.bm25 = bm25
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k

    def retrieve(
        self, question: str, top_k: int = 5, doc_ids: list[str] | None = None, mode: Mode = "rerank"
    ) -> list[RetrievedChunk]:
        """Return the `top_k` best chunks for `question` using the requested pipeline `mode`."""
        vec_hits = self.vectors.query(self.embedder.embed_query(question), self.candidates, doc_ids)
        if mode == "vector":
            return [RetrievedChunk(chunk=c, vector_score=s) for c, s in vec_hits[:top_k]]

        bm25_hits = self.bm25.search(question, self.candidates, doc_ids)
        vec_by_id = {c.chunk_id: (c, s) for c, s in vec_hits}
        bm25_by_id = dict(bm25_hits)
        missing = [cid for cid in bm25_by_id if cid not in vec_by_id]
        chunks = {cid: c for cid, (c, _) in vec_by_id.items()} | self.vectors.get_chunks(missing)

        fused = rrf_fuse([[c.chunk_id for c, _ in vec_hits], [cid for cid, _ in bm25_hits]], self.rrf_k)
        merged = [
            RetrievedChunk(
                chunk=chunks[cid],
                vector_score=vec_by_id[cid][1] if cid in vec_by_id else None,
                bm25_score=bm25_by_id.get(cid),
                rrf_score=score,
            )
            for cid, score in fused[: self.candidates]
            if cid in chunks
        ]
        if mode == "hybrid":
            return merged[:top_k]
        return self.reranker.rerank(question, merged, top_k)
