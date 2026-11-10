"""Retrieval entry point: user-scoped hybrid search, the relevance gate and structured logging."""

import logging

from app.retrieval.hybrid import HybridRetriever, Mode, RetrievalResult
from app.schemas import RetrievedChunk

log = logging.getLogger("rag-chat.retrieval")


def passes_gate(results: list[RetrievedChunk], min_score: float) -> bool:
    """True when the best reranked chunk is relevant enough to be worth sending to the LLM."""
    scores = [r.rerank_score for r in results if r.rerank_score is not None]
    return bool(scores) and max(scores) >= min_score


class RetrievalService:
    """Wraps the retriever so every call is scoped to one user and logged without document text."""

    def __init__(self, retriever: HybridRetriever, final_k: int = 5) -> None:
        self.retriever = retriever
        self.final_k = final_k

    def search(
        self, user_id: str, question: str, doc_ids: list[str] | None = None, mode: Mode = "rerank"
    ) -> RetrievalResult:
        """Retrieve the user's best chunks and log ids, scores and latencies."""
        result = self.retriever.retrieve(user_id, question, self.final_k, doc_ids, mode)
        log.info(
            "retrieval",
            extra={
                "user_id": user_id,
                "timings_ms": result.timings_ms,
                "chunk_ids": [r.chunk.chunk_id for r in result.chunks],
                "rerank_scores": [round(r.rerank_score, 3) for r in result.chunks if r.rerank_score is not None],
            },
        )
        return result
