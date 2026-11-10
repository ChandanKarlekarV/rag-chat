"""Cross-encoder reranker, loaded once per process."""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.schemas import RetrievedChunk


class Reranker:
    """Scores (question, passage) pairs jointly; higher is more relevant (raw logits)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name, device="cpu", max_length=512)

    def score(self, question: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage."""
        if not passages:
            return []
        scores = self._model.predict([(question, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(self, question: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Attach `rerank_score` to every candidate and return the best `top_k`."""
        scores = self.score(question, [c.chunk.text for c in candidates])
        scored = [c.model_copy(update={"rerank_score": s}) for c, s in zip(candidates, scores)]
        scored.sort(key=lambda c: c.rerank_score, reverse=True)
        return scored[:top_k]


@lru_cache
def get_reranker(model_name: str) -> Reranker:
    """Return the singleton reranker for `model_name`."""
    return Reranker(model_name)
