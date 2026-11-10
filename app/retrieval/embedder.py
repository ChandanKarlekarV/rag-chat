"""Sentence-transformers embedding wrapper, loaded once per process."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Encodes passages and queries into L2-normalised vectors."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages (no prefix) in batches."""
        if not texts:
            return []
        vecs = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query, adding the bge instruction prefix when applicable."""
        vec = self._model.encode(self._query_prefix + text, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()


@lru_cache
def get_embedder(model_name: str, device: str = "cpu") -> Embedder:
    """Return the singleton embedder for `model_name` on `device`."""
    return Embedder(model_name, device)
