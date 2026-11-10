"""Application settings loaded from environment variables / `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables for the RAG pipeline. Override any field via env var of the same name."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = 2048
    llm_timeout_s: float = 60.0

    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    candidates: int = 20
    rrf_k: int = 60
    min_rerank_score: float = -2.0

    max_upload_mb: int = 20
    data_dir: Path = Path("./data")

    @property
    def uploads_dir(self) -> Path:
        """Directory where raw uploaded PDFs are kept."""
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        """Directory for the persistent Chroma database."""
        return self.data_dir / "chroma"

    @property
    def sqlite_path(self) -> Path:
        """Path of the SQLite metadata database."""
        return self.data_dir / "rag.sqlite3"

    @property
    def bm25_path(self) -> Path:
        """Path of the pickled BM25 corpus."""
        return self.data_dir / "bm25.pkl"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
