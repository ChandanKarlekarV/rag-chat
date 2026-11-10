"""Application settings loaded from environment variables / `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_SECRET = "dev-only-insecure-secret-change-me-before-deploying"  # refused when APP_ENV=prod


class Settings(BaseSettings):
    """All tunables. Override any field with an env var of the same name."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"  # "prod" enables Secure cookies and refuses the dev secret
    secret_key: str = DEV_SECRET
    database_url: str = ""  # empty -> sqlite file inside data_dir
    data_dir: Path = Path("./data")

    anthropic_api_key: str = ""  # optional shared key; users can bring their own in the UI
    llm_provider: str = "anthropic"  # "anthropic" | "fake" (offline demo, extractive answers)
    llm_model: str = "claude-sonnet-5-5"  # verified on platform.claude.com models overview
    llm_max_tokens: int = 2048
    llm_timeout_s: float = 60.0

    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieve_k: int = 20
    final_k: int = 5
    rrf_k: int = 60
    min_rerank_score: float = -5.0

    max_upload_mb: int = 20
    max_docs_user: int = 50
    max_docs_guest: int = 10
    guest_ttl_hours: int = 24
    session_days: int = 7
    auth_rate_per_min: int = 5
    ask_rate_per_min: int = 20

    @property
    def is_prod(self) -> bool:
        """True when running in production mode."""
        return self.app_env.lower() == "prod"

    @property
    def sqlalchemy_url(self) -> str:
        """Database URL, defaulting to a SQLite file in the data directory."""
        return self.database_url or f"sqlite:///{(self.data_dir / 'app.db').as_posix()}"

    @property
    def uploads_dir(self) -> Path:
        """Directory where uploaded PDFs are stored under random names."""
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        """Directory for the persistent Chroma database."""
        return self.data_dir / "chroma"

    @property
    def bm25_dir(self) -> Path:
        """Directory holding one pickled BM25 corpus per user."""
        return self.data_dir / "bm25"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
