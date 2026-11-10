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

    # --- answer modes (ANSWER_MODES.md) ---
    # Local (extractive) mode. Most accurate on our eval (see README); smaller options:
    # deepset/roberta-base-squad2, distilbert-base-cased-distilled-squad (no "no answer" training).
    qa_model: str = "deepset/deberta-v3-large-squad2"
    min_qa_score: float = 0.05  # probability that the span beats "no answer"; tuned by scripts/evaluate.py
    qa_weight: float = 0.6  # span score = QA_WEIGHT * qa_score + (1 - QA_WEIGHT) * normalized rerank score
    max_sentences: int = 3  # supporting sentences quoted under the answer
    load_local_model: bool = True  # False skips loading the QA model (Local mode then unavailable)

    # Ollama mode (optional; the app checks whether it's reachable).
    ollama_base_url: str = "http://localhost:11434"  # in Docker: http://host.docker.internal:11434
    ollama_model: str = "llama3.2"
    ollama_enabled: bool = True

    # API mode. Optional shared key; users can paste their own in Settings (never stored server-side).
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5-5"  # verified on platform.claude.com models overview
    llm_max_tokens: int = 2048
    llm_timeout_s: float = 60.0

    # Where local models run: "auto" (NVIDIA GPU if available, else CPU), "cuda" or "cpu".
    device: str = "auto"
    torch_threads: int = 0  # cap CPU threads used by the models (0 = all cores)

    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieve_k: int = 20
    final_k: int = 5
    rrf_k: int = 60
    min_rerank_score: float = -5.0

    # Multi-server mode (all optional; empty = single-process defaults).
    redis_url: str = ""  # shared rate limits, ingest progress and per-user locks
    chroma_url: str = ""  # Chroma server instead of the embedded store (required with >1 worker)

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
