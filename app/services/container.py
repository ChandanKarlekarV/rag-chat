"""Builds and holds all long-lived services (models are loaded exactly once)."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.db.session import Database
from app.generation.factory import make_llm
from app.generation.llm_base import LLMClient
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid import ChunkReranker, HybridRetriever, QueryEmbedder
from app.retrieval.vector_store import VectorStore
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.ingest_service import DocEmbedder, IngestService
from app.services.rate_limit import RateLimiter
from app.services.retrieval_service import RetrievalService

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = ROOT / "samples" / "halcyon_h2_handbook.pdf"

LLMFactory = Callable[[str], LLMClient]


class Embedder(QueryEmbedder, DocEmbedder, Protocol):
    """Structural type: something that embeds both queries and documents."""


@dataclass
class Container:
    """Everything the HTTP layer needs."""

    settings: Settings
    db: Database
    auth: AuthService
    ingest: IngestService
    retrieval: RetrievalService
    chat: ChatService
    llm_factory: LLMFactory
    auth_limiter: RateLimiter
    ask_limiter: RateLimiter
    server_llm: LLMClient | None = None
    models_loaded: bool = True

    def resolve_llm(self, user_key: str | None) -> tuple[LLMClient | None, str]:
        """Prefer the key the browser sent, else the server's own LLM. Returns (llm, source)."""
        if user_key and user_key.strip() and self.settings.llm_provider != "fake":
            return self.llm_factory(user_key.strip()), "browser"
        return self.server_llm, ("server" if self.server_llm else "none")


def build_container(
    settings: Settings,
    embedder: Embedder | None = None,
    reranker: ChunkReranker | None = None,
    llm_factory: LLMFactory | None = None,
    sample_pdf: Path = SAMPLE_PDF,
) -> Container:
    """Wire the app. Tests pass light fakes for the models and the LLM factory."""
    if embedder is None or reranker is None:
        from app.retrieval.embedder import get_embedder  # heavy imports only when real models are used
        from app.retrieval.reranker import get_reranker

        embedder = embedder or get_embedder(settings.embed_model)
        reranker = reranker or get_reranker(settings.rerank_model)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.sqlalchemy_url)
    db.create_all()
    vectors = VectorStore(settings.chroma_dir)
    bm25 = BM25Store(settings.bm25_dir)
    ingest = IngestService(db, vectors, bm25, embedder, settings)
    retriever = HybridRetriever(embedder, vectors, bm25, reranker, settings.retrieve_k, settings.rrf_k)
    retrieval = RetrievalService(retriever, settings.final_k)
    factory = llm_factory or (lambda key: make_llm(settings, key))
    server_llm = None
    if settings.llm_provider == "fake" or settings.anthropic_api_key:
        server_llm = factory(settings.anthropic_api_key)
    return Container(
        settings=settings,
        db=db,
        auth=AuthService(db, settings, ingest, sample_pdf),
        ingest=ingest,
        retrieval=retrieval,
        chat=ChatService(db, retrieval, settings.min_rerank_score),
        llm_factory=factory,
        auth_limiter=RateLimiter(settings.auth_rate_per_min),
        ask_limiter=RateLimiter(settings.ask_rate_per_min),
        server_llm=server_llm,
    )
