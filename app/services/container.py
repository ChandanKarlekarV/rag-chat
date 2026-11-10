"""Builds and holds all long-lived services (models are loaded exactly once per process)."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings
from app.db.session import Database
from app.generation.answerer import Answerer, LLMAnswerer, LocalAnswerer, SpanReader
from app.generation.factory import make_llm
from app.generation.llm_base import LLMClient
from app.generation.rewrite import HeuristicRewriter, LLMRewriter, QueryRewriter
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid import ChunkReranker, HybridRetriever, QueryEmbedder
from app.retrieval.vector_store import VectorStore
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.coordination import MemoryLocks, MemoryProgress, RedisLocks, RedisProgress
from app.services.ingest_service import DocEmbedder, IngestService
from app.services.rate_limit import MemoryRateLimiter, RateLimiter, RedisRateLimiter
from app.services.retrieval_service import RetrievalService

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = ROOT / "samples" / "halcyon_h2_handbook.pdf"

LLMFactory = Callable[[str], LLMClient]


class Embedder(QueryEmbedder, DocEmbedder, Protocol):
    """Structural type: something that embeds both queries and documents."""


@dataclass(frozen=True)
class Engine:
    """How a request will be answered."""

    answerer: Answerer | None
    rewriter: QueryRewriter
    source: str  # "browser" | "server" | "local" | "none"


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
    local: LocalAnswerer | None = None
    redis: Any = None
    models_loaded: bool = True

    def engine(self, user_key: str | None) -> Engine:
        """Pick the answerer for this request.

        A key sent by the browser always wins (Claude). Otherwise the server's own key, then the
        local reader, depending on `LLM_PROVIDER`.
        """
        provider = self.settings.llm_provider
        key = (user_key or "").strip()
        if provider == "local":
            return self._local()
        if key and provider in ("auto", "anthropic"):
            llm = self.llm_factory(key)
            return Engine(LLMAnswerer(llm), LLMRewriter(llm), "browser")
        if self.server_llm is not None:
            return Engine(LLMAnswerer(self.server_llm), LLMRewriter(self.server_llm), "server")
        if provider == "auto":
            return self._local()
        return Engine(None, HeuristicRewriter(), "none")

    def _local(self) -> Engine:
        return Engine(self.local, HeuristicRewriter(), "local" if self.local else "none")


def build_container(
    settings: Settings,
    embedder: Embedder | None = None,
    reranker: ChunkReranker | None = None,
    llm_factory: LLMFactory | None = None,
    reader: SpanReader | None = None,
    sample_pdf: Path = SAMPLE_PDF,
    redis_client: Any = None,
) -> Container:
    """Wire the app. Tests pass light fakes for the models, the reader and the LLM factory."""
    if embedder is None or reranker is None:
        from app.retrieval.embedder import get_embedder  # heavy imports only when real models are used
        from app.retrieval.reranker import get_reranker

        embedder = embedder or get_embedder(settings.embed_model)
        reranker = reranker or get_reranker(settings.rerank_model)
    if reader is None and settings.llm_provider in ("auto", "local"):
        from app.generation.reader import get_reader

        reader = get_reader(settings.reader_model)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.sqlalchemy_url)
    db.migrate()
    vectors = VectorStore(settings.chroma_dir, settings.chroma_url)
    bm25 = BM25Store(settings.bm25_dir)
    if redis_client is None and settings.redis_url:
        import redis as redis_lib

        redis_client = redis_lib.Redis.from_url(settings.redis_url)
    if redis_client is not None:
        progress, locks = RedisProgress(redis_client), RedisLocks(redis_client)
        auth_limiter: RateLimiter = RedisRateLimiter(redis_client, settings.auth_rate_per_min)
        ask_limiter: RateLimiter = RedisRateLimiter(redis_client, settings.ask_rate_per_min)
    else:
        progress, locks = MemoryProgress(), MemoryLocks()
        auth_limiter = MemoryRateLimiter(settings.auth_rate_per_min)
        ask_limiter = MemoryRateLimiter(settings.ask_rate_per_min)
    ingest = IngestService(db, vectors, bm25, embedder, settings, progress, locks)
    retriever = HybridRetriever(embedder, vectors, bm25, reranker, settings.retrieve_k, settings.rrf_k)
    retrieval = RetrievalService(retriever, settings.final_k)
    factory = llm_factory or (lambda key: make_llm(settings, key))
    server_llm = None
    if settings.llm_provider == "fake" or settings.anthropic_api_key:
        server_llm = factory(settings.anthropic_api_key)
    local = (
        LocalAnswerer(reader, settings.reader_model, settings.reader_min_score, settings.reader_rerank_weight)
        if reader is not None
        else None
    )
    return Container(
        settings=settings,
        db=db,
        auth=AuthService(db, settings, ingest, sample_pdf),
        ingest=ingest,
        retrieval=retrieval,
        chat=ChatService(db, retrieval, settings.min_rerank_score),
        llm_factory=factory,
        auth_limiter=auth_limiter,
        ask_limiter=ask_limiter,
        redis=redis_client,
        server_llm=server_llm,
        local=local,
    )
