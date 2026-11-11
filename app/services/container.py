"""Builds and holds all long-lived services (models are loaded exactly once per process)."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings
from app.db.session import Database
from app.generation.base import AnswerMode
from app.generation.extractive import LocalExtractiveProvider, SentenceScorer, SpanReader
from app.generation.llm_anthropic import AnthropicLLM
from app.generation.llm_base import LLMClient
from app.generation.llm_openai import OpenAICompatLLM
from app.generation.ollama import OllamaLLM
from app.generation.providers import ApiCredentials, resolve_credentials
from app.generation.router import AnswerRouter, KeyInput
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid import ChunkReranker, HybridRetriever, QueryEmbedder
from app.retrieval.vector_store import VectorStore
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService, Engine
from app.services.coordination import MemoryLocks, MemoryProgress, RedisLocks, RedisProgress
from app.services.ingest_service import DocEmbedder, IngestService
from app.services.rate_limit import MemoryRateLimiter, RateLimiter, RedisRateLimiter
from app.services.retrieval_service import RetrievalService
from app.services.study_service import StudyService

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = ROOT / "samples" / "halcyon_h2_handbook.pdf"

LLMFactory = Callable[[ApiCredentials], LLMClient]
OllamaFactory = Callable[[str], OllamaLLM]


class Embedder(QueryEmbedder, DocEmbedder, Protocol):
    """Structural type: something that embeds both queries and documents."""


class ScoringReranker(ChunkReranker, Protocol):
    """A reranker that can also score raw sentences (used for supporting sentences)."""

    def score(self, question: str, passages: list[str]) -> list[float]: ...


@dataclass
class Container:
    """Everything the HTTP layer needs."""

    settings: Settings
    db: Database
    auth: AuthService
    ingest: IngestService
    retrieval: RetrievalService
    chat: ChatService
    router: AnswerRouter
    study: StudyService
    auth_limiter: RateLimiter
    ask_limiter: RateLimiter
    redis: Any = None
    models_loaded: bool = True
    device: str = "cpu"  # where the local models run

    def engine(
        self, key: KeyInput | None, requested: str | None, preferred: str | None, ollama_model: str | None
    ) -> Engine:
        """Resolve the answer mode for one request into what the chat service needs."""
        pref = AnswerMode(preferred) if preferred in ("local", "ollama", "api") else None
        choice = self.router.pick(AnswerMode(requested) if requested else None, key, pref, ollama_model)
        return Engine(choice.mode.value, choice.provider, choice.rewriter, choice.error_code, choice.error_message)


def make_api_llm(settings: Settings, creds: ApiCredentials) -> LLMClient:
    """The HTTP client for the provider the credentials belong to."""
    if creds.spec.kind == "anthropic":
        return AnthropicLLM(
            creds.key, creds.model or settings.llm_model, settings.llm_max_tokens, settings.llm_timeout_s
        )
    return OpenAICompatLLM(creds.spec, creds.key, creds.model, settings.llm_timeout_s)


def server_credentials(settings: Settings) -> ApiCredentials | None:
    """The server's shared key, if one is configured (ANTHROPIC_API_KEY, else LLM_API_KEY)."""
    if settings.anthropic_api_key:
        return ApiCredentials(settings.anthropic_api_key, "anthropic", settings.llm_model)
    if settings.llm_api_key:
        return resolve_credentials(settings.llm_api_key, settings.llm_provider, settings.llm_api_model)
    return None


def build_container(
    settings: Settings,
    embedder: Embedder | None = None,
    reranker: ScoringReranker | None = None,
    llm_factory: LLMFactory | None = None,
    reader: SpanReader | None = None,
    sample_pdf: Path = SAMPLE_PDF,
    redis_client: Any = None,
    ollama_factory: OllamaFactory | None = None,
) -> Container:
    """Wire the app. Tests pass light fakes for the models, the reader and the LLM/Ollama factories."""
    device = "cpu"
    if embedder is None or reranker is None or (reader is None and settings.load_local_model):
        from app.devices import limit_cpu_threads, resolve_device  # heavy imports only for real models

        limit_cpu_threads(settings.torch_threads)
        device = resolve_device(settings.device)
        if embedder is None or reranker is None:
            from app.retrieval.embedder import get_embedder
            from app.retrieval.reranker import get_reranker

            embedder = embedder or get_embedder(settings.embed_model, device)
            reranker = reranker or get_reranker(settings.rerank_model, device)
        if reader is None and settings.load_local_model:
            from app.generation.reader import get_reader

            reader = get_reader(settings.qa_model, device)
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

    scorer: SentenceScorer = reranker.score
    local = (
        LocalExtractiveProvider(
            reader, settings.qa_model, scorer, settings.min_qa_score, settings.qa_weight, settings.max_sentences
        )
        if reader is not None
        else None
    )
    api_factory = llm_factory or (lambda creds: make_api_llm(settings, creds))
    if ollama_factory is None and settings.ollama_enabled:
        ollama_factory = lambda model: OllamaLLM(settings.ollama_base_url, model)  # noqa: E731
    router = AnswerRouter(local, api_factory, server_credentials(settings), ollama_factory, settings.ollama_model)
    return Container(
        settings=settings,
        db=db,
        auth=AuthService(db, settings, ingest, sample_pdf),
        ingest=ingest,
        retrieval=retrieval,
        chat=ChatService(db, retrieval, settings.min_rerank_score),
        router=router,
        study=StudyService(db, settings, embedder),
        auth_limiter=auth_limiter,
        ask_limiter=ask_limiter,
        redis=redis_client,
        device=device,
    )
