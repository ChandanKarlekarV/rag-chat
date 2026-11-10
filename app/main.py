"""FastAPI application: routes, model loading at startup, static frontend."""

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.db import DocumentDB
from app.generation.answer import NO_DOCS_MESSAGE, answer_stream
from app.generation.llm import LLM, AnthropicLLM, LLMError
from app.index import DocumentIndex
from app.ingest.loader import IngestError
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embedder import get_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import get_reranker
from app.retrieval.vector_store import VectorStore
from app.schemas import AskRequest, DocumentMeta, HealthResponse, LLMCheckResponse, UploadResponse

log = logging.getLogger("rag-chat")
STATIC_DIR = Path(__file__).parent / "static"


API_KEY_HEADER = "X-Anthropic-Api-Key"
LLMFactory = Callable[[str], LLM]


@dataclass
class Services:
    """Long-lived components shared by all requests."""

    settings: Settings
    index: DocumentIndex
    retriever: HybridRetriever
    llm: LLM | None  # built from the server's .env key, if any
    llm_factory: LLMFactory  # builds an LLM from a key the user pasted into the UI

    def resolve_llm(self, user_key: str | None) -> tuple[LLM | None, str]:
        """Prefer the key sent by the browser; fall back to the server key. Returns (llm, source)."""
        if user_key and user_key.strip():
            return self.llm_factory(user_key.strip()), "browser"
        return self.llm, ("server" if self.llm else "none")


def anthropic_factory(settings: Settings) -> LLMFactory:
    """Factory that creates an Anthropic client for a given key with the configured model."""
    return lambda key: AnthropicLLM(key, settings.llm_model, settings.llm_max_tokens, settings.llm_timeout_s)


def build_services(settings: Settings, llm: LLM | None = None) -> Services:
    """Load models once and wire the stores together."""
    embedder = get_embedder(settings.embed_model)
    reranker = get_reranker(settings.rerank_model)
    vectors = VectorStore(settings.chroma_dir)
    bm25 = BM25Store(settings.bm25_path)
    if not bm25.chunk_ids and vectors.count():
        bm25.rebuild(vectors.all_chunks())
    index = DocumentIndex(
        DocumentDB(settings.sqlite_path),
        vectors,
        bm25,
        embedder,
        settings.uploads_dir,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.max_upload_mb,
    )
    retriever = HybridRetriever(embedder, vectors, bm25, reranker, settings.candidates, settings.rrf_k)
    factory = anthropic_factory(settings)
    if llm is None and settings.anthropic_api_key:
        llm = factory(settings.anthropic_api_key)
    return Services(settings, index, retriever, llm, factory)


def create_app(services: Services | None = None) -> FastAPI:
    """Application factory. Tests inject `services`; production builds them at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services or build_services(get_settings())
        log.info("rag-chat ready")
        yield

    # No CORS middleware on purpose: the UI is same-origin, and other sites must not be able to
    # spend the server's key or relay a user's key through this API.
    app = FastAPI(title="rag-chat", version="1.0.0", description="Chat with your PDFs.", lifespan=lifespan)

    def svc(request: Request) -> Services:
        return request.app.state.services

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        s = svc(request)
        return HealthResponse(
            status="ok",
            documents=s.index.db.count(),
            llm_configured=s.llm is not None,
            llm_model=s.settings.llm_model,
        )

    @app.post("/llm/check", response_model=LLMCheckResponse)
    def llm_check(
        request: Request, user_key: str | None = Header(default=None, alias=API_KEY_HEADER)
    ) -> LLMCheckResponse:
        """Check whether the browser's key (or else the server's key) can reach the model."""
        s = svc(request)
        llm, source = s.resolve_llm(user_key)
        model = s.settings.llm_model
        if llm is None:
            return LLMCheckResponse(ok=False, source=source, model=model, message="No API key has been added yet.")
        try:
            llm.check()
        except LLMError as exc:
            return LLMCheckResponse(ok=False, source=source, model=model, message=str(exc))
        return LLMCheckResponse(ok=True, source=source, model=model, message=f"Connected to {model}.")

    @app.post("/upload", response_model=UploadResponse)
    def upload(request: Request, file: UploadFile = File(...)) -> UploadResponse:
        s = svc(request)
        limit = s.settings.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)
        try:
            meta, duplicate = s.index.add_pdf(data, file.filename or "document.pdf")
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return UploadResponse(document=meta, duplicate=duplicate)

    @app.get("/documents", response_model=list[DocumentMeta])
    def documents(request: Request) -> list[DocumentMeta]:
        return svc(request).index.list()

    @app.delete("/documents/{doc_id}")
    def delete_document(request: Request, doc_id: str) -> dict[str, str]:
        if not svc(request).index.delete(doc_id):
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"status": "deleted", "doc_id": doc_id}

    @app.post("/ask")
    def ask(
        request: Request, body: AskRequest, user_key: str | None = Header(default=None, alias=API_KEY_HEADER)
    ) -> StreamingResponse:
        s = svc(request)
        llm, _ = s.resolve_llm(user_key)

        def events() -> Iterator[str]:
            if s.index.db.count() == 0:
                yield _sse("error", {"message": NO_DOCS_MESSAGE})
                return
            try:
                top_k = body.top_k or s.settings.top_k
                sources = s.retriever.retrieve(body.question, top_k, body.doc_ids or None)
            except Exception:
                log.exception("retrieval failed")
                yield _sse("error", {"message": "Search failed unexpectedly. Please try again."})
                return
            for ev in answer_stream(body.question, sources, llm, s.settings.min_rerank_score):
                yield _sse(ev["event"], ev["data"])

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers", "chromadb"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
app = create_app()
