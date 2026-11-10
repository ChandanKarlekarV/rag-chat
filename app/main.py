"""FastAPI application: routes, model loading at startup, static frontend."""

import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.db import DocumentDB
from app.generation.answer import NO_DOCS_MESSAGE, answer_stream
from app.generation.llm import LLM, AnthropicLLM
from app.index import DocumentIndex
from app.ingest.loader import IngestError
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embedder import get_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import get_reranker
from app.retrieval.vector_store import VectorStore
from app.schemas import AskRequest, DocumentMeta, HealthResponse, UploadResponse

log = logging.getLogger("rag-chat")
STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class Services:
    """Long-lived components shared by all requests."""

    settings: Settings
    index: DocumentIndex
    retriever: HybridRetriever
    llm: LLM | None


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
    if llm is None and settings.anthropic_api_key:
        llm = AnthropicLLM(settings.anthropic_api_key, settings.llm_model, settings.llm_max_tokens, settings.llm_timeout_s)
    return Services(settings, index, retriever, llm)


def create_app(services: Services | None = None) -> FastAPI:
    """Application factory. Tests inject `services`; production builds them at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services or build_services(get_settings())
        log.info("rag-chat ready")
        yield

    app = FastAPI(title="rag-chat", version="1.0.0", description="Chat with your PDFs.", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def svc(request: Request) -> Services:
        return request.app.state.services

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        s = svc(request)
        return HealthResponse(
            status="ok",
            documents=s.index.db.count(),
            llm_configured=s.llm is not None,
        )

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
    def ask(request: Request, body: AskRequest) -> StreamingResponse:
        s = svc(request)

        def events() -> Iterator[str]:
            if s.index.db.count() == 0:
                yield _sse("error", {"message": NO_DOCS_MESSAGE})
                return
            try:
                sources = s.retriever.retrieve(body.question, body.top_k, body.doc_ids or None)
            except Exception:
                log.exception("retrieval failed")
                yield _sse("error", {"message": "Search failed unexpectedly. Please try again."})
                return
            for ev in answer_stream(body.question, sources, s.llm, s.settings.min_rerank_score):
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
app = create_app()
