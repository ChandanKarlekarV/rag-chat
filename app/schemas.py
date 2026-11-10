"""Pydantic v2 data models shared across ingest, retrieval, generation and the API."""

from typing import Optional

from pydantic import BaseModel, Field


class PageText(BaseModel):
    """Cleaned text of one PDF page."""

    page: int  # 1-indexed
    text: str


class Chunk(BaseModel):
    """A retrievable slice of a document."""

    chunk_id: str  # f"{doc_id}:{index}"
    doc_id: str
    filename: str
    page_start: int
    page_end: int
    index: int
    text: str
    token_count: int


class DocumentMeta(BaseModel):
    """Row of the `documents` table as exposed by the API."""

    doc_id: str
    filename: str
    pages: int
    chunks: int
    uploaded_at: str


class RetrievedChunk(BaseModel):
    """A chunk plus every score it collected on its way through the pipeline."""

    chunk: Chunk
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class Citation(BaseModel):
    """A source the answer actually cited."""

    n: int
    filename: str
    page_start: int
    page_end: int
    snippet: str  # first ~300 chars, for compact previews
    text: str = ""  # full chunk text, shown in the source drawer
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class AskRequest(BaseModel):
    """Body of `POST /ask`."""

    question: str = Field(min_length=1, max_length=2000)
    doc_ids: Optional[list[str]] = None
    top_k: int = Field(default=5, ge=1, le=10)


class AskResponse(BaseModel):
    """Final, non-streamed shape of an answer (also the payload of the SSE `done` event)."""

    answer: str
    citations: list[Citation]
    answerable: bool


class UploadResponse(BaseModel):
    """Result of `POST /upload`."""

    document: DocumentMeta
    duplicate: bool = False


class HealthResponse(BaseModel):
    """Result of `GET /health`."""

    status: str
    documents: int
    llm_configured: bool
