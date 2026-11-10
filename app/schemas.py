"""Pydantic v2 models for the pipeline and the HTTP API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------- pipeline ----------


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


class RetrievedChunk(BaseModel):
    """A chunk plus every score it collected on its way through the pipeline."""

    chunk: Chunk
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    """A source shown to the user."""

    n: int
    doc_id: str
    filename: str
    page_start: int
    page_end: int
    snippet: str  # first ~300 chars, for compact previews
    text: str  # full chunk text, shown in the source drawer
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


# ---------- auth ----------


class Credentials(BaseModel):
    """Register / login body."""

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class UserOut(BaseModel):
    """The signed-in user."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    is_guest: bool


# ---------- documents ----------

Status = Literal["processing", "ready", "failed"]
Stage = Literal["queued", "reading", "splitting", "understanding", "ready", "failed"]


class DocumentOut(BaseModel):
    """A document as listed in the sidebar."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    pages: int
    chunks: int
    status: Status
    error: str | None = None
    created_at: datetime


class DocumentStatus(BaseModel):
    """Ingest progress for polling."""

    id: str
    status: Status
    stage: Stage
    progress: int = Field(ge=0, le=100)
    error: str | None = None
    pages: int = 0
    chunks: int = 0


# ---------- chat ----------


class AskRequest(BaseModel):
    """Body of `POST /chat/ask`."""

    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    doc_ids: list[str] | None = None


class MessageOut(BaseModel):
    """A persisted chat message."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    answerable: bool | None = None
    citations: list[Citation] = []
    created_at: datetime


class ConversationOut(BaseModel):
    """A conversation in the history list."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime


class ConversationDetail(ConversationOut):
    """A conversation with its messages."""

    messages: list[MessageOut]


class RenameRequest(BaseModel):
    """Body of `PATCH /conversations/{id}`."""

    title: str = Field(min_length=1, max_length=120)


class LLMCheckResponse(BaseModel):
    """Result of `POST /llm/check`."""

    ok: bool
    source: Literal["browser", "server", "local", "none"]
    provider: str
    model: str
    message: str


class HealthResponse(BaseModel):
    """Result of `GET /health`."""

    status: Literal["ok", "degraded"]
    database: bool
    redis: bool | None = None  # None when Redis isn't configured
    models_loaded: bool
    llm_configured: bool  # True when the server has its own key (or the fake provider)
    llm_provider: str
    llm_model: str
