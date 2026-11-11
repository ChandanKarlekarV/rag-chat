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
    highlight: list[int] | None = None  # [start, end) of the extracted answer span inside 	ext, if any
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


# ---------- auth ----------

ModeChoice = Literal["auto", "local", "ollama", "api"]


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
    preferred_mode: ModeChoice = "auto"
    ollama_model: str | None = None


class PreferencesIn(BaseModel):
    """Body of `PUT /me/preferences` (the API key is never stored server-side)."""

    default_mode: ModeChoice | None = None
    ollama_model: str | None = Field(default=None, max_length=120)


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
    mode: Literal["local", "ollama", "api"] | None = None  # None = saved default, else auto


class MessageOut(BaseModel):
    """A persisted chat message."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    answerable: bool | None = None
    citations: list[Citation] = []
    mode: str | None = None
    label: str | None = None
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


class ModeOut(BaseModel):
    """One answer mode and whether it can run now."""

    mode: Literal["local", "ollama", "api"]
    title: str
    available: bool
    reason: str
    description: str


class ProviderOut(BaseModel):
    """An AI provider whose API keys are accepted."""

    id: str
    name: str
    key_url: str


class ModesResponse(BaseModel):
    """Result of `GET /chat/modes`."""

    modes: list[ModeOut]
    recommended: Literal["local", "ollama", "api"]
    default_mode: ModeChoice
    ollama_models: list[str]
    ollama_model: str
    providers: list[ProviderOut] = []


class TestKeyResponse(BaseModel):
    """Result of `POST /chat/test-key`."""

    ok: bool
    message: str
    provider: str | None = None  # detected or chosen provider id
    provider_name: str | None = None
    models: list[str] = []  # chat models the key can use
    model: str | None = None  # the model that will be used


# ---------- study material ----------

StudyKind = Literal["notes", "sticky", "mindmap", "flow", "glossary", "flashcards"]


class Heading(BaseModel):
    """A heading found in a PDF (1 = top level)."""

    model_config = ConfigDict(frozen=True)

    page: int
    level: int
    title: str


class StudyRequest(BaseModel):
    """Body of `POST /study`."""

    doc_id: str
    kinds: list[StudyKind] = Field(min_length=1, max_length=6)
    mode: Literal["local", "ollama", "api"] | None = None  # None = saved default, else auto


class NoteItem(BaseModel):
    """One short-note bullet."""

    text: str = Field(min_length=1, max_length=600)
    page: int | None = None


class NoteSection(BaseModel):
    """Short notes for one section of the document."""

    title: str = Field(min_length=1, max_length=200)
    page: int | None = None
    points: list[NoteItem] = Field(min_length=1)


class StickyNote(BaseModel):
    """One key fact on a sticky note."""

    title: str = Field(default="Key point", max_length=80)
    text: str = Field(min_length=1, max_length=400)
    page: int | None = None
    tone: int = Field(default=1, ge=1, le=4)  # which of the four note colours


class MindNode(BaseModel):
    """A node of the mind map."""

    label: str = Field(min_length=1, max_length=120)
    page: int | None = None
    children: list["MindNode"] = []


class FlowStep(BaseModel):
    """One step of a process."""

    text: str = Field(min_length=1, max_length=400)
    page: int | None = None


class Flow(BaseModel):
    """A step-by-step process, drawn as a flowchart."""

    title: str = Field(min_length=1, max_length=200)
    steps: list[FlowStep] = Field(min_length=2)


class GlossaryTerm(BaseModel):
    """A key term and its definition."""

    term: str = Field(min_length=1, max_length=120)
    definition: str = Field(min_length=1, max_length=600)
    page: int | None = None


class Flashcard(BaseModel):
    """A question on the front, the answer on the back."""

    front: str = Field(min_length=1, max_length=400)
    back: str = Field(min_length=1, max_length=600)
    page: int | None = None


class StudyPack(BaseModel):
    """Result of `POST /study`."""

    doc_id: str
    filename: str
    title: str
    pages: int
    mode: Literal["local", "ollama", "api"]
    label: str
    kinds: list[StudyKind]
    notes: list[NoteSection] = []
    sticky: list[StickyNote] = []
    mindmap: MindNode | None = None
    flows: list[Flow] = []
    glossary: list[GlossaryTerm] = []
    flashcards: list[Flashcard] = []
    warnings: list[str] = []
    seconds: float


class HealthResponse(BaseModel):
    """Result of `GET /health`."""

    status: Literal["ok", "degraded"]
    database: bool
    redis: bool | None = None  # None when Redis isn't configured
    models_loaded: bool
    local_model: str | None  # the extractive QA model, when loaded
    device: str  # where the local models run: "cuda" (GPU) or "cpu"
    server_key: bool  # the server has its own ANTHROPIC_API_KEY
