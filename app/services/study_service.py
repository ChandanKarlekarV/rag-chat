"""Study material for one of the user's documents (notes, sticky notes, mind map, flows, terms, cards)."""

import logging
import time

from app.config import Settings
from app.db.repositories import DocumentRepo
from app.db.session import Database
from app.errors import AppError, NotFound
from app.generation.base import AnswerMode
from app.generation.generative import GenerativeProvider
from app.generation.llm_base import LLMError
from app.generation.study import LLMStudyBuilder, LocalStudyBuilder, SentenceEmbedder
from app.ingest.cleaner import clean_pages
from app.ingest.loader import IngestError, load_pdf
from app.ingest.outline import document_title, extract_outline
from app.schemas import StudyPack, StudyRequest
from app.services.chat_service import Engine

log = logging.getLogger("rag-chat.study")

# How much document text (characters) each kind of model is given.
CONTEXT_CHARS = {AnswerMode.API.value: 24000, AnswerMode.OLLAMA.value: 12000}


class StudyService:
    """Reads a ready document and builds the requested study material."""

    def __init__(self, db: Database, settings: Settings, embedder: SentenceEmbedder) -> None:
        self.db = db
        self.settings = settings
        self.local = LocalStudyBuilder(embedder)

    def build(self, user_id: str, req: StudyRequest, engine: Engine) -> StudyPack:
        """Study material for `req.doc_id` (owned by `user_id`) made with the engine's mode."""
        if engine.error_code:
            raise AppError(engine.error_code, engine.error_message or "This mode isn't available right now.")
        with self.db.session() as s:
            doc = DocumentRepo(s).get(user_id, req.doc_id)
            if doc is None:
                raise NotFound("not_found", "This document no longer exists.")
            if doc.status != "ready":
                raise AppError("not_ready", "This document is still being prepared. Try again when it's ready.")
            filename, doc_id = doc.filename, doc.id
        path = self.settings.uploads_dir / f"{doc_id}.pdf"
        if not path.exists():
            raise AppError("file_missing", "The original file is gone. Please upload it again.")
        started = time.perf_counter()
        data = path.read_bytes()
        try:
            raw = load_pdf(data, filename)
        except IngestError as exc:
            raise AppError("invalid_file", str(exc)) from exc
        pages = clean_pages(raw)
        meta_title, headings = extract_outline(data)
        title = document_title(meta_title, filename)
        kinds = list(dict.fromkeys(req.kinds))  # unique, in the order asked

        builder: LocalStudyBuilder | LLMStudyBuilder = self.local
        if engine.mode != AnswerMode.LOCAL.value and isinstance(engine.provider, GenerativeProvider):
            builder = LLMStudyBuilder(engine.provider.llm, self.local, CONTEXT_CHARS.get(engine.mode, 12000))
        try:
            content = builder.build(title, pages, raw, headings, kinds)
        except LLMError as exc:
            raise AppError("llm_error", str(exc), 502) from exc
        seconds = round(time.perf_counter() - started, 2)
        log.info("study material built", extra={"doc_id": doc_id, "mode": engine.mode, "seconds": seconds})
        return StudyPack(
            doc_id=doc_id,
            filename=filename,
            title=title,
            pages=len(pages),
            mode=engine.mode,  # type: ignore[arg-type]
            label=builder.label,
            kinds=kinds,
            topics=content.topics,
            notes=content.notes,
            sticky=content.sticky,
            mindmap=content.mindmap,
            flows=content.flows,
            glossary=content.glossary,
            flashcards=content.flashcards,
            warnings=content.warnings,
            seconds=seconds,
        )
