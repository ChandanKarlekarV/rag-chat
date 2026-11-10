"""Document ingestion: validate → store → (background) read → split → embed → index."""

import logging
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.db.models import Document, User
from app.db.repositories import DocumentRepo
from app.db.session import Database
from app.errors import AppError, Conflict, NotFound, TooMany
from app.ingest.chunker import chunk_pages
from app.ingest.cleaner import clean_pages
from app.ingest.loader import IngestError, load_pdf, sha256_of, validate_pdf_bytes
from app.retrieval.bm25_store import BM25Store
from app.retrieval.vector_store import VectorStore
from app.schemas import DocumentStatus
from app.services.coordination import LockProvider, MemoryLocks, MemoryProgress, ProgressStore

log = logging.getLogger("rag-chat.ingest")
EMBED_BATCH = 32


class DocEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class IngestService:
    """Owns the write path for documents and keeps SQLite, Chroma, BM25 and files consistent."""

    def __init__(
        self,
        db: Database,
        vectors: VectorStore,
        bm25: BM25Store,
        embedder: DocEmbedder,
        settings: Settings,
        progress: ProgressStore | None = None,
        locks: LockProvider | None = None,
    ) -> None:
        self.db = db
        self.vectors = vectors
        self.bm25 = bm25
        self.embedder = embedder
        self.settings = settings
        self.progress = progress or MemoryProgress()
        self.locks = locks or MemoryLocks()
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    def _lock(self, user_id: str) -> AbstractContextManager[object]:
        return self.locks.lock(f"user:{user_id}")

    def _file(self, doc_id: str) -> Path:
        return self.settings.uploads_dir / f"{doc_id}.pdf"  # random id, never the user's filename

    # ---- write path ----

    def submit(self, user: User, data: bytes, filename: str) -> Document:
        """Validate and register an upload. The caller schedules `process(doc.id)` afterwards."""
        filename = Path(filename or "document.pdf").name[:255] or "document.pdf"
        try:
            validate_pdf_bytes(data, filename, self.settings.max_upload_mb)
        except IngestError as exc:
            raise AppError("invalid_file", str(exc)) from exc
        digest = sha256_of(data)
        limit = self.settings.max_docs_guest if user.is_guest else self.settings.max_docs_user
        with self.db.session() as s:
            repo = DocumentRepo(s)
            if existing := repo.by_sha(user.id, digest):
                raise Conflict("duplicate", f"You already uploaded this PDF as '{existing.filename}'.")
            if repo.count(user.id) >= limit:
                raise TooMany("doc_limit", f"You can keep up to {limit} documents. Delete one to upload another.")
            doc = repo.add(Document(user_id=user.id, filename=filename, sha256=digest, status="processing"))
        self._file(doc.id).write_bytes(data)
        self.progress.set(doc.id, "queued", 0)
        return doc

    def process(self, doc_id: str) -> None:
        """Run the ingest stages for a submitted document. Never raises; failures are recorded."""
        with self.db.session() as s:
            doc = DocumentRepo(s).by_id_unscoped(doc_id)
            if doc is None:
                return
            user_id, filename = doc.user_id, doc.filename
        try:
            self._set(doc_id, "reading", 5)
            pages = clean_pages(load_pdf(self._file(doc_id).read_bytes(), filename))
            if not any(p.text.strip() for p in pages):
                raise IngestError("This PDF has no selectable text (it may be a scan). Try a text-based PDF.")
            self._set(doc_id, "splitting", 20)
            chunks = chunk_pages(pages, doc_id, filename, self.settings.chunk_size, self.settings.chunk_overlap)
            embeddings: list[list[float]] = []
            for start in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[start : start + EMBED_BATCH]
                embeddings += self.embedder.embed_documents([c.text for c in batch])
                self._set(doc_id, "understanding", 25 + int(70 * (start + len(batch)) / max(len(chunks), 1)))
            with self._lock(user_id):
                with self.db.session() as s:
                    if DocumentRepo(s).by_id_unscoped(doc_id) is None:
                        return  # deleted while processing
                self.vectors.add_chunks(user_id, chunks, embeddings)
                self.bm25.rebuild(user_id, self.vectors.user_chunks(user_id))
                with self.db.session() as s:
                    doc = DocumentRepo(s).by_id_unscoped(doc_id)
                    if doc is not None:
                        doc.pages, doc.chunks, doc.status, doc.error = len(pages), len(chunks), "ready", None
            self._set(doc_id, "ready", 100)
            log.info("document ready", extra={"doc_id": doc_id, "pages": len(pages), "chunks": len(chunks)})
        except IngestError as exc:
            self._fail(doc_id, user_id, str(exc))
        except Exception:
            log.exception("ingest failed", extra={"doc_id": doc_id})
            self._fail(doc_id, user_id, "Something went wrong while reading this PDF. Try uploading it again.")

    def retry(self, user_id: str, doc_id: str) -> Document:
        """Mark a failed document for another ingest attempt. The caller schedules `process`."""
        with self.db.session() as s:
            doc = DocumentRepo(s).get(user_id, doc_id)
            if doc is None:
                raise NotFound("not_found", "This document no longer exists.")
            if doc.status != "failed":
                raise AppError("not_failed", "Only documents that failed can be retried.")
            if not self._file(doc_id).exists():
                raise AppError("file_missing", "The original file is gone. Please upload it again.")
            doc.status, doc.error = "processing", None
        self.progress.set(doc_id, "queued", 0)
        return doc

    def ingest_now(self, user: User, data: bytes, filename: str) -> Document:
        """Submit and process synchronously (used to preload the guest sample)."""
        doc = self.submit(user, data, filename)
        self.process(doc.id)
        return doc

    def _set(self, doc_id: str, stage: str, percent: int) -> None:
        self.progress.set(doc_id, stage, min(percent, 100))

    def _fail(self, doc_id: str, user_id: str, message: str) -> None:
        self._set(doc_id, "failed", 100)
        with self._lock(user_id):
            self.vectors.delete_doc(user_id, doc_id)
            with self.db.session() as s:
                doc = DocumentRepo(s).by_id_unscoped(doc_id)
                if doc is not None:
                    doc.status, doc.error = "failed", message

    # ---- read / delete ----

    def list_documents(self, user_id: str) -> list[Document]:
        """The user's documents, newest first."""
        with self.db.session() as s:
            return DocumentRepo(s).for_user(user_id)

    def status(self, user_id: str, doc_id: str) -> DocumentStatus:
        """Current ingest stage for one of the user's documents."""
        with self.db.session() as s:
            doc = DocumentRepo(s).get(user_id, doc_id)
            if doc is None:
                raise NotFound("not_found", "This document no longer exists.")
            prog = self.progress.get(doc_id)
            if doc.status != "processing":
                stage, pct = doc.status, 100
            else:
                stage, pct = prog or ("queued", 0)
            return DocumentStatus(
                id=doc.id,
                status=doc.status,
                stage=stage,
                progress=pct,  # type: ignore[arg-type]
                error=doc.error,
                pages=doc.pages,
                chunks=doc.chunks,
            )

    def delete(self, user_id: str, doc_id: str) -> None:
        """Delete one of the user's documents everywhere."""
        with self._lock(user_id):
            with self.db.session() as s:
                repo = DocumentRepo(s)
                doc = repo.get(user_id, doc_id)
                if doc is None:
                    raise NotFound("not_found", "This document no longer exists.")
                repo.delete(doc)
            self.vectors.delete_doc(user_id, doc_id)
            self.bm25.rebuild(user_id, self.vectors.user_chunks(user_id))
        self._file(doc_id).unlink(missing_ok=True)
        self.progress.delete(doc_id)

    def purge_user(self, user_id: str) -> None:
        """Remove a user's vectors, BM25 index and files (rows are removed by the caller)."""
        with self.db.session() as s:
            doc_ids = [d.id for d in DocumentRepo(s).for_user(user_id)]
        with self._lock(user_id):
            self.vectors.delete_user(user_id)
            self.bm25.delete_user(user_id)
        for doc_id in doc_ids:
            self._file(doc_id).unlink(missing_ok=True)
            self.progress.delete(doc_id)
