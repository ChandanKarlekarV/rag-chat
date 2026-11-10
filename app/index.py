"""Document index: keeps SQLite, Chroma, BM25 and the uploads folder consistent."""

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.db import DocumentDB
from app.ingest.chunker import chunk_pages
from app.ingest.cleaner import clean_pages
from app.ingest.loader import IngestError, load_pdf, sha256_of, validate_pdf_bytes
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.schemas import DocumentMeta


class DocumentIndex:
    """Ingests and deletes documents. Writes are serialised with a lock."""

    def __init__(
        self,
        db: DocumentDB,
        vectors: VectorStore,
        bm25: BM25Store,
        embedder: Embedder,
        uploads_dir: Path,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        max_upload_mb: int = 20,
    ) -> None:
        self.db = db
        self.vectors = vectors
        self.bm25 = bm25
        self.embedder = embedder
        self.uploads_dir = uploads_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_upload_mb = max_upload_mb
        self._lock = threading.Lock()
        uploads_dir.mkdir(parents=True, exist_ok=True)

    def add_pdf(self, data: bytes, filename: str) -> tuple[DocumentMeta, bool]:
        """Ingest a PDF. Returns (meta, duplicate) where duplicate=True means it was already indexed."""
        filename = Path(filename).name or "document.pdf"
        validate_pdf_bytes(data, filename, self.max_upload_mb)
        digest = sha256_of(data)
        if existing := self.db.get_by_sha(digest):
            return existing, True

        pages = clean_pages(load_pdf(data, filename))
        if not any(p.text.strip() for p in pages):
            raise IngestError(
                f"'{filename}' has no extractable text. It is probably a scanned PDF; OCR is not supported yet."
            )
        doc_id = uuid.uuid4().hex
        chunks = chunk_pages(pages, doc_id, filename, self.chunk_size, self.chunk_overlap)
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        meta = DocumentMeta(
            doc_id=doc_id,
            filename=filename,
            pages=len(pages),
            chunks=len(chunks),
            uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        with self._lock:
            if existing := self.db.get_by_sha(digest):  # lost a race with an identical upload
                return existing, True
            self.vectors.add_chunks(chunks, embeddings)
            try:
                self.db.insert(meta, digest)
            except Exception:
                self.vectors.delete_doc(doc_id)
                raise
            (self.uploads_dir / f"{doc_id}.pdf").write_bytes(data)
            self.bm25.rebuild(self.vectors.all_chunks())
        return meta, False

    def delete(self, doc_id: str) -> bool:
        """Remove a document everywhere. Returns False if it did not exist."""
        with self._lock:
            if not self.db.delete(doc_id):
                return False
            self.vectors.delete_doc(doc_id)
            (self.uploads_dir / f"{doc_id}.pdf").unlink(missing_ok=True)
            self.bm25.rebuild(self.vectors.all_chunks())
        return True

    def list(self) -> list[DocumentMeta]:
        """All indexed documents, newest first."""
        return self.db.list()
