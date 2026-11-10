"""SQLite helpers for the `documents` metadata table."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.schemas import DocumentMeta

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  pages INTEGER,
  chunks INTEGER,
  sha256 TEXT UNIQUE,
  uploaded_at TEXT
);
"""
_COLS = "doc_id, filename, pages, chunks, uploaded_at"


class DocumentDB:
    """Thin wrapper over a SQLite file; one short-lived connection per call (thread-safe)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            with conn:  # commits or rolls back
                yield conn
        finally:
            conn.close()

    def insert(self, meta: DocumentMeta, sha256: str) -> None:
        """Insert a new document row."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, filename, pages, chunks, sha256, uploaded_at) VALUES (?,?,?,?,?,?)",
                (meta.doc_id, meta.filename, meta.pages, meta.chunks, sha256, meta.uploaded_at),
            )

    def get(self, doc_id: str) -> DocumentMeta | None:
        """Return the document with `doc_id`, if any."""
        with self._conn() as conn:
            row = conn.execute(f"SELECT {_COLS} FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return _row(row) if row else None

    def get_by_sha(self, sha256: str) -> DocumentMeta | None:
        """Return the document whose content hash is `sha256`, if any."""
        with self._conn() as conn:
            row = conn.execute(f"SELECT {_COLS} FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        return _row(row) if row else None

    def list(self) -> list[DocumentMeta]:
        """Return all documents, newest first."""
        with self._conn() as conn:
            rows = conn.execute(f"SELECT {_COLS} FROM documents ORDER BY uploaded_at DESC").fetchall()
        return [_row(r) for r in rows]

    def delete(self, doc_id: str) -> bool:
        """Delete a row; returns True if something was deleted."""
        with self._conn() as conn:
            return conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,)).rowcount > 0

    def count(self) -> int:
        """Number of documents."""
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])


def _row(row: tuple) -> DocumentMeta:
    return DocumentMeta(doc_id=row[0], filename=row[1], pages=row[2], chunks=row[3], uploaded_at=row[4])
