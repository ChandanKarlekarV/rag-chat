"""ChromaDB wrapper storing chunk vectors, texts and metadata."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.schemas import Chunk

COLLECTION = "chunks"


def _metadata(chunk: Chunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "index": chunk.index,
        "token_count": chunk.token_count,
    }


def chunk_from_record(chunk_id: str, text: str, meta: dict[str, Any]) -> Chunk:
    """Rebuild a `Chunk` from a Chroma id/document/metadata triple."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=str(meta["doc_id"]),
        filename=str(meta["filename"]),
        page_start=int(meta["page_start"]),
        page_end=int(meta["page_end"]),
        index=int(meta["index"]),
        text=text,
        token_count=int(meta.get("token_count", 0)),
    )


class VectorStore:
    """Persistent cosine-similarity vector index."""

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        """Number of stored chunks."""
        return self._col.count()

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert chunks with their precomputed embeddings."""
        if not chunks:
            return
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[_metadata(c) for c in chunks],
        )

    def query(
        self, embedding: list[float], k: int, doc_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        """Return up to `k` (chunk, cosine similarity) pairs, best first."""
        n = min(k, self.count())
        if n == 0 or doc_ids == []:
            return []
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            (chunk_from_record(cid, doc, meta), 1.0 - float(dist))
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def get_chunks(self, ids: list[str]) -> dict[str, Chunk]:
        """Fetch chunks by id."""
        if not ids:
            return {}
        res = self._col.get(ids=ids, include=["documents", "metadatas"])
        return {
            cid: chunk_from_record(cid, doc, meta)
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        }

    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk (used to rebuild the BM25 index)."""
        res = self._col.get(include=["documents", "metadatas"])
        return [chunk_from_record(c, d, m) for c, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def delete_doc(self, doc_id: str) -> None:
        """Remove all chunks belonging to `doc_id`."""
        self._col.delete(where={"doc_id": doc_id})
