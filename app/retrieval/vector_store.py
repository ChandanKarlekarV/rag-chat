"""ChromaDB wrapper. Every read is filtered by `user_id` (isolation rule)."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.schemas import Chunk

COLLECTION = "chunks"


def _metadata(chunk: Chunk, user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
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


def _where(user_id: str, doc_ids: list[str] | None) -> dict[str, Any]:
    if doc_ids:
        return {"$and": [{"user_id": user_id}, {"doc_id": {"$in": doc_ids}}]}
    return {"user_id": user_id}


class VectorStore:
    """Persistent cosine-similarity vector index shared by all users, partitioned by metadata."""

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        self._col = self._client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    def add_chunks(self, user_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert a document's chunks with precomputed embeddings."""
        if chunks:
            self._col.add(
                ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks],
                embeddings=embeddings,
                metadatas=[_metadata(c, user_id) for c in chunks],
            )

    def query(
        self, user_id: str, embedding: list[float], k: int, doc_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        """Up to `k` of the user's (chunk, cosine similarity) pairs, best first."""
        if doc_ids == []:
            return []
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=k,
            where=_where(user_id, doc_ids),
            include=["documents", "metadatas", "distances"],
        )
        return [
            (chunk_from_record(cid, doc, meta), 1.0 - float(dist))
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def get_chunks(self, user_id: str, ids: list[str]) -> dict[str, Chunk]:
        """Fetch the user's chunks by id (ids owned by others are silently dropped)."""
        if not ids:
            return {}
        res = self._col.get(ids=ids, where={"user_id": user_id}, include=["documents", "metadatas"])
        return {
            cid: chunk_from_record(cid, doc, meta)
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        }

    def user_chunks(self, user_id: str) -> list[Chunk]:
        """All of a user's chunks (used to rebuild their BM25 index)."""
        res = self._col.get(where={"user_id": user_id}, include=["documents", "metadatas"])
        return [chunk_from_record(c, d, m) for c, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def delete_doc(self, user_id: str, doc_id: str) -> None:
        """Remove one document's chunks."""
        self._col.delete(where={"$and": [{"user_id": user_id}, {"doc_id": doc_id}]})

    def delete_user(self, user_id: str) -> None:
        """Remove every chunk of a user."""
        self._col.delete(where={"user_id": user_id})
