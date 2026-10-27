"""Embeddings and the Qdrant vector store.

Qdrant runs as a server in docker-compose (`QDRANT_URL`) and in embedded local mode
otherwise (same client API, data under `QDRANT_PATH`), so the project runs and is
tested without Docker.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import Settings
from app.ingest import Chunk


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed: no API key, no GPU, deterministic."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        # query_embed applies the model's query instruction prefix (BGE models use one).
        return next(iter(self._model.query_embed(text))).tolist()


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for offline tests. Lexical only."""

    def __init__(self, dim: int = 512) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vec[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float
    page: int | None = None


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], batch_size: int = 256) -> None:
        points = [
            PointStruct(
                # Stable ids make re-ingesting idempotent.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                vector=v,
                payload={
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "section": c.section,
                    "text": c.text,
                    "page": c.page,
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points=points[i : i + batch_size])

    def search(self, vector: list[float], limit: int) -> list[Hit]:
        result = self.client.query_points(self.collection, query=vector, limit=limit, with_payload=True)
        hits = []
        for point in result.points:
            p = point.payload or {}
            hits.append(
                Hit(
                    p["chunk_id"],
                    p["doc_id"],
                    p["title"],
                    p.get("section") or "",
                    p["text"],
                    point.score,
                    p.get("page"),
                )
            )
        return hits


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 5) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        return self.store.search(self.embedder.embed_query(query), k or self.top_k)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(settings.embedding_model)


def build_store(settings: Settings) -> VectorStore:
    if settings.qdrant_url:
        client = QdrantClient(url=settings.qdrant_url)
    else:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        try:
            client = QdrantClient(path=str(settings.qdrant_path))
        except RuntimeError as exc:
            # Embedded mode allows one process at a time (e.g. the dev server holds the lock).
            raise RuntimeError(
                f"The local index at {settings.qdrant_path} is in use by another process. "
                "Stop the running API (scripts/dev.sh) first, or set QDRANT_URL to use a Qdrant server."
            ) from exc
    return VectorStore(client, settings.collection)
