"""Embeddings, the Qdrant vector store, and the retrieval pipeline.

Each chunk is stored with a dense vector (fastembed `bge-small`) and a sparse BM25
vector. A `RetrievalConfig` picks the pipeline at query time, so variants can be
compared on one index:

- `dense`: cosine search on the dense vector
- `hybrid`: dense and BM25 candidates fused with Reciprocal Rank Fusion inside Qdrant
- `cards`: include the per-document metadata cards (authors, status, dates)
- `rerank`: re-score the top candidates with a local cross-encoder

Qdrant runs as a server in docker-compose (`QDRANT_URL`) and in embedded local mode
otherwise (same client API, data under `QDRANT_PATH`), so the project runs and is
tested without Docker.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.ingest import Chunk

DENSE = "dense"
SPARSE = "bm25"

SparseVector = tuple[list[int], list[float]]


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SparseEmbedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


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


class FastEmbedBM25:
    """BM25 term weights as sparse vectors; Qdrant applies IDF at query time."""

    name = "Qdrant/bm25"

    def __init__(self) -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(self.name)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [(v.indices.tolist(), v.values.tolist()) for v in self._model.embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> SparseVector:
        v = next(iter(self._model.query_embed(text)))
        return v.indices.tolist(), v.values.tolist()


class CrossEncoderReranker:
    """Local ONNX cross-encoder: scores each (query, passage) pair jointly."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self._model = TextCrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts)]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bucket(token: str, size: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest()[:4], "big") % size


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for offline tests. Lexical only."""

    def __init__(self, dim: int = 512) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokens(text):
            vec[_bucket(token, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class HashSparseEmbedder:
    """Term-frequency sparse vectors over hashed tokens, for offline tests."""

    name = "hash-sparse"

    def _embed(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for token in _tokens(text):
            idx = _bucket(token, 2**20)
            counts[idx] = counts.get(idx, 0.0) + 1.0
        return list(counts), list(counts.values())

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._embed(text)


class OverlapReranker:
    """Scores by query-term overlap, for offline tests."""

    name = "overlap"

    def score(self, query: str, texts: list[str]) -> list[float]:
        terms = set(_tokens(query))
        return [float(len(terms & set(_tokens(t)))) for t in texts]


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float
    page: int | None = None
    kind: str = "text"


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "dense"  # dense | hybrid
    cards: bool = False
    rerank: bool = False
    candidates: int = 20  # fetched per retriever before fusion / reranking

    def label(self) -> str:
        return "+".join([self.mode, *(["cards"] if self.cards else []), *(["rerank"] if self.rerank else [])])


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    def schema_ok(self) -> bool:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        sparse = info.config.params.sparse_vectors or {}
        return isinstance(vectors, dict) and DENSE in vectors and SPARSE in sparse

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and (recreate or not self.schema_ok()):
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection,
                vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            with warnings.catch_warnings():
                # Embedded Qdrant ignores payload indexes and warns; the server uses it for the card filter.
                warnings.simplefilter("ignore", UserWarning)
                self.client.create_payload_index(self.collection, "kind", models.PayloadSchemaType.KEYWORD)

    def count(self) -> int:
        """Indexed points, or 0 when the collection is missing or uses an outdated schema."""
        if not self.client.collection_exists(self.collection) or not self.schema_ok():
            return 0
        return self.client.count(self.collection).count

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None = None,
        batch_size: int = 256,
    ) -> None:
        points = []
        for i, (c, v) in enumerate(zip(chunks, vectors, strict=True)):
            vector: dict[str, Any] = {DENSE: v}
            if sparse_vectors is not None:
                indices, values = sparse_vectors[i]
                vector[SPARSE] = models.SparseVector(indices=indices, values=values)
            points.append(
                models.PointStruct(
                    # Stable ids make re-ingesting idempotent.
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                    vector=vector,
                    payload={
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "title": c.title,
                        "section": c.section,
                        "text": c.text,
                        "page": c.page,
                        "kind": c.kind,
                    },
                )
            )
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points=points[i : i + batch_size])

    def search(
        self,
        dense: list[float],
        limit: int,
        sparse: SparseVector | None = None,
        include_cards: bool = True,
        candidates: int = 20,
    ) -> list[Hit]:
        query_filter = None
        if not include_cards:
            query_filter = models.Filter(
                must_not=[models.FieldCondition(key="kind", match=models.MatchValue(value="card"))]
            )
        if sparse is None:
            result = self.client.query_points(
                self.collection, query=dense, using=DENSE, limit=limit, query_filter=query_filter, with_payload=True
            )
        else:
            fetch = max(limit, candidates)
            result = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dense, using=DENSE, limit=fetch, filter=query_filter),
                    models.Prefetch(
                        query=models.SparseVector(indices=sparse[0], values=sparse[1]),
                        using=SPARSE,
                        limit=fetch,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
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
                    p.get("kind", "text"),
                )
            )
        return hits


@dataclass
class Stage:
    """One timed step of a retrieval, recorded so it can be emitted as a trace span."""

    name: str
    start_ns: int
    end_ns: int
    attributes: dict[str, Any]


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 5,
        sparse: SparseEmbedder | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.sparse = sparse
        self.reranker = reranker
        self.config = config or RetrievalConfig()
        if self.config.mode == "hybrid" and sparse is None:
            raise ValueError("hybrid retrieval needs a sparse embedder")
        if self.config.rerank and reranker is None:
            raise ValueError("reranking needs a reranker")

    def with_config(self, config: RetrievalConfig) -> Retriever:
        return Retriever(self.embedder, self.store, self.top_k, self.sparse, self.reranker, config)

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        return self.search_with_stages(query, k)[0]

    def search_with_stages(self, query: str, k: int | None = None) -> tuple[list[Hit], list[Stage]]:
        cfg = self.config
        k = k or self.top_k
        stages: list[Stage] = []

        def timed(name: str, fn, **attributes):
            start = time.time_ns()
            value = fn()
            stages.append(Stage(name, start, time.time_ns(), attributes))
            return value

        dense = timed(
            "retrieval.embed_query", lambda: self.embedder.embed_query(query), **{"embedding.model": self.embedder.name}
        )
        sparse = None
        if cfg.mode == "hybrid":
            assert self.sparse is not None
            sparse = timed(
                "retrieval.sparse_query", lambda: self.sparse.embed_query(query), **{"sparse.model": self.sparse.name}
            )
        fetch = max(k, cfg.candidates) if cfg.rerank else k
        hits = timed(
            "retrieval.vector_search",
            lambda: self.store.search(dense, fetch, sparse, cfg.cards, cfg.candidates),
            **{"retrieval.query": query, "retrieval.mode": cfg.mode, "retrieval.limit": fetch},
        )
        if cfg.rerank and hits:
            assert self.reranker is not None
            scores = timed(
                "retrieval.rerank",
                lambda: self.reranker.score(query, [h.text for h in hits]),
                **{"rerank.model": self.reranker.name, "rerank.candidates": len(hits)},
            )
            for hit, score in zip(hits, scores, strict=True):
                hit.score = score
            hits = sorted(hits, key=lambda h: h.score, reverse=True)
        return hits[:k], stages


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(settings.embedding_model)


def build_sparse_embedder(settings: Settings) -> SparseEmbedder:
    if settings.embedding_backend == "hash":
        return HashSparseEmbedder()
    return FastEmbedBM25()


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.rerank:
        return None
    if settings.embedding_backend == "hash":
        return OverlapReranker()
    return CrossEncoderReranker(settings.rerank_model)


def retrieval_config(settings: Settings) -> RetrievalConfig:
    return RetrievalConfig(
        settings.retrieval_mode, settings.retrieval_cards, settings.rerank, settings.rerank_candidates
    )


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
