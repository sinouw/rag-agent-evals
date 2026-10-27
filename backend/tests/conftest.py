from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app.ingest import ingest
from app.retrieval import HashEmbedder, Retriever, VectorStore
from app.tools import SqlTool
from app.tracing import Tracing

REPO_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A three-document slice of the real corpus."""
    out = tmp_path / "corpus"
    out.mkdir()
    for name in ("pep-0008.rst", "pep-0572.rst", "pep-0618.rst"):
        (out / name).write_text((REPO_CORPUS / name).read_text(encoding="utf-8"), encoding="utf-8")
    return out


@pytest.fixture
def indexed(corpus: Path, tmp_path: Path):
    embedder = HashEmbedder()
    store = VectorStore(QdrantClient(":memory:"), "test")
    db = tmp_path / "metadata.db"
    stats = ingest(corpus, store, embedder, db, size=1200, overlap=200)
    return Retriever(embedder, store, top_k=5), SqlTool(db), stats


@pytest.fixture
def tracing() -> Tracing:
    return Tracing(jsonl_path=None)
