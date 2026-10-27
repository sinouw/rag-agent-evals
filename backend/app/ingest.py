"""Load documents, split them into section-aware chunks, embed them into the vector store,
and record per-document metadata in SQLite for the SQL tool.

Run: `uv run python -m app.ingest [--recreate]`
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".rst", ".txt"}

_HEADER_LINE = re.compile(r"^([A-Z][A-Za-z-]+):\s*(.*)$")
_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)
    # For PDFs: (page_number, text) so chunks can cite pages.
    pages: list[tuple[int, str]] | None = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    page: int | None = None
    kind: str = "text"  # "text" for body chunks, "card" for the per-document metadata card

    def embedding_text(self) -> str:
        # Prefixing the document and section titles gives each chunk the context
        # it loses when cut out of its document ("contextual chunk headers").
        heading = f"{self.title} — {self.section}" if self.section else self.title
        return f"{heading}\n\n{self.text}"


def parse_header_block(text: str) -> tuple[dict[str, str], str]:
    """Parse an RFC 822-style `Key: value` block at the top of a file (PEP headers, front matter)."""
    lines = text.splitlines()
    headers: dict[str, str] = {}
    key = None
    i = 0
    for i, line in enumerate(lines):  # noqa: B007 - i is used after the loop
        if not line.strip():
            break
        match = _HEADER_LINE.match(line)
        if match:
            key = match.group(1)
            headers[key] = match.group(2).strip()
        elif key and line[:1].isspace():
            headers[key] += " " + line.strip()
        else:
            return {}, text
    else:
        i = len(lines)
    if len(headers) < 2:
        return {}, text
    return headers, "\n".join(lines[i + 1 :])


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.name}")
    doc_id = path.stem
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [(n, (page.extract_text() or "").strip()) for n, page in enumerate(reader.pages, start=1)]
        title = (reader.metadata.title if reader.metadata else None) or path.stem
        return Document(doc_id, title, "\n\n".join(t for _, t in pages), path.name, {}, pages)

    raw = path.read_text(encoding="utf-8")
    headers, body = parse_header_block(raw)
    title = headers.get("Title") or _first_heading(body) or path.stem
    if "PEP" in headers:
        title = f"PEP {headers['PEP']}: {title}"
    return Document(doc_id, title, body, path.name, headers)


def _first_heading(text: str) -> str | None:
    for _, heading, _ in _split_sections(text):
        if heading:
            return heading
    return None


def _split_sections(text: str) -> list[tuple[int, str, str]]:
    """Split RST or Markdown text into (index, heading, body) sections."""
    lines = text.splitlines()
    sections: list[tuple[int, str, str]] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body or heading:
            sections.append((len(sections), heading, body))

    i = 0
    while i < len(lines):
        line = lines[i]
        md = _MD_HEADING.match(line)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if md:
            flush()
            heading, buf = md.group(2), []
        elif line.strip() and _RST_UNDERLINE.match(nxt) and len(nxt.strip()) >= len(line.strip()) - 2:
            flush()
            heading, buf = line.strip(), []
            i += 1  # skip the underline
        elif _RST_UNDERLINE.match(line) and not buf:
            pass  # overline of an RST title
        else:
            buf.append(line)
        i += 1
    flush()
    return sections


def _pack(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into chunks of ~`size` chars, carrying `overlap` chars forward."""
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > size:  # hard-split oversized paragraphs
            head, para = para[:size], para[size - overlap :]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        cut = tail.find(" ")
        tail = tail[cut + 1 :] if cut != -1 else tail
        current = f"{tail}\n\n{para}" if tail and len(tail) + len(para) + 2 <= size else para
    if current.strip():
        chunks.append(current)
    return chunks


def chunk_document(doc: Document, size: int = 1200, overlap: int = 200) -> list[Chunk]:
    units: list[tuple[str, str, int | None]] = []  # (section, text, page)
    if doc.pages is not None:
        units = [("", text, page) for page, text in doc.pages if text]
    else:
        units = [(heading, body, None) for _, heading, body in _split_sections(doc.text) if body]

    chunks: list[Chunk] = []
    for section, text, page in units:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for piece in _pack(paragraphs, size, overlap):
            chunk_id = f"{doc.doc_id}#{len(chunks)}"
            chunks.append(Chunk(chunk_id, doc.doc_id, doc.title, section, piece, page))
    return chunks


CARD_FIELDS = ("Title", "Author", "Sponsor", "PEP-Delegate", "Status", "Type", "Topic", "Created", "Python-Version")


def document_card(doc: Document) -> Chunk | None:
    """One searchable chunk per document holding its header fields (authors, delegate,
    status, dates). Body chunks never contain these, so without a card questions like
    "who wrote X?" cannot be answered from retrieval."""
    lines = []
    for name in CARD_FIELDS:
        value = doc.metadata.get(name)
        if value:
            # Drop email addresses: names are what users ask about.
            value = re.sub(r"\s*<[^>]*>", "", value).strip()
            lines.append(f"{name}: {value}")
    if len(lines) < 2:
        return None
    return Chunk(f"{doc.doc_id}#card", doc.doc_id, doc.title, "Document metadata", "\n".join(lines), kind="card")


def load_corpus(corpus_dir: Path) -> list[Document]:
    paths = sorted(p for p in corpus_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    return [load_document(p) for p in paths]


def _iso_date(value: str | None) -> str | None:
    if not value:
        return None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def write_metadata_db(docs: list[Document], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(tmp) as conn:
        conn.execute(
            """CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                pep_number INTEGER,
                title TEXT NOT NULL,
                status TEXT,
                type TEXT,
                created DATE,
                python_version TEXT,
                author_count INTEGER,
                word_count INTEGER NOT NULL
            )"""
        )
        for doc in docs:
            meta = doc.metadata
            authors = meta.get("Author")
            conn.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.doc_id,
                    int(meta["PEP"]) if meta.get("PEP", "").isdigit() else None,
                    meta.get("Title") or doc.title,
                    meta.get("Status"),
                    meta.get("Type"),
                    _iso_date(meta.get("Created")),
                    meta.get("Python-Version"),
                    len([a for a in authors.split(",") if a.strip()]) if authors else None,
                    len(doc.text.split()),
                ),
            )
    tmp.replace(db_path)


def ingest(
    corpus_dir: Path,
    store,
    embedder,
    db_path: Path,
    size: int,
    overlap: int,
    recreate: bool = False,
    sparse=None,
) -> dict:
    """Index a corpus: body chunks plus one metadata card per document, with dense and
    (when a sparse embedder is given) BM25 vectors. Returns counts and timing."""
    started = time.perf_counter()
    docs = load_corpus(corpus_dir)
    if not docs:
        raise RuntimeError(f"No supported documents in {corpus_dir}")
    chunks = [c for doc in docs for c in chunk_document(doc, size, overlap)]
    cards = [card for doc in docs if (card := document_card(doc))]
    everything = chunks + cards
    texts = [c.embedding_text() for c in everything]
    store.ensure_collection(embedder.dim, recreate=recreate)
    sparse_vectors = sparse.embed_documents(texts) if sparse else None
    store.upsert(everything, embedder.embed_documents(texts), sparse_vectors)
    write_metadata_db(docs, db_path)
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "cards": len(cards),
        "seconds": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    from app.config import get_settings
    from app.retrieval import build_embedder, build_sparse_embedder, build_store

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Index the corpus into the vector store.")
    parser.add_argument("--corpus", type=Path, default=settings.corpus_dir)
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    args = parser.parse_args()

    stats = ingest(
        args.corpus,
        build_store(settings),
        build_embedder(settings),
        settings.sqlite_path,
        settings.chunk_size,
        settings.chunk_overlap,
        recreate=args.recreate,
        sparse=build_sparse_embedder(settings),
    )
    print(
        f"Indexed {stats['documents']} documents as {stats['chunks']} chunks "
        f"+ {stats['cards']} metadata cards in {stats['seconds']}s"
    )


if __name__ == "__main__":
    main()
