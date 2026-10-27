from pathlib import Path

from app.ingest import _split_sections, chunk_document, load_document, parse_header_block
from tests.conftest import REPO_CORPUS


def test_pep_header_is_parsed_into_metadata_and_title():
    doc = load_document(REPO_CORPUS / "pep-0572.rst")
    assert doc.title == "PEP 572: Assignment Expressions"
    assert doc.metadata["Status"] == "Final"
    assert doc.metadata["Python-Version"] == "3.8"
    # continuation lines are folded into the header value
    assert "Guido van Rossum" in doc.metadata["Author"]
    assert not doc.text.lstrip().startswith("PEP:")


def test_text_without_header_block_is_left_alone():
    headers, body = parse_header_block("Just a paragraph.\nWith: a colon")
    assert headers == {}
    assert body.startswith("Just a paragraph")


def test_sections_split_on_rst_and_markdown_headings():
    rst = "Intro\n=====\n\nHello.\n\nDetails\n-------\n\nMore text."
    assert [(h, b) for _, h, b in _split_sections(rst)] == [("Intro", "Hello."), ("Details", "More text.")]
    md = "# Title\n\nBody one.\n\n## Sub\n\nBody two."
    assert [(h, b) for _, h, b in _split_sections(md)] == [("Title", "Body one."), ("Sub", "Body two.")]


def test_chunks_respect_size_and_carry_section(tmp_path: Path):
    paragraphs = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(30))
    path = tmp_path / "doc.md"
    path.write_text(f"# Guide\n\n## Setup\n\n{paragraphs}")
    chunks = chunk_document(load_document(path), size=600, overlap=100)
    assert len(chunks) > 5
    assert all(len(c.text) <= 600 for c in chunks)
    assert {c.section for c in chunks} == {"Setup"}
    assert chunks[0].chunk_id == "doc#0"
    assert chunks[1].embedding_text().startswith("Guide — Setup")


def test_consecutive_chunks_overlap(tmp_path: Path):
    path = tmp_path / "doc.md"
    path.write_text("\n\n".join(f"Sentence number {i} is here." for i in range(200)))
    chunks = chunk_document(load_document(path), size=300, overlap=80)
    tail = chunks[0].text[-40:].split("\n\n")[-1]
    assert tail in chunks[1].text


def _minimal_pdf(text: str) -> bytes:
    """Build a one-page PDF with a single line of text (no PDF library needed)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_pdf_pages_become_chunks_with_page_numbers(tmp_path: Path):
    path = tmp_path / "manual.pdf"
    path.write_bytes(_minimal_pdf("The retry budget is three attempts."))
    doc = load_document(path)
    chunks = chunk_document(doc)
    assert "retry budget is three attempts" in chunks[0].text
    assert chunks[0].page == 1


def test_ingest_builds_index_and_metadata(indexed):
    retriever, sql, stats = indexed
    assert stats["documents"] == 3
    assert retriever.store.count() == stats["chunks"]
    result = sql.run("SELECT pep_number, python_version FROM documents ORDER BY pep_number")
    assert result.data["rows"] == [[8, None], [572, "3.8"], [618, "3.10"]]


def test_reingest_is_idempotent(indexed, corpus, tmp_path):
    from app.ingest import ingest

    retriever, _, stats = indexed
    ingest(corpus, retriever.store, retriever.embedder, tmp_path / "m2.db", size=1200, overlap=200)
    assert retriever.store.count() == stats["chunks"]
