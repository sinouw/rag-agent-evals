from app.ingest import document_card, load_document
from app.retrieval import RetrievalConfig
from tests.conftest import REPO_CORPUS


def test_document_card_holds_header_fields_without_emails():
    card = document_card(load_document(REPO_CORPUS / "pep-0668.rst"))
    assert card is not None and card.kind == "card"
    assert "PEP-Delegate: Paul Moore" in card.text
    assert "Pradyun Gedam" in card.text
    assert "@" not in card.text


def test_cards_are_excluded_unless_enabled(indexed):
    retriever, _, _ = indexed
    query = "Author Chris Angelico Tim Peters Guido van Rossum"
    without = retriever.search(query, 10)
    assert all(h.kind == "text" for h in without)
    with_cards = retriever.with_config(RetrievalConfig(cards=True)).search(query, 10)
    assert with_cards[0].chunk_id == "pep-0572#card"


def test_hybrid_fuses_dense_and_sparse(indexed):
    retriever, _, _ = indexed
    hybrid = retriever.with_config(RetrievalConfig(mode="hybrid", cards=True))
    hits, stages = hybrid.search_with_stages("strict zip ValueError", 5)
    assert hits[0].doc_id == "pep-0618"
    assert [s.name for s in stages] == ["retrieval.embed_query", "retrieval.sparse_query", "retrieval.vector_search"]


def test_rerank_reorders_candidates_and_records_a_stage(indexed):
    retriever, _, _ = indexed
    reranked = retriever.with_config(RetrievalConfig(rerank=True, candidates=15))
    hits, stages = reranked.search_with_stages("line length 79 characters", 3)
    assert len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    rerank = next(s for s in stages if s.name == "retrieval.rerank")
    assert rerank.attributes["rerank.candidates"] == 15
    assert rerank.end_ns >= rerank.start_ns


def test_config_labels():
    assert RetrievalConfig().label() == "dense"
    assert RetrievalConfig("hybrid", True, True).label() == "hybrid+cards+rerank"
