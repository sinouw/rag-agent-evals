import json

import pytest

from app.agent import Agent
from app.llm import ScriptedLLM, ToolCall
from evals.run import (
    GoldenItem,
    agent_eval,
    first_relevant_rank,
    load_golden,
    parse_judgement,
    render_markdown,
    retrieval_eval,
)


def test_golden_set_is_well_formed():
    items = load_golden()
    assert len(items) >= 30
    assert len({i.id for i in items}) == len(items)
    for item in items:
        assert item.category in {"lookup", "multi_doc", "metadata_sql", "calculation", "unanswerable"}
        if item.category == "unanswerable":
            assert not item.expected_docs


def test_first_relevant_rank():
    assert first_relevant_rank(["a", "b", "c"], ["c"]) == 3
    assert first_relevant_rank(["a", "b"], ["z"]) is None


def test_parse_judgement_tolerates_wrapping_and_garbage():
    assert parse_judgement('```json\n{"correct": true, "faithful": false, "reason": "r"}\n```') == {
        "correct": True,
        "faithful": False,
        "reason": "r",
    }
    assert parse_judgement("nope")["correct"] is False


def test_retrieval_eval_metrics(indexed):
    retriever, _, _ = indexed
    items = [
        GoldenItem("a", "lookup", "zip strict ValueError lengths", "x", ["pep-0618"], evidence="ValueError"),
        GoldenItem("b", "lookup", "walrus assignment expressions", "x", ["pep-0572"]),
        GoldenItem("c", "unanswerable", "?", "x", []),
    ]
    result = retrieval_eval(retriever, items, k=5)
    summary = result["summary"]
    assert summary["questions"] == 2
    assert summary["hit_at_5"] == 1.0
    assert summary["evidence_questions"] == 1


@pytest.mark.asyncio
async def test_agent_eval_scores_and_renders(indexed, tracing):
    retriever, sql, _ = indexed
    agent = Agent(
        ScriptedLLM([[ToolCall("c1", "search_docs", '{"query": "zip strict"}')], "ValueError [S1]."]),
        retriever,
        sql,
        tracing,
        "corpus",
    )
    judge = ScriptedLLM([json.dumps({"correct": True, "faithful": True, "reason": "matches"})])
    item = GoldenItem("a", "lookup", "zip strict?", "ValueError", ["pep-0618"], expected_tool="search_docs")
    e2e = await agent_eval(agent, judge, [item])
    row = e2e["rows"][0]
    assert row["correct"] and row["faithful"] and row["tool_ok"]
    assert e2e["summary"]["correctness"] == 1.0

    retrieval = retrieval_eval(retriever, [item], k=5)
    meta = {
        "timestamp": "t",
        "embedding_model": "hash",
        "k": 5,
        "documents": 3,
        "chunks": 10,
        "golden_questions": 1,
        "provider": "scripted",
        "model": "m",
        "judge": "scripted/m",
    }
    markdown = render_markdown({"meta": meta, "retrieval": retrieval, "e2e": e2e})
    assert "## End-to-end agent" in markdown and "| a | lookup | ✅ | ✅ |" in markdown


def test_update_readme_replaces_only_the_marked_block(tmp_path):
    from evals.run import README_END, README_START, update_readme

    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{README_START}\npending real run\n{README_END}\n\n## After\n")
    by_cat = {
        "questions": 1,
        "correctness": 1.0,
        "faithfulness": 1.0,
        "citation_precision": None,
        "tool_selection": None,
    }
    summary = {
        **by_cat,
        "errors": 0,
        "p50_latency_ms": 900,
        "p95_latency_ms": 1500,
        "avg_input_tokens": 2000,
        "avg_output_tokens": 80,
        "avg_cost_usd": 0.0012,
        "by_category": {"lookup": by_cat},
    }
    meta = {"timestamp": "t", "golden_questions": 1, "provider": "p", "model": "m", "judge": "j/m"}
    update_readme(readme, {"meta": meta, "e2e": {"summary": summary}}, "r.md")
    text = readme.read_text()
    assert "pending real run" not in text
    assert "| 100% | 100% | — | — | 900 / 1500 ms | 2000 / 80 | $0.00120 | 0 |" in text
    assert text.startswith("# Title") and text.endswith("## After\n")


@pytest.mark.asyncio
async def test_agent_eval_stops_after_daily_quota_error(indexed, tracing):
    from app.llm import ScriptedLLM as Base

    class QuotaLLM(Base):
        async def stream(self, messages, tools=None):
            raise RuntimeError("Rate limit reached on tokens per day (TPD): Limit 200000")
            yield  # pragma: no cover

    retriever, sql, _ = indexed
    agent = Agent(QuotaLLM([]), retriever, sql, tracing, "corpus")
    items = [GoldenItem(f"q{i}", "lookup", "?", "x", ["pep-0618"]) for i in range(3)]
    e2e = await agent_eval(agent, ScriptedLLM([]), items)
    assert e2e["quota_exhausted"] is True
    assert e2e["rows"][1]["error"] == "skipped: provider daily quota exhausted"
    assert e2e["summary"]["errors"] == 3
