import json

import pytest

from app.agent import Agent, extract_citation_labels
from app.llm import ScriptedLLM, ToolCall

CORPUS = "a test corpus"


async def collect(agent: Agent, question: str, **kwargs) -> list[dict]:
    return [event async for event in agent.run(question, **kwargs)]


def test_extract_citation_labels():
    text = "Lines max 79 [S2]. Also [S1, S3] and again [S2]; not [note] or S9."
    assert extract_citation_labels(text) == ["S2", "S1", "S3"]


@pytest.mark.asyncio
async def test_agent_searches_then_answers_with_resolved_citations(indexed, tracing):
    retriever, sql, _ = indexed
    llm = ScriptedLLM(
        [
            [ToolCall("c1", "search_docs", json.dumps({"query": "maximum line length"}))],
            "Limit lines to 79 characters [S1].",
        ]
    )
    agent = Agent(llm, retriever, sql, tracing, CORPUS)
    events = await collect(agent, "What is the max line length?", include_context=True)

    types = [e["type"] for e in events]
    assert types[0] == "start"
    assert types.index("tool_call") < types.index("tool_result") < types.index("token")
    done = events[-1]
    assert done["type"] == "done"
    assert done["answer"] == "Limit lines to 79 characters [S1]."
    assert done["citations"][0]["label"] == "S1"
    assert done["citations"][0]["doc_id"].startswith("pep-")
    assert done["tools"] == [{"name": "search_docs", "arguments": {"query": "maximum line length"}, "error": False}]
    assert "[S1]" in done["context"][0]["content"]
    # the tool result is passed back to the model on the next turn
    assert llm.requests[1][-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_agent_runs_sql_and_calculator(indexed, tracing):
    retriever, sql, _ = indexed
    llm = ScriptedLLM(
        [
            [ToolCall("c1", "sql_query", json.dumps({"sql": "SELECT COUNT(*) FROM documents"}))],
            [ToolCall("c2", "calculator", json.dumps({"expression": "3 * 2"}))],
            "There are 3 documents; doubled that is 6.",
        ]
    )
    events = await collect(Agent(llm, retriever, sql, tracing, CORPUS), "count?")
    results = [e for e in events if e["type"] == "tool_result"]
    assert results[0]["data"]["rows"] == [[3]]
    assert results[1]["data"]["result"] == 6
    assert events[-1]["citations"] == []


@pytest.mark.asyncio
async def test_agent_stops_offering_tools_at_step_budget(indexed, tracing):
    retriever, sql, _ = indexed
    loop = [ToolCall(f"c{i}", "calculator", '{"expression": "1+1"}') for i in range(3)]
    llm = ScriptedLLM([[loop[0]], [loop[1]], "Done."])
    events = await collect(Agent(llm, retriever, sql, tracing, CORPUS, max_steps=3), "q")
    assert events[-1]["answer"] == "Done."
    assert events[-1]["steps"] == 3
    assert "Tool budget exhausted" in llm.requests[-1][-1]["content"]


@pytest.mark.asyncio
async def test_unknown_citations_and_bad_tool_arguments_are_handled(indexed, tracing):
    retriever, sql, _ = indexed
    llm = ScriptedLLM([[ToolCall("c1", "search_docs", "not json")], "Answer citing [S7]."])
    events = await collect(Agent(llm, retriever, sql, tracing, CORPUS), "q")
    assert next(e for e in events if e["type"] == "tool_result")["error"] is True
    assert events[-1]["citations"] == []


@pytest.mark.asyncio
async def test_trace_records_llm_tool_and_retrieval_spans(indexed, tracing):
    retriever, sql, _ = indexed
    llm = ScriptedLLM([[ToolCall("c1", "search_docs", '{"query": "zip strict"}')], "Raises ValueError [S1]."])
    events = await collect(Agent(llm, retriever, sql, tracing, CORPUS), "zip?")
    spans = tracing.store.get(events[-1]["trace_id"])
    names = [s["name"] for s in spans]
    assert names.count("llm.chat") == 2
    assert {"agent.run", "tool.search_docs", "retrieval.embed_query", "retrieval.vector_search"} <= set(names)
    root = next(s for s in spans if s["name"] == "agent.run")
    assert root["parent_id"] is None
    assert root["attributes"]["gen_ai.usage.output_tokens"] == 20
    tool = next(s for s in spans if s["name"] == "tool.search_docs")
    assert json.loads(tool["attributes"]["retrieval.results"])
    search = next(s for s in spans if s["name"] == "retrieval.vector_search")
    assert search["parent_id"] == tool["span_id"]
    assert search["attributes"]["retrieval.mode"] == "dense"
    assert all(s["trace_id"] == root["trace_id"] for s in spans)


@pytest.mark.asyncio
async def test_provider_errors_become_error_events(indexed, tracing):
    retriever, sql, _ = indexed

    class Failing(ScriptedLLM):
        async def stream(self, messages, tools=None):
            raise RuntimeError("rate limited")
            yield  # pragma: no cover

    events = await collect(Agent(Failing([]), retriever, sql, tracing, CORPUS), "q")
    assert events[-1]["type"] == "error"
    assert "rate limited" in events[-1]["message"]


def test_error_messages_redact_key_like_strings():
    from app.agent import redact

    assert redact("Incorrect API key provided: gsk_abc123***xyz9.") == "Incorrect API key provided: [redacted-key]."
    assert "AIza" not in redact("bad key AIzaSyD-abcdefgh")


def test_fullwidth_citation_brackets_are_normalized():
    from app.agent import normalize_citations

    assert extract_citation_labels(normalize_citations("Raises ValueError【S1】 and 【S2, S3】.")) == ["S1", "S2", "S3"]


def test_error_messages_redact_account_ids():
    from app.agent import redact

    assert (
        redact("limit for organization `org_0000examplefake` reached")
        == "limit for organization `[redacted-id]` reached"
    )
