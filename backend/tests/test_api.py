import json

from fastapi.testclient import TestClient

from app.agent import Agent
from app.config import Settings
from app.llm import ScriptedLLM, ToolCall
from app.main import Components, create_app


def make_client(indexed, tracing, turns) -> TestClient:
    retriever, sql, _ = indexed
    agent = Agent(ScriptedLLM(turns), retriever, sql, tracing, "a test corpus")
    settings = Settings(sqlite_path=sql.db_path, trace_log_path=None)
    return TestClient(create_app(Components(settings, agent, retriever, tracing)))


def parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        data = [line[5:].strip() for line in block.splitlines() if line.startswith("data:")]
        if data:
            events.append(json.loads("".join(data)))
    return events


def test_ask_streams_events_and_trace_is_retrievable(indexed, tracing):
    turns = [[ToolCall("c1", "search_docs", '{"query": "line length"}')], "79 characters [S1]."]
    with make_client(indexed, tracing, turns) as client:
        response = client.post("/api/ask", json={"question": "Max line length?"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.text)
        assert [e["type"] for e in events][:2] == ["start", "tool_call"]
        done = events[-1]
        assert done["type"] == "done" and done["citations"]

        trace = client.get(f"/api/traces/{done['trace_id']}").json()
        assert {s["name"] for s in trace} >= {"agent.run", "llm.chat", "tool.search_docs"}
        assert client.get("/api/traces").json()[0]["trace_id"] == done["trace_id"]


def test_health_documents_and_search(indexed, tracing):
    with make_client(indexed, tracing, []) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok" and health["indexed_chunks"] > 0
        assert len(client.get("/api/documents").json()) == 3
        hits = client.get("/api/search", params={"q": "zip strict ValueError", "k": 3}).json()
        assert len(hits) == 3 and hits[0]["doc_id"] == "pep-0618"


def test_validation_and_missing_trace(indexed, tracing):
    with make_client(indexed, tracing, []) as client:
        assert client.post("/api/ask", json={"question": ""}).status_code == 422
        assert client.get("/api/traces/does-not-exist").status_code == 404


def test_ask_without_llm_returns_503(indexed, tracing):
    retriever, sql, _ = indexed
    settings = Settings(sqlite_path=sql.db_path, trace_log_path=None)
    app = create_app(Components(settings, None, retriever, tracing, llm_error="No API key for groq"))
    with TestClient(app) as client:
        response = client.post("/api/ask", json={"question": "hi"})
        assert response.status_code == 503
        assert client.get("/api/health").json()["status"] == "degraded"
