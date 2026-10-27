"""FastAPI app: streams agent answers over Server-Sent Events and serves traces.

Run: `uv run uvicorn app.main:app --reload`
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.agent import Agent
from app.config import Settings, get_settings
from app.ingest import ingest
from app.llm import LLMClient, create_llm
from app.retrieval import Retriever, build_embedder, build_store
from app.tools import SqlTool
from app.tracing import Tracing

log = logging.getLogger("rag-agent")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@dataclass
class Components:
    settings: Settings
    agent: Agent | None
    retriever: Retriever
    tracing: Tracing
    llm_error: str | None = None


def build_components(settings: Settings, llm: LLMClient | None = None) -> Components:
    tracing = Tracing(
        settings.service_name, settings.otlp_endpoint, settings.trace_log_path, settings.trace_capture_content
    )
    embedder = build_embedder(settings)
    store = build_store(settings)
    if settings.auto_ingest and (store.count() == 0 or not settings.sqlite_path.exists()):
        stats = ingest(
            settings.corpus_dir, store, embedder, settings.sqlite_path, settings.chunk_size, settings.chunk_overlap
        )
        log.warning("Indexed %(documents)s documents as %(chunks)s chunks in %(seconds)ss", stats)
    retriever = Retriever(embedder, store, settings.top_k)
    llm_error = None
    if llm is None:
        try:
            llm = create_llm(
                settings.llm_provider,
                settings.llm_model,
                settings.api_key_for(settings.llm_provider),
                settings.llm_temperature,
            )
        except (RuntimeError, ValueError) as exc:
            # Keep serving retrieval/health so the misconfiguration is visible, not a crash loop.
            llm_error = str(exc)
            log.error("LLM unavailable: %s", exc)
    agent = None
    if llm is not None:
        agent = Agent(
            llm,
            retriever,
            SqlTool(settings.sqlite_path),
            tracing,
            settings.corpus_description,
            settings.max_agent_steps,
        )
    return Components(settings, agent, retriever, tracing, llm_error)


def create_app(components: Components | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.components = components or await asyncio.to_thread(build_components, get_settings())
        yield
        app.state.components.tracing.shutdown()

    app = FastAPI(title="RAG agent", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def comps(request: Request) -> Components:
        return request.app.state.components

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        c = comps(request)
        return {
            "status": "ok" if c.agent else "degraded",
            "llm": {"provider": c.agent.llm.provider, "model": c.agent.llm.model} if c.agent else None,
            "llm_error": c.llm_error,
            "embedding_model": c.retriever.embedder.name,
            "indexed_chunks": c.retriever.store.count(),
        }

    @app.post("/api/ask")
    async def ask(body: AskRequest, request: Request) -> EventSourceResponse:
        c = comps(request)
        if c.agent is None:
            raise HTTPException(503, c.llm_error or "LLM not configured")
        agent = c.agent

        async def events():
            async for event in agent.run(body.question):
                yield {"event": event["type"], "data": json.dumps(event, default=str)}

        return EventSourceResponse(events(), ping=15)

    @app.get("/api/search")
    async def search(q: str, request: Request, k: int = 5) -> list[dict[str, Any]]:
        hits = await asyncio.to_thread(comps(request).retriever.search, q, max(1, min(k, 20)))
        return [h.__dict__ for h in hits]

    @app.get("/api/traces")
    def recent_traces(request: Request, limit: int = 20) -> list[dict[str, Any]]:
        return comps(request).tracing.store.recent(limit)

    @app.get("/api/traces/{trace_id}")
    def get_trace(trace_id: str, request: Request) -> list[dict[str, Any]]:
        spans = comps(request).tracing.store.get(trace_id)
        if spans is None:
            raise HTTPException(404, "trace not found")
        return spans

    @app.get("/api/documents")
    def documents(request: Request) -> list[dict[str, Any]]:
        path = comps(request).settings.sqlite_path
        if not path.exists():
            return []
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM documents ORDER BY pep_number, doc_id")]

    return app


app = create_app()
