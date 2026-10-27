"""The agent loop: the model decides which tools to call (document search, SQL over
metadata, calculator), sees their results, and streams a cited answer.

`Agent.run` is an async generator of events consumed by the SSE endpoint and the eval
runner:

    start       {trace_id}
    tool_call   {id, name, arguments, step}
    tool_result {id, name, data, error}
    token       {text}
    discard_text {}            text streamed before the model chose to call a tool
    done        {answer, citations, tools, usage, cost_usd, latency_ms, ...}
    error       {message}
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from app.llm import LLMClient, TextDelta, ToolCall, TurnEnd, Usage, estimate_cost
from app.retrieval import Retriever
from app.tools import SourceRegistry, SqlTool, ToolResult, calculator, search_result, tool_schemas
from app.tracing import Tracing, trace_id_of

SYSTEM_PROMPT = """You answer questions about {corpus}.

Tools:
- search_docs: find passages in the documents. Search before answering any content question; \
search again with different wording if the first results don't contain the answer.
- sql_query: metadata across documents (counts, creation dates, statuses, types, Python versions, word counts).
- calculator: any arithmetic, including date differences. Never do arithmetic in your head: \
whenever the answer is a number you derive (a difference, sum, average, count of years), \
compute it with the calculator, even if it looks trivial.

Rules:
- Base every factual statement on tool results, and cite the passages you used inline with \
their source ids in square brackets, e.g. "... introduced in Python 3.8 [S2]." Only search_docs \
results have source ids; facts from sql_query or the calculator need no citation.
- If the tools don't give you the answer, say that the documents don't say. Do not guess.
- Be concise: a direct answer first, then brief supporting detail."""

FINAL_STEP_NOTE = (
    "Tool budget exhausted. Answer now from the results above, or say the documents don't contain the answer."
)

_CITATION_GROUP = re.compile(r"\[([^\[\]]+)\]")
_LABEL = re.compile(r"\bS\d+\b")


# Provider error messages can echo (partially masked) keys back; never forward them.
_KEY_LIKE = re.compile(r"\b(gsk_|sk-|AIza)[A-Za-z0-9_\-*]{6,}")


_ACCOUNT_ID = re.compile(r"\b(org|proj|user)_[A-Za-z0-9]{8,}")


def redact(text: str) -> str:
    return _ACCOUNT_ID.sub("[redacted-id]", _KEY_LIKE.sub("[redacted-key]", text))


def normalize_citations(answer: str) -> str:
    """Some models (e.g. gpt-oss) write citations with fullwidth brackets: 【S1】 -> [S1]."""
    return answer.replace("【", "[").replace("】", "]")


def extract_citation_labels(answer: str) -> list[str]:
    """Labels cited in an answer, in order of first appearance. Handles [S1] and [S1, S3]."""
    seen: list[str] = []
    for group in _CITATION_GROUP.findall(answer):
        for label in _LABEL.findall(group):
            if label not in seen:
                seen.append(label)
    return seen


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        sql_tool: SqlTool,
        tracing: Tracing,
        corpus_description: str,
        max_steps: int = 5,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.sql_tool = sql_tool
        self.tracing = tracing
        self.max_steps = max_steps
        self.system_prompt = SYSTEM_PROMPT.format(corpus=corpus_description)
        self.tools = tool_schemas(sql_tool.schema())

    async def run(self, question: str, include_context: bool = False) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        registry = SourceRegistry()
        usage = Usage()
        tool_log: list[dict[str, Any]] = []
        context: list[dict[str, str]] = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        tr = self.tracing
        with tr.span(
            "agent.run",
            **{
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.system": self.llm.provider,
                "gen_ai.request.model": self.llm.model,
                "agent.question": question,
            },
        ) as root:
            trace_id = trace_id_of(root)
            yield {"type": "start", "trace_id": trace_id}
            answer = ""
            steps = 0
            try:
                for step in range(1, self.max_steps + 1):
                    steps = step
                    last = step == self.max_steps
                    if last:
                        messages.append({"role": "user", "content": FINAL_STEP_NOTE})
                    end: TurnEnd | None = None
                    async for item in self._llm_turn(messages, None if last else self.tools, step, root):
                        if isinstance(item, TextDelta):
                            yield {"type": "token", "text": item.text}
                        else:
                            end = item
                    assert end is not None
                    usage.add(end.usage)
                    if not end.tool_calls:
                        answer = normalize_citations(end.text.strip())
                        break
                    if end.text:
                        yield {"type": "discard_text"}
                    messages.append(_assistant_tool_message(end))
                    for call in end.tool_calls:
                        args = call.parsed_arguments()
                        yield {"type": "tool_call", "id": call.id, "name": call.name, "arguments": args, "step": step}
                        result = await self._run_tool(call.name, args, registry, root)
                        tool_log.append({"name": call.name, "arguments": args, "error": result.error})
                        context.append({"tool": call.name, "content": result.content})
                        yield {
                            "type": "tool_result",
                            "id": call.id,
                            "name": call.name,
                            "data": result.data,
                            "error": result.error,
                        }
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": result.content})
            except Exception as exc:  # surface provider/tool failures to the client, keep them in the trace
                root.record_exception(exc)
                root.set_attribute("error.type", type(exc).__name__)
                yield {"type": "error", "message": redact(f"{type(exc).__name__}: {exc}"), "trace_id": trace_id}
                return

            citations = []
            for label in extract_citation_labels(answer):
                hit = registry.by_label.get(label)
                if hit:
                    citations.append(
                        {
                            "label": label,
                            "doc_id": hit.doc_id,
                            "title": hit.title,
                            "section": hit.section,
                            "page": hit.page,
                            "score": round(hit.score, 4),
                            "snippet": hit.text[:600],
                        }
                    )
            cost = estimate_cost(self.llm.model, usage)
            latency_ms = round((time.perf_counter() - started) * 1000)
            root.set_attributes(
                {
                    "agent.steps": steps,
                    "agent.tool_calls": len(tool_log),
                    "agent.citations": len(citations),
                    "gen_ai.usage.input_tokens": usage.input_tokens,
                    "gen_ai.usage.output_tokens": usage.output_tokens,
                    "llm.cost_usd": cost if cost is not None else -1.0,
                    "agent.latency_ms": latency_ms,
                }
            )
            if (content := tr.content(answer)) is not None:
                root.set_attribute("agent.answer", content)
            done: dict[str, Any] = {
                "type": "done",
                "trace_id": trace_id,
                "answer": answer,
                "citations": citations,
                "tools": tool_log,
                "steps": steps,
                "latency_ms": latency_ms,
                "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
                "cost_usd": cost,
                "provider": self.llm.provider,
                "model": self.llm.model,
            }
            if include_context:
                done["context"] = context
            yield done

    async def _llm_turn(self, messages, tools, step: int, root) -> AsyncIterator[TextDelta | TurnEnd]:
        tr = self.tracing
        with tr.span(
            "llm.chat",
            parent=root,
            **{
                "gen_ai.operation.name": "chat",
                "gen_ai.system": self.llm.provider,
                "gen_ai.request.model": self.llm.model,
                "agent.step": step,
                "llm.tools_offered": bool(tools),
            },
        ) as span:
            if (content := tr.content(messages)) is not None:
                span.add_event("gen_ai.prompt", {"content": content})
            started = time.perf_counter()
            first_token: float | None = None
            async for item in self.llm.stream(messages, tools):
                if isinstance(item, TextDelta) and first_token is None:
                    first_token = time.perf_counter()
                if isinstance(item, TurnEnd):
                    cost = estimate_cost(self.llm.model, item.usage)
                    span.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": item.usage.input_tokens,
                            "gen_ai.usage.output_tokens": item.usage.output_tokens,
                            "gen_ai.response.finish_reasons": [item.finish_reason or "unknown"],
                            "llm.cost_usd": cost if cost is not None else -1.0,
                            "llm.tool_calls": [c.name for c in item.tool_calls],
                        }
                    )
                    if first_token is not None:
                        span.set_attribute("llm.time_to_first_token_ms", round((first_token - started) * 1000, 1))
                    completion = item.text or json.dumps([c.__dict__ for c in item.tool_calls])
                    if (content := tr.content(completion)) is not None:
                        span.add_event("gen_ai.completion", {"content": content})
                yield item

    async def _run_tool(self, name: str, args: dict[str, Any], registry: SourceRegistry, root) -> ToolResult:
        tr = self.tracing
        with tr.span(
            f"tool.{name}", parent=root, **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name}
        ) as span:
            span.set_attribute("tool.arguments", json.dumps(args))
            if name == "search_docs":
                query = str(args.get("query") or "").strip()
                if not query:
                    result = ToolResult("Error: 'query' is required.", error=True)
                else:
                    k = max(1, min(int(args.get("k") or self.retriever.top_k), 10))
                    hits, stages = await asyncio.to_thread(self.retriever.search_with_stages, query, k)
                    for stage in stages:
                        tr.record(stage.name, span, stage.start_ns, stage.end_ns, **stage.attributes)
                    span.set_attribute("retrieval.config", self.retriever.config.label())
                    span.set_attribute(
                        "retrieval.results",
                        json.dumps([{"chunk_id": h.chunk_id, "score": round(h.score, 4)} for h in hits]),
                    )
                    result = search_result(query, k, hits, registry)
            elif name == "sql_query":
                result = await asyncio.to_thread(self.sql_tool.run, str(args.get("sql") or ""))
            elif name == "calculator":
                result = calculator(str(args.get("expression") or ""))
            else:
                result = ToolResult(f"Error: unknown tool {name!r}.", error=True)
            span.set_attribute("tool.error", result.error)
            if (content := tr.content(result.content)) is not None:
                span.add_event("tool.output", {"content": content})
            return result


def _assistant_tool_message(end: TurnEnd) -> dict[str, Any]:
    calls: list[ToolCall] = end.tool_calls
    return {
        "role": "assistant",
        "content": end.text or None,
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments or "{}"}}
            for c in calls
        ],
    }
