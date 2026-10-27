"""OpenTelemetry tracing for the agent.

Every request produces one trace: an `agent.run` root span with child spans for each
LLM call (model, tokens, cost, latency, prompt), each tool call, and the retrieval
steps inside `search_docs` (query, top-k hits with scores). Spans go to:

- an in-memory store, served at `GET /api/traces/{id}` and drawn by the UI,
- a JSONL file (`TRACE_LOG_PATH`) for offline inspection,
- any OTLP backend when `OTLP_ENDPOINT` is set (Jaeger in docker-compose).

Attribute names follow the OpenTelemetry GenAI semantic conventions where one exists.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span, Status, StatusCode


def span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    start, end = span.start_time or 0, span.end_time or 0
    return {
        "trace_id": format(span.context.trace_id, "032x"),
        "span_id": format(span.context.span_id, "016x"),
        "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
        "name": span.name,
        "start_ns": start,
        "duration_ms": round((end - start) / 1e6, 2),
        "status": span.status.status_code.name,
        "attributes": dict(span.attributes or {}),
        "events": [{"name": e.name, "attributes": dict(e.attributes or {})} for e in span.events],
    }


class InMemoryTraceStore(SpanExporter):
    """Keeps the most recent traces in memory, grouped by trace id."""

    def __init__(self, max_traces: int = 200) -> None:
        self._traces: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._max = max_traces
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            for span in spans:
                data = span_to_dict(span)
                self._traces.setdefault(data["trace_id"], []).append(data)
                self._traces.move_to_end(data["trace_id"])
            while len(self._traces) > self._max:
                self._traces.popitem(last=False)
        return SpanExportResult.SUCCESS

    def get(self, trace_id: str) -> list[dict[str, Any]] | None:
        with self._lock:
            spans = self._traces.get(trace_id)
            return sorted(spans, key=lambda s: s["start_ns"]) if spans else None

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(self._traces)[-limit:]
            roots = []
            for trace_id in reversed(ids):
                root = next((s for s in self._traces[trace_id] if s["parent_id"] is None), None)
                if root:
                    roots.append(root)
            return roots


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            for span in spans:
                fh.write(json.dumps(span_to_dict(span), default=str) + "\n")
        return SpanExportResult.SUCCESS


class Tracing:
    def __init__(
        self,
        service_name: str = "rag-agent",
        otlp_endpoint: str | None = None,
        jsonl_path: Path | None = None,
        capture_content: bool = True,
    ) -> None:
        self.store = InMemoryTraceStore()
        self.capture_content = capture_content
        self.provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        # Synchronous so a trace is complete by the time the response's final event is sent.
        self.provider.add_span_processor(SimpleSpanProcessor(self.store))
        if jsonl_path:
            self.provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(jsonl_path)))
        if otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            self.provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
        self.tracer = self.provider.get_tracer("rag-agent")

    @contextmanager
    def span(self, name: str, parent: Span | None = None, **attributes: Any) -> Iterator[Span]:
        """Start a span under an explicit parent (or as a new root when parent is None).

        Parents are passed explicitly instead of through the ambient context because the
        agent is an async generator: the context can't be safely attached across `yield`s.
        """
        ctx = trace.set_span_in_context(parent) if parent else otel_context.Context()
        span = self.tracer.start_span(name, context=ctx, attributes=_clean(attributes))
        try:
            yield span
        except BaseException as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            span.end()

    def content(self, value: Any, limit: int = 20_000) -> str | None:
        """Serialize prompt/response content for a span, unless content capture is off."""
        if not self.capture_content:
            return None
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        return text[:limit]

    def shutdown(self) -> None:
        self.provider.shutdown()


def _clean(attributes: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in attributes.items() if v is not None}


def trace_id_of(span: Span) -> str:
    return format(span.get_span_context().trace_id, "032x")
