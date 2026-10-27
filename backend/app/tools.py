"""Tools the agent can call: document search, a read-only SQL query over document
metadata, and a calculator. Each tool returns text for the model plus structured
data for the API/trace."""

from __future__ import annotations

import ast
import math
import operator
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.retrieval import Hit


@dataclass
class ToolResult:
    content: str  # what the model sees
    data: dict[str, Any] = field(default_factory=dict)  # what the UI/trace sees
    error: bool = False


# ---------------------------------------------------------------- calculator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt}


class CalculatorError(ValueError):
    pass


def safe_eval(expression: str) -> float | int:
    """Evaluate arithmetic without `eval`: only numbers, + - * / // % **, parentheses
    and a few functions are allowed."""
    if len(expression) > 200:
        raise CalculatorError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"invalid expression: {exc.msg}") from exc

    def ev(node: ast.AST):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise CalculatorError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise CalculatorError(f"unsupported syntax: {ast.dump(node)[:60]}")

    try:
        return ev(tree)
    except ZeroDivisionError as exc:
        raise CalculatorError("division by zero") from exc


def calculator(expression: str) -> ToolResult:
    try:
        value = safe_eval(expression)
    except CalculatorError as exc:
        return ToolResult(f"Error: {exc}", {"expression": expression}, error=True)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return ToolResult(f"{expression} = {value}", {"expression": expression, "result": value})


# ---------------------------------------------------------------- SQL


class SqlTool:
    """Read-only SELECT access to the document metadata table."""

    max_rows = 50

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def schema(self) -> str:
        if not self.db_path.exists():
            return "(metadata database not built yet)"
        with self._connect() as conn:
            rows = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table'").fetchall()
        return "\n".join(r[0] for r in rows)

    def _connect(self) -> sqlite3.Connection:
        # mode=ro opens the file read-only at the OS level; the authorizer below is a second guard.
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.set_authorizer(_read_only_authorizer)
        return conn

    def run(self, sql: str) -> ToolResult:
        statement = sql.strip().rstrip(";").strip()
        if ";" in statement:
            return ToolResult("Error: only a single statement is allowed.", {"sql": sql}, error=True)
        if not statement.lower().startswith(("select", "with")):
            return ToolResult("Error: only SELECT queries are allowed.", {"sql": sql}, error=True)
        try:
            with self._connect() as conn:
                cursor = conn.execute(statement)
                columns = [d[0] for d in cursor.description or []]
                rows = cursor.fetchmany(self.max_rows + 1)
        except sqlite3.Error as exc:
            return ToolResult(f"SQL error: {exc}", {"sql": sql}, error=True)
        truncated = len(rows) > self.max_rows
        rows = rows[: self.max_rows]
        lines = [" | ".join(columns)] + [" | ".join("NULL" if v is None else str(v) for v in row) for row in rows]
        if truncated:
            lines.append(f"(truncated to {self.max_rows} rows)")
        return ToolResult(
            "\n".join(lines),
            {"sql": statement, "columns": columns, "rows": [list(r) for r in rows], "truncated": truncated},
        )


def _read_only_authorizer(action: int, *_args: Any) -> int:
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


# ---------------------------------------------------------------- search


class SourceRegistry:
    """Numbers retrieved chunks [S1], [S2], ... across all searches in one request,
    so citations in the final answer can be resolved back to chunks."""

    def __init__(self) -> None:
        self.by_label: dict[str, Hit] = {}
        self._by_chunk: dict[str, str] = {}

    def label(self, hit: Hit) -> str:
        if hit.chunk_id not in self._by_chunk:
            label = f"S{len(self._by_chunk) + 1}"
            self._by_chunk[hit.chunk_id] = label
            self.by_label[label] = hit
        return self._by_chunk[hit.chunk_id]


def format_hits(hits: list[Hit], registry: SourceRegistry) -> str:
    if not hits:
        return "No matching passages found."
    blocks = []
    for hit in hits:
        location = f" › {hit.section}" if hit.section else ""
        if hit.page:
            location += f" (page {hit.page})"
        blocks.append(f"[{registry.label(hit)}] {hit.title}{location}\n{hit.text}")
    return "\n\n---\n\n".join(blocks)


def search_result(query: str, k: int, hits: list[Hit], registry: SourceRegistry) -> ToolResult:
    sources = [
        {
            "label": registry.label(h),
            "doc_id": h.doc_id,
            "title": h.title,
            "section": h.section,
            "score": round(h.score, 4),
        }
        for h in hits
    ]
    return ToolResult(format_hits(hits, registry), {"query": query, "k": k, "sources": sources})


# ---------------------------------------------------------------- schemas


def tool_schemas(sql_schema: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": (
                    "Semantic search over the document corpus. Returns the most relevant passages, "
                    "each labelled with a source id like [S1] to cite in the answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "A focused search query."},
                        "k": {"type": "integer", "description": "Number of passages (1-10, default 5)."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sql_query",
                "description": (
                    "Run a read-only SQLite SELECT over the document metadata table. Use it for "
                    "questions about many documents at once (counts, dates, statuses, versions, "
                    "lengths). Schema:\n" + sql_schema
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
                    "required": ["sql"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate an arithmetic expression, e.g. '(2023 - 2001) * 12'. Use for any arithmetic.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        },
    ]
