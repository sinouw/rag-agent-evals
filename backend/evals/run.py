"""Eval suite: retrieval quality, answer correctness, faithfulness, citations and tool use.

    uv run python -m evals.run                    # full suite (needs an LLM API key)
    uv run python -m evals.run --retrieval-only   # retrieval metrics only, no API key

Writes `evals/results/<timestamp>-<model>.{json,md}` and copies the markdown to
`evals/results/latest.md` (or `latest-retrieval.md`). Exits non-zero when a
`--min-*` threshold is missed, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from app.agent import Agent
from app.llm import LLMClient, Usage, estimate_cost
from app.retrieval import Retriever

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

JUDGE_PROMPT = """You are grading an AI assistant's answer to a question about a document collection.

Question: {question}

Reference answer (ground truth): {reference}

Assistant's answer: {answer}

Context the assistant retrieved with its tools:
<context>
{context}
</context>

Grade two things independently:
1. correct: does the assistant's answer agree in substance with the reference answer? Extra correct
   detail is fine; missing the key fact, or contradicting it, is not. If the reference says the
   documents do not contain the answer, the assistant is correct only if it declines or says the
   documents don't say, rather than inventing an answer.
2. faithful: is every factual claim in the assistant's answer supported by the context above?
   An answer that only says the information isn't available is faithful.

Reply with JSON only: {{"correct": true|false, "faithful": true|false, "reason": "<one sentence>"}}"""


@dataclass
class GoldenItem:
    id: str
    category: str
    question: str
    reference_answer: str
    expected_docs: list[str] = field(default_factory=list)
    evidence: str | None = None  # text the answer-bearing chunk must contain
    expected_tool: str | None = None


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    return [GoldenItem(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- retrieval


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def first_relevant_rank(retrieved_docs: list[str], expected: list[str]) -> int | None:
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in expected:
            return rank
    return None


def retrieval_eval(retriever: Retriever, items: list[GoldenItem], k: int = 5) -> dict[str, Any]:
    """Doc-level retrieval metrics over items that name the documents holding the answer."""
    rows = []
    for item in items:
        if not item.expected_docs:
            continue
        started = time.perf_counter()
        hits = retriever.search(item.question, k)
        latency_ms = (time.perf_counter() - started) * 1000
        docs = [h.doc_id for h in hits]
        rank = first_relevant_rank(docs, item.expected_docs)
        evidence_rank = None
        if item.evidence:
            needle = _normalize(item.evidence)
            evidence_rank = next(
                (n for n, h in enumerate(hits, 1) if h.doc_id in item.expected_docs and needle in _normalize(h.text)),
                None,
            )
        rows.append(
            {
                "id": item.id,
                "category": item.category,
                "rank": rank,
                "evidence_rank": evidence_rank,
                "has_evidence": bool(item.evidence),
                "recall": len(set(docs) & set(item.expected_docs)) / len(item.expected_docs),
                "retrieved": docs,
                "latency_ms": round(latency_ms, 1),
            }
        )
    n = len(rows) or 1
    with_evidence = [r for r in rows if r["has_evidence"]]
    n_ev = len(with_evidence) or 1
    summary = {
        "questions": len(rows),
        "k": k,
        "hit_at_1": sum(1 for r in rows if r["rank"] == 1) / n,
        "hit_at_3": sum(1 for r in rows if r["rank"] and r["rank"] <= 3) / n,
        f"hit_at_{k}": sum(1 for r in rows if r["rank"]) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        f"recall_at_{k}": sum(r["recall"] for r in rows) / n,
        "evidence_questions": len(with_evidence),
        f"evidence_hit_at_{k}": sum(1 for r in with_evidence if r["evidence_rank"]) / n_ev,
        "evidence_mrr": sum(1 / r["evidence_rank"] for r in with_evidence if r["evidence_rank"]) / n_ev,
        "p50_latency_ms": round(statistics.median(r["latency_ms"] for r in rows), 1) if rows else 0.0,
    }
    return {"summary": summary, "rows": rows}


# ---------------------------------------------------------------- end-to-end


def parse_judgement(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return {"correct": False, "faithful": False, "reason": f"unparseable judge output: {text[:200]}"}
    return {
        "correct": bool(data.get("correct")),
        "faithful": bool(data.get("faithful")),
        "reason": str(data.get("reason", "")),
    }


async def judge_answer(
    judge: LLMClient, item: GoldenItem, answer: str, context: list[dict[str, str]]
) -> tuple[dict, Usage]:
    context_text = "\n\n".join(f"[{c['tool']}]\n{c['content']}" for c in context)[:16_000] or "(no tool calls)"
    prompt = JUDGE_PROMPT.format(
        question=item.question, reference=item.reference_answer, answer=answer or "(empty)", context=context_text
    )
    text, usage = await judge.complete([{"role": "user", "content": prompt}], json_mode=True)
    return parse_judgement(text), usage


async def run_item(agent: Agent, judge: LLMClient, item: GoldenItem) -> dict[str, Any]:
    done: dict[str, Any] | None = None
    error = None
    async for event in agent.run(item.question, include_context=True):
        if event["type"] == "done":
            done = event
        elif event["type"] == "error":
            error = event["message"]
    if done is None:
        return {
            "id": item.id,
            "category": item.category,
            "error": error or "no answer",
            "correct": False,
            "faithful": False,
        }

    verdict, judge_usage = await judge_answer(judge, item, done["answer"], done.get("context", []))
    cited_docs = [c["doc_id"] for c in done["citations"]]
    tools_used = [t["name"] for t in done["tools"]]
    row: dict[str, Any] = {
        "id": item.id,
        "category": item.category,
        "question": item.question,
        "answer": done["answer"],
        **verdict,
        "cited_docs": cited_docs,
        "citation_precision": (
            sum(1 for d in cited_docs if d in item.expected_docs) / len(cited_docs)
            if cited_docs and item.expected_docs
            else None
        ),
        "tools_used": tools_used,
        "tool_ok": (item.expected_tool in tools_used) if item.expected_tool else None,
        "steps": done["steps"],
        "latency_ms": done["latency_ms"],
        "input_tokens": done["usage"]["input_tokens"],
        "output_tokens": done["usage"]["output_tokens"],
        "cost_usd": done["cost_usd"],
        "judge_cost_usd": estimate_cost(judge.model, judge_usage),
        "trace_id": done["trace_id"],
    }
    return row


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(1 for v in values if v) / len(values) if values else None


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct * (len(ordered) - 1)))]


def summarize_e2e(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    latencies = [r["latency_ms"] for r in ok]
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["category"], []).append(r)

    def block(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "questions": len(group),
            "correctness": _rate(group, "correct"),
            "faithfulness": _rate(group, "faithful"),
            "citation_precision": _mean(group, "citation_precision"),
            "tool_selection": _rate(group, "tool_ok"),
        }

    costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
    return {
        **block(rows),
        "errors": len(rows) - len(ok),
        "p50_latency_ms": round(_percentile(latencies, 0.5)) if latencies else None,
        "p95_latency_ms": round(_percentile(latencies, 0.95)) if latencies else None,
        "avg_input_tokens": round(_mean(ok, "input_tokens") or 0),
        "avg_output_tokens": round(_mean(ok, "output_tokens") or 0),
        "avg_cost_usd": statistics.fmean(costs) if costs else None,
        "total_cost_usd": sum(costs) if costs else None,
        "by_category": {name: block(group) for name, group in sorted(groups.items())},
    }


async def agent_eval(agent: Agent, judge: LLMClient, items: list[GoldenItem], concurrency: int = 1) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(item: GoldenItem) -> dict[str, Any]:
        async with semaphore:
            row = await run_item(agent, judge, item)
            mark = "ok " if row.get("correct") else "ERR" if "error" in row else "x  "
            print(f"  {mark} {item.id} {item.category:<13} {row.get('latency_ms', '-')}ms", file=sys.stderr)
            return row

    rows = await asyncio.gather(*(bounded(i) for i in items))
    return {"summary": summarize_e2e(rows), "rows": rows}


# ---------------------------------------------------------------- report


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def e2e_summary_lines(s: dict[str, Any], heading: str) -> list[str]:
    """Summary and per-category tables for the end-to-end run (also written into the README)."""
    cost = "—" if s["avg_cost_usd"] is None else f"${s['avg_cost_usd']:.5f}"
    lines = [
        "| Correctness | Faithfulness | Citation precision | Tool selection | p50 / p95 latency | Avg tokens (in / out) | Avg cost / question | Errors |",
        "|---|---|---|---|---|---|---|---|",
        f"| {_pct(s['correctness'])} | {_pct(s['faithfulness'])} | {_pct(s['citation_precision'])} | {_pct(s['tool_selection'])} "
        f"| {s['p50_latency_ms']} / {s['p95_latency_ms']} ms | {s['avg_input_tokens']} / {s['avg_output_tokens']} | {cost} | {s['errors']} |",
        "",
        f"{heading} By category",
        "",
        "| Category | Questions | Correctness | Faithfulness | Citation precision | Tool selection |",
        "|---|---|---|---|---|---|",
    ]
    for name, b in s["by_category"].items():
        lines.append(
            f"| {name} | {b['questions']} | {_pct(b['correctness'])} | {_pct(b['faithfulness'])} "
            f"| {_pct(b['citation_precision'])} | {_pct(b['tool_selection'])} |"
        )
    return lines


README_START = "<!-- E2E_RESULTS:START -->"
README_END = "<!-- E2E_RESULTS:END -->"


def update_readme(readme: Path, result: dict[str, Any], report_name: str) -> None:
    """Replace the README's end-to-end results block with the numbers from this run."""
    text = readme.read_text()
    start, end = text.index(README_START) + len(README_START), text.index(README_END)
    meta = result["meta"]
    body = [
        "",
        f"Measured {meta['timestamp']} on all {meta['golden_questions']} golden questions. "
        f"Answer model `{meta['provider']}/{meta['model']}`, judge `{meta['judge']}`. "
        f"Full per-question report: [`backend/evals/results/{report_name}`](backend/evals/results/{report_name}).",
        "",
        f"Run on Groq's free tier: the provider returned {meta.get('rate_limited_responses', 0)} rate-limit (HTTP 429) "
        "responses, which the client retried after waiting. Those waits are included in the latency figures.",
        "",
        *e2e_summary_lines(result["e2e"]["summary"], heading="####"),
        "",
    ]
    readme.write_text(text[:start] + "\n".join(body) + text[end:])


def render_markdown(result: dict[str, Any]) -> str:
    meta = result["meta"]
    lines = [
        f"# Eval report — {meta['timestamp']}",
        "",
        f"- Answer model: `{meta.get('provider')}/{meta.get('model')}`"
        if meta.get("model")
        else "- Answer model: not run",
        f"- Judge model: `{meta.get('judge')}`" if meta.get("judge") else "",
        f"- Embeddings: `{meta['embedding_model']}`, top-k {meta['k']}, {meta['chunks']} chunks from {meta['documents']} documents",
        f"- Golden set: {meta['golden_questions']} questions (`evals/golden.jsonl`)",
        f"- Provider rate-limit (HTTP 429) responses during the run: {meta['rate_limited_responses']} "
        "(retried automatically; the waits are included in latency)"
        if "rate_limited_responses" in meta
        else None,
        "",
        "## Retrieval",
        "",
    ]
    r = result["retrieval"]["summary"]
    k = r["k"]
    lines += [
        "| Level | Questions | Hit@1 | Hit@3 | Hit@{k} | MRR | Recall@{k} |".replace("{k}", str(k)),
        "|---|---|---|---|---|---|---|",
        f"| Document | {r['questions']} | {_pct(r['hit_at_1'])} | {_pct(r['hit_at_3'])} | {_pct(r[f'hit_at_{k}'])} "
        f"| {r['mrr']:.2f} | {_pct(r[f'recall_at_{k}'])} |",
        f"| Answer-bearing chunk | {r['evidence_questions']} | — | — | {_pct(r[f'evidence_hit_at_{k}'])} "
        f"| {r['evidence_mrr']:.2f} | — |",
        "",
        "- **Document**: a chunk from a document that holds the answer was retrieved.",
        "- **Answer-bearing chunk**: a retrieved chunk from that document contains the evidence text "
        "recorded for the question in the golden set (stricter).",
        f"- p50 retrieval latency (embed query + vector search): {r['p50_latency_ms']} ms.",
        "",
    ]
    misses = [
        row
        for row in result["retrieval"]["rows"]
        if not row["rank"] or (row["has_evidence"] and not row["evidence_rank"])
    ]
    if misses:
        lines += [
            "Misses: " + ", ".join(f"{m['id']} (got {', '.join(dict.fromkeys(m['retrieved']))})" for m in misses),
            "",
        ]

    e2e = result.get("e2e")
    if e2e:
        lines += ["## End-to-end agent", "", *e2e_summary_lines(e2e["summary"], heading="###")]
        lines += [
            "",
            "### Per question",
            "",
            "| ID | Category | Correct | Faithful | Tools | Cited | Latency | Judge's reason |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in e2e["rows"]:
            if "error" in row:
                lines.append(f"| {row['id']} | {row['category']} | error | | | | | {row['error'][:120]} |")
                continue
            tools = ", ".join(row["tools_used"]) or "—"
            cited = ", ".join(dict.fromkeys(row["cited_docs"])) or "—"
            reason = row["reason"].replace("|", "/")
            lines.append(
                f"| {row['id']} | {row['category']} | {'✅' if row['correct'] else '❌'} | {'✅' if row['faithful'] else '❌'} "
                f"| {tools} | {cited} | {row['latency_ms']} ms | {reason} |"
            )
    return "\n".join(line for line in lines if line is not None) + "\n"


# ---------------------------------------------------------------- CLI


def main() -> int:
    from app.config import get_settings
    from app.llm import create_llm
    from app.main import build_components

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-only", action="store_true", help="skip the agent and judge (no API key needed)")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--ids", help="comma-separated question ids to run")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--update-readme", action="store_true", help="write the end-to-end results into README.md")
    parser.add_argument("--min-hit-rate", type=float, help="fail if Hit@k is below this (0-1)")
    parser.add_argument("--min-correctness", type=float, help="fail if answer correctness is below this (0-1)")
    parser.add_argument("--min-faithfulness", type=float, help="fail if faithfulness is below this (0-1)")
    args = parser.parse_args()

    settings = get_settings()
    items = load_golden(args.golden)
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [i for i in items if i.id in wanted]
    if args.limit:
        items = items[: args.limit]

    if args.retrieval_only:
        settings = settings.model_copy(
            update={"llm_provider": "openai", "llm_api_key": SecretStr("unused-for-retrieval")}
        )
    components = build_components(settings)
    retriever = components.retriever

    from app.ingest import chunk_document, load_corpus

    docs = load_corpus(settings.corpus_dir)
    meta: dict[str, Any] = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "embedding_model": retriever.embedder.name,
        "k": args.k,
        "documents": len(docs),
        "chunks": sum(len(chunk_document(d, settings.chunk_size, settings.chunk_overlap)) for d in docs),
        "golden_questions": len(items),
    }
    result: dict[str, Any] = {"meta": meta}
    print(f"Retrieval eval on {len(items)} questions…", file=sys.stderr)
    result["retrieval"] = retrieval_eval(retriever, items, args.k)

    if not args.retrieval_only:
        if components.agent is None:
            print(f"Cannot run the agent: {components.llm_error}", file=sys.stderr)
            return 2
        agent = components.agent
        judge_provider = settings.judge_provider or settings.llm_provider
        judge = create_llm(judge_provider, settings.judge_model, settings.api_key_for(judge_provider))
        meta.update(provider=agent.llm.provider, model=agent.llm.model, judge=f"{judge.provider}/{judge.model}")
        print(
            f"Agent eval with {agent.llm.provider}/{agent.llm.model}, judge {judge.provider}/{judge.model}…",
            file=sys.stderr,
        )
        result["e2e"] = asyncio.run(agent_eval(agent, judge, items, args.concurrency))
        meta["rate_limited_responses"] = getattr(agent.llm, "rate_limited_responses", 0) + getattr(
            judge, "rate_limited_responses", 0
        )

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    label = "retrieval" if args.retrieval_only else str(meta["model"]).replace("/", "_")
    markdown = render_markdown(result)
    (args.out / f"{stamp}-{label}.json").write_text(json.dumps(result, indent=2, default=str))
    (args.out / f"{stamp}-{label}.md").write_text(markdown)
    (args.out / ("latest-retrieval.md" if args.retrieval_only else "latest.md")).write_text(markdown)
    print(markdown)
    if args.update_readme and "e2e" in result:
        update_readme(EVAL_DIR.parents[1] / "README.md", result, f"{stamp}-{label}.md")
        print("README.md updated with these results.", file=sys.stderr)

    failures = []
    hit_rate = result["retrieval"]["summary"][f"hit_at_{args.k}"]
    if args.min_hit_rate is not None and hit_rate < args.min_hit_rate:
        failures.append(f"Hit@{args.k} {hit_rate:.2f} < {args.min_hit_rate}")
    if "e2e" in result:
        s = result["e2e"]["summary"]
        if args.min_correctness is not None and (s["correctness"] or 0) < args.min_correctness:
            failures.append(f"correctness {s['correctness']:.2f} < {args.min_correctness}")
        if args.min_faithfulness is not None and (s["faithfulness"] or 0) < args.min_faithfulness:
            failures.append(f"faithfulness {s['faithfulness']:.2f} < {args.min_faithfulness}")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
