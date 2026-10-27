"""Compare retrieval pipelines on the golden set, one change at a time (no API key needed).

    uv run python -m evals.compare_retrieval

Writes `evals/results/retrieval-comparison.{md,json}`.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any

from app.config import get_settings
from app.main import build_components
from app.retrieval import RetrievalConfig
from evals.run import RESULTS_DIR, _pct, load_golden, retrieval_eval

CONFIGS = [
    RetrievalConfig("dense"),
    RetrievalConfig("dense", cards=True),
    RetrievalConfig("hybrid", cards=True),
    RetrievalConfig("dense", cards=True, rerank=True),
    RetrievalConfig("hybrid", cards=True, rerank=True),
]


def main() -> int:
    settings = get_settings().model_copy(update={"rerank": True})
    components = build_components(settings.model_copy(update={"llm_provider": "openai", "llm_api_key": None}))
    base = components.retriever
    reranker = base.reranker
    items = load_golden()
    k = 5

    results: dict[str, Any] = {}
    for config in CONFIGS:
        retriever = base.with_config(config)
        print(f"  {config.label()}…", file=sys.stderr)
        results[config.label()] = retrieval_eval(retriever, items, k)["summary"]

    lines = [
        f"# Retrieval comparison — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"Golden set: {len(items)} questions; embeddings `{base.embedder.name}`; top-{k}; "
        f"reranker `{reranker.name if reranker else '-'}` over {CONFIGS[-1].candidates} candidates.",
        "Answer chunk = a top-5 chunk from the right document contains the question's evidence text.",
        "",
        "| Pipeline | Doc Hit@1 | Doc MRR | Answer chunk Hit@5 | Answer chunk MRR (all) | Answer chunk MRR (core) "
        "| Answer chunk MRR (hard) | p50 latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, r in results.items():
        by = r["by_difficulty"]
        lines.append(
            f"| `{label}` | {_pct(r['hit_at_1'])} | {r['mrr']:.2f} | {_pct(r['evidence_hit_at_5'])} "
            f"| {r['evidence_mrr']:.2f} | {by['core']['evidence_mrr']:.2f} | {by['hard']['evidence_mrr']:.2f} "
            f"| {r['p50_latency_ms']:.0f} ms |"
        )
    markdown = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval-comparison.md").write_text(markdown)
    (RESULTS_DIR / "retrieval-comparison.json").write_text(json.dumps(results, indent=2))
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
