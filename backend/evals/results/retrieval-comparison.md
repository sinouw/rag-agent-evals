# Retrieval comparison — 2026-10-27 15:23 UTC

Golden set: 65 questions; embeddings `BAAI/bge-small-en-v1.5`; top-5; reranker `Xenova/ms-marco-MiniLM-L-6-v2` over 20 candidates.
Answer chunk = a top-5 chunk from the right document contains the question's evidence text.

| Pipeline | Doc Hit@1 | Doc MRR | Answer chunk Hit@5 | Answer chunk MRR (all) | Answer chunk MRR (core) | Answer chunk MRR (hard) | p50 latency |
|---|---|---|---|---|---|---|---|
| `dense` | 96% | 0.98 | 76% | 0.55 | 0.76 | 0.32 | 141 ms |
| `dense+cards` | 96% | 0.98 | 87% | 0.62 | 0.76 | 0.48 | 124 ms |
| `hybrid+cards` | 96% | 0.98 | 89% | 0.72 | 0.88 | 0.56 | 177 ms |
| `dense+cards+rerank` | 94% | 0.95 | 84% | 0.74 | 0.85 | 0.63 | 3325 ms |
| `hybrid+cards+rerank` | 94% | 0.95 | 84% | 0.75 | 0.84 | 0.66 | 4494 ms |
