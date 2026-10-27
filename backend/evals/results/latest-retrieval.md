# Eval report — 2026-10-27 14:25 UTC

- Answer model: not run

- Embeddings: `BAAI/bge-small-en-v1.5`, top-k 5, 1226 chunks from 25 documents
- Golden set: 32 questions (`evals/golden.jsonl`)

## Retrieval

| Level | Questions | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 |
|---|---|---|---|---|---|---|
| Document | 23 | 96% | 100% | 100% | 0.98 | 98% |
| Answer-bearing chunk | 23 | — | — | 100% | 0.76 | — |

- **Document**: a chunk from a document that holds the answer was retrieved.
- **Answer-bearing chunk**: a retrieved chunk from that document contains the evidence text recorded for the question in the golden set (stricter).
- p50 retrieval latency (embed query + vector search): 238.0 ms.

