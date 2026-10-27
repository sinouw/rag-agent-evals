# Eval report — 2026-10-27 15:48 UTC

- Answer model: not run

- Embeddings: `BAAI/bge-small-en-v1.5`, top-k 5, 1226 chunks from 25 documents
- Golden set: 65 questions (`evals/golden.jsonl`)

## Retrieval

Retrieval config: `hybrid+cards`

| Questions | Doc Hit@1 | Doc Hit@5 | Doc MRR | Answer chunk Hit@5 | Answer chunk MRR |
|---|---|---|---|---|---|
| all (48; 45 with evidence) | 96% | 100% | 0.98 | 89% | 0.72 |
| core (23; 23 with evidence) | 96% | 100% | 0.97 | 100% | 0.88 |
| hard (25; 22 with evidence) | 96% | 100% | 0.98 | 77% | 0.56 |

- **Document**: a chunk from a document that holds the answer was retrieved.
- **Answer-bearing chunk**: a retrieved chunk from that document contains the evidence text recorded for the question in the golden set (stricter).
- p50 retrieval latency (embed query + vector search): 87.8 ms.

Misses: h10 (got pep-0492), h11 (got pep-3333), h13 (got pep-0380, pep-0695, pep-0498), h16 (got pep-0657), a01 (got pep-0557, pep-0572)

