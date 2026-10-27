# Eval report — 2026-10-27 14:47 UTC

- Answer model: `groq/openai/gpt-oss-120b`
- Judge model: `groq/qwen/qwen3.8-27b`
- Embeddings: `BAAI/bge-small-en-v1.5`, top-k 5, 1226 chunks from 25 documents
- Golden set: 32 questions (`evals/golden.jsonl`)
- Provider rate-limit (HTTP 429) responses during the run: 50 (retried automatically; the waits are included in latency)

## Retrieval

| Level | Questions | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 |
|---|---|---|---|---|---|---|
| Document | 23 | 96% | 100% | 100% | 0.98 | 98% |
| Answer-bearing chunk | 23 | — | — | 100% | 0.76 | — |

- **Document**: a chunk from a document that holds the answer was retrieved.
- **Answer-bearing chunk**: a retrieved chunk from that document contains the evidence text recorded for the question in the golden set (stricter).
- p50 retrieval latency (embed query + vector search): 51.3 ms.

## End-to-end agent

| Correctness | Faithfulness | Citation precision | Tool selection | p50 / p95 latency | Avg tokens (in / out) | Avg cost / question | Errors |
|---|---|---|---|---|---|---|---|
| 100% | 100% | 98% | 86% | 10278 / 47898 ms | 3184 / 271 | $0.00068 | 0 |

### By category

| Category | Questions | Correctness | Faithfulness | Citation precision | Tool selection |
|---|---|---|---|---|---|
| calculation | 3 | 100% | 100% | 100% | 67% |
| lookup | 20 | 100% | 100% | 98% | — |
| metadata_sql | 4 | 100% | 100% | — | 100% |
| multi_doc | 2 | 100% | 100% | 100% | — |
| unanswerable | 3 | 100% | 100% | — | — |

### Per question

| ID | Category | Correct | Faithful | Tools | Cited | Latency | Judge's reason |
|---|---|---|---|---|---|---|---|
| q01 | lookup | ✅ | ✅ | search_docs | pep-0008 | 1198 ms | The assistant correctly identifies the 79-character limit as recommended by PEP 8, which is directly supported by the retrieved context [S1]. |
| q02 | lookup | ✅ | ✅ | search_docs | pep-0008 | 3920 ms | The assistant correctly identifies that the Python style guide recommends 4 spaces per indentation level, which is explicitly stated in the provided context [S2]. |
| q03 | lookup | ✅ | ✅ | search_docs | pep-0257, pep-0008 | 3603 ms | The assistant correctly identifies triple double quotes as the standard and raw triple double quotes for backslashes, which aligns with the reference answer and is fully supported by the retrieved context (specifically S3 and S1). |
| q04 | lookup | ✅ | ✅ | search_docs | pep-0343 | 2581 ms | The assistant correctly identifies the two required methods as __enter__() and __exit__(), which is explicitly supported by the retrieved context in source [S1]. |
| q05 | lookup | ✅ | ✅ | search_docs | pep-0618 | 7357 ms | The assistant correctly identifies that a ValueError is raised when iterables have different lengths with strict=True, and this claim is fully supported by the provided context [S1]. |
| q06 | lookup | ✅ | ✅ | search_docs | pep-0020 | 14716 ms | The assistant correctly identifies the numbers 20 and 19 as stated in the reference, and these facts are directly supported by the retrieved context [S1]. |
| q07 | lookup | ✅ | ✅ | search_docs | pep-0703 | 13669 ms | The assistant correctly identifies the `--disable-gil` option and the `Py_GIL_DISABLED` macro, both of which are explicitly supported by the provided context in [S1]. |
| q08 | lookup | ✅ | ✅ | search_docs | pep-0518 | 10079 ms | The assistant correctly identifies the [build-system] table and the required 'requires' key, and both claims are directly supported by the provided context [S2]. |
| q09 | lookup | ✅ | ✅ | search_docs, search_docs | pep-0723 | 23114 ms | The assistant correctly identifies the start marker as `# /// <type>` and the end marker as `# ///`, which aligns with the reference answer and is fully supported by the specification details and examples in context [S2]. |
| q10 | lookup | ✅ | ✅ | search_docs | pep-0668 | 9342 ms | The assistant correctly identifies the file name as EXTERNALLY-MANAGED and accurately states it is placed in the standard library directory, both of which are supported by the provided context. |
| q11 | lookup | ✅ | ✅ | search_docs | pep-0657 | 19596 ms | The assistant correctly identifies the command-line option `python -Xno_debug_ranges` as specified in the reference answer, and this fact is explicitly supported by the retrieved context in [S2]. |
| q12 | lookup | ✅ | ✅ | search_docs | pep-0709 | 10350 ms | The assistant's answer correctly identifies the ~2x (specifically 1.96x) speedup for the microbenchmark and the 11% speedup for the real-world benchmark, both of which are explicitly supported by the provided context in sources [S2] and [S6]. |
| q13 | lookup | ✅ | ✅ | search_docs | pep-0584 | 13698 ms | The assistant correctly identifies the `/` and `/=` operators as the merge and update operators added to dict, which aligns with the reference answer and is fully supported by the provided context (specifically S1 and S4). |
| q14 | lookup | ✅ | ✅ | search_docs | pep-0526 | 14881 ms | The assistant correctly identifies `typing.ClassVar` as the method to annotate class variables, which aligns with the reference answer, and all specific claims and code examples are supported by the provided context (specifically S1 and S3). |
| q15 | lookup | ✅ | ✅ | search_docs | pep-3107 | 15888 ms | The assistant correctly identifies that function annotations are stored in the `__annotations__` attribute as a dictionary mapping parameter names and 'return' to their values, which aligns with the reference answer and is fully supported by the provided context [S2]. |
| q16 | lookup | ✅ | ✅ | search_docs | pep-0492 | 13369 ms | The assistant correctly identifies `__aenter__` and `__aexit__` as the two magic methods, which aligns with the reference answer and is fully supported by the provided context [S1]. |
| q17 | lookup | ✅ | ✅ | search_docs | pep-3333 | 10604 ms | The assistant correctly identifies the two arguments (environ dictionary and start_response callable) as stated in the reference, and all details provided are directly supported by the retrieved context [S1]. |
| q18 | lookup | ✅ | ✅ | search_docs | pep-0380 | 8073 ms | The assistant correctly identifies the `yield from <expr>` syntax as per the reference and context, and all additional details regarding delegation and return values are supported by the provided text. |
| q19 | lookup | ✅ | ✅ | search_docs | pep-0703 | 14597 ms | The assistant correctly identifies the PYTHONGIL environment variable and accurately describes its behavior (0 to disable, 1 to enable) based on the provided context [S4]. |
| q20 | lookup | ✅ | ✅ | search_docs, search_docs | pep-0701 | 41709 ms | The assistant correctly identifies that reusing the delimiting quote character was impossible before PEP 701, which aligns with the reference answer's primary point, and the specific example and error message provided are directly supported by the retrieved context [S1]. |
| q21 | multi_doc | ✅ | ✅ | search_docs | pep-0484 | 10278 ms | The assistant correctly identifies that PEP 484 standardized type hints and the typing module on top of PEP 3107's syntax, which aligns with the reference answer and is fully supported by the provided context [S2]. |
| q22 | multi_doc | ✅ | ✅ | search_docs | pep-0604 | 5289 ms | The assistant correctly identifies that PEP 604 uses the `/` operator while PEP 484 uses `typing.Union`, and all claims are directly supported by the provided context documents [S1] and [S2]. |
| q23 | metadata_sql | ✅ | ✅ | sql_query | — | 2208 ms | The assistant's answer matches the reference answer of 22, and the context provided (SQL query result showing 22) supports this claim. |
| q24 | metadata_sql | ✅ | ✅ | sql_query | — | 2199 ms | The assistant correctly identifies PEP 695, 701, and 709 as the targets for Python 3.12, which matches the reference answer, and the specific PEP numbers and titles are directly supported by the provided SQL query context. |
| q25 | metadata_sql | ✅ | ✅ | sql_query | — | 6287 ms | The assistant correctly identifies PEP 257 and the date 2001-05-29, which matches the reference answer and is fully supported by the provided SQL query context. |
| q26 | metadata_sql | ✅ | ✅ | sql_query | — | 6467 ms | The assistant correctly identifies PEP 484 as the longest PEP, matching the reference answer, and the specific word count provided is directly supported by the retrieved SQL query context. |
| q27 | calculation | ✅ | ✅ | search_docs | pep-0008 | 9947 ms | The assistant correctly identifies the 7-character difference between the 79-character code limit and the 72-character docstring/comment limit, and both specific values are explicitly supported by the provided context [S1]. |
| q28 | calculation | ✅ | ✅ | sql_query, calculator | — | 4635 ms | The assistant correctly identifies the creation years (2001 and 2023) from the context, calculates the difference as 22 years, and matches the reference answer. |
| q29 | calculation | ✅ | ✅ | sql_query, calculator | — | 4294 ms | The assistant's answer matches the reference answer's calculation (14143 / 3 ≈ 4714.33) and all factual claims are directly supported by the provided SQL query results and calculator output. |
| q30 | unanswerable | ✅ | ✅ | search_docs, search_docs, search_docs, search_docs | — | 124628 ms | The assistant correctly identifies that the provided documents (which are PEP 8) do not contain information about the Google Python Style Guide, aligning with the reference answer's core point that the corpus lacks the specific document requested. |
| q31 | unanswerable | ✅ | ✅ | search_docs, search_docs | pep-0572 | 47898 ms | The assistant correctly identifies that the documents do not support the premise of a walrus operator for Python 2.7 list slicing, and accurately cites PEP 572 as the source for the walrus operator in Python 3.8, which is supported by the provided context. |
| q32 | unanswerable | ✅ | ✅ | search_docs, search_docs, search_docs, search_docs | — | 85952 ms | The assistant correctly identifies that the provided documents do not contain the requested download statistics, which aligns with the reference answer and is supported by the lack of such data in the retrieved context. |
