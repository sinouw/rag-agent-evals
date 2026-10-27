> Core-set subset of run 20261027-154535 (the hard set hit the provider's daily token cap).

# Eval report — 2026-10-27 15:25 UTC

- Answer model: `groq/openai/gpt-oss-120b`
- Judge model: `groq/qwen/qwen3.8-27b`
- Embeddings: `BAAI/bge-small-en-v1.5`, top-k 5, 1226 chunks from 25 documents
- Golden set: the 32 core questions (`evals/golden.jsonl`)
- Provider rate-limit (HTTP 429) responses during the run: 106 (retried automatically; the waits are included in latency)

## Retrieval

Retrieval config: `hybrid+cards`

| Questions | Doc Hit@1 | Doc Hit@5 | Doc MRR | Answer chunk Hit@5 | Answer chunk MRR |
|---|---|---|---|---|---|
| all (23; 23 with evidence) | 96% | 100% | 0.97 | 100% | 0.88 |

- **Document**: a chunk from a document that holds the answer was retrieved.
- **Answer-bearing chunk**: a retrieved chunk from that document contains the evidence text recorded for the question in the golden set (stricter).
- p50 retrieval latency (embed query + vector search): 213.3 ms.

## End-to-end agent

| Correctness | Faithfulness | Citation precision | Tool selection | p50 / p95 latency | Avg tokens (in / out) | Avg cost / question | Errors |
|---|---|---|---|---|---|---|---|
| 100% | 100% | 98% | 86% | 9693 / 22967 ms | 2772 / 265 | $0.00061 | 0 |

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
| q01 | lookup | ✅ | ✅ | search_docs | pep-0008 | 1874 ms | The assistant correctly identifies the 79-character limit for code as per the reference and context, and accurately notes the 72-character limit for docstrings and comments which is also supported by the retrieved text. |
| q02 | lookup | ✅ | ✅ | search_docs | pep-0008 | 1605 ms | The assistant correctly identifies that the Python style guide recommends 4 spaces per indentation level, which is explicitly supported by the retrieved context [S1]. |
| q03 | lookup | ✅ | ✅ | search_docs | pep-0257, pep-0008 | 1970 ms | The assistant correctly identifies triple double quotes as the standard and raw triple double quotes for backslashes, which aligns with the reference answer and is fully supported by the retrieved context from PEP 257 and PEP 8. |
| q04 | lookup | ✅ | ✅ | search_docs | pep-0343 | 3684 ms | The assistant correctly identifies the two required methods as __enter__() and __exit__(), which is explicitly supported by the provided context in sources [S1] and [S2]. |
| q05 | lookup | ✅ | ✅ | search_docs | pep-0618 | 1436 ms | The assistant correctly identifies that a ValueError is raised, which matches the reference answer, and all specific details about the error message and timing are supported by the provided context (specifically S1 and S3). |
| q06 | lookup | ✅ | ✅ | search_docs | pep-0020 | 2798 ms | The assistant correctly identifies the number of aphorisms (20) and the number written down (19), which matches the reference answer and is directly supported by the text in context [S1]. |
| q07 | lookup | ✅ | ✅ | search_docs | pep-0703 | 8021 ms | The assistant correctly identifies the `--disable-gil` flag and the `Py_GIL_DISABLED` macro, both of which are explicitly supported by the provided context documents [S1] and [S2]. |
| q08 | lookup | ✅ | ✅ | search_docs | pep-0518 | 22621 ms | The assistant correctly identifies the [build-system] table and the required 'requires' key, and the description of the key as a list of PEP 508 dependency strings is directly supported by context [S2]. |
| q09 | lookup | ✅ | ✅ | search_docs | pep-0723 | 9693 ms | The assistant correctly identifies the start line format (`# /// TYPE`) and end line format (`# ///`) as well as the requirement for intermediate lines to be comments, which aligns perfectly with the reference answer and is fully supported by the provided context (specifically S3 and S10). |
| q10 | lookup | ✅ | ✅ | search_docs | pep-0668 | 3818 ms | The assistant correctly identifies the file name as EXTERNALLY-MANAGED and accurately describes its location and effect based on the provided context. |
| q11 | lookup | ✅ | ✅ | search_docs | pep-0657 | 11771 ms | The assistant correctly identifies the command-line option `python -Xno_debug_ranges` as specified in the reference and supported by context [S2], and accurately describes its function of disabling fine-grained error location data in tracebacks. |
| q12 | lookup | ✅ | ✅ | search_docs | pep-0709 | 14376 ms | The assistant correctly identifies the 2x (specifically 1.96x) speedup for the microbenchmark and the 11% speedup for the real-world benchmark, both of which are explicitly supported by the provided context in S1 and S4. |
| q13 | lookup | ✅ | ✅ | search_docs | pep-0584 | 11038 ms | The assistant correctly identifies the `/` and `/=` operators as the merge and update operators for dict, which aligns with the reference answer and is fully supported by the provided context (specifically S1 and S4). |
| q14 | lookup | ✅ | ✅ | search_docs | pep-0526 | 11692 ms | The assistant correctly identifies `typing.ClassVar` as the method to annotate class variables, matching the reference answer, and all specific details (syntax, examples, PEP 526 reference) are directly supported by the provided context. |
| q15 | lookup | ✅ | ✅ | search_docs | pep-3107 | 10656 ms | The assistant correctly identifies that function annotations are stored in the function's `__annotations__` attribute as a dictionary mapping parameter names and 'return' to their evaluated values, which is fully supported by the provided context (specifically S4). |
| q16 | lookup | ✅ | ✅ | search_docs | pep-0492 | 10215 ms | The assistant correctly identifies __aenter__ and __aexit__ as the two magic methods, which aligns with the reference answer and is fully supported by the provided context [S1]. |
| q17 | lookup | ✅ | ✅ | search_docs | pep-3333 | 10343 ms | The assistant correctly identifies the two arguments (environ dictionary and start_response callable) as stated in the reference, and all details provided are directly supported by the retrieved context [S2] and [S3]. |
| q18 | lookup | ✅ | ✅ | search_docs | pep-0380 | 11647 ms | The assistant correctly identifies the `yield from <expr>` syntax as the answer, and all details provided (such as the expression evaluating to an iterable and the delegation mechanism) are directly supported by the retrieved context [S3]. |
| q19 | lookup | ✅ | ✅ | search_docs | pep-0703 | 8383 ms | The assistant correctly identifies the PYTHONGIL environment variable and accurately describes its behavior (0 disables, 1 enables) based on the provided context [S3]. |
| q20 | lookup | ✅ | ✅ | search_docs | pep-0701 | 9879 ms | The assistant correctly identifies that reusing the delimiting quote character was impossible, and accurately notes that backslashes and comments were also prohibited, all of which are explicitly supported by the provided context. |
| q21 | multi_doc | ✅ | ✅ | search_docs | pep-0484 | 11200 ms | The assistant correctly identifies that PEP 484 standardized type hints on top of PEP 3107's syntax, and this claim is fully supported by the retrieved context [S3] which states PEP 484 introduces a standard vocabulary and tools for type hints. |
| q22 | multi_doc | ✅ | ✅ | search_docs, search_docs | pep-0604, pep-0484 | 22967 ms | The assistant correctly identifies that PEP 484 uses `Union[X, Y]` and PEP 604 introduces the `X / Y` syntax, and all specific claims (including the equivalence and the use of `Optional` as a shortcut in PEP 484) are supported by the provided context. |
| q23 | metadata_sql | ✅ | ✅ | sql_query | — | 6991 ms | The assistant's answer matches the reference answer of 22, and the context provided (SQL query result showing 22) supports this claim. |
| q24 | metadata_sql | ✅ | ✅ | sql_query | — | 3865 ms | The assistant correctly identifies PEPs 695, 701, and 709 as targeting Python 3.12, which matches the reference answer, and all specific details (titles and PEP numbers) are directly supported by the provided SQL query context. |
| q25 | metadata_sql | ✅ | ✅ | sql_query | — | 1127 ms | The assistant correctly identified PEP 257 and the date May 29, 2001, which matches the reference answer and is fully supported by the provided SQL query context. |
| q26 | metadata_sql | ✅ | ✅ | sql_query | — | 3814 ms | The assistant correctly identifies PEP 484 as the longest PEP, matching the reference answer, and the specific word count provided is directly supported by the retrieved SQL query context. |
| q27 | calculation | ✅ | ✅ | search_docs | pep-0008 | 7240 ms | The assistant correctly identifies the 79-character limit for code and 72-character limit for docstrings/comments, calculating the 7-character difference, which is fully supported by the provided context [S1]. |
| q28 | calculation | ✅ | ✅ | sql_query, calculator | — | 11411 ms | The assistant correctly identifies the creation years (2001 and 2023) from the context, calculates the difference as 22 years, and matches the reference answer. |
| q29 | calculation | ✅ | ✅ | sql_query, calculator | — | 3164 ms | The assistant's answer matches the reference answer's calculation and value, and all factual claims (total words, count of PEPs, and the resulting average) are directly supported by the provided SQL query results and calculator output. |
| q30 | unanswerable | ✅ | ✅ | search_docs, search_docs, search_docs, search_docs | — | 112645 ms | The assistant correctly identifies that the provided documents (which contain PEP 8, not the Google Style Guide) do not contain the specific information requested, aligning with the reference answer and the retrieved context. |
| q31 | unanswerable | ✅ | ✅ | search_docs, search_docs, search_docs | pep-0572 | 65453 ms | The assistant correctly identifies that the premise is false (PEP 572 targets Python 3.8, not 2.7, and is not about list slicing), which aligns with the reference answer, and all factual claims are supported by the provided context (specifically S1 and S11). |
| q32 | unanswerable | ✅ | ✅ | — | — | 4883 ms | The assistant correctly states that the documents do not contain the requested information, which aligns with the reference answer and is faithful to the lack of retrieved context. |
