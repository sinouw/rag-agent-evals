# rag-agent-evals

A document question-answering agent with retrieval-augmented generation, tool use, tracing, and an eval suite that measures it.

Ask a question about a document collection. The agent searches a vector index, queries document metadata with SQL, does arithmetic with a calculator, and streams back an answer that cites the passages it used. Every request is traced (retrieval hits, prompts, tokens, cost, latency). One command runs a golden-set eval and writes a report.

The demo corpus is 25 Python Enhancement Proposals (PEPs), which are public domain. Any folder of PDF, Markdown, reStructuredText or text files works.

![The UI answering a question: streamed answer with a citation chip, the expanded source passage, the agent's search call with scored hits, and the request's trace waterfall](docs/screenshot.jpg)

*A real answer from `openai/gpt-oss-120b` on Groq: 1.09 s and 2,152 input / 248 output tokens. The trace shows two LLM calls around one search (query embedding 56 ms, vector search 11 ms). This answer also shows a known weakness: the model stated the 79 − 72 difference without calling the calculator.*

## What it does

- **Ingestion**: loads PDF (per page), Markdown, RST and text; splits on headings, then packs paragraphs into ~1,200-character chunks with overlap; prefixes each chunk with its document and section title before embedding; stores per-document metadata (status, type, created date, Python version, word count) in SQLite.
- **Retrieval**: local embeddings with [fastembed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`, ONNX, no API key) into **Qdrant** (cosine). Qdrant runs as a server under Docker Compose and in embedded mode otherwise, with the same client code.
- **Agent loop**: the model picks tools through native function calling:
  - `search_docs`: semantic search; results are labelled `[S1]`, `[S2]`… so citations resolve back to chunks
  - `sql_query`: read-only SELECT over the metadata table (opened `mode=ro` plus an SQLite authorizer that denies writes)
  - `calculator`: arithmetic through an AST whitelist, no `eval`
- **Streaming API**: FastAPI with Server-Sent Events: tool calls, tool results, answer tokens, then a final event with citations, token usage, estimated cost and latency.
- **Observability**: OpenTelemetry spans per request (`agent.run` → `llm.chat` / `tool.*` → `retrieval.embed_query` / `retrieval.vector_search`) with GenAI semantic-convention attributes: model, input/output tokens, cost, time to first token, prompts and completions, retrieved chunk ids and scores. Spans go to an in-memory store (shown as a waterfall in the UI), a JSONL file, and any OTLP backend (Jaeger in Compose).
- **Swappable LLM provider**: one OpenAI-compatible client covers Groq, Gemini, OpenAI and Ollama; switch with `LLM_PROVIDER` / `LLM_MODEL`.
- **Evals**: 32 hand-written questions with reference answers. Each answer was checked against the PEP text, across five categories: single-document lookup, multi-document, metadata (SQL), calculation, and unanswerable (the agent should decline).
- **UI**: React + TypeScript (Vite): streamed answer with clickable citation chips and source snippets, the agent's tool calls, latency/tokens/cost, and the trace waterfall.

## Architecture

```mermaid
flowchart LR
  UI[React UI] -- POST /api/ask<br/>SSE stream --> API[FastAPI]
  API --> Agent[Agent loop]
  Agent <--> LLM[(LLM provider<br/>Groq / Gemini)]
  Agent --> T1[search_docs]
  Agent --> T2[sql_query]
  Agent --> T3[calculator]
  T1 --> Emb[fastembed<br/>bge-small] --> Q[(Qdrant)]
  T2 --> DB[(SQLite metadata)]
  Ingest[Ingestion CLI] --> Emb
  Ingest --> DB
  Agent -. OpenTelemetry spans .-> Traces[(In-memory store / JSONL<br/>Jaeger via OTLP)]
  Evals[Eval runner] --> Agent
  Evals --> Judge[(LLM judge)]
```

## Eval results

### End-to-end agent

<!-- E2E_RESULTS:START -->
Measured 2026-10-27 14:47 UTC on all 32 golden questions. Answer model `groq/openai/gpt-oss-120b`, judge `groq/qwen/qwen3.8-27b`. Full per-question report: [`backend/evals/results/20261027-145651-openai_gpt-oss-120b.md`](backend/evals/results/20261027-145651-openai_gpt-oss-120b.md).

Run on Groq's free tier: the provider returned 50 rate-limit (HTTP 429) responses, which the client retried after waiting. Those waits are included in the latency figures.

| Correctness | Faithfulness | Citation precision | Tool selection | p50 / p95 latency | Avg tokens (in / out) | Avg cost / question | Errors |
|---|---|---|---|---|---|---|---|
| 100% | 100% | 98% | 86% | 10278 / 47898 ms | 3184 / 271 | $0.00068 | 0 |

#### By category

| Category | Questions | Correctness | Faithfulness | Citation precision | Tool selection |
|---|---|---|---|---|---|
| calculation | 3 | 100% | 100% | 100% | 67% |
| lookup | 20 | 100% | 100% | 98% | — |
| metadata_sql | 4 | 100% | 100% | — | 100% |
| multi_doc | 2 | 100% | 100% | 100% | — |
| unanswerable | 3 | 100% | 100% | — | — |
<!-- E2E_RESULTS:END -->

### Retrieval

Measured on the 23 golden questions that name the documents holding the answer (no API key involved). Generated by `uv run python -m evals.run --retrieval-only` and committed in [`backend/evals/results/latest-retrieval.md`](backend/evals/results/latest-retrieval.md):

| Level | Questions | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 |
|---|---|---|---|---|---|---|
| Document | 23 | 96% | 100% | 100% | 0.98 | 98% |
| Answer-bearing chunk | 23 | — | — | 100% | 0.76 | — |

- *Document*: a chunk from a document that holds the answer is in the top 5.
- *Answer-bearing chunk*: a top-5 chunk from that document contains the evidence text recorded for the question. This is stricter; MRR 0.76 means the right passage is usually, but not always, ranked first.
- p50 retrieval latency (query embedding + vector search) measured 51–238 ms across runs on a 2019 Intel MacBook Pro CPU.

**Reading the end-to-end numbers (run of 2026-10-27).**

- I read all 32 answers and agree with each of the judge's verdicts.
- The one tool-selection miss: on q27 the model stated a 79 − 72 = 7 difference without calling the calculator, despite the prompt asking it to.
- The one citation outside the expected document: on q03, besides PEP 257, the model also cited PEP 8, which does discuss docstrings.
- Latency is dominated by the free tier: the 50 rate-limited responses add their waits to p50 and p95. The three unanswerable questions were the slowest (48–125 s), because the agent searched 2–4 times before declining.

**Caveats.** The golden set is small (32 questions) and I wrote it myself after reading the documents, so it is likely easier than real user questions. Each PEP covers one distinct topic, which makes document-level retrieval easy; the answer-bearing-chunk metric is the more informative number. The judge is an LLM, and LLM judges make mistakes; per-question verdicts and reasons are in the report so they can be checked.

## Run it

### Without Docker (verified)

This is the path I ran and verified, on macOS (Intel) with Python 3.12 via [uv](https://docs.astral.sh/uv/) and Node 24.

```bash
cp .env.example .env        # then set GROQ_API_KEY and/or GEMINI_API_KEY
./scripts/dev.sh            # API on :8000, UI on http://localhost:5173
```

If those ports are taken, set `API_PORT` / `WEB_PORT`, e.g. `API_PORT=8011 WEB_PORT=5181 ./scripts/dev.sh`.

The first start indexes the corpus into `data/qdrant` (embedded Qdrant) and `data/metadata.db`. That took about 4–6 minutes on a 2019 Intel MacBook Pro CPU; later starts reuse the index. Without an API key, the API still starts and serves search and health, and the UI shows "LLM not configured".

### With Docker Compose (not verified locally)

```bash
cp .env.example .env
docker compose up --build   # UI http://localhost:3000, API :8000, Jaeger http://localhost:16686
```

Runs Qdrant as a server, the API (embedding model baked into the image), the UI behind nginx (which proxies `/api` with buffering off for SSE), and Jaeger receiving the API's traces over OTLP. **I have not run this locally**: Docker isn't installed on the machine this was built on. CI builds both images on every push.

## Run the evals

One command produces the real end-to-end numbers and writes them into this README:

```bash
./scripts/run-evals.sh
```

It reads `GROQ_API_KEY` / `GEMINI_API_KEY` from `.env` and answers with `LLM_PROVIDER`: by default Groq, model `openai/gpt-oss-120b`. A model never grades its own answers. When both keys are set, the other provider judges; with only a Groq key, `qwen/qwen3.8-27b` judges. Stop `scripts/dev.sh` first: the embedded index allows one process at a time. Free-tier rate limits can slow the run. The client retries on HTTP 429, and anything that still fails is listed as an error in the report.

Lower-level commands:

```bash
cd backend
uv run python -m evals.run --retrieval-only     # retrieval metrics, no API key
uv run python -m evals.run                      # full suite: agent + LLM judge (needs a key)
uv run python -m evals.run --ids q05,q23        # a subset
```

The full suite runs the agent on each question, then asks a judge model to grade **correctness** against the reference answer and **faithfulness** of the answer to the retrieved context. Alongside these it computes deterministic metrics:

- **citation precision**: cited chunks that come from an expected document
- **tool selection**: the expected tool was called, for SQL and calculator questions
- **latency**: p50/p95
- **tokens and estimated cost** per question

Reports go to `backend/evals/results/` as Markdown and JSON. `--min-hit-rate`, `--min-correctness` and `--min-faithfulness` make the command exit non-zero below a threshold, which is how CI gates on it. Set `JUDGE_PROVIDER` / `JUDGE_MODEL` to grade with a different model from the one answering.

## Tests and CI

```bash
cd backend && uv run pytest     # 50 tests: chunking, PDF loading, tools, agent loop, SSE API, tracing, eval metrics
cd frontend && npm run lint && npm run build
```

Backend tests mock every LLM call with a scripted model, and use a hashing embedder and in-memory Qdrant, so they need no API key and run offline in seconds. This covers the agent loop, the SSE endpoint, tracing, and the eval runner including the judge.

`.github/workflows/ci.yml` runs five jobs:

- **backend**: lint and tests
- **retrieval eval**: gated on Hit@5 ≥ 0.9, no key needed
- **agent eval**: makes real LLM calls, so it runs only when a `GROQ_API_KEY` or `GEMINI_API_KEY` repository secret is set; without one it skips
- **frontend**: lint and build
- **docker**: builds the Compose images

## Configuration

All settings are environment variables (or `.env`); see [`backend/app/config.py`](backend/app/config.py) and [`.env.example`](.env.example). The main ones: `LLM_PROVIDER`, `LLM_MODEL`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `JUDGE_PROVIDER`, `JUDGE_MODEL`, `QDRANT_URL`, `CORPUS_DIR`, `TOP_K`, `CHUNK_SIZE`, `OTLP_ENDPOINT`, `TRACE_CAPTURE_CONTENT` (set `false` to keep prompts out of traces).

API keys are read as secrets: they are masked in settings output, and key-like strings are redacted from error messages sent to the client.

To use your own documents, point `CORPUS_DIR` at a folder and run `uv run python -m app.ingest --recreate`. Also update `CORPUS_DESCRIPTION`, which goes into the system prompt.

## Layout

```
backend/app/        config, ingest, retrieval, tools, agent loop, tracing, FastAPI app
backend/evals/      golden.jsonl, eval runner, committed results
backend/tests/      pytest suite
frontend/src/       React UI (SSE client, citations, trace waterfall)
data/corpus/        25 PEPs (public domain / CC0), fetched by scripts/fetch_corpus.py at a pinned commit
```

## Limitations and next steps

- Dense retrieval only; adding BM25 hybrid search and a reranker, then measuring them with the same eval, is the obvious next experiment.
- Cost figures are estimates from the price table in `backend/app/llm.py` ($0.15 input / $0.75 output per million tokens for `openai/gpt-oss-120b`). I could not confirm Groq's current price list, so check it before relying on the cost numbers. Judge calls are not included in cost per question.
- The eval set is small and self-written; growing it with real user questions would make the numbers more meaningful.
