#!/usr/bin/env bash
# Run the stack without Docker: API on :8000 (embedded Qdrant under data/qdrant), UI on :5173.
# Needs uv and Node 20+. The first start indexes the corpus, which takes a few minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

(cd backend && uv sync --quiet)
(cd frontend && [ -d node_modules ] || npm ci --silent)

trap 'kill 0' EXIT INT TERM
(cd backend && uv run uvicorn app.main:app --port 8000) &
(cd frontend && npm run dev -- --port 5173 --strictPort) &
wait
