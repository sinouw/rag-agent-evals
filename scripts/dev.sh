#!/usr/bin/env bash
# Run the stack without Docker: API on :8000 (embedded Qdrant under data/qdrant), UI on :5173.
# Override the ports with API_PORT / WEB_PORT if they are taken.
# Needs uv and Node 20+. The first start indexes the corpus, which takes a few minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

(cd backend && uv sync --quiet)
(cd frontend && [ -d node_modules ] || npm ci --silent)

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"

trap 'kill 0' EXIT INT TERM
(cd backend && uv run uvicorn app.main:app --port "$API_PORT") &
(cd frontend && API_PROXY_TARGET="http://localhost:$API_PORT" npm run dev -- --port "$WEB_PORT" --strictPort) &
wait
