#!/usr/bin/env bash
# Produce the real end-to-end eval numbers and write them into README.md.
#
#   ./scripts/run-evals.sh            # all 32 golden questions
#   ./scripts/run-evals.sh --limit 5  # extra args go to `python -m evals.run`
#
# Reads GROQ_API_KEY / GEMINI_API_KEY from .env (values are never printed).
# The answering provider is LLM_PROVIDER (default groq, model openai/gpt-oss-120b). If both
# keys are set, the other provider judges; with only a Groq key, qwen/qwen3.8-27b judges.
# Either way a model never grades its own answers.
set -euo pipefail
cd "$(dirname "$0")/.."

has_key() { [ -n "${!1:-}" ] || grep -Eq "^$1=[^[:space:]]+" .env 2>/dev/null; }
env_value() { grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' || true; }

if ! has_key GROQ_API_KEY && ! has_key GEMINI_API_KEY; then
  echo "No API key: add GROQ_API_KEY and/or GEMINI_API_KEY to .env (see .env.example)." >&2
  exit 2
fi

provider="${LLM_PROVIDER:-$(env_value LLM_PROVIDER)}"
if [ -z "$provider" ]; then
  if has_key GROQ_API_KEY; then provider=groq; else provider=gemini; fi
fi
export LLM_PROVIDER="$provider"
if [ -z "${JUDGE_PROVIDER:-$(env_value JUDGE_PROVIDER)}" ]; then
  if [ "$provider" = groq ] && has_key GEMINI_API_KEY; then export JUDGE_PROVIDER=gemini; fi
  if [ "$provider" = gemini ] && has_key GROQ_API_KEY; then export JUDGE_PROVIDER=groq; fi
fi
# Only one provider available: judge with a different model family on it.
if [ -z "${JUDGE_PROVIDER:-}" ] && [ -z "${JUDGE_MODEL:-$(env_value JUDGE_MODEL)}" ] && [ "$provider" = groq ]; then
  export JUDGE_MODEL=qwen/qwen3.8-27b
fi

cd backend
uv sync --quiet
# The first run indexes the corpus automatically (a few minutes on CPU).
uv run python -m evals.run --update-readme "$@"
echo "Reports: backend/evals/results/ (latest.md). README.md eval section updated." >&2
