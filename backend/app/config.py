from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. Every field can be set from the environment or a `.env` file."""

    model_config = SettingsConfigDict(env_file=(REPO_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore")

    # Answering model. Any OpenAI-compatible provider works; see app/llm.py for presets.
    llm_provider: str = "groq"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_temperature: float = 0.0

    # Provider keys, read from the environment or `.env`. SecretStr keeps them out of reprs and logs.
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # Judge model used by the eval suite (defaults to the answering provider).
    judge_provider: str | None = None
    judge_model: str | None = None

    # Retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_backend: str = "fastembed"  # fastembed | hash (offline, used in tests)
    qdrant_url: str | None = None  # e.g. http://qdrant:6333; unset = embedded local mode
    qdrant_path: Path = REPO_ROOT / "data" / "qdrant"
    collection: str = "docs"
    top_k: int = 5
    retrieval_mode: str = "hybrid"  # dense | hybrid (dense + BM25 fused with RRF)
    retrieval_cards: bool = True  # search the per-document metadata cards too
    rerank: bool = False  # re-score candidates with a local cross-encoder
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Corpus + structured metadata (queried by the SQL tool)
    corpus_dir: Path = REPO_ROOT / "data" / "corpus"
    sqlite_path: Path = REPO_ROOT / "data" / "metadata.db"
    corpus_description: str = (
        "a collection of Python Enhancement Proposals (PEPs), the design documents "
        "that describe new Python features, processes and conventions"
    )
    auto_ingest: bool = True

    # Agent
    max_agent_steps: int = 5

    # Observability
    otlp_endpoint: str | None = None  # e.g. http://jaeger:4318/v1/traces
    trace_log_path: Path | None = REPO_ROOT / "data" / "traces.jsonl"
    trace_capture_content: bool = True
    service_name: str = "rag-agent"

    def api_key_for(self, provider: str) -> str | None:
        """LLM_API_KEY applies to the answering provider; otherwise use the provider's own key."""
        key = self.llm_api_key if provider == self.llm_provider and self.llm_api_key else None
        key = key or getattr(self, f"{provider}_api_key", None)
        return key.get_secret_value() if isinstance(key, SecretStr) else key


@lru_cache
def get_settings() -> Settings:
    return Settings()
