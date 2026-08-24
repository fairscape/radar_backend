"""Service settings.

Single ``Settings`` model loaded from environment + ``.env``. Singleton
via ``@lru_cache`` so dependencies share one instance per process. Tests
that need an override clear the cache and inject their own values.

Field naming matches the env var directly (``env_prefix=""``) so an
operator reading ``.env.example`` can map every line straight to a
class attribute.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Mirrors rag_lib.rag.llm.SUPPORTED_PROVIDERS; kept duplicated to avoid
# importing the RAG module just for validation (it would pull httpx +
# pydantic-ai during settings construction).
_SUPPORTED_LLM_PROVIDERS = {"ollama", "anthropic", "openai"}


class Settings(BaseSettings):
    RADAR_DB_PATH: Path = Path("data/radar.db")
    RADAR_DEFAULT_MAILTO: str = "demo@example.com"
    RADAR_VAULT_DIR: Path = Path("data/vault")
    RADAR_CHROMA_DIR: Path = Path("data/chroma")
    # Default LLM provider for /api/chat. One of: ollama | anthropic | openai.
    # Per-request override is honored by the chat endpoint; this just sets
    # the fallback when the request omits ``provider``.
    RADAR_LLM_PROVIDER: str = "ollama"
    RADAR_OLLAMA_URL: str = "http://localhost:11434"
    RADAR_OLLAMA_MODEL: str = "llama3.1:8b"
    # Per-request timeout for the chat LLM call. A 7B model answering a
    # full 20-chunk RAG prompt (~60k chars in) on a small GPU can run
    # well past 60s — the prior default — so the ceiling is generous.
    RADAR_OLLAMA_TIMEOUT: float = 6000.0
    # Third-party providers. API keys live ONLY in the gitignored .env
    # file — never in the DB, never returned by any API, never logged.
    # The provider's client raises LLMNotConfigured at construction
    # time when its key is missing, mapped to a 503 by the router with
    # an actionable hint.
    RADAR_ANTHROPIC_API_KEY: str | None = None
    RADAR_ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
    RADAR_OPENAI_API_KEY: str | None = None
    RADAR_OPENAI_MODEL: str = "gpt-4o-mini"
    # Shared per-request timeout (seconds) for the non-Ollama providers.
    # Ollama keeps its own much larger ceiling because local models are
    # CPU-bound for many seconds even on small prompts.
    RADAR_LLM_TIMEOUT: float = 120.0
    # Embedding model used by upload, chat retrieval, and the wizard's
    # draft creation. All three must agree, otherwise coherence joins
    # (which filter paper_embeddings by profile.embedding_model) come
    # back empty.
    RADAR_DEFAULT_EMBEDDING_MODEL: str = "specter2"
    # Embedder used specifically for the chat retrieval path. Indexed
    # into a parallel per-user ``vault_chat`` collection at upload time.
    # When set, PDF upload tries to write to both ``vault`` (SPECTER2,
    # for selectors/centroids) and ``vault_chat`` (this model, for RAG).
    # The chat-side index is *best-effort*: if ollama is unreachable or
    # the model isn't pulled, we log a warning and skip — upload still
    # succeeds. Set to empty string to disable the parallel index.
    RADAR_CHAT_EMBEDDING_MODEL: str = "mxbai-embed-large"
    # Default selector key when the wizard's create-draft body omits one.
    # Keep symmetric with RADAR_DEFAULT_EMBEDDING_MODEL so operators can
    # swap defaults without code changes.
    RADAR_DEFAULT_SELECTOR: str = "centroid"
    RADAR_CORS_ORIGINS: list[str] = ["*"]
    RADAR_LOG_JSON: bool = True
    # Optional path to a log file. When set, every record (request lines,
    # tracebacks, structlog events) is also written here via a rotating
    # handler so logs survive container restarts.
    RADAR_LOG_FILE: str | None = None
    RADAR_LOG_FILE_MAX_BYTES: int = 10_000_000
    RADAR_LOG_FILE_BACKUP_COUNT: int = 5
    RADAR_SCHEDULER_ENABLED: bool = True
    RADAR_HOST: str = "127.0.0.1"
    RADAR_PORT: int = 8000
    # Phase 12 — when False (dev default) requests without ``X-User-Email``
    # fall back to the demo user (id=1). Production ``.env`` flips this on
    # so the API rejects unauthenticated traffic with 401.
    RADAR_REQUIRE_AUTH: bool = False

    # UMLS concept extraction (Phase B+). On by default — it is part of the
    # upload path now, not an opt-in extra; the comment here said "disabled
    # by default" long after that stopped being true. Without the [umls]
    # extras and the SciSpacy model, each upload logs
    # ``vault.umls_extraction_skipped`` and stores no concepts; set this to
    # False to skip the attempt instead of failing it once per paper.
    RADAR_UMLS_ENABLED: bool = True
    RADAR_UMLS_SPACY_MODEL: str = "en_core_sci_lg"
    RADAR_UMLS_MIN_CONFIDENCE: float = 0.7
    RADAR_UMLS_MAX_CONCEPTS: int = 30
    # UMLS topics are a *candidate list* the user prunes in wizard step 3,
    # not a curated set, so the cost of an extra wrong one is a toggle
    # while the cost of a missing right one is that it can never be
    # chosen. Four was too tight to serve that: on a type-2 diabetes
    # profile the four highest-similarity matches were all generic
    # ("Various Academic Research Studies", "Ethics in Clinical
    # Research" — generic concepts match generic topic names, so they
    # score above a specific one) and every diabetes topic fell outside
    # the cut.
    RADAR_UMLS_MAX_TOPIC_ADDITIONS: int = 10
    RADAR_UMLS_MIN_TOPIC_SIMILARITY: float = 0.40
    RADAR_UMLS_EMBEDDING_MODEL: str = "mxbai-embed-large"
    RADAR_UMLS_CACHE_DIR: Path = Path("data/umls_cache")

    # MedCPT cross-encoder reranker. Disabled by default; enable after
    # installing the [reranker] extras (transformers + torch).
    RADAR_RERANKER_ENABLED: bool = False
    RADAR_DEFAULT_RERANKER: str = "medcpt"
    RADAR_RERANKER_ALPHA: float = 0.4
    RADAR_RERANKER_BETA: float = 0.6
    RADAR_RERANKER_DEVICE: str = "cpu"
    RADAR_RERANKER_BATCH_SIZE: int = 64
    RADAR_RERANKER_MAX_QUERIES: int = 30
    RADAR_RERANKER_MIN_UMLS_CONFIDENCE: float = 0.7
    RADAR_RERANKER_AGGREGATION: str = "mean"
    RADAR_RERANKER_MODEL_ID: str = "ncbi/MedCPT-Cross-Encoder"
    # Query mode for the cross-encoder reranker.
    # "topic"   — use profile topic display names as queries (original)
    # "article" — use seed paper content (build_embedding_input) as queries
    RADAR_RERANKER_QUERY_MODE: str = "topic"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    @field_validator("RADAR_LLM_PROVIDER", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> str:
        if value is None or value == "":
            return "ollama"
        name = str(value).strip().lower()
        if name not in _SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"RADAR_LLM_PROVIDER must be one of "
                f"{sorted(_SUPPORTED_LLM_PROVIDERS)}, got {value!r}"
            )
        return name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
