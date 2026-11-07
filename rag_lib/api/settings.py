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

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Mirrors rag_lib.rag.llm.SUPPORTED_PROVIDERS; kept duplicated to avoid
# importing the RAG module just for validation (it would pull httpx +
# pydantic-ai during settings construction).
_SUPPORTED_LLM_PROVIDERS = {"ollama", "anthropic", "openai"}


class Settings(BaseSettings):
    RADAR_DB_PATH: Path = Path("data/radar.db")
    RADAR_DEFAULT_MAILTO: str = "demo@example.com"
    RADAR_VAULT_DIR: Path = Path("data/vault")
    # Largest PDF the vault takes, in megabytes. Uploads had no limit, and a
    # 30 MB file of random bytes was written to disk as-is.
    RADAR_UPLOAD_MAX_MB: int = 50
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
    # OpenAlex API key (free, https://openalex.org/rest-api). Sent as
    # ``api_key`` on every OpenAlex request when set: anonymous search is
    # rate-limited whenever OpenAlex is under load (2026-11-05, user's own
    # key). Never logged: rag_lib.openalex_client scrubs it from errors.
    RADAR_OPENALEX_API_KEY: str | None = None
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
    # Prosopia instance the seed-corpus importer reads
    # (``POST /api/import/prosopia``). Per-request ``base_url``
    # overrides it, so this is the default a plain slug resolves against.
    RADAR_PROSOPIA_BASE_URL: str = "https://prosopia.databio.org"
    # Any other Prosopia instance a request's ``base_url`` may name. The
    # server fetches that URL, so an open ``base_url`` let any caller make
    # it read loopback ports or a cloud metadata address.
    RADAR_PROSOPIA_ALLOWED_BASE_URLS: list[str] = []
    RADAR_CORS_ORIGINS: list[str] = ["*"]
    RADAR_LOG_JSON: bool = True
    # Optional path to a log file. When set, every record (request lines,
    # tracebacks, structlog events) is also written here via a rotating
    # handler so logs survive container restarts.
    RADAR_LOG_FILE: str | None = None
    RADAR_LOG_FILE_MAX_BYTES: int = 10_000_000
    RADAR_LOG_FILE_BACKUP_COUNT: int = 5
    RADAR_SCHEDULER_ENABLED: bool = True
    # The nightly per-interest scans (profile_schedules). Off -- the default
    # since 2026-10-15, when daily scans were switched off -- leaves the
    # scheduler running for scans, imports and sweeps someone starts, and
    # keeps every schedule row as it is for when this is turned back on.
    # Off in code, not only in a deployment's .env: a checkout without that
    # line would have registered a 04:00 job for every older interest.
    RADAR_NIGHTLY_GATHER_ENABLED: bool = False
    RADAR_HOST: str = "127.0.0.1"
    RADAR_PORT: int = 8000
    # Phase 12 — when False (dev default) requests without ``X-User-Email``
    # fall back to the demo user (id=1). Production ``.env`` flips this on
    # so the API rejects unauthenticated traffic with 401.
    RADAR_REQUIRE_AUTH: bool = False
    # With RADAR_REQUIRE_AUTH on and no sign-in provider, identity would be
    # whatever X-User-Email the request carries. That is only safe where a
    # proxy overwrites it (the test site's per-port Caddy), so it has to be
    # asked for here; otherwise the process refuses to start. Production
    # signs people in with the buttons and leaves this off, so a client
    # secret missing from its .env stops the backend instead of letting
    # anyone send the header and be anyone.
    RADAR_TRUST_EMAIL_HEADER: bool = False
    # Sign-in through OpenID Connect. Each provider is on when its client
    # secret is set; with any on, identity comes from the session cookie
    # the /auth routes set and there is no demo-user fallback. Prosopia's
    # issuer is RADAR_PROSOPIA_BASE_URL; Google's is accounts.google.com.
    BIOXPLORER_PROSOPIA_CLIENT_ID: str = "bioxplorer"
    BIOXPLORER_PROSOPIA_CLIENT_SECRET: str | None = None
    BIOXPLORER_GOOGLE_CLIENT_ID: str | None = None
    BIOXPLORER_GOOGLE_CLIENT_SECRET: str | None = None
    # The one callback every provider returns to; register it with each
    # (https://<host>/auth/login/callback).
    BIOXPLORER_OIDC_REDIRECT_URI: str | None = None
    # HMAC key for the session cookie. Required when any provider is on.
    BIOXPLORER_SESSION_SECRET: str | None = None
    BIOXPLORER_SESSION_DAYS: int = 14
    # True behind https; False lets the cookie ride plain-http dev.
    BIOXPLORER_COOKIE_SECURE: bool = False
    # Ceiling on how many candidates one gather may pull from OpenAlex
    # (trial scan, scan-now, and the daily job alike). None = each caller's
    # own default (1000 for the trial scan, 500 for scans). A CPU-only
    # deployment embeds every candidate in-process, so this is the knob
    # that bounds a scan's wall time. A publication-year backfill is not
    # bounded by it: it takes up to 100 papers per year asked for.
    RADAR_GATHER_MAX_CANDIDATES: int | None = None

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
    # The UMLS topics an interest gets (FUNCTIONAL_SPEC U.2,
    # rag_lib.umls.topic_mapper.merge_umls_topics): each concept's nearest
    # topic; from each seed its concepts of MIN_CONFIDENCE or more, PER_PAPER
    # of them by confidence; MAX_TOPIC_ADDITIONS for the interest. They are
    # added switched off: words matched literally (a misread "Baseline" is
    # dental cement at 1.0), so the user picks.
    RADAR_UMLS_TOPIC_MIN_CONFIDENCE: float = 0.9
    # ...and only a concept whose nearest topic is this close to its name
    # (UT-1): "BaseLine dental cement" -> "Dental materials and
    # restorations" is 0.665.
    RADAR_UMLS_TOPIC_MIN_SIMILARITY: float = 0.80
    RADAR_UMLS_TOPICS_PER_PAPER: int = 5
    RADAR_UMLS_MAX_TOPIC_ADDITIONS: int = 20
    RADAR_UMLS_EMBEDDING_MODEL: str = "mxbai-embed-large"
    # OpenAlex topics mapped from LLM-generated research topics (rag_lib.llm_topics):
    # a label's best match from MIN, more from EXTRA; at most MAX added per run.
    RADAR_LLM_TOPIC_MIN_SIMILARITY: float = 0.65
    RADAR_LLM_TOPIC_EXTRA_SIMILARITY: float = 0.80
    RADAR_LLM_MAX_TOPIC_ADDITIONS: int = 10
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

    @field_validator(
        "RADAR_ANTHROPIC_API_KEY", "RADAR_OPENAI_API_KEY", "RADAR_OPENALEX_API_KEY", "RADAR_LOG_FILE",
        "RADAR_GATHER_MAX_CANDIDATES", "BIOXPLORER_PROSOPIA_CLIENT_SECRET",
        "BIOXPLORER_GOOGLE_CLIENT_ID", "BIOXPLORER_GOOGLE_CLIENT_SECRET",
        "BIOXPLORER_OIDC_REDIRECT_URI", "BIOXPLORER_SESSION_SECRET", mode="before",
    )
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        # "KEY=" in .env means unset, as .env.example presents it; an int
        # field would otherwise refuse "" and stop the backend starting.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "RADAR_UMLS_MIN_CONFIDENCE", "RADAR_UMLS_TOPIC_MIN_CONFIDENCE", "RADAR_UMLS_TOPIC_MIN_SIMILARITY",
        "RADAR_RERANKER_MIN_UMLS_CONFIDENCE", "RADAR_LLM_TOPIC_MIN_SIMILARITY", "RADAR_LLM_TOPIC_EXTRA_SIMILARITY",
    )
    @classmethod
    def _a_score_bar(cls, value: float, info) -> float:
        # A confidence or similarity bar: 80 for 0.80 would quietly drop
        # every concept or topic (review 2026-11-06 L4), so it stops the start.
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{info.field_name} must be between 0 and 1, got {value!r}")
        return value

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

    @model_validator(mode="after")
    def _oidc_complete(self) -> "Settings":
        if bool(self.BIOXPLORER_GOOGLE_CLIENT_ID) != bool(self.BIOXPLORER_GOOGLE_CLIENT_SECRET):
            raise ValueError("set both BIOXPLORER_GOOGLE_CLIENT_ID and BIOXPLORER_GOOGLE_CLIENT_SECRET, or neither")
        if self.BIOXPLORER_PROSOPIA_CLIENT_SECRET or self.BIOXPLORER_GOOGLE_CLIENT_SECRET:
            missing = [
                name for name in ("BIOXPLORER_OIDC_REDIRECT_URI", "BIOXPLORER_SESSION_SECRET")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"sign-in is on but {', '.join(missing)} is not set")
            if len(self.BIOXPLORER_SESSION_SECRET) < 32:
                raise ValueError("BIOXPLORER_SESSION_SECRET must be at least 32 characters")
            if (self.BIOXPLORER_OIDC_REDIRECT_URI.startswith("https://")
                    and not self.BIOXPLORER_COOKIE_SECURE):
                raise ValueError("BIOXPLORER_OIDC_REDIRECT_URI is https, so set BIOXPLORER_COOKIE_SECURE=true")
            if self.RADAR_TRUST_EMAIL_HEADER:
                raise ValueError("sign-in is on, so X-User-Email is ignored: unset RADAR_TRUST_EMAIL_HEADER")
        elif self.BIOXPLORER_OIDC_REDIRECT_URI:
            # A sign-in deployment whose client secrets went missing. With
            # RADAR_REQUIRE_AUTH also gone it would otherwise start as the
            # single-user demo, everyone sharing account 1.
            raise ValueError(
                "BIOXPLORER_OIDC_REDIRECT_URI is set but no sign-in provider is "
                "(BIOXPLORER_GOOGLE_CLIENT_ID/_SECRET or BIOXPLORER_PROSOPIA_CLIENT_SECRET)"
            )
        elif self.RADAR_REQUIRE_AUTH and not self.RADAR_TRUST_EMAIL_HEADER:
            raise ValueError(
                "RADAR_REQUIRE_AUTH is on but no sign-in provider is set "
                "(BIOXPLORER_GOOGLE_CLIENT_ID/_SECRET or BIOXPLORER_PROSOPIA_CLIENT_SECRET); "
                "set RADAR_TRUST_EMAIL_HEADER=true only behind a proxy that overwrites X-User-Email"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
