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

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    RADAR_DB_PATH: Path = Path("data/radar.db")
    RADAR_DEFAULT_MAILTO: str = "demo@example.com"
    RADAR_VAULT_DIR: Path = Path("data/vault")
    RADAR_CHROMA_DIR: Path = Path("data/chroma")
    RADAR_OLLAMA_URL: str = "http://localhost:11434"
    RADAR_OLLAMA_MODEL: str = "llama3.1:8b"
    # Embedding model used by upload, chat retrieval, and the wizard's
    # draft creation. All three must agree, otherwise coherence joins
    # (which filter paper_embeddings by profile.embedding_model) come
    # back empty.
    RADAR_DEFAULT_EMBEDDING_MODEL: str = "specter2"
    RADAR_CORS_ORIGINS: list[str] = ["*"]
    RADAR_LOG_JSON: bool = True
    RADAR_SCHEDULER_ENABLED: bool = True
    RADAR_HOST: str = "127.0.0.1"
    RADAR_PORT: int = 8000
    # Phase 12 — when False (dev default) requests without ``X-User-Email``
    # fall back to the demo user (id=1). Production ``.env`` flips this on
    # so the API rejects unauthenticated traffic with 401.
    RADAR_REQUIRE_AUTH: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
