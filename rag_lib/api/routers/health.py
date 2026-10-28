"""GET /api/health — liveness + version + active LLM."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Annotated

from fastapi import APIRouter, Depends

from ..deps import get_settings
from ..schemas import Health
from ..settings import Settings
from ... import openalex_status
from ...rag import llm as rag_llm

router = APIRouter()


def _service_version() -> str:
    try:
        return pkg_version("rag_lib")
    except PackageNotFoundError:
        return "0.0.0+unknown"


@router.get("/health", response_model=Health)
def health(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Health:
    provider = rag_llm.resolve_provider(settings)
    model = rag_llm.provider_model(settings, provider)
    return Health(
        status="ok",
        version=_service_version(),
        # Back-compat: populate only when the active provider is Ollama
        # so legacy frontends keep reading a meaningful model label.
        ollama_model=settings.RADAR_OLLAMA_MODEL if provider == "ollama" else None,
        llm_provider=provider,
        llm_model=model,
    )


@router.get("/openalex/status")
def openalex_availability() -> dict:
    """Whether OpenAlex is paused (quota used up, API key wanted), and until
    when: the frontend's banner, and its upload buttons, follow this."""
    return openalex_status.status()
