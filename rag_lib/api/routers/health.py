"""GET /api/health — liveness + version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Annotated

from fastapi import APIRouter, Depends

from ..deps import get_settings
from ..schemas import Health
from ..settings import Settings

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
    return Health(
        status="ok",
        version=_service_version(),
        ollama_model=settings.RADAR_OLLAMA_MODEL,
    )
