"""GET /api/health — liveness + version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as pkg_version

from fastapi import APIRouter

from ..schemas import Health

router = APIRouter()


def _service_version() -> str:
    try:
        return pkg_version("rag_lib")
    except PackageNotFoundError:
        return "0.0.0+unknown"


@router.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", version=_service_version())
