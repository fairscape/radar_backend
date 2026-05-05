"""FastAPI application factory.

Wires settings → logging → CORS → routers, runs DB migrations on
startup, and reserves the lifespan slot Phase 7 will use to launch the
APScheduler instance. The module-level ``app`` is what uvicorn imports
(``rag_lib.api.app:app``).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as pkg_version

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_lib.db import apply_migrations, connect
from rag_lib.scheduler import build_scheduler, start as start_scheduler, stop as stop_scheduler

from .logging import configure_logging
from .routers import chat, health, profiles, radar, users, vault
from .settings import get_settings


def _service_version() -> str:
    try:
        return pkg_version("rag_lib")
    except PackageNotFoundError:
        return "0.0.0+unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(json=settings.RADAR_LOG_JSON)
    log = structlog.get_logger("rag_lib.api")

    conn = connect(settings.RADAR_DB_PATH)
    try:
        applied = apply_migrations(conn)
        if applied:
            log.info("migrations.applied", versions=applied)
        else:
            log.info("migrations.up_to_date")
    finally:
        conn.close()

    if settings.RADAR_SCHEDULER_ENABLED:
        scheduler = build_scheduler(settings)
        start_scheduler(scheduler, settings)
        app.state.scheduler = scheduler
    else:
        app.state.scheduler = None
        log.info("scheduler.disabled")

    log.info(
        "api.startup",
        version=_service_version(),
        db=str(settings.RADAR_DB_PATH),
        cors_origins=settings.RADAR_CORS_ORIGINS,
        scheduler_enabled=settings.RADAR_SCHEDULER_ENABLED,
    )

    try:
        yield
    finally:
        stop_scheduler(app.state.scheduler)
        log.info("api.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(json=settings.RADAR_LOG_JSON)

    app = FastAPI(
        title="RADAR API",
        version=_service_version(),
        lifespan=lifespan,
    )

    # Dev-only: allow any origin. Switch back to an explicit allow-list
    # before exposing this service publicly.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    access_log = structlog.get_logger("rag_lib.api.access")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            access_log.exception(
                "request.unhandled",
                method=request.method,
                path=request.url.path,
                query=request.url.query or None,
                client=request.client.host if request.client else None,
                duration_ms=duration_ms,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "internal server error"},
            )
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_fn = access_log.warning if response.status_code >= 500 else access_log.info
        log_fn(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(profiles.router, prefix="/api/profiles", tags=["profiles"])
    app.include_router(radar.router, prefix="/api/radar", tags=["radar"])
    app.include_router(vault.router, prefix="/api/vault", tags=["vault"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(users.router, prefix="/api/users", tags=["users"])

    # Top-level liveness for load-balancer / docker healthcheck use.
    @app.get("/health", include_in_schema=False)
    def root_health() -> dict:
        return {"status": "ok", "version": _service_version()}

    logging.getLogger("rag_lib.api").debug("app.created")
    return app


app = create_app()
