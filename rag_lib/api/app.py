"""FastAPI application factory.

Wires settings → logging → CORS → routers, runs DB migrations on
startup, and reserves the lifespan slot Phase 7 will use to launch the
APScheduler instance. The module-level ``app`` is what uvicorn imports
(``rag_lib.api.app:app``).
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
from importlib.metadata import PackageNotFoundError, version as pkg_version

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_lib.db import apply_migrations, connect
from rag_lib.scheduler import build_scheduler, start as start_scheduler, stop as stop_scheduler

from . import draft_lock, oidc, write_lease
from .logging import configure_logging
from .routers import auth, chat, health, profiles, prosopia, radar, researchers, users, vault
from .settings import get_settings


def _service_version() -> str:
    try:
        return pkg_version("rag_lib")
    except PackageNotFoundError:
        return "0.0.0+unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(
        json=settings.RADAR_LOG_JSON,
        file_path=settings.RADAR_LOG_FILE,
        file_max_bytes=settings.RADAR_LOG_FILE_MAX_BYTES,
        file_backup_count=settings.RADAR_LOG_FILE_BACKUP_COUNT,
    )
    log = structlog.get_logger("rag_lib.api")

    conn = connect(settings.RADAR_DB_PATH)
    try:
        applied = apply_migrations(conn)
        if applied:
            log.info("migrations.applied", versions=applied)
        else:
            log.info("migrations.up_to_date")
        # Only a process that runs the jobs owns the open runs: one started
        # with the scheduler off (tests, a second API worker) leaves them.
        if settings.RADAR_SCHEDULER_ENABLED:
            # Research-topic scratch left by older builds or a generation a
            # restart cut off (seed text and vectors, review BT3 / R7):
            # nothing generates during boot, so all of it goes.
            from pathlib import Path

            from ..interest_terms import _sweep_left_behind
            _sweep_left_behind(Path(settings.RADAR_VAULT_DIR).parent / "rp_profiles", older_than_s=0)
            from ..db.repos.gather_runs import close_interrupted, open_scan_profiles

            # Before closing: the scans a restart cut off had committed part
            # of their batch with the batch's own percentiles (as a failed
            # scan does, which re-scores before closing).
            cut_off = open_scan_profiles(conn)
            n_closed = close_interrupted(conn)
            if n_closed:
                log.warning("gather_runs.closed_interrupted", n=n_closed)
            for pid in cut_off:
                try:
                    from ..scheduler.jobs import rescore_pool
                    rescore_pool(conn, pid, settings)
                except Exception:  # noqa: BLE001 -- best effort at boot
                    log.exception("gather_runs.interrupted_rescore_failed", profile_id=pid)
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
        from ..embedders import get_embedder
        emb_model = settings.RADAR_DEFAULT_EMBEDDING_MODEL
        if emb_model and emb_model != "placeholder-v1":
            log.info("api.preload_embedder", model=emb_model)
            get_embedder(emb_model)("warmup")
            log.info("api.preload_embedder.done", model=emb_model)
    except Exception as exc:
        log.warning("api.preload_embedder.failed", error=str(exc)[:200])

    if settings.RADAR_UMLS_ENABLED:
        try:
            log.info("api.preload_umls")
            from ..umls.extractor import extract_umls_concepts
            extract_umls_concepts(
                "warmup",
                min_confidence=settings.RADAR_UMLS_MIN_CONFIDENCE,
                spacy_model=settings.RADAR_UMLS_SPACY_MODEL,
                cache_dir=str(settings.RADAR_UMLS_CACHE_DIR),
            )
            log.info("api.preload_umls.done")
        except Exception as exc:
            log.warning("api.preload_umls.failed", error=str(exc)[:200])

    try:
        yield
    finally:
        stop_scheduler(app.state.scheduler)
        log.info("api.shutdown")



_SECRET_FIELDS = {"api_key"}


def _holds_secret(tp, depth: int = 0) -> bool:
    """``tp`` (a body annotation) has a secret field at any depth: one
    named in ``_SECRET_FIELDS`` or typed ``SecretStr``."""
    import typing

    from pydantic import BaseModel, SecretStr

    if depth > 6 or tp is None:
        return False
    if tp is SecretStr:
        return True
    if any(_holds_secret(a, depth + 1) for a in typing.get_args(tp)):
        return True
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return any(name in _SECRET_FIELDS or _holds_secret(f.annotation, depth + 1)
                   for name, f in tp.model_fields.items())
    return False


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(
        json=settings.RADAR_LOG_JSON,
        file_path=settings.RADAR_LOG_FILE,
        file_max_bytes=settings.RADAR_LOG_FILE_MAX_BYTES,
        file_backup_count=settings.RADAR_LOG_FILE_BACKUP_COUNT,
    )

    app = FastAPI(
        title="BioXplorer API",
        version=_service_version(),
        lifespan=lifespan,
        # Which tab each request comes from: a draft open in one tab is
        # changed from that tab only (draft_lock, FUNCTIONAL_SPEC W.L).
        dependencies=[Depends(draft_lock.request_tab)],
    )

    # CORS: only the origins RADAR_CORS_ORIGINS lists (live: the vite dev
    # server; the SPA itself is same-origin and needs none). This used to
    # allow every origin with credentials, which was harmless while the
    # identity was a header a page elsewhere could not set, and is not with
    # the sign-in cookie: any page the browser sends that cookie from could
    # have read the answers. "*" stays possible for a header-model dev box,
    # but never with credentials.
    _origins = [o for o in settings.RADAR_CORS_ORIGINS if o != "*"]
    _any_origin = "*" in settings.RADAR_CORS_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if _any_origin else _origins,
        allow_credentials=not _any_origin,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # A 422 echoes what was sent ("input"). For a field holding a secret --
    # the Anthropic key typed in for one topic generation -- that would hand
    # the key back in the response and to anything that records responses.
    # SecretStr masks a valid string only; a key sent in the wrong shape (a
    # list, a number) came back verbatim. And an error about the body as a
    # whole (not an object, a missing field) echoes all of it, key
    # included (L16). So: when the route's body model holds a secret field
    # anywhere, no error of that request carries input or ctx.
    from fastapi.exceptions import RequestValidationError
    from fastapi.encoders import jsonable_encoder

    _secret_routes: dict[int, bool] = {}

    def _body_params(dependant, depth: int = 0):
        """The route's body parameters, its sub-dependencies' included."""
        yield from getattr(dependant, "body_params", None) or ()
        if depth < 8:
            for sub in getattr(dependant, "dependencies", None) or ():
                yield from _body_params(sub, depth + 1)

    def _route_holds_secret(request: Request) -> bool:
        route = request.scope.get("route")
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            return False
        if id(route) not in _secret_routes:
            try:
                _secret_routes[id(route)] = any(
                    p.name in _SECRET_FIELDS or _holds_secret(p.field_info.annotation)
                    for p in _body_params(dependant)
                )
            except Exception:     # unreadable: say less, never 500 a 422
                _secret_routes[id(route)] = True
        return _secret_routes[id(route)]

    def _json_safe(value, depth: int = 0):
        """An echoed input JSON can encode: NaN / inf as text, long text cut.

        A NaN threshold failed validation and then failed again here --
        ``JSONResponse`` refuses NaN -- so the 422 became a 500.
        """
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, str) and len(value) > 200:
            return value[:200] + "…"
        if depth < 4 and isinstance(value, dict):
            return {k: _json_safe(v, depth + 1) for k, v in list(value.items())[:50]}
        if depth < 4 and isinstance(value, (list, tuple)):
            return [_json_safe(v, depth + 1) for v in list(value)[:50]]
        return value

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        errors = []
        secret_body = _route_holds_secret(request)
        for err in exc.errors():
            if secret_body or _SECRET_FIELDS & {str(part) for part in err.get("loc", ())}:
                err = {k: v for k, v in err.items() if k not in ("input", "ctx")}
            errors.append(_json_safe(jsonable_encoder(err)))
        return JSONResponse(status_code=422, content={"detail": errors})

    from .services.wizard import TopicsChanging

    @app.exception_handler(TopicsChanging)
    async def _topics_changing(request: Request, exc: TopicsChanging):
        # The list was applied/removed under a rebuild three times over;
        # the rebuilt list is not written (it could undo one). Try again.
        return JSONResponse(status_code=409, content={
            "detail": "the interest's topics changed while the list was being rebuilt; try again"})

    from .services.wizard import SeedsChanging

    @app.exception_handler(SeedsChanging)
    async def _seeds_changing(request: Request, exc: SeedsChanging):
        # Seeds were added or removed in another tab under three fits in a
        # row; none was written (each would undo a change). Try again.
        return JSONResponse(status_code=409, content={
            "detail": "the interest's seeds changed while it was being fitted; try again"})

    @app.exception_handler(write_lease.EditingElsewhere)
    async def _editing_elsewhere(request: Request, exc: write_lease.EditingElsewhere):
        # ``editingElsewhere`` turns the page view-only (W.A R2).
        return JSONResponse(status_code=409, content={"detail": str(exc), "editingElsewhere": True})

    @app.exception_handler(draft_lock.DraftOpenElsewhere)
    async def _draft_open_elsewhere(request: Request, exc: draft_lock.DraftOpenElsewhere):
        # Refused wherever it was reached (the profiles router's guard, an
        # upload, an import, a Vault or researcher delete). ``openElsewhere``
        # lets the wizard show its "open in another tab" page.
        return JSONResponse(status_code=409, content={"detail": str(exc), "openElsewhere": True})

    from .services.wizard import SavedInterest

    @app.exception_handler(SavedInterest)
    async def _saved_interest(request: Request, exc: SavedInterest):
        # Refused in the service, whichever route reached it (spec E.6 R12).
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    from .services.researchers import ResearcherExists

    @app.exception_handler(ResearcherExists)
    async def _researcher_exists(request: Request, exc: ResearcherExists):
        # The Profiles page's import of a person already stored (O.2 D).
        # ``researcher`` lets the page open that profile.
        return JSONResponse(status_code=409, content={
            "detail": str(exc), "researcher": {"id": exc.researcher_id, "name": exc.name}})

    from ..openalex_status import OpenAlexUnavailable

    @app.exception_handler(OpenAlexUnavailable)
    async def _openalex_unavailable(request: Request, exc: OpenAlexUnavailable):
        # OpenAlex did not answer (quota, API key, outage): whatever the
        # request was doing stops, nothing is stored as if it had answered.
        # ``openalex`` lets the frontend show the pause banner at once.
        from .. import openalex_status
        return JSONResponse(status_code=503, content={
            "detail": exc.user_message, "openalex": openalex_status.status()})

    @app.exception_handler(OverflowError)
    async def _overflow(request: Request, exc: OverflowError):
        # An id past SQLite's 64-bit integers (2**63 and up) cannot name a
        # row; it was a 500 from the driver.
        return JSONResponse(status_code=404, content={"detail": "not found"})

    @app.exception_handler(sqlite3.IntegrityError)
    async def _integrity(request: Request, exc: sqlite3.IntegrityError):
        # A write that names a row deleted meanwhile (another tab deleted
        # the interest, draft or researcher mid-request) fails its foreign
        # key. Each route found so far turned this into a 404 one at a time;
        # this is the backstop for the ones not found. The request's own
        # connection is closed uncommitted by get_db.
        if "FOREIGN KEY" in str(exc):
            structlog.get_logger(__name__).warning(
                "request.target_gone", method=request.method, path=request.url.path)
            return JSONResponse(status_code=404, content={
                "detail": "what this request works on no longer exists (deleted in another tab?)"})
        structlog.get_logger(__name__).warning(
            "request.conflict", method=request.method, path=request.url.path, error=str(exc))
        return JSONResponse(status_code=409, content={"detail": "conflicting change; reload and try again"})

    # Cross-site writes. With the sign-in cookie the browser attaches the
    # identity by itself, so a page on another site (or another
    # *.bioxplorer.org subdomain, which SameSite=Lax counts as the same
    # site) could submit a form or fetch that deletes an interest as the
    # person viewing it. The SPA only ever writes same-origin, so a write
    # the browser marks cross-origin, or whose Origin is not ours, is
    # refused before any route runs. Reads are left alone: CORS keeps
    # their answers from other origins.
    _own_origins = set(_origins)
    if settings.BIOXPLORER_OIDC_REDIRECT_URI:
        _u = urlsplit(settings.BIOXPLORER_OIDC_REDIRECT_URI)
        _own_origins.add(f"{_u.scheme}://{_u.netloc}")

    @app.middleware("http")
    async def refuse_cross_site_writes(request: Request, call_next):
        if (request.method not in ("GET", "HEAD", "OPTIONS")
                and oidc.providers(settings)):
            site = request.headers.get("sec-fetch-site")
            origin = request.headers.get("origin")
            if (site is not None and site not in ("same-origin", "none")) or (
                    origin is not None and origin not in _own_origins):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "cross-site request refused"},
                )
        return await call_next(request)

    # Sliding session (records/21 M1): a successful API request on a session
    # past half its lifetime gets a fresh cookie, so only
    # BIOXPLORER_SESSION_DAYS without a visit signs someone out, never day
    # 14 of daily use.
    # Not on errors, and not where the route set a cookie itself: that is
    # what keeps /auth/logout's clearing (and sign-in's own cookie) from
    # being overwritten by a renewal.
    # Not for a tab nobody has used for 10 minutes (``X-Radar-Idle: 1``,
    # review BL-3): its polls and lease heartbeats would otherwise keep a
    # forgotten tab signed in for ever.
    @app.middleware("http")
    async def renew_session(request: Request, call_next):
        response = await call_next(request)
        if (oidc.providers(settings)
                and request.headers.get("X-Radar-Idle") != "1"
                and response.status_code < 400
                and "set-cookie" not in response.headers):
            session = oidc.request_session(settings, request)
            token = oidc.renewed_session(settings, session) if session else None
            if token:
                oidc.set_session_cookie(settings, response, token)
        return response

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
    # One tab at a time changes an account (write_lease, FUNCTIONAL_SPEC
    # W.A): every router that can change something goes through one guard.
    wl = [Depends(write_lease.guard)]
    app.include_router(profiles.router, prefix="/api/profiles", tags=["profiles"], dependencies=wl)
    app.include_router(radar.router, prefix="/api/radar", tags=["radar"], dependencies=wl)
    app.include_router(vault.router, prefix="/api/vault", tags=["vault"], dependencies=wl)
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"], dependencies=wl)
    app.include_router(users.router, prefix="/api/users", tags=["users"], dependencies=wl)
    app.include_router(prosopia.router, prefix="/api/import", tags=["import"], dependencies=wl)
    app.include_router(researchers.router, prefix="/api/researchers", tags=["researchers"], dependencies=wl)
    app.include_router(auth.router, prefix="/auth", tags=["auth"])

    # Top-level liveness for load-balancer / docker healthcheck use.
    @app.get("/health", include_in_schema=False)
    def root_health() -> dict:
        return {"status": "ok", "version": _service_version()}

    logging.getLogger("rag_lib.api").debug("app.created")
    return app


app = create_app()
