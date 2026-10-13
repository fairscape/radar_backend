"""Prosopia import routes.

Its own router rather than three more endpoints on ``profiles.py``:
the import is a self-contained integration with an outside service, and
keeping it separate means the Prosopia client, its failure modes and its
test seams never have to be understood by anyone reading the profile
CRUD.

The flow is the wizard dry-run's, for the same reason. The expensive
part — an OpenAlex resolution and an embedding per paper, 82 of them for
the reference profile — cannot run inside a request. So:

  ``POST /api/import/prosopia``      reads the profile, creates the
                                     draft, opens a ``gather_runs`` row
                                     and hands the rest to the scheduler.
                                     Returns ``{draft_slug, run_id}``.
  ``GET  /api/import/prosopia/{id}`` the audit row, plus ``result`` once
                                     the job finished cleanly.

Splitting it this way also puts the errors where they belong: a slug the
Prosopia instance has never heard of is a 404 on the POST, not a run row
the caller has to poll to discover was pointless.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..deps import get_current_user, get_db, get_settings
from ..schemas import (
    GatherRun,
    ProsopiaImportRequest,
    ProsopiaImportResult,
    ProsopiaImportStart,
    ProsopiaImportStatus,
)
from ..settings import Settings

router = APIRouter()


TIER = "prosopia_import"


def get_prosopia_client():
    """Prosopia client dependency.

    ``None`` in production so the service builds one from the request's
    ``base_url`` / settings. Tests override this in
    ``app.dependency_overrides`` to inject a fake and stay offline.
    """
    return None


def get_import_openalex_client():
    """OpenAlex client for the import path.

    ``None`` in production. When a test overrides it, the import also
    runs *inline* rather than on the scheduler — the same bargain the
    wizard's dry-run route strikes with its fixture gatherer, so the
    test harness never has to drive APScheduler to see a result.
    """
    return None


@router.post("/prosopia", response_model=ProsopiaImportStart)
def import_prosopia(
    body: ProsopiaImportRequest,
    request: Request,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    prosopia_client: Annotated[object | None, Depends(get_prosopia_client)] = None,
    openalex_client: Annotated[
        object | None, Depends(get_import_openalex_client)
    ] = None,
) -> ProsopiaImportStart:
    """Start importing a published Prosopia profile as a draft's seeds.

    The draft exists by the time this returns, so the caller can route
    straight to the wizard and watch the seeds arrive. Poll
    ``GET /api/import/prosopia/{run_id}`` for progress and the per-rung
    result.

    400 means the ref was empty; 404 means the Prosopia instance does
    not have that profile; 502 means it could not be read at all.
    Anything that goes wrong *after* the job starts lands on
    ``run.error`` instead — by then the draft is real and the caller
    needs the run row, not an exception.
    """
    from ...scheduler.jobs import import_prosopia_profile
    from ..services import prosopia as prosopia_service
    from ...db.repos import gather_runs as gather_runs_repo

    try:
        plan = prosopia_service.prepare_import(
            db,
            settings,
            user_id=int(user["id"]),
            ref=body.ref,
            base_url=body.base_url or settings.RADAR_PROSOPIA_BASE_URL,
            name=body.name,
            embedding_model=body.embedding_model,
            prosopia_client=prosopia_client,
        )
    except prosopia_service.ProfileNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"prosopia profile '{body.ref}' not found",
        ) from exc
    except prosopia_service.ProsopiaError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"could not read prosopia profile '{body.ref}': {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc

    # Opened eagerly so the response can echo a real run_id even on the
    # inline path below.
    run_id = gather_runs_repo.start(
        db,
        profile_id=plan.profile_id,
        user_id=int(user["id"]),
        tier_used=TIER,
    )

    if openalex_client is not None:
        # Test path: run synchronously with the injected client so tests
        # don't have to coordinate with APScheduler. The response shape
        # is unchanged — the run has simply already finished by the time
        # the first poll arrives.
        try:
            result = prosopia_service.run_import(
                db, settings, plan=plan, openalex_client=openalex_client,
            )
        except Exception as exc:  # noqa: BLE001 — mirror the job's audit write
            gather_runs_repo.finish(
                db, run_id, n_fetched=0, n_new=0, n_redup=0,
                tier_used=TIER, error=f"{type(exc).__name__}: {exc}",
            )
            raise
        gather_runs_repo.finish(
            db, run_id,
            n_fetched=result["drafted"],
            n_new=result["drafted"],
            n_redup=0,
            tier_used=TIER,
            result_json=json.dumps(result),
        )
        return ProsopiaImportStart(draft_slug=plan.draft_slug, run_id=run_id)

    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        gather_runs_repo.finish(
            db, run_id, n_fetched=0, n_new=0, n_redup=0, tier_used=TIER,
            error="scheduler disabled (set RADAR_SCHEDULER_ENABLED=true)",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="scheduler disabled (set RADAR_SCHEDULER_ENABLED=true)",
        )
    scheduler.add_job(
        import_prosopia_profile,
        trigger="date",
        run_date=datetime.now(tz=timezone.utc),
        args=[plan],
        kwargs={"run_id": run_id},
        id=f"prosopia-import:{plan.profile_id}:{run_id}",
        replace_existing=False,
        max_instances=1,
    )
    return ProsopiaImportStart(draft_slug=plan.draft_slug, run_id=run_id)


@router.get("/prosopia/{run_id}", response_model=ProsopiaImportStatus)
def import_prosopia_status(
    run_id: int,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> ProsopiaImportStatus:
    """Progress and result for one import run.

    ``result`` stays null until the job succeeded — ``finished_at`` set,
    ``error`` null, ``result_json`` parsed. A failed job returns the run
    row with ``error`` populated, which is where the caller reads what
    went wrong.
    """
    from ...db.repos import gather_runs as gather_runs_repo
    from ...db.repos import profiles as profiles_repo

    row = gather_runs_repo.get(db, run_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"import run {run_id} not found",
        )
    # Runs are addressed by a bare id, so ownership has to be checked
    # against the profile rather than assumed from the URL.
    profile = profiles_repo.get(db, int(row["profile_id"]))
    if profile is None or int(profile["user_id"]) != int(user["id"]):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"import run {run_id} not found",
        )

    run = GatherRun(
        id=int(row["id"]),
        profile_id=int(row["profile_id"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        since_date=row["since_date"],
        filter_string=row["filter_string"],
        tier_used=row["tier_used"],
        n_fetched=row["n_fetched"],
        n_new=row["n_new"],
        n_redup=row["n_redup"],
        api_calls=row["api_calls"],
        error=row["error"],
        current_step=row["current_step"],
        n_processed=row["n_processed"],
        n_total=row["n_total"],
        last_message=row["last_message"],
    )

    result: ProsopiaImportResult | None = None
    if row["finished_at"] and not row["error"] and row["result_json"]:
        try:
            result = ProsopiaImportResult.model_validate(
                json.loads(row["result_json"]),
            )
        except (ValueError, TypeError):
            # A malformed result_json must not 500 the poll endpoint:
            # the run row still tells the caller the job finished, and
            # result=None renders as "no result" rather than spinning.
            result = None

    return ProsopiaImportStatus(run=run, result=result)
