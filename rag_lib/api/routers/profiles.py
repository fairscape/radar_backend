"""Profiles routes.

Phase 5 shipped the read paths (list / get / detail). Phase 11 added
the wizard-draft write paths (``/draft/...`` + commit on ``POST /``).
Every endpoint depends on ``get_current_user`` so Phase 12's auth swap
is a single-dep change.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from datetime import datetime, timezone

from ..deps import get_current_user, get_db, get_scheduler, get_settings
from ..schemas import (
    CommitDraftRequest,
    Draft,
    DraftCoherence,
    DraftCreateRequest,
    DraftDryRun,
    DraftDryRunRequest,
    DryRunResponse,
    FeedbackEventOut,
    GatherNowResponse,
    GatherRun,
    Profile,
    ProfileDetail,
    ProfileThresholdUpdate,
    RefitResponse,
    Schedule,
    ScheduleUpdate,
    Topic,
)
from ..services import profiles as profiles_service
from ..services import wizard as wizard_service
from ..settings import Settings

router = APIRouter()


def get_wizard_gatherer():
    """Dry-run gatherer dependency.

    Returns ``None`` by default so the wizard service falls back to a
    real ``OpenAlexGatherer``. Tests override this in
    ``app.dependency_overrides`` to inject a FixtureGatherer.
    """
    return None


# ---------------------------------------------------------------------------
# Phase 11 — wizard draft endpoints. Declared before ``/{key}`` so the
# ``/draft`` prefix isn't shadowed by the slug-route catch-all.
# ---------------------------------------------------------------------------


@router.post("/draft", response_model=Draft)
def create_draft(
    body: DraftCreateRequest,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Draft:
    if not body.name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="profile name must be non-empty",
        )
    result = wizard_service.create_draft(
        db,
        user_id=int(user["id"]),
        name=body.name.strip(),
        embedding_model=settings.RADAR_DEFAULT_EMBEDDING_MODEL,
    )
    return Draft(**result)


@router.post("/draft/{slug}/coherence", response_model=DraftCoherence)
def draft_coherence(
    slug: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> DraftCoherence:
    try:
        return wizard_service.compute_draft_coherence(
            db, user_id=int(user["id"]), slug=slug,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
        ) from exc


@router.get("/draft/{slug}/topics", response_model=list[Topic])
def draft_topics(
    slug: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> list[Topic]:
    try:
        topic_filters = wizard_service.aggregate_draft_topics(
            db, user_id=int(user["id"]), slug=slug,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
        ) from exc
    out: list[Topic] = []
    for entry in (topic_filters.get("topics") or []):
        tid = entry.get("id") or ""
        if not tid:
            continue
        out.append(Topic(
            id=tid,
            name=entry.get("display_name") or "",
            count=int(entry.get("count") or 0),
            on=True,
        ))
    return out


@router.post("/draft/{slug}/dry-run", response_model=DraftDryRun)
def draft_dry_run(
    slug: str,
    body: DraftDryRunRequest,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    gatherer: Annotated[object | None, Depends(get_wizard_gatherer)] = None,
) -> DraftDryRun:
    """Sweep + preview against the last ``days`` of OpenAlex output.

    Tests inject a fixture gatherer via
    ``app.dependency_overrides[get_wizard_gatherer]``.
    """
    try:
        return wizard_service.dry_run_draft(
            db, settings,
            user_id=int(user["id"]), slug=slug,
            days=body.days, thresholds=body.thresholds,
            gatherer=gatherer,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
        ) from exc


@router.delete("/draft/{slug}")
def delete_draft(
    slug: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> dict:
    deleted = wizard_service.delete_draft(
        db, user_id=int(user["id"]), slug=slug,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"draft '{slug}' not found for current user",
        )
    return {"ok": True}


@router.post("", response_model=Profile)
def commit_draft(
    body: CommitDraftRequest,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> Profile:
    """Commit a wizard draft → live profile.

    Flips ``is_draft=0``, fits the CentroidSelector, and registers a
    schedule so the next scheduler boot picks the profile up. The
    schedule defaults to daily 04:00 UTC; override via ``cron`` / ``tz``.
    """
    try:
        wizard_service.commit_draft(
            db,
            user_id=int(user["id"]),
            slug=body.slug,
            threshold=body.threshold,
            selected_topic_ids=body.selected_topic_ids,
            cron=body.cron,
            tz=body.tz,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
        ) from exc
    profile = profiles_service.get_profile(db, int(user["id"]), body.slug)
    if profile is None:
        # Should be unreachable — commit just flipped the row to live.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="profile vanished mid-commit",
        )
    return profile


@router.get("", response_model=list[Profile])
def list_profiles(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> list[Profile]:
    return profiles_service.list_profiles(db, int(user["id"]))


@router.get("/{key}", response_model=Profile | None)
def get_profile(
    key: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> Profile | None:
    return profiles_service.get_profile(db, int(user["id"]), key)


@router.get("/{key}/detail", response_model=ProfileDetail | None)
def get_profile_detail(
    key: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> ProfileDetail | None:
    return profiles_service.get_profile_detail(db, int(user["id"]), key)


@router.patch("/{key}", response_model=Profile)
def update_profile(
    key: str,
    body: ProfileThresholdUpdate,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> Profile:
    """Update editable fields on a committed profile.

    Today only ``threshold`` is editable inline from the profile
    detail page; the dry-run histogram lets the user see how the
    chosen θ affects candidate volume before saving.
    """
    from rag_lib.db.repos import profiles as profiles_repo

    profile_id = _resolve_profile_id(db, int(user["id"]), key)
    profiles_repo.update_threshold(db, profile_id, float(body.threshold))
    updated = profiles_service.get_profile(db, int(user["id"]), key)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"profile '{key}' not found for current user",
        )
    return updated


# Recompute aliases — let the profile detail page re-run coherence and
# topic aggregation after the user uploads more seeds. Both work for
# drafts AND committed profiles (the wizard service uses
# ``_require_profile`` rather than ``_require_draft``).
@router.post("/{key}/recompute-coherence", response_model=DraftCoherence)
def recompute_coherence(
    key: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> DraftCoherence:
    try:
        return wizard_service.compute_draft_coherence(
            db, user_id=int(user["id"]), slug=key,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
        ) from exc


@router.post("/{key}/recompute-topics", response_model=list[Topic])
def recompute_topics(
    key: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> list[Topic]:
    try:
        topic_filters = wizard_service.aggregate_draft_topics(
            db, user_id=int(user["id"]), slug=key,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
        ) from exc
    out: list[Topic] = []
    for entry in (topic_filters.get("topics") or []):
        tid = entry.get("id") or ""
        if not tid:
            continue
        out.append(Topic(
            id=tid,
            name=entry.get("display_name") or "",
            count=int(entry.get("count") or 0),
            on=True,
        ))
    return out


@router.post("/{key}/refit", response_model=RefitResponse)
def refit_profile(
    key: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    autotune: int = Query(default=0, ge=0, le=1),
) -> RefitResponse:
    """Re-fit the selector for the given profile.

    Phase 5 returns a placeholder ``cost`` string so the UI gets the
    expected shape. Phase 11 (wizard) will replace the body with a real
    selector.fit call followed by ``store_profile_from_object``.

    ``?autotune=1`` (Phase 8) runs ``feedback.autotune.apply_recommended``
    after the refit and reports the new threshold inline in ``cost``.
    """
    profile = profiles_service.get_profile(db, int(user["id"]), key)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"profile '{key}' not found for current user",
        )
    cost = "0.00 s · 0 vecs"
    if autotune:
        from rag_lib.feedback.autotune import apply_recommended

        profile_id = _resolve_profile_id(db, int(user["id"]), key)
        result = apply_recommended(db, profile_id)
        if result["applied"]:
            old_str = (
                f"{result['old']:.2f}" if result["old"] is not None else "—"
            )
            cost = f"{cost} · autotune {old_str}→{result['new']:.2f}"
        else:
            cost = f"{cost} · autotune skipped (n={result['n_events']})"
    return RefitResponse(ok=True, key=key, cost=cost)


@router.post("/{key}/dry-run", response_model=DryRunResponse)
def dry_run_profile(
    key: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> DryRunResponse:
    """Dry-run the gather + score loop for the profile.

    Phase 5 returns the persisted candidate count so the UI has a
    meaningful number to display. Phase 11 will replace this with a
    real ``radar.dry_run`` invocation.
    """
    profile = profiles_service.get_profile(db, int(user["id"]), key)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"profile '{key}' not found for current user",
        )
    from rag_lib.db.repos import (
        candidates as candidates_repo,
        profiles as profiles_repo,
    )
    row = profiles_repo.get_by_slug(db, int(user["id"]), key)
    if row is None:
        return DryRunResponse(ok=True, key=key, n=0, scores=[])
    profile_id = int(row["id"])
    scores = candidates_repo.scores_for_profile(db, profile_id)
    return DryRunResponse(ok=True, key=key, n=len(scores), scores=scores)


def _resolve_profile_id(db: sqlite3.Connection, user_id: int, key: str) -> int:
    """Look up a profile id by slug, 404ing if it doesn't belong to ``user``."""
    from rag_lib.db.repos import profiles as profiles_repo

    row = profiles_repo.get_by_slug(db, user_id, key)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"profile '{key}' not found for current user",
        )
    return int(row["id"])


@router.post("/{key}/gather-now", response_model=GatherNowResponse)
def gather_now(
    key: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    scheduler: Annotated[object, Depends(get_scheduler)],
    days: int = Query(default=1, ge=1, le=365),
    limit: int = Query(default=500, ge=1, le=5000),
) -> GatherNowResponse:
    """Enqueue an immediate one-shot gather for the profile.

    Opens a ``gather_runs`` row eagerly so the response can echo a real
    ``run_id`` (and the audit trail captures the trigger even if the
    background thread later errors). ``days`` controls how far back to
    look for new candidates, ``limit`` caps the per-run fetch size.
    """
    from rag_lib.db.repos import gather_runs as gather_runs_repo
    from rag_lib.scheduler.jobs import gather_for_profile

    profile_id = _resolve_profile_id(db, int(user["id"]), key)
    # Open the audit row eagerly so we can echo a real run_id; the
    # background job reuses it via the ``run_id`` kwarg.
    run_id = gather_runs_repo.start(
        db,
        profile_id=profile_id,
        user_id=int(user["id"]),
        tier_used="manual",
    )
    scheduler.add_job(
        gather_for_profile,
        trigger="date",
        run_date=datetime.now(tz=timezone.utc),
        args=[int(user["id"]), profile_id],
        kwargs={
            "run_id": run_id,
            "tier": "manual",
            "days": days,
            "limit": limit,
        },
        id=f"gather-now:{profile_id}:{run_id}",
        replace_existing=False,
        max_instances=1,
    )
    return GatherNowResponse(ok=True, run_id=run_id)


@router.get("/{key}/feedback", response_model=list[FeedbackEventOut])
def list_feedback(
    key: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    limit: int = Query(default=20, ge=1, le=200),
    action: str | None = Query(default=None),
) -> list[FeedbackEventOut]:
    """Recent feedback events for a profile (Phase 8).

    ``action`` filters to one of ``saved`` / ``dismissed`` / ``snoozed``;
    omit it to see every action.
    """
    from rag_lib.db.repos import feedback as feedback_repo

    profile_id = _resolve_profile_id(db, int(user["id"]), key)
    rows = feedback_repo.recent_for_profile(
        db, profile_id, limit=limit, action=action,
    )
    return [
        FeedbackEventOut(
            id=int(r["id"]),
            profile_id=int(r["profile_id"]),
            openalex_id=r["openalex_id"],
            doi=r["doi"],
            action=r["action"],
            score=r["score"],
            selector=r["selector"],
            selector_config_hash=r["selector_config_hash"],
            benchmark_run_id=r["benchmark_run_id"],
            ts=r["ts"],
        )
        for r in rows
    ]


@router.get("/{key}/runs", response_model=list[GatherRun])
def list_runs(
    key: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    limit: int = 20,
) -> list[GatherRun]:
    from rag_lib.db.repos import gather_runs as gather_runs_repo

    profile_id = _resolve_profile_id(db, int(user["id"]), key)
    rows = gather_runs_repo.recent_for_profile(db, profile_id, limit=limit)
    return [
        GatherRun(
            id=int(r["id"]),
            profile_id=int(r["profile_id"]),
            started_at=r["started_at"],
            finished_at=r["finished_at"],
            since_date=r["since_date"],
            filter_string=r["filter_string"],
            tier_used=r["tier_used"],
            n_fetched=r["n_fetched"],
            n_new=r["n_new"],
            n_redup=r["n_redup"],
            api_calls=r["api_calls"],
            error=r["error"],
        )
        for r in rows
    ]


@router.patch("/{key}/schedule", response_model=Schedule)
def update_schedule(
    key: str,
    body: ScheduleUpdate,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    scheduler: Annotated[object, Depends(get_scheduler)],
) -> Schedule:
    """Update the cron / tz / enabled flag for a profile and re-register the job."""
    from apscheduler.triggers.cron import CronTrigger

    from rag_lib.db.repos import schedules as schedules_repo
    from rag_lib.scheduler.jobs import gather_for_profile

    profile_id = _resolve_profile_id(db, int(user["id"]), key)

    if body.cron is not None:
        try:
            CronTrigger.from_crontab(body.cron, timezone=body.tz or "UTC")
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"invalid cron '{body.cron}': {exc}",
            ) from exc

    row = schedules_repo.upsert(
        db,
        profile_id=profile_id,
        cron=body.cron,
        tz=body.tz,
        enabled=body.enabled,
    )

    job_id = f"gather:{profile_id}"
    if row["enabled"]:
        scheduler.add_job(
            gather_for_profile,
            trigger=CronTrigger.from_crontab(row["cron"], timezone=row["tz"]),
            args=[int(user["id"]), profile_id],
            id=job_id,
            replace_existing=True,
            max_instances=1,
        )
    else:
        try:
            scheduler.remove_job(job_id)
        except Exception:  # noqa: BLE001 — job may not be registered yet
            pass

    return Schedule(
        profile_id=int(row["profile_id"]),
        cron=row["cron"],
        tz=row["tz"],
        enabled=bool(row["enabled"]),
        updated_at=row["updated_at"],
    )
