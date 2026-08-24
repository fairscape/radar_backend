"""Profiles routes.

Phase 5 shipped the read paths (list / get / detail). Phase 11 added
the wizard-draft write paths (``/draft/...`` + commit on ``POST /``).
Every endpoint depends on ``get_current_user`` so Phase 12's auth swap
is a single-dep change.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from datetime import datetime, timezone

from ..deps import get_current_user, get_db, get_scheduler, get_settings
from ..schemas import (
    CommitDraftRequest,
    Draft,
    DraftCoherence,
    DraftCreateRequest,
    DraftDryRun,
    DraftDryRunRequest,
    DraftDryRunStart,
    DraftDryRunStatus,
    DryRunResponse,
    FeedbackEventOut,
    GatherNowResponse,
    GatherRun,
    Profile,
    ProfileDetail,
    ProfileThresholdUpdate,
    RerankerCandidate,
    RerankerComparisonResponse,
    RefitResponse,
    Schedule,
    ScheduleUpdate,
    Topic,
    TopicYield,
    TopicYieldResponse,
    WizardOption,
    WizardOptions,
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
# ``/draft`` and ``/wizard`` prefixes aren't shadowed by the slug-route
# catch-all.
# ---------------------------------------------------------------------------


# Human-readable copy for the wizard dropdowns. Keys must match the
# registry keys; missing keys render with the bare key as the label so
# plugin-registered embedders/selectors still appear in the dropdown
# without code changes here.
_EMBEDDER_LABELS: dict[str, tuple[str, str]] = {
    "specter2": (
        "SPECTER2 (proximity)",
        "Scientific paper embeddings with the proximity adapter. Recommended.",
    ),
    "placeholder-v1": (
        "Placeholder (hash-only)",
        "Deterministic hash-seeded vectors. For tests / offline runs only.",
    ),
}

_SELECTOR_LABELS: dict[str, tuple[str, str]] = {
    "centroid": (
        "Centroid",
        "Cosine similarity to the seed mean vector. Lightweight default.",
    ),
    "max_seed": (
        "Max-seed similarity",
        "Score against the closest matching seed. Better when seeds are multimodal.",
    ),
}


def _wizard_options(default_embedder: str, default_selector: str) -> WizardOptions:
    from ...embedders import EMBEDDERS
    from ...selectors import SELECTORS

    def _row(key: str, labels: dict[str, tuple[str, str]], default_key: str) -> WizardOption:
        label, desc = labels.get(key, (key, ""))
        return WizardOption(
            key=key, label=label, description=desc, default=(key == default_key),
        )

    embedders = [_row(k, _EMBEDDER_LABELS, default_embedder) for k in sorted(EMBEDDERS)]
    selectors = [_row(k, _SELECTOR_LABELS, default_selector) for k in sorted(SELECTORS)]
    # Belt-and-braces: if the configured default isn't actually
    # registered (operator typo in .env), no row will carry default=True.
    # Fall back to flagging the first row so the UI still has a
    # selectable default.
    if embedders and not any(e.default for e in embedders):
        embedders[0] = embedders[0].model_copy(update={"default": True})
    if selectors and not any(s.default for s in selectors):
        selectors[0] = selectors[0].model_copy(update={"default": True})
    return WizardOptions(embedders=embedders, selectors=selectors)


@router.get("/wizard/options", response_model=WizardOptions)
def get_wizard_options(
    settings: Annotated[Settings, Depends(get_settings)],
) -> WizardOptions:
    """List registered embedders + selectors with the current defaults.

    The wizard's name step renders these in two dropdowns; pre-selecting
    the row with ``default: true`` matches what the create-draft endpoint
    would pick when ``embedding_model`` / ``selector`` are omitted.
    """
    return _wizard_options(
        default_embedder=settings.RADAR_DEFAULT_EMBEDDING_MODEL,
        default_selector=settings.RADAR_DEFAULT_SELECTOR,
    )


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
    embedding_model = body.embedding_model or settings.RADAR_DEFAULT_EMBEDDING_MODEL
    selector_name = body.selector or settings.RADAR_DEFAULT_SELECTOR
    try:
        result = wizard_service.create_draft(
            db,
            user_id=int(user["id"]),
            name=body.name.strip(),
            embedding_model=embedding_model,
            selector=selector_name,
        )
    except ValueError as e:
        # Unknown embedder/selector key — surfaces the registry error so
        # the partner team's CI sees exactly which key is missing.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
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
            on=bool(entry.get("on", True)),
            source=entry.get("source"),
        ))
    return out


@router.post("/draft/{slug}/dry-run", response_model=DraftDryRunStart)
def draft_dry_run(
    slug: str,
    body: DraftDryRunRequest,
    request: Request,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    gatherer: Annotated[object | None, Depends(get_wizard_gatherer)] = None,
) -> DraftDryRunStart:
    """Kick off an async dry-run; returns the run_id to poll.

    The wizard's calibrate step polls
    ``GET /api/profiles/draft/{slug}/dry-run/{run_id}`` for stage +
    progress, then reads ``result`` once ``run.finished_at`` is set.

    When tests inject a fixture gatherer via
    ``app.dependency_overrides[get_wizard_gatherer]``, the dry-run is
    executed inline (synchronously) so the test harness doesn't need
    to drive APScheduler. The response shape is unchanged — the
    fixture run finishes before the response returns and the
    status-poll endpoint returns the cached result on first call.
    """
    from rag_lib.db.repos import gather_runs as gather_runs_repo
    from rag_lib.scheduler.jobs import dry_run_for_draft

    try:
        profile_row = wizard_service._require_draft(db, int(user["id"]), slug)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
        ) from exc
    profile_id = int(profile_row["id"])

    # Open the audit row eagerly so we can echo a real run_id even when
    # the inline / test path runs synchronously.
    run_id = gather_runs_repo.start(
        db,
        profile_id=profile_id,
        user_id=int(user["id"]),
        tier_used="dry_run",
    )

    if gatherer is not None:
        # Test path: run synchronously with the injected gatherer so
        # tests don't have to coordinate with APScheduler. Errors
        # propagate as 500s — the test harness expects them surfaced
        # rather than buried in the audit row.
        result = wizard_service.dry_run_draft(
            db, settings,
            user_id=int(user["id"]), slug=slug,
            days=body.days, thresholds=body.thresholds,
            gatherer=gatherer,
        )
        gather_runs_repo.finish(
            db, run_id,
            n_fetched=len(result.scores),
            n_new=0, n_redup=0,
            tier_used="dry_run",
            result_json=result.model_dump_json(),
        )
        return DraftDryRunStart(ok=True, run_id=run_id)

    # Pull the scheduler off app.state lazily — when the scheduler is
    # disabled (e.g., test runs with RADAR_SCHEDULER_ENABLED=false) we
    # only 503 here on the async path; the inline-with-fixture path
    # above never reaches this branch.
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        gather_runs_repo.finish(
            db, run_id,
            n_fetched=0, n_new=0, n_redup=0,
            tier_used="dry_run",
            error="scheduler disabled (set RADAR_SCHEDULER_ENABLED=true)",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="scheduler disabled (set RADAR_SCHEDULER_ENABLED=true)",
        )
    scheduler.add_job(
        dry_run_for_draft,
        trigger="date",
        run_date=datetime.now(tz=timezone.utc),
        args=[int(user["id"]), profile_id, slug],
        kwargs={
            "run_id": run_id,
            "days": body.days,
            "thresholds": body.thresholds,
        },
        id=f"dry-run:{profile_id}:{run_id}",
        replace_existing=False,
        max_instances=1,
    )
    return DraftDryRunStart(ok=True, run_id=run_id)


@router.get(
    "/draft/{slug}/dry-run/{run_id}", response_model=DraftDryRunStatus,
)
def draft_dry_run_status(
    slug: str,
    run_id: int,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> DraftDryRunStatus:
    """Status + result for one async dry-run.

    Returns the gather_runs row reshaped as ``GatherRun`` plus a
    ``result`` field populated only when the job succeeded — i.e.
    ``finished_at`` is set, ``error`` is null, and ``result_json``
    parsed cleanly. The wizard polls this until ``run.finished_at``
    flips, then renders ``result``.
    """
    import json
    from rag_lib.db.repos import gather_runs as gather_runs_repo

    try:
        profile_row = wizard_service._require_draft(db, int(user["id"]), slug)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
        ) from exc

    row = gather_runs_repo.get(db, run_id)
    if row is None or int(row["profile_id"]) != int(profile_row["id"]):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"dry-run {run_id} not found for draft '{slug}'",
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

    result: DraftDryRun | None = None
    if row["finished_at"] and not row["error"] and row["result_json"]:
        try:
            result = DraftDryRun.model_validate(json.loads(row["result_json"]))
        except (ValueError, TypeError):
            # A malformed result_json shouldn't 500 the poll endpoint —
            # the run row still tells the frontend the job finished, and
            # surfacing result=None lets the UI render a graceful "no
            # result" path rather than spinning forever.
            result = None

    return DraftDryRunStatus(run=run, result=result)


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
            on=bool(entry.get("on", True)),
            source=entry.get("source"),
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
            current_step=r["current_step"],
            n_processed=r["n_processed"],
            n_total=r["n_total"],
            last_message=r["last_message"],
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


def _reranker_queries_for(db: sqlite3.Connection, profile_row) -> list[str]:
    """The queries the configured reranker would actually issue.

    Must be derived from the reranker itself, not assumed. This used to
    hard-code topic display names, which is only correct for
    ``query_mode="topic"`` — under the ``article`` mode the real queries
    are seed-paper content, so the panel was listing queries that were
    never sent. Showing fabricated inputs is worse than showing none:
    it invites tuning the topic list to fix a ranking the topic list
    never influenced.
    """
    if profile_row is None:
        return []
    try:
        from ...scheduler.jobs import _build_reranker, _load_profile

        settings = get_settings()
        reranker = _build_reranker(profile_row, settings)
        loader = getattr(reranker, "_load_queries", None)
        if loader is None:  # NoopReranker issues nothing
            return []
        return list(loader(_load_profile(db, profile_row)))
    except Exception:
        # Diagnostics must never take the endpoint down.
        return []


@router.get("/{key}/reranker-comparison", response_model=RerankerComparisonResponse)
def reranker_comparison(
    key: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    limit: int = Query(default=50, ge=1, le=500),
) -> RerankerComparisonResponse:
    """Compare candidate rankings before vs after MedCPT reranking."""
    from rag_lib.db.repos import candidates as candidates_repo, profiles as profiles_repo

    profile_id = _resolve_profile_id(db, int(user["id"]), key)

    profile_row = profiles_repo.get(db, profile_id)
    _queries_used = _reranker_queries_for(db, profile_row)

    # Load candidates that have reranker scores.
    # Use score_raw (original selector cosine) for "before" ranking,
    # score_blended (alpha*sel_norm + beta*rr_norm) for "after" ranking.
    # NOTE: pc.score is overwritten with the blended value by insert_dedup,
    # so we must use score_raw for the true selector score.
    # Both rankings are computed over every scored candidate, then the
    # top ``limit`` by blended score are returned. Ranking inside the
    # sample instead would have measured the reranker against itself: the
    # sample is chosen by blended score, so a paper lifted from selector
    # rank 400 into the top ten reports a "before" of at most ``limit``,
    # and every paper the reranker pushed out of the top ``limit`` is
    # absent — the one direction ``max_rank_down`` is supposed to show.
    all_rows = db.execute(
        """
        SELECT pc.openalex_id, pc.score_raw, pc.score_blended,
               p.title
        FROM profile_candidates pc
        JOIN papers p USING (openalex_id)
        WHERE pc.profile_id = ?
          AND pc.score_blended IS NOT NULL
          AND pc.score_raw IS NOT NULL
        """,
        (profile_id,),
    ).fetchall()

    if not all_rows:
        return RerankerComparisonResponse(ok=True, key=key, n=0, queries_used=_queries_used)

    by_selector = sorted(all_rows, key=lambda r: r["score_raw"], reverse=True)
    by_blended = sorted(all_rows, key=lambda r: r["score_blended"], reverse=True)

    selector_rank = {r["openalex_id"]: i + 1 for i, r in enumerate(by_selector)}
    blended_rank = {r["openalex_id"]: i + 1 for i, r in enumerate(by_blended)}

    rows = by_blended[:limit]

    # Summary stats span every candidate, not just the ones listed. A
    # demotion out of the top ``limit`` is exactly what max_rank_down is
    # for, and it is never visible in the listed rows.
    deltas = [
        selector_rank[r["openalex_id"]] - blended_rank[r["openalex_id"]]
        for r in all_rows
    ]

    candidates: list[RerankerCandidate] = []
    for r in rows:
        oid = r["openalex_id"]
        rb = selector_rank[oid]
        ra = blended_rank[oid]
        candidates.append(RerankerCandidate(
            openalex_id=oid,
            title=r["title"] or "",
            score_selector=float(r["score_raw"]),
            score_blended=float(r["score_blended"]),
            rank_before=rb,
            rank_after=ra,
        ))

    avg_change = sum(abs(d) for d in deltas) / len(deltas) if deltas else 0.0
    max_up = max(deltas, default=0)
    max_down = abs(min(deltas, default=0))

    return RerankerComparisonResponse(
        ok=True,
        key=key,
        n=len(candidates),
        candidates=candidates,
        avg_rank_change=round(avg_change, 2),
        max_rank_up=max_up,
        max_rank_down=max_down,
        queries_used=_queries_used,
    )


@router.get("/{key}/topic-yield", response_model=TopicYieldResponse)
def topic_yield(
    key: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    days: int = Query(default=30, ge=1, le=365),
) -> TopicYieldResponse:
    """What each topic's gather quota has actually brought in.

    Pairs the profile's topic list with per-topic candidate tallies so
    the Step 3 toggles can be judged on output rather than on the topic
    name. Topics that have never sourced a candidate are still listed
    (all-zero) — "this topic yields nothing" is exactly the case the
    user needs to see.
    """
    from datetime import timedelta

    from rag_lib.db.repos import candidates as candidates_repo
    from rag_lib.db.repos import profiles as profiles_repo

    profile_id = _resolve_profile_id(db, int(user["id"]), key)
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    tallies = {
        row["topic_id"]: row
        for row in candidates_repo.topic_yield_for_profile(
            db, profile_id, since=since,
        )
    }

    filters = profiles_repo.topic_filters(db, profile_id) or {}
    out: list[TopicYield] = []
    for entry in filters.get("topics") or []:
        tid = entry.get("id")
        if not tid:
            continue
        row = tallies.pop(tid, None)
        out.append(TopicYield(
            topic_id=tid,
            display_name=entry.get("display_name") or "",
            on=bool(entry.get("on", True)),
            n_candidates=int(row["n_candidates"]) if row else 0,
            n_shown=int(row["n_shown"] or 0) if row else 0,
            n_saved=int(row["n_saved"] or 0) if row else 0,
            n_dismissed=int(row["n_dismissed"] or 0) if row else 0,
            last_fetched_at=row["last_fetched_at"] if row else None,
        ))

    # Topics that sourced candidates but have since been dropped from
    # topic_filters (re-aggregation after the seed set changed). Their
    # papers are still in the radar, so the yield stays visible.
    for tid, row in tallies.items():
        out.append(TopicYield(
            topic_id=tid,
            display_name="(no longer in profile)",
            on=False,
            n_candidates=int(row["n_candidates"]),
            n_shown=int(row["n_shown"] or 0),
            n_saved=int(row["n_saved"] or 0),
            n_dismissed=int(row["n_dismissed"] or 0),
            last_fetched_at=row["last_fetched_at"],
        ))

    out.sort(key=lambda t: (-t.n_candidates, t.display_name))
    return TopicYieldResponse(ok=True, key=key, days=days, topics=out)
