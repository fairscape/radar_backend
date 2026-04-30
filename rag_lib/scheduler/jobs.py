"""Background-job bodies dispatched by APScheduler.

Each function here is the entry point a scheduled job lands in. They
must be importable by their dotted path (APScheduler ``add_job`` stores
the callable as a reference) and own their DB connection — the
foreground request connection isn't safe to share across threads.

Phase 8 will wrap ``gather_for_profile`` to chain an autotune pass on
each successful run; keep the gather body factored so that hook can
slot in without refactoring the call site.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from ..db import connect
from ..db.repos import (
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)
from ..gatherers.openalex import OpenAlexGatherer
from ..paper import Paper
from ..persistence.db_store import dedup_and_insert_candidates
from ..profile import Profile
from ..radar import _days_ago_iso
from ..selectors.centroid import CentroidSelector

log = structlog.get_logger("rag_lib.scheduler.jobs")


GATHER_DAYS_DEFAULT = 1
GATHER_LIMIT_DEFAULT = 500


def _resolve_mailto(user_row, default_mailto: str) -> str:
    if user_row is not None and user_row["mailto"]:
        return user_row["mailto"]
    if user_row is not None and user_row["email"]:
        return user_row["email"]
    return default_mailto


def _load_profile(conn, profile_row) -> Profile:
    """Reconstruct a domain ``Profile`` from DB rows.

    Seed papers and embeddings are pulled so the selector's ``fit`` /
    ``select`` paths have what they need. The fitted centroid is also
    available via ``selector_config_json`` for fast-path reconstruction.
    """
    topic_filters = (
        json.loads(profile_row["topic_filters_json"])
        if profile_row["topic_filters_json"]
        else {}
    )

    seed_ids = profiles_repo.list_seed_openalex_ids(conn, int(profile_row["id"]))
    embedding_model = profile_row["embedding_model"]
    papers: list[Paper] = []
    for oa_id in seed_ids:
        prow = papers_repo.get_by_openalex_id(conn, oa_id)
        if prow is None:
            continue
        topics = papers_repo.decode_topics(prow)
        paper = Paper.from_dict(
            {
                "doi": prow["doi"],
                "openalex_id": prow["openalex_id"],
                "title": prow["title"],
                "abstract": prow["abstract"],
                "year": prow["year"],
                "venue": prow["venue"],
                "primary_topic": topics.get("primary_topic"),
                "topics": topics.get("topics") or [],
                "local_path": prow["local_path"],
                "body_text": prow["body_text"],
                "source": prow["source"],
            }
        )
        vec = embeddings_repo.get(conn, oa_id, embedding_model)
        if vec is not None:
            paper.embeddings[embedding_model] = vec.tolist()
        papers.append(paper)

    return Profile(
        name=profile_row["name"],
        papers=papers,
        topic_filters=topic_filters,
        embedding_model=embedding_model,
        threshold=profile_row["threshold"],
    )


def _build_selector(profile_row, profile: Profile) -> CentroidSelector:
    """Reuse the persisted centroid when available, else refit from seeds."""
    sel_json = profile_row["selector_config_json"]
    if sel_json:
        try:
            cfg = json.loads(sel_json)
        except json.JSONDecodeError:
            cfg = None
        if cfg and cfg.get("centroid"):
            return CentroidSelector.from_config(cfg)

    selector = CentroidSelector(embedding_model=profile.embedding_model)
    selector.fit(profile)
    return selector


def gather_for_profile(
    user_id: int,
    profile_id: int,
    *,
    settings: Any | None = None,
    days: int = GATHER_DAYS_DEFAULT,
    limit: int = GATHER_LIMIT_DEFAULT,
    run_id: int | None = None,
    tier: str = "scheduled",
) -> int | None:
    """Run one OpenAlex gather + score + persist cycle for a profile.

    When ``run_id`` is supplied (e.g., the ``gather-now`` endpoint
    pre-created the row so it could echo the id back to the caller), the
    existing row is reused. Otherwise a new row is opened. Returns the
    ``gather_runs.id`` actually used. On error the row is closed with
    the exception message in ``error`` and the id is still returned.
    """
    if settings is None:
        from ..api.settings import get_settings  # avoid circular at import time
        settings = get_settings()

    conn = connect(settings.RADAR_DB_PATH)
    try:
        user_row = users_repo.get(conn, user_id)
        mailto = _resolve_mailto(user_row, settings.RADAR_DEFAULT_MAILTO)

        profile_row = profiles_repo.get(conn, profile_id)
        if profile_row is None:
            log.error("scheduler.profile_missing", profile_id=profile_id)
            return None

        since = _days_ago_iso(days)
        if run_id is None:
            run_id = gather_runs_repo.start(
                conn,
                profile_id=profile_id,
                user_id=user_id,
                since_date=since,
                tier_used=tier,
            )

        try:
            profile = _load_profile(conn, profile_row)
            selector = _build_selector(profile_row, profile)
            gatherer = OpenAlexGatherer(mailto=mailto)

            candidates = gatherer.fetch(profile, since=since, limit=limit)
            ranked = selector.select(candidates, profile, threshold=profile.threshold)
            n_new, n_redup = dedup_and_insert_candidates(
                conn,
                profile_id=profile_id,
                gather_run_id=run_id,
                ranked=ranked,
                tier_used="scheduled",
            )
            api_calls = gatherer.cost().get("api_calls", 0)
            gather_runs_repo.finish(
                conn,
                run_id,
                n_fetched=len(candidates),
                n_new=n_new,
                n_redup=n_redup,
                api_calls=int(api_calls),
                tier_used="scheduled",
            )
            log.info(
                "scheduler.gather_ok",
                profile_id=profile_id,
                run_id=run_id,
                n_fetched=len(candidates),
                n_new=n_new,
                n_redup=n_redup,
            )
            return run_id
        except Exception as exc:  # noqa: BLE001 — surface into audit row
            gather_runs_repo.finish(
                conn,
                run_id,
                n_fetched=0,
                n_new=0,
                n_redup=0,
                error=f"{type(exc).__name__}: {exc}",
            )
            log.exception(
                "scheduler.gather_failed",
                profile_id=profile_id,
                run_id=run_id,
            )
            return run_id
    finally:
        conn.close()


__all__ = ["gather_for_profile"]
