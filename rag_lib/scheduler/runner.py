"""APScheduler boot + shutdown for the in-process daily-gather loop.

We use a ``BackgroundScheduler`` so each job body runs in its own
thread; the gatherer is sync (``requests``-based) and that interacts
better with FastAPI's threadpool model than ``AsyncIOScheduler`` would.

Jobstore is in-memory: schedules live in SQLite (``profile_schedules``),
and ``start`` re-hydrates jobs from there on every boot. That keeps the
APScheduler state derivable from the source of truth instead of being a
second copy that can drift.
"""

from __future__ import annotations

import logging

import structlog

from ..db import connect
from ..db.repos import schedules as schedules_repo
from .jobs import gather_for_profile

log = structlog.get_logger("rag_lib.scheduler")


def build_scheduler(settings):
    """Construct an APScheduler ``BackgroundScheduler`` with sane defaults.

    ``max_instances=1`` per job id is set per-job at registration time
    (``start``), not here, because the global default doesn't apply
    retroactively to jobs added later.
    """
    # Imported here so a missing apscheduler install doesn't break module
    # import for callers that only need ``gather_for_profile``.
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler(
        timezone="UTC",
        job_defaults={
            "coalesce": True,           # collapse missed firings into one
            "max_instances": 1,         # one run per job at a time
            "misfire_grace_time": 600,  # 10 min grace after a missed fire
        },
    )
    return scheduler


def valid_cron(cron: str, tz: str | None) -> str | None:
    """None if ``cron``/``tz`` make a trigger, else the reason they do not."""
    from apscheduler.triggers.cron import CronTrigger

    try:
        CronTrigger.from_crontab(cron, timezone=tz or "UTC")
    except Exception as exc:  # noqa: BLE001
        # Any failure, not ValueError/TypeError: an unknown zone raises
        # ZoneInfoNotFoundError, a KeyError. Let through, a stored bad tz
        # made register_gather_job raise inside start() -- and the backend
        # then failed to boot at all.
        return f"{type(exc).__name__}: {exc}"
    return None


def register_gather_job(
    scheduler, *, user_id: int, profile_id: int, cron: str, tz: str | None,
    slug: str | None = None,
) -> bool:
    """Add or replace one profile's nightly gather job. False if the cron is bad.

    The one place this is built. Boot, a schedule edit and a commit each had
    their own copy, which had already drifted: only boot named the job, and
    only boot survived a bad cron -- the commit copy raised after the draft
    had gone live, answering 500 to a commit that had in fact happened.
    """
    from apscheduler.triggers.cron import CronTrigger

    problem = valid_cron(cron, tz)
    if problem is not None:
        log.error("scheduler.invalid_cron", profile_id=profile_id, cron=cron, tz=tz, error=problem)
        return False
    scheduler.add_job(
        gather_for_profile,
        trigger=CronTrigger.from_crontab(cron, timezone=tz or "UTC"),
        args=[user_id, profile_id],
        id=f"gather:{profile_id}",
        name=f"gather:{slug or profile_id}",
        replace_existing=True,
        max_instances=1,
    )
    log.info("scheduler.job_registered", profile_id=profile_id, slug=slug, cron=cron, tz=tz)
    return True


def start(scheduler, settings) -> None:
    """Register one job per enabled profile schedule, then start the scheduler.

    Re-entrant: ``replace_existing=True`` on every ``add_job`` call lets
    a hot-reload re-register without colliding with prior boots.
    """
    conn = connect(settings.RADAR_DB_PATH)
    try:
        rows = schedules_repo.list_enabled(conn)
    finally:
        conn.close()

    for row in rows:
        register_gather_job(
            scheduler,
            user_id=int(row["user_id"]), profile_id=int(row["profile_id"]),
            cron=row["cron"], tz=row["tz"], slug=row["slug"],
        )
    # Not per profile: one sweep over every seed still missing UMLS. First
    # run a few minutes after boot, so a restart also clears any backlog.
    from datetime import datetime, timedelta, timezone

    from apscheduler.triggers.interval import IntervalTrigger

    from .jobs import backfill_seed_umls_job

    scheduler.add_job(
        backfill_seed_umls_job,
        trigger=IntervalTrigger(minutes=15),
        id="umls-backfill",
        name="umls-backfill",
        replace_existing=True,
        max_instances=1,
        # Aware and UTC, like the scheduler: a naive now() is local time,
        # which APScheduler read as UTC -- on this EDT host, ~4 hours ago.
        next_run_time=datetime.now(timezone.utc) + timedelta(minutes=3),
    )
    scheduler.start()
    log.info("scheduler.started", n_jobs=len(rows))


def stop(scheduler) -> None:
    """Graceful shutdown for the FastAPI lifespan teardown."""
    if scheduler is None:
        return
    try:
        scheduler.shutdown(wait=False)
    except Exception:  # noqa: BLE001 — log-and-swallow during teardown
        logging.getLogger("rag_lib.scheduler").exception("scheduler.shutdown_failed")
