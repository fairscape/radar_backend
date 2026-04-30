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


def start(scheduler, settings) -> None:
    """Register one job per enabled profile schedule, then start the scheduler.

    Re-entrant: ``replace_existing=True`` on every ``add_job`` call lets
    a hot-reload re-register without colliding with prior boots.
    """
    from apscheduler.triggers.cron import CronTrigger

    conn = connect(settings.RADAR_DB_PATH)
    try:
        rows = schedules_repo.list_enabled(conn)
    finally:
        conn.close()

    for row in rows:
        profile_id = int(row["profile_id"])
        user_id = int(row["user_id"])
        cron = row["cron"]
        tz = row["tz"]
        try:
            trigger = CronTrigger.from_crontab(cron, timezone=tz)
        except (ValueError, TypeError) as exc:
            log.error(
                "scheduler.invalid_cron",
                profile_id=profile_id,
                cron=cron,
                tz=tz,
                error=str(exc),
            )
            continue
        scheduler.add_job(
            gather_for_profile,
            trigger=trigger,
            args=[user_id, profile_id],
            id=f"gather:{profile_id}",
            name=f"gather:{row['slug']}",
            replace_existing=True,
            max_instances=1,
        )
        log.info(
            "scheduler.job_registered",
            profile_id=profile_id,
            slug=row["slug"],
            cron=cron,
            tz=tz,
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
