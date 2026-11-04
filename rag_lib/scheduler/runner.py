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
import threading
from typing import Callable

import structlog

from ..db import connect
from ..db.repos import schedules as schedules_repo
from .jobs import gather_for_profile

log = structlog.get_logger("rag_lib.scheduler")


# What to do if a job is dropped unrun (CR.6). A job queued behind busy
# workers past ``misfire_grace_time`` is skipped by APScheduler: its body
# never runs, so whatever the request opened for it -- a run row, a
# "running" flag -- stayed open (a scan read as running for two hours,
# 409-ing every "Scan now"). Whoever hands a job over says here how to
# close what it opened; the listener does it when the job is missed.
_on_missed: dict[str, Callable[[], None]] = {}
_on_missed_lock = threading.Lock()

MISSED = {
    "scan": "This scan did not start: the server was too busy. Nothing was changed. Please try again.",
    "import": "This import did not start: the server was too busy. Nothing was changed. Please try again.",
}


def on_missed(job_id: str, callback: Callable[[], None]) -> None:
    """Run ``callback`` if job ``job_id`` is dropped without running.
    Register before ``add_job``; ``forget_missed`` if ``add_job`` fails."""
    with _on_missed_lock:
        _on_missed[job_id] = callback


def forget_missed(job_id: str) -> None:
    with _on_missed_lock:
        _on_missed.pop(job_id, None)


def close_run_if_missed(job_id: str, run_id: int, *, what: str = "scan",
                        undo: Callable | None = None) -> None:
    """``on_missed`` for a job whose run row the request opened: the row is
    closed with ``MISSED[what]``, and ``undo(conn)`` removes what the
    request made for the job (an import's new draft or researcher)."""
    def close() -> None:
        from ..api.settings import get_settings
        from ..db.repos import gather_runs as gather_runs_repo

        conn = connect(get_settings().RADAR_DB_PATH)
        try:
            row = gather_runs_repo.get(conn, run_id)
            if row is None or row["finished_at"] is not None:
                return
            gather_runs_repo.finish(conn, run_id, n_fetched=0, n_new=0, n_redup=0, error=MISSED[what])
            if undo is not None:
                undo(conn)
        finally:
            conn.close()
    on_missed(job_id, close)


def _job_event(event) -> None:
    from apscheduler.events import EVENT_JOB_MISSED

    with _on_missed_lock:
        callback = _on_missed.pop(event.job_id, None)
    if callback is None or event.code != EVENT_JOB_MISSED:
        return
    log.warning("scheduler.job_missed", job_id=event.job_id)
    try:
        callback()
    except Exception:  # noqa: BLE001 -- the scheduler thread must go on
        log.exception("scheduler.job_missed_close_failed", job_id=event.job_id)


def build_scheduler(settings):
    """Construct an APScheduler ``BackgroundScheduler`` with sane defaults.

    ``max_instances=1`` per job id is set per-job at registration time
    (``start``), not here, because the global default doesn't apply
    retroactively to jobs added later.
    """
    # Imported here so a missing apscheduler install doesn't break module
    # import for callers that only need ``gather_for_profile``.
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler(
        timezone="UTC",
        # The default pool is 10 threads. Scans, backfills, trial scans,
        # imports, seed-UMLS extractions and rerank refreshes share it; a
        # job queued behind ten long ones sat with its row open and no
        # heartbeat until it counted as stale.
        executors={"default": ThreadPoolExecutor(max_workers=24)},
        job_defaults={
            "coalesce": True,           # collapse missed firings into one
            "max_instances": 1,         # one run per job at a time
            "misfire_grace_time": 600,  # 10 min grace after a missed fire
        },
    )
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED

    # Ran (or failed) -> its callback is dropped; missed -> it runs (CR.6).
    scheduler.add_listener(_job_event, EVENT_JOB_MISSED | EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
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

    from ..api.settings import get_settings

    problem = valid_cron(cron, tz)
    if problem is not None:
        log.error("scheduler.invalid_cron", profile_id=profile_id, cron=cron, tz=tz, error=problem)
        return False
    if not get_settings().RADAR_NIGHTLY_GATHER_ENABLED:
        # Paused for the whole server; the row stays enabled, so turning
        # the setting back on and restarting brings every job back.
        log.info("scheduler.nightly_paused", profile_id=profile_id, slug=slug)
        return True
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
    # Reranker logits that went stale (queries changed) or were never
    # computed (papers that started passing after their scan); requests
    # from the routes run at once, this catches whatever they missed.
    from . import jobs as jobs_module

    scheduler.add_job(
        jobs_module.refresh_rerankers_sweep,
        trigger=IntervalTrigger(minutes=30),
        id="rerank-refresh-sweep",
        name="rerank-refresh-sweep",
        replace_existing=True,
        max_instances=1,
        next_run_time=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    jobs_module._SCHEDULER = scheduler
    scheduler.start()
    log.info("scheduler.started", n_jobs=len(rows))


def stop(scheduler) -> None:
    """Graceful shutdown for the FastAPI lifespan teardown."""
    if scheduler is None:
        return
    from . import jobs as jobs_module

    if jobs_module._SCHEDULER is scheduler:
        jobs_module._SCHEDULER = None
    try:
        scheduler.shutdown(wait=False)
    except Exception:  # noqa: BLE001 — log-and-swallow during teardown
        logging.getLogger("rag_lib.scheduler").exception("scheduler.shutdown_failed")
