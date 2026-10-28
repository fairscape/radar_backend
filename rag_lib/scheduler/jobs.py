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
import sqlite3
import threading
import time
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
from ..embedders import embed_progress
from ..gatherers.openalex import OpenAlexGatherer
from ..openalex_tiers import enabled_topic_ids
from ..paper import Paper
from ..persistence.db_store import dedup_and_insert_candidates
from ..scoring.pool import KEEP_ALL, in_years, passes, rescore_profile_pool
from ..profile import Profile
from ..radar import _days_ago_iso
from ..selector import Selector
from ..selectors import get_selector, selector_from_config

log = structlog.get_logger("rag_lib.scheduler.jobs")


GATHER_DAYS_DEFAULT = 1
# No total cap: a scan takes each topic's most cited (gatherers.openalex.
# PER_TOPIC); RADAR_GATHER_MAX_CANDIDATES, when set, still caps the total.
GATHER_LIMIT_DEFAULT: int | None = None

# How often the embedding-loop tick writes to gather_runs. Each write is
# a tiny SQLite UPDATE so the cost is small, but with hundreds of
# candidates and a 2.5s frontend poll there's no point updating more
# often than the UI can show.
_EMBED_TICK_EVERY = 5


class _ProgressReporter:
    """Bind a (conn, run_id) so jobs can mark phases and tick the
    embedding counter without threading bookkeeping through every call.

    The class owns nothing the gather body doesn't already own — the
    conn passed in is the same connection ``gather_for_profile`` uses
    for its other writes, so progress updates land on the same
    transaction discipline as the audit row itself.
    """

    def __init__(self, conn, run_id: int) -> None:
        self._conn = conn
        self._run_id = run_id
        self._embed_done = 0
        self._last_total: int | None = None

    def step(
        self,
        name: str,
        *,
        total: int | None = None,
        message: str | None = None,
    ) -> None:
        self._embed_done = 0
        self._last_total = total
        gather_runs_repo.set_step(
            self._conn, self._run_id, name, n_total=total, message=message,
        )

    def tick(self, *, n_processed: int, message: str | None = None) -> None:
        """Report absolute progress within the current step.

        ``make_embed_tick`` counts for callers that only know "one more
        done"; this is for loops that already know where they are, such
        as the Prosopia import walking a fixed list of works. Both land
        on the same column, so the frontend needs no second idiom.
        """
        gather_runs_repo.tick(
            self._conn, self._run_id,
            n_processed=n_processed, message=message,
        )

    def make_embed_tick(self, total: int | None = None) -> "callable":
        """Return a zero-arg callback for embed_progress().

        ``total`` is optional — if the caller already knows the total
        candidate count it can pass it here, otherwise the tick reads
        the most recent total set via ``step()`` (so a later
        ``step("embedding", total=...)`` call still produces a correct
        N/M message).
        """
        # Reset the counter on each fresh embedding pass so a retried
        # gather doesn't carry forward a stale value.
        self._embed_done = 0
        if total is not None:
            self._last_total = total

        def _tick() -> None:
            self._embed_done += 1
            current_total = self._last_total
            if self._embed_done % _EMBED_TICK_EVERY != 0 and (
                current_total is None or self._embed_done != current_total
            ):
                return
            msg = (
                f"Embedded {self._embed_done} / {current_total} candidates"
                if current_total
                else f"Embedded {self._embed_done} candidates"
            )
            gather_runs_repo.tick(
                self._conn, self._run_id,
                n_processed=self._embed_done, message=msg,
            )

        return _tick


def _resolve_mailto(user_row, default_mailto: str) -> str:
    return users_repo.mailto_from_row(user_row, default_mailto)


def _paper_from_row(prow) -> Paper:
    """A ``papers`` row as a domain ``Paper`` (text and topics, no vectors)."""
    topics = papers_repo.decode_topics(prow)
    return Paper.from_dict(
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
        paper = _paper_from_row(prow)
        vec = embeddings_repo.get(conn, oa_id, embedding_model)
        if vec is not None:
            paper.embeddings[embedding_model] = vec.tolist()
        papers.append(paper)

    from ..interest_terms import load as load_terms, query_labels

    return Profile(
        name=profile_row["name"],
        papers=papers,
        topic_filters=topic_filters,
        embedding_model=embedding_model,
        threshold=profile_row["threshold"],
        interest_terms=query_labels(load_terms(profile_row)),
    )


def _build_selector(profile_row, profile: Profile) -> Selector:
    """Reuse the persisted selector config when available, else refit.

    Dispatches by ``selector_config_json["type"]`` through the selector
    registry, so any registered selector (centroid, max_seed, or a
    plugin) hydrates without changes here. Falls back to a fresh
    centroid fit when the profile has no persisted config (legacy rows
    written before the registry landed).
    """
    sel_json = profile_row["selector_config_json"]
    cfg: dict | None = None
    if sel_json:
        try:
            cfg = json.loads(sel_json)
        except json.JSONDecodeError:
            cfg = None

    if cfg and cfg.get("type"):
        selector = selector_from_config(cfg)
        # If the persisted config carries fitted state, use it as-is.
        # Otherwise fit now against the live profile so the scheduler
        # never hands an unfit selector to ``select()``.
        if not _is_fit_payload(cfg):
            selector.fit(profile)
        return selector

    cls = get_selector("centroid")
    selector = cls(embedding_model=profile.embedding_model)
    selector.fit(profile)
    return selector


def _is_fit_payload(cfg: dict) -> bool:
    """Heuristic: a persisted config is "fit" if it carries any of the
    state keys produced by ``fit()``. Adding a selector with new state
    keys means extending this check."""
    return any(
        cfg.get(k) is not None
        for k in ("centroid", "seed_matrix")
    )


def rescore_pool(conn, profile_id: int, settings=None, *, reranker=None) -> int:
    """``rescore_profile_pool`` with the interest's own reranker weights.

    For every caller that changes the pool or its threshold: the gather,
    the CLI gather, and a threshold edit. Building the reranker only reads
    its config; the model loads on first use.
    """
    if settings is None:
        from ..api.settings import get_settings  # avoid circular import
        settings = get_settings()
    if reranker is None:
        prow = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        reranker = _build_reranker(prow, settings)
    # The pool may now hold papers never reranked, or reranked against
    # queries that just changed; rerank those in the background.
    if reranker.name != "noop":
        request_rerank_refresh(profile_id)
    return rescore_profile_pool(
        conn, profile_id,
        alpha=float(getattr(reranker, "alpha", getattr(settings, "RADAR_RERANKER_ALPHA", 0.4))),
        beta=float(getattr(reranker, "beta", getattr(settings, "RADAR_RERANKER_BETA", 0.6))),
    )


# Papers reranked per ``refresh_reranker`` call. The whole live pool was
# ~1,500 rows on 2026-10-14, so one call normally clears an interest; the
# cap bounds a GPU pass and the sweep picks up the rest.
RERANK_REFRESH_LIMIT = 600

# The live BackgroundScheduler, set by ``runner.start``; None in tests and
# the CLI, where a refresh request is a no-op and the sweep never runs.
_SCHEDULER = None
_refresh_locks: dict[int, Any] = {}
# Interests asked for while their refresh was running: APScheduler skips a
# job whose previous run is still going, so the job re-runs itself instead.
_refresh_again: set[int] = set()


def stale_rerank_ids(conn, profile_id: int, key: str, limit: int | None = None) -> list[str]:
    """Pool rows (what the feed can show) whose reranker logit is missing or
    was computed against another query set, best cosine first."""
    from ..db.repos.candidates import IN_YEARS, NOT_SEED, PASSES

    sql = f"""
        SELECT pc.openalex_id
        FROM profile_candidates pc
        JOIN profiles pr ON pr.id = pc.profile_id
        JOIN papers p ON p.openalex_id = pc.openalex_id
        WHERE pc.profile_id = ? AND {PASSES} AND {IN_YEARS} AND {NOT_SEED}
          AND (pc.score_reranker_raw IS NULL OR pc.reranker_query_key IS NULL
               OR pc.reranker_query_key != ?)
        ORDER BY pc.score_raw DESC
    """
    params: list = [profile_id, key]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [r[0] for r in conn.execute(sql, params).fetchall()]


def refresh_reranker(conn, profile_id: int, settings=None, *, reranker=None,
                     profile: Profile | None = None, limit: int = RERANK_REFRESH_LIMIT) -> int:
    """Rerank the pool rows whose logit is stale or missing, then rescore the pool.

    Two ways a pool row ended up without a current logit. Its queries
    changed (a seed, topic or research-topic edit) after it was reranked,
    so its logit was on another scale but normalised with the rest. Or it
    started passing after its scan (a lowered threshold, a refit, a wider
    year range): a scan reranks only what passes at the time, so it stood
    at the pool's median for good. Returns the number reranked.
    """
    if settings is None:
        from ..api.settings import get_settings  # avoid circular import
        settings = get_settings()
    import threading

    lock = _refresh_locks.setdefault(profile_id, threading.Lock())
    if not lock.acquire(blocking=False):
        # One is running for this interest: have it go round once more,
        # since what it read may predate whatever asked for this one.
        _refresh_again.add(profile_id)
        return 0
    try:
        n = 0
        for _ in range(3):      # bounded: each pass reranks only what went stale
            _refresh_again.discard(profile_id)
            n += _refresh_once(conn, profile_id, settings, reranker=reranker, profile=profile, limit=limit)
            if profile_id not in _refresh_again:
                break
            # Asked again meanwhile: the queries may have changed since
            # ``reranker`` / ``profile`` were built, so rebuild both.
            reranker = profile = None
        return n
    finally:
        lock.release()


def _refresh_once(conn, profile_id: int, settings, *, reranker=None,
                  profile: Profile | None = None, limit: int = RERANK_REFRESH_LIMIT) -> int:
    """One ``refresh_reranker`` pass; the caller holds the interest's lock."""
    prow = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    if prow is None or prow["is_draft"]:
        return 0
    if reranker is None:
        reranker = _build_reranker(prow, settings)
    if reranker.name == "noop" or not hasattr(reranker, "query_key"):
        return 0
    if profile is None:
        profile = _load_profile(conn, prow)
    key = reranker.query_key(profile)
    if key is None:
        return 0
    ids = stale_rerank_ids(conn, profile_id, key, limit)
    if not ids:
        return 0
    rows = {r["openalex_id"]: r for r in conn.execute(
        f"SELECT pc.openalex_id, pc.score_raw, pc.score, pc.reranker_query_key FROM profile_candidates pc "
        f"WHERE pc.profile_id = ? AND pc.openalex_id IN ({','.join('?' * len(ids))})",
        (profile_id, *ids),
    ).fetchall()}
    entries = []
    for oa in ids:
        prow_p = papers_repo.get_by_openalex_id(conn, oa)
        if prow_p is None:
            continue
        raw = rows[oa]["score_raw"] if rows[oa]["score_raw"] is not None else rows[oa]["score"]
        entries.append((float(raw or 0.0), _paper_from_row(prow_p), {"score_raw": raw}))
    out = reranker.rerank(entries, profile, conn=None)
    # Only rows nobody rewrote meanwhile: a scan that reranked a paper
    # against newer queries while this ran must keep its logit.
    conn.executemany(
        "UPDATE profile_candidates SET score_reranker_raw = ?, reranker_query_key = ? "
        "WHERE profile_id = ? AND openalex_id = ? AND reranker_query_key IS ?",
        [(bd["score_reranker_raw"], bd.get("reranker_query_key", key), profile_id, p.openalex_id,
          rows[p.openalex_id]["reranker_query_key"])
         for _, p, bd in out if bd.get("score_reranker_raw") is not None],
    )
    conn.commit()
    rescore_profile_pool(
        conn, profile_id,
        alpha=float(getattr(reranker, "alpha", 0.4)), beta=float(getattr(reranker, "beta", 0.6)),
    )
    log.info("scheduler.rerank_refreshed", profile_id=profile_id, n=len(out))
    return len(out)


def refresh_reranker_job(profile_id: int) -> None:
    """Background entry point for ``refresh_reranker`` (own connection; logs, never raises)."""
    from ..api.settings import get_settings

    settings = get_settings()
    conn = connect(settings.RADAR_DB_PATH)
    try:
        refresh_reranker(conn, profile_id, settings)
    except Exception:  # noqa: BLE001 -- a failed refresh leaves the old logits
        log.exception("scheduler.rerank_refresh_failed", profile_id=profile_id)
    finally:
        conn.close()


def refresh_rerankers_sweep() -> None:
    """Every live interest with stale logits: the net under ``request_rerank_refresh``
    (a request made while a refresh ran, a restart, rows from before 0019)."""
    from ..api.settings import get_settings

    settings = get_settings()
    if not getattr(settings, "RADAR_RERANKER_ENABLED", False):
        return
    conn = connect(settings.RADAR_DB_PATH)
    try:
        pids = [int(r[0]) for r in conn.execute("SELECT id FROM profiles WHERE is_draft = 0").fetchall()]
    finally:
        conn.close()
    for pid in pids:
        refresh_reranker_job(pid)


def request_rerank_refresh(profile_id: int) -> None:
    """Refresh an interest's reranker logits soon, in the background.

    Called wherever the pool or the queries change. A no-op without a
    running scheduler; a request while one is pending replaces it, and one
    while it runs makes it run again when done.
    """
    if _SCHEDULER is None:
        return
    from datetime import datetime, timezone

    # Also for a refresh already running: APScheduler skips (and, the
    # trigger being a date, drops) the job below while one with its id is
    # mid-execution, so the request was lost until the 30-minute sweep.
    # The running loop goes round once more when it sees this.
    _refresh_again.add(profile_id)
    try:
        _SCHEDULER.add_job(
            refresh_reranker_job, trigger="date", run_date=datetime.now(tz=timezone.utc),
            args=[profile_id], id=f"rerank-refresh:{profile_id}",
            replace_existing=True, max_instances=1,
        )
    except Exception:  # noqa: BLE001 -- the sweep catches up
        log.exception("scheduler.rerank_refresh_request_failed", profile_id=profile_id)


def _build_reranker(profile_row, settings):
    """Construct a reranker from profile config or global settings.

    Returns a ``NoopReranker`` when the global toggle is off, otherwise
    checks for a per-profile ``reranker_config_json`` override before
    falling back to the operator's default settings.

    Settings are read via ``getattr`` with defaults: callers legitimately
    pass partial config objects (tests, the CLI), and a missing key must
    degrade to "no reranking" rather than raise — an AttributeError here
    would surface as a failed gather run for the whole profile.
    """
    from ..rerankers import get_reranker, reranker_from_config

    def _opt(name, default):
        return getattr(settings, name, default)

    if not _opt("RADAR_RERANKER_ENABLED", False):
        return get_reranker("noop")()

    rr_json = profile_row["reranker_config_json"] if profile_row else None
    if rr_json:
        try:
            cfg = json.loads(rr_json)
            return reranker_from_config(cfg)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    cls = get_reranker(_opt("RADAR_DEFAULT_RERANKER", "medcpt"))
    return cls(
        model_id=_opt("RADAR_RERANKER_MODEL_ID", "ncbi/MedCPT-Cross-Encoder"),
        alpha=_opt("RADAR_RERANKER_ALPHA", 0.4),
        beta=_opt("RADAR_RERANKER_BETA", 0.6),
        device=_opt("RADAR_RERANKER_DEVICE", "cpu"),
        batch_size=_opt("RADAR_RERANKER_BATCH_SIZE", 64),
        max_queries=_opt("RADAR_RERANKER_MAX_QUERIES", 30),
        min_umls_confidence=_opt("RADAR_RERANKER_MIN_UMLS_CONFIDENCE", 0.7),
        aggregation=_opt("RADAR_RERANKER_AGGREGATION", "mean"),
        query_mode=_opt("RADAR_RERANKER_QUERY_MODE", "topic"),
    )


def gather_for_profile(
    user_id: int,
    profile_id: int,
    *,
    settings: Any | None = None,
    days: int = GATHER_DAYS_DEFAULT,
    limit: int | None = GATHER_LIMIT_DEFAULT,
    run_id: int | None = None,
    tier: str = "scheduled",
    years: tuple[int, int | None] | None = None,
    on_success: Any | None = None,
) -> int | None:
    """Run one OpenAlex gather + score + persist cycle for a profile.

    ``on_success(conn)`` runs once the scan has succeeded, just before its
    run row is closed as done: a draft's first scan makes it live there
    (``save_draft``), so the run reads done only once it is. If it raises,
    the run is closed as failed with that error.

    ``years`` (``(year_from, year_to)``, ``year_to`` None = this year) makes
    it a backfill: papers from those years (``fetch_years``, most cited
    first, ``years.BACKFILL_PER_YEAR`` a year) instead of the last ``days``.
    Everything after the fetch -- scoring, keeping papers below the
    threshold, reranking what passes, the pool rescore -- is the same.

    When ``run_id`` is supplied (e.g., the ``gather-now`` endpoint
    pre-created the row so it could echo the id back to the caller), the
    existing row is reused. Otherwise a new row is opened. Returns the
    ``gather_runs.id`` actually used. On error the row is closed with
    the exception message in ``error`` and the id is still returned.
    """
    if settings is None:
        from ..api.settings import get_settings  # avoid circular at import time
        settings = get_settings()
    # Deployment-wide ceiling on the fetch (RADAR_GATHER_MAX_CANDIDATES);
    # the route's ``limit`` query param and the daily default both bow to it.
    from ..api.services.wizard import capped_fetch_limit
    limit = capped_fetch_limit(settings, limit)

    lock = _scan_locks.setdefault(profile_id, threading.Lock())
    if not lock.acquire(blocking=False):
        # Another scan of this interest is in this process right now (the
        # nightly one and a "Scan now", or the wizard's first scan).
        conn = connect(settings.RADAR_DB_PATH)
        try:
            if run_id is None:
                run_id = gather_runs_repo.start(
                    conn, profile_id=profile_id, user_id=user_id, tier_used=tier,
                )
            gather_runs_repo.finish(
                conn, run_id, n_fetched=0, n_new=0, n_redup=0,
                error=SCAN_BUSY,
            )
        finally:
            conn.close()
        log.info("scheduler.gather_skipped", profile_id=profile_id, run_id=run_id, reason="scan_running")
        return run_id
    try:
        return _gather_for_profile_locked(
            user_id, profile_id, settings=settings, days=days, limit=limit,
            run_id=run_id, tier=tier, years=years, on_success=on_success,
        )
    except Exception as exc:
        # Died before its own try (the database could not be opened, the
        # profile row read failed): the pre-created row stayed open and
        # the interest read "scan running" for two hours.
        _close_run_after_failure(settings, run_id, exc)
        raise
    finally:
        lock.release()


def _close_run_after_failure(settings, run_id: int | None, exc: BaseException) -> None:
    """Close a job's pre-created run row with the error, on a fresh
    connection, when the job failed outside the code that closes it
    itself. Best effort: the failure being reported comes first."""
    if run_id is None:
        return
    try:
        conn = connect(settings.RADAR_DB_PATH)
    except Exception:  # noqa: BLE001
        log.exception("scheduler.run_close_failed", run_id=run_id)
        return
    try:
        row = gather_runs_repo.get(conn, run_id)
        if row is not None and row["finished_at"] is None:
            gather_runs_repo.finish(conn, run_id, n_fetched=0, n_new=0, n_redup=0,
                                    error=run_error(exc))
    except Exception:  # noqa: BLE001
        log.exception("scheduler.run_close_failed", run_id=run_id)
    finally:
        conn.close()


def run_error(exc: BaseException) -> str:
    """What a failed scan's run row says: OpenAlex not answering in words
    (it ends "Nothing was changed."), anything else as the exception."""
    from ..openalex_status import OpenAlexUnavailable

    if isinstance(exc, OpenAlexUnavailable):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


SCAN_BUSY = "Skipped: another scan of this interest was running."
_scan_locks: dict[int, threading.Lock] = {}


def _close_failed_scan(conn, profile_id: int, run_id: int, settings, exc: BaseException) -> None:
    """Close a scan's run as failed, with the pool on one scale.

    The candidates are committed a hundred at a time, each chunk carrying
    the batch's own ``score_pct``/``score_blended``; a failure between
    chunks left those rows ranked against the pool on another scale until
    the next clean scan. The pool is re-scored first, best effort.
    """
    try:
        rescore_pool(conn, profile_id, settings)
    except Exception:  # noqa: BLE001 -- the failure being reported comes first
        log.exception("scheduler.gather_failed.rescore_failed", profile_id=profile_id, run_id=run_id)
    for attempt in (1, 2):
        try:
            gather_runs_repo.finish(
                conn, run_id, n_fetched=0, n_new=0, n_redup=0,
                error=run_error(exc),
            )
            break
        except sqlite3.OperationalError:
            # Locked past the busy timeout: once more after a moment. Open,
            # the row would count as a running scan for two hours.
            if attempt == 2:
                log.exception("scheduler.gather_failed.close_failed", profile_id=profile_id, run_id=run_id)
            else:
                time.sleep(2)
    log.error("scheduler.gather_failed", profile_id=profile_id, run_id=run_id,
              error=f"{type(exc).__name__}: {exc}")


def _refit_if_fit_stale(conn, user_id: int, profile_id: int, run_id: int | None, *, when: str) -> None:
    """Fit the interest on the seeds it has if the stored fit was made from
    other seeds. A seed attached or removed while a scan runs (an upload,
    "add as seeds") is not fitted then -- the refit refuses under a running
    scan -- and no scan fits it on its own: every scan hydrates the stored
    fit. Called at the start of a scan (under the scan lock, with the own
    run row open) and when it closes, clean or failed. Best effort."""
    try:
        row = profiles_repo.get(conn, profile_id)
        if row is None or row["is_draft"]:
            return
        from ..api.services import wizard as wizard_service

        fitted = wizard_service.fit_seed_ids(row)
        now = sorted(profiles_repo.list_seed_openalex_ids(conn, profile_id))
        if fitted is not None:
            if now == fitted:
                return
        elif not _legacy_fit_is_stale(row, len(now)):
            return
        log.info("scheduler.gather_refit_stale_fit", profile_id=profile_id, run_id=run_id, when=when,
                 legacy=fitted is None,
                 n_added=len(set(now) - set(fitted or [])), n_removed=len(set(fitted or []) - set(now)))
        try:
            wizard_service.refit_profile(conn, user_id=user_id, slug=row["slug"], own_run_id=run_id)
        except wizard_service.ScanRunning:
            # Another run opened as this one closed; that scan refits at its start.
            log.info("scheduler.gather_refit_deferred", profile_id=profile_id, run_id=run_id, when=when)
        except ValueError as exc:
            # No embeddable seed: the scan goes on with the stored fit. A
            # line, not a traceback -- a legacy fit says this every scan.
            log.warning("scheduler.gather_refit_skipped", profile_id=profile_id, run_id=run_id, when=when, error=str(exc))
    except Exception:  # noqa: BLE001
        log.exception("scheduler.gather_refit_stale_fit_failed", profile_id=profile_id, run_id=run_id, when=when)


def _legacy_fit_is_stale(row, n_seeds_now: int) -> bool:
    """A fit from before the seed ids were recorded: the one thing it tells
    about its seeds is how many it embedded (``seed_matrix``). Different
    count -> stale; no matrix (a CLI-written centroid, the full-text
    selector, a plugin) -> nothing to compare, so refit once: that records
    the seed ids and the exact comparison applies from then on."""
    sel_json = row["selector_config_json"]
    if not sel_json:
        return False
    try:
        cfg = json.loads(sel_json)
    except (ValueError, TypeError):
        return False
    matrix = cfg.get("seed_matrix") if isinstance(cfg, dict) else None
    return not isinstance(matrix, list) or len(matrix) != n_seeds_now


def _gather_for_profile_locked(
    user_id: int,
    profile_id: int,
    *,
    settings: Any,
    days: int,
    limit: int | None,
    run_id: int | None,
    tier: str,
    years: tuple[int, int | None] | None,
    on_success: Any | None = None,
) -> int | None:
    conn = connect(settings.RADAR_DB_PATH)
    try:
        user_row = users_repo.get(conn, user_id)
        mailto = _resolve_mailto(user_row, settings.RADAR_DEFAULT_MAILTO)

        profile_row = profiles_repo.get(conn, profile_id)
        if profile_row is None:
            log.error("scheduler.profile_missing", profile_id=profile_id)
            return None
        since = _days_ago_iso(days)
        span = None
        if years is not None:
            from .. import years as years_rules
            span = years_rules.fetch_span(years[0], years[1])
            since = f"{span[0]}-01-01"
        if run_id is None:
            run_id = gather_runs_repo.start(
                conn,
                profile_id=profile_id,
                user_id=user_id,
                since_date=since,
                tier_used=tier,
            )

        reporter = _ProgressReporter(conn, run_id)
        try:
            # Seeds that arrived since the fit (under an earlier scan, or
            # between "Scan now" and this job) are fitted before scoring.
            _refit_if_fit_stale(conn, user_id, profile_id, run_id, when="start")
            profile_row = profiles_repo.get(conn, profile_id) or profile_row
            reporter.step(
                "loading_profile",
                message="Loading profile and seed embeddings",
            )
            profile = _load_profile(conn, profile_row)

            # Every topic switched off is a deliberate "gather nothing",
            # so stop here rather than running the rest of the pipeline
            # over an empty candidate list. Bailing out before the
            # selector is built also skips reconstructing the centroid
            # and loading the reranker model for a run that can't
            # produce anything. Not an error — the run closes cleanly
            # with a message the profile page can show.
            topics = (profile.topic_filters or {}).get("topics") or []
            if topics and not enabled_topic_ids(profile.topic_filters):
                msg = (
                    f"All {len(topics)} topics are switched off — "
                    f"enable at least one to gather."
                )
                reporter.step("done", message=msg)
                if on_success is not None:
                    on_success(conn)
                gather_runs_repo.finish(
                    conn, run_id,
                    n_fetched=0, n_new=0, n_redup=0, api_calls=0,
                    tier_used="no-enabled-topics",
                )
                log.info(
                    "scheduler.gather_skipped",
                    profile_id=profile_id, run_id=run_id,
                    reason="no_enabled_topics", n_topics=len(topics),
                )
                return run_id

            selector = _build_selector(profile_row, profile)
            gatherer = OpenAlexGatherer(mailto=mailto)

            reporter.step("fetching", message="Querying OpenAlex")
            if span is not None:
                from .. import years as years_rules
                candidates = gatherer.fetch_years(
                    profile, span[0], span[1], per_year=years_rules.BACKFILL_PER_YEAR,
                )
            else:
                candidates = gatherer.fetch(profile, since=since, limit=limit)
            # An interest's own seeds are not news to it.
            candidates = profiles_repo.without_seeds(
                candidates, profiles_repo.seed_keys(conn, profile_id))
            embeddings_repo.attach_stored(
                conn, candidates, getattr(selector, "embedding_model", None) or profile.embedding_model)

            reporter.step(
                "embedding",
                total=len(candidates),
                message=f"Embedding {len(candidates)} candidates",
            )
            with embed_progress(reporter.make_embed_tick(len(candidates))):
                scored = selector.select(candidates, profile, threshold=KEEP_ALL)
            # Keep the papers below the threshold too. They used to be
            # dropped here, so the pool held only what cleared the threshold
            # of the day: lowering it later showed nothing new, and the
            # threshold histogram could only ever say "100% pass". They are
            # stored but kept out of the feed (candidates.PASSES), and are
            # not reranked -- the cross-encoder costs per paper, and a scan
            # fetches ~5x more than passes.
            # Outside the interest's years counts as below too: the feed
            # (candidates.IN_YEARS) never shows those, yet they were reranked
            # and counted as the run's "new" ("5 new papers", feed unchanged).
            yf, yt = profile_row["year_from"], profile_row["year_to"]

            def _reaches_feed(e) -> bool:
                return passes(e[2].get("score_raw", e[0]), profile.threshold) and in_years(e[1].year, yf, yt)

            ranked = [e for e in scored if _reaches_feed(e)]
            below = [e for e in scored if not _reaches_feed(e)]
            # --- Reranker stage ---
            reranker = _build_reranker(profile_row, settings)
            if reranker.name != "noop" and ranked:
                log.debug(
                    "scheduler.rerank.before",
                    profile_id=profile_id,
                    reranker=reranker.name,
                    n=len(ranked),
                    top5=[
                        {"score": round(s, 4), "title": (p.title or "")[:60]}
                        for s, p, _ in ranked[:5]
                    ],
                )
                reporter.step(
                    "reranking",
                    total=len(ranked),
                    message=f"Reranking {len(ranked)} candidates with {reranker.name}",
                )
                ranked = reranker.rerank(ranked, profile, conn=conn)
                log.info(
                    "scheduler.rerank.done",
                    profile_id=profile_id,
                    reranker=reranker.name,
                    n=len(ranked),
                    diagnostics=reranker.diagnostics(),
                    cost=reranker.cost(),
                    top5=[
                        {
                            "blended": round(s, 4),
                            "rr_norm": bd.get("score_reranker_norm"),
                            "title": (p.title or "")[:60],
                        }
                        for s, p, bd in ranked[:5]
                    ],
                )
            else:
                log.debug(
                    "scheduler.rerank.skipped",
                    profile_id=profile_id,
                    reranker=reranker.name,
                    n=len(ranked) if ranked else 0,
                )

            # Carry the actual tier the gatherer landed on (e.g.
            # "must-have-AND") through to the audit row + per-candidate
            # rows. Falls back to ``tier`` (the scheduler's calling
            # context — "scheduled" / "manual") when no tier ran (empty
            # topic_filters, gatherer.last_tier_used is None).
            # getattr, not attribute access: last_tier_used is an
            # optional reporting hook, not part of the Gatherer protocol,
            # so a custom or stub gatherer that omits it must not fail
            # the run.
            tier_label = getattr(gatherer, "last_tier_used", None) or tier
            reporter.step(
                "persisting",
                total=len(ranked) + len(below),
                message=f"Saving {len(ranked)} candidates above the threshold and {len(below)} below",
            )
            # Passing papers first: the in-batch title dedup keeps the first
            # copy, and only they count as new -- the run's "new" is what
            # reached the feed.
            n_new, n_redup = dedup_and_insert_candidates(
                conn,
                profile_id=profile_id,
                gather_run_id=run_id,
                ranked=ranked + below,
                tier_used=tier_label,
                source_topics=getattr(gatherer, "last_source_topics", None),
                n_counted=len(ranked),
            )
            # The blend and percentile just written are relative to this
            # batch only; the feed ranks the whole pool. Put every row back
            # on one scale (see scoring/pool.py for what went wrong).
            # Seeds changed while this ran (a refit rewrote the fit): this
            # batch was scored against the old centroid, and its rows went
            # in after the refit had rescored the pool. Score every stored
            # candidate against the current seeds again, and rerank with
            # queries rebuilt from them.
            now_row = profiles_repo.get(conn, profile_id)
            if now_row is not None and now_row["selector_config_json"] != profile_row["selector_config_json"]:
                from ..scoring.rescore import rescore_candidates

                log.info("scheduler.gather_seeds_changed", profile_id=profile_id, run_id=run_id)
                rescore_candidates(conn, profile_id)
                reranker, profile = _build_reranker(now_row, settings), None
                # "New" is what reached the feed; count it on the new scores.
                from ..db.repos.candidates import IN_YEARS, NOT_SEED, PASSES

                n_new = int(conn.execute(
                    f"""
                    SELECT count(*) FROM profile_candidates pc
                    JOIN profiles pr ON pr.id = pc.profile_id
                    JOIN papers p ON p.openalex_id = pc.openalex_id
                    JOIN gather_runs g ON g.id = pc.gather_run_id
                    WHERE pc.profile_id = ? AND pc.gather_run_id = ?
                      AND pc.fetched_at >= g.started_at
                      AND {PASSES} AND {IN_YEARS} AND {NOT_SEED}
                    """,
                    (profile_id, run_id),
                ).fetchone()[0])
            # Pool papers this batch did not rerank (they started passing
            # since their own scan, or their queries changed) get a current
            # logit now; the rest of the pool was blended at the median.
            if reranker.name != "noop":
                reporter.step("reranking", message="Reranking older papers in the pool")
                try:
                    if not refresh_reranker(conn, profile_id, settings, reranker=reranker, profile=profile):
                        # Nothing stale, or a background refresh holds the
                        # interest: have it go round once more after this.
                        request_rerank_refresh(profile_id)
                except Exception:  # noqa: BLE001 -- the scan itself succeeded; the sweep retries
                    log.exception("scheduler.rerank_refresh_failed", profile_id=profile_id)
            rescore_profile_pool(
                conn, profile_id,
                alpha=float(getattr(reranker, "alpha", 0.4)), beta=float(getattr(reranker, "beta", 0.6)),
            )
            api_calls = gatherer.cost().get("api_calls", 0)
            if on_success is not None:
                on_success(conn)
            gather_runs_repo.finish(
                conn,
                run_id,
                n_fetched=len(candidates),
                n_new=n_new,
                n_redup=n_redup,
                api_calls=int(api_calls),
                tier_used=tier_label,
            )
            log.info(
                "scheduler.gather_ok",
                profile_id=profile_id,
                run_id=run_id,
                n_fetched=len(candidates),
                n_new=n_new,
                n_redup=n_redup,
            )
            _refit_if_fit_stale(conn, user_id, profile_id, run_id, when="end")
            return run_id
        except Exception as exc:  # noqa: BLE001 — surface into audit row
            _close_failed_scan(conn, profile_id, run_id, settings, exc)
            _refit_if_fit_stale(conn, user_id, profile_id, run_id, when="failed")
            return run_id
    finally:
        conn.close()


def save_draft(
    user_id: int,
    profile_id: int,
    *,
    run_id: int,
    settings: Any | None = None,
    days: int = 7,
    limit: int | None = GATHER_LIMIT_DEFAULT,
    years: tuple[int, int | None] | None = None,
    cron: str | None = None,
    tz: str | None = None,
    register: Any | None = None,
) -> int | None:
    """``POST /api/profiles``' job: the draft's first scan, and the draft
    goes live only when it succeeds (FUNCTIONAL_SPEC D.2-5, F2.3 Q1).

    The scan is the usual one (the last ``days``, or ``years`` as a
    backfill). Its success makes the draft live inside the scan's own
    close, so its run reads done only once the interest is live. A scan
    that fails -- OpenAlex unavailable among others -- leaves a draft with
    nothing from the scan in it: the feed rows it stored are removed, the
    run keeps the error, and saving again starts over. ``register()`` then
    registers the nightly job (the route's scheduler).
    """
    if settings is None:
        from ..api.settings import get_settings  # avoid circular import
        settings = get_settings()
    from ..api.services import wizard as wizard_service

    def live(conn) -> None:
        wizard_service.go_live(conn, profile_id, cron=cron, tz=tz)

    run_id = gather_for_profile(
        user_id, profile_id, settings=settings, days=days, limit=limit,
        run_id=run_id, tier="backfill" if years else "first_scan", years=years, on_success=live,
    )
    conn = connect(settings.RADAR_DB_PATH)
    try:
        row = profiles_repo.get(conn, profile_id)
        if row is not None and row["is_draft"]:
            conn.execute("DELETE FROM profile_candidates WHERE profile_id = ?", (profile_id,))
            conn.commit()
            log.info("save_draft.failed", profile_id=profile_id, run_id=run_id)
        elif row is not None and register is not None:
            try:
                register()
            except Exception:  # noqa: BLE001 -- live; the next boot registers it
                log.exception("save_draft.register_failed", profile_id=profile_id)
    finally:
        conn.close()
    return run_id


def dry_run_for_draft(
    user_id: int,
    profile_id: int,
    slug: str,
    *,
    settings: Any | None = None,
    days: int = 30,
    thresholds: list[float] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    selected_topic_ids: list[str] | None = None,
    run_id: int,
) -> int | None:
    """Async dry-run body for the wizard's calibrate step.

    Mirrors :func:`gather_for_profile` for the dry-run flow: opens its
    own connection, drives the same ``_ProgressReporter`` plumbing, and
    closes the audit row with the serialized DraftDryRun stashed in
    ``result_json`` for the status-poll endpoint to hand back.

    The kickoff route pre-creates ``run_id`` so it can echo it to the
    caller; we always reuse it (no slug→profile re-lookup needed since
    the route already has the row).
    """
    if settings is None:
        from ..api.settings import get_settings  # avoid circular import
        settings = get_settings()

    # Local import keeps scheduler/jobs.py free of an api/services
    # dependency at module load (rag_lib is occasionally imported from
    # contexts that don't pull api in).
    from ..api.services.wizard import compute_dry_run_for_profile
    from ..db.repos import profiles as profiles_repo

    try:
        conn = connect(settings.RADAR_DB_PATH)
    except Exception as exc:
        _close_run_after_failure(settings, run_id, exc)
        raise
    try:
        profile_row = profiles_repo.get(conn, profile_id)
        if profile_row is None:
            log.error("dry_run.profile_missing", profile_id=profile_id)
            gather_runs_repo.finish(
                conn, run_id,
                n_fetched=0, n_new=0, n_redup=0,
                error=f"profile {profile_id} not found at dispatch time",
            )
            return run_id

        reporter = _ProgressReporter(conn, run_id)
        try:
            with embed_progress(reporter.make_embed_tick(None)):
                # The reporter doesn't know the embed total ahead of
                # time — compute_dry_run_for_profile sets it via
                # reporter.step("embedding", total=...) once the
                # candidate list is known.
                result = compute_dry_run_for_profile(
                    conn, settings,
                    profile_row=profile_row,
                    slug=slug,
                    days=days,
                    thresholds=thresholds,
                    year_from=year_from,
                    year_to=year_to,
                    selected_topic_ids=selected_topic_ids,
                    reporter=reporter,
                )
            reporter.step("persisting", message="Saving dry-run result")
            gather_runs_repo.finish(
                conn, run_id,
                n_fetched=len(result.scores),
                n_new=0,
                n_redup=0,
                tier_used="dry_run",
                result_json=result.model_dump_json(),
            )
            log.info(
                "dry_run.ok",
                profile_id=profile_id, run_id=run_id,
                n=len(result.scores),
            )
            return run_id
        except Exception as exc:  # noqa: BLE001 — surface into audit row
            gather_runs_repo.finish(
                conn, run_id,
                n_fetched=0, n_new=0, n_redup=0,
                tier_used="dry_run",
                error=run_error(exc),
            )
            log.exception(
                "dry_run.failed",
                profile_id=profile_id, run_id=run_id,
            )
            return run_id
    finally:
        conn.close()


def import_prosopia_profile(
    plan: Any,
    *,
    settings: Any | None = None,
    run_id: int,
) -> int | None:
    """Async body for ``POST /api/import/prosopia``.

    The expensive half of the import: an OpenAlex round trip per DOI
    batch and one embedding per paper. For the reference profile that is
    82 papers behind a cold SPECTER2 load, which is minutes — far past
    what a request can hold open, and with no way to show progress if it
    tried.

    The kickoff route has already read the Prosopia profile and created
    the draft, so ``plan`` carries everything needed and there is no
    second chance for the slug to 404 somewhere the caller cannot see.
    Like the dry-run job we reuse the pre-created ``run_id`` and close
    the row either way — a failure lands on ``gather_runs.error``, which
    is what the poll endpoint shows.
    """
    if settings is None:
        from ..api.settings import get_settings  # avoid circular import
        settings = get_settings()

    from ..api.services.prosopia import run_import

    try:
        conn = connect(settings.RADAR_DB_PATH)
    except Exception as exc:
        _close_run_after_failure(settings, run_id, exc)
        raise
    try:
        # The kick-off route opened the row under the tier that says where
        # the seeds come from (prosopia_import / orcid_import); keep it.
        row = gather_runs_repo.get(conn, run_id)
        tier = (row["tier_used"] if row is not None else None) or "prosopia_import"
        reporter = _ProgressReporter(conn, run_id)
        try:
            reporter.step(
                "loading_profile",
                total=len(plan.records),
                message=f"Importing {len(plan.records)} works from '{plan.slug}'",
            )
            result = run_import(
                conn, settings, plan=plan, reporter=reporter, run_id=run_id,
            )
            reporter.step("persisting", message="Saving import result")
            gather_runs_repo.finish(
                conn, run_id,
                n_fetched=result["drafted"],
                n_new=result["drafted"],
                n_redup=0,
                tier_used=tier,
                result_json=json.dumps(result),
            )
            log.info(
                "prosopia_import.ok",
                run_id=run_id, slug=plan.slug,
                draft_slug=plan.draft_slug, drafted=result["drafted"],
                resolved_by=result["resolved_by"],
            )
            # After finish(), not before: the draft is usable now, and this
            # can take a few seconds a paper. Doing it here means wizard
            # step 3 -- usually opened minutes later -- finds the seeds
            # already extracted instead of fitting what it can into its
            # 30-second budget. Anything left is the periodic job's.
            _backfill_import_umls(conn, settings, plan)
            return run_id
        except Exception as exc:  # noqa: BLE001 — surface into audit row
            from ..api.services.prosopia import import_error, undo_import

            gather_runs_repo.finish(
                conn, run_id,
                n_fetched=0, n_new=0, n_redup=0,
                tier_used=tier,
                error=import_error(exc),
            )
            log.exception(
                "prosopia_import.failed", run_id=run_id, slug=plan.slug,
            )
            # Nothing half-built stays: the draft and researcher this
            # import's request made, and what it stored (FUNCTIONAL_SPEC D).
            try:
                undo_import(conn, plan, run_id=run_id)
            except Exception:  # noqa: BLE001 -- the run already says it failed
                log.exception("prosopia_import.undo_failed", run_id=run_id)
            return run_id
    finally:
        conn.close()


def _backfill_import_umls(conn, settings, plan) -> None:
    """UMLS for the papers one import brought in. Best-effort."""
    from ..api.services.vault import backfill_seed_umls

    try:
        if plan.profile_id is not None:
            ids = profiles_repo.list_seed_openalex_ids(conn, int(plan.profile_id))
        elif plan.researcher_id is not None:
            ids = [r[0] for r in conn.execute(
                "SELECT openalex_id FROM researcher_papers WHERE researcher_id = ?",
                (int(plan.researcher_id),),
            )]
        else:
            return
        backfill_seed_umls(conn, settings, openalex_ids=ids, limit=max(1, len(ids)))
    except Exception:  # noqa: BLE001 -- the import itself already succeeded
        log.exception("prosopia_import.umls_backfill_failed", slug=plan.slug)


# Interests whose seeds ``extract_seed_umls_job`` is working through.
class _RunningFlags:
    """Interests whose seed-UMLS extraction is running, in memory (the
    extraction has no run row). An entry outlives its job only when the
    job never ran (APScheduler dropped it) or died outside its own
    ``finally``; such an entry expires, so step 3 is not stuck on "running"
    until the next restart."""

    MAX_MINUTES = 30

    def __init__(self) -> None:
        self._since: dict[int, float] = {}

    def add(self, profile_id: int) -> None:
        self._since[int(profile_id)] = time.monotonic()

    def discard(self, profile_id: int) -> None:
        self._since.pop(int(profile_id), None)

    def clear(self) -> None:
        self._since.clear()

    def __contains__(self, profile_id: object) -> bool:
        since = self._since.get(int(profile_id)) if isinstance(profile_id, int) else None
        if since is None:
            return False
        if time.monotonic() - since > self.MAX_MINUTES * 60:
            self._since.pop(int(profile_id), None)
            return False
        return True


SEED_UMLS_RUNNING = _RunningFlags()


def extract_seed_umls_job(profile_id: int, settings: Any | None = None) -> int:
    """Extract UMLS for every seed of one interest that has none yet.

    Wizard step 3 starts this and waits for it, so the topic list is built
    from every seed's concepts. Step 3 used to extract inside its own
    request within a 30-second budget and show whatever had landed; an
    import's extraction ran after the import reported done, and what
    finished later never reached the draft's topics.
    """
    if settings is None:
        from ..api.settings import get_settings  # avoid circular import
        settings = get_settings()
    from ..api.services.vault import backfill_seed_umls

    conn = None
    try:
        # Inside the try: a failure to open the database used to leave the
        # flag set, and step 3 waited on it until the next restart.
        SEED_UMLS_RUNNING.add(profile_id)
        conn = connect(settings.RADAR_DB_PATH)
        ids = profiles_repo.list_seed_openalex_ids(conn, profile_id)
        return backfill_seed_umls(conn, settings, openalex_ids=ids, limit=max(1, len(ids)))
    except Exception:  # noqa: BLE001 -- the seeds stay pending; the sweep retries
        log.exception("scheduler.seed_umls_failed", profile_id=profile_id)
        return 0
    finally:
        if conn is not None:
            conn.close()
        SEED_UMLS_RUNNING.discard(profile_id)


def backfill_seed_umls_job(settings: Any | None = None) -> int:
    """Periodic: extract UMLS for seeds that have never been extracted.

    The safety net under every path a seed can arrive by -- an import
    whose thread died, a paper added from the vault or the feed, an
    upload whose extraction failed, step 3 running out of budget. Before
    it existed nothing ever came back for them. Bounded per run.

    It first reads the files of uploads migrated without their own text
    (``versions.parse_pending``), so their concepts come from that text,
    and embeds the public papers 0029 left without a vector
    (``versions.embed_queued``).
    """
    if settings is None:
        from ..api.settings import get_settings  # avoid circular import
        settings = get_settings()
    from ..api.services.vault import backfill_seed_umls

    from ..db.repos.versions import embed_queued, parse_pending

    conn = connect(settings.RADAR_DB_PATH)
    try:
        try:
            parse_pending(conn)
            embed_queued(conn, default_model=settings.RADAR_DEFAULT_EMBEDDING_MODEL)
        except Exception:  # noqa: BLE001 -- the UMLS pass still runs
            log.exception("scheduler.parse_pending_failed")
        return backfill_seed_umls(conn, settings, limit=40)
    except Exception:  # noqa: BLE001 -- next run tries again
        log.exception("scheduler.umls_backfill_failed")
        return 0
    finally:
        conn.close()


__all__ = ["gather_for_profile", "dry_run_for_draft", "import_prosopia_profile",
           "backfill_seed_umls_job", "extract_seed_umls_job"]
