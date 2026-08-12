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
from ..embedders import embed_progress
from ..gatherers.openalex import OpenAlexGatherer
from ..openalex_tiers import enabled_topic_ids
from ..paper import Paper
from ..persistence.db_store import dedup_and_insert_candidates
from ..profile import Profile
from ..radar import _days_ago_iso
from ..selector import Selector
from ..selectors import get_selector, selector_from_config

log = structlog.get_logger("rag_lib.scheduler.jobs")


GATHER_DAYS_DEFAULT = 1
GATHER_LIMIT_DEFAULT = 500

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

        reporter = _ProgressReporter(conn, run_id)
        try:
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
            candidates = gatherer.fetch(profile, since=since, limit=limit)

            reporter.step(
                "embedding",
                total=len(candidates),
                message=f"Embedding {len(candidates)} candidates",
            )
            with embed_progress(reporter.make_embed_tick(len(candidates))):
                ranked = selector.select(
                    candidates, profile, threshold=profile.threshold,
                )
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
                total=len(ranked),
                message=f"Saving {len(ranked)} ranked candidates",
            )
            n_new, n_redup = dedup_and_insert_candidates(
                conn,
                profile_id=profile_id,
                gather_run_id=run_id,
                ranked=ranked,
                tier_used=tier_label,
                source_topics=getattr(gatherer, "last_source_topics", None),
            )
            api_calls = gatherer.cost().get("api_calls", 0)
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


def dry_run_for_draft(
    user_id: int,
    profile_id: int,
    slug: str,
    *,
    settings: Any | None = None,
    days: int = 30,
    thresholds: list[float] | None = None,
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

    conn = connect(settings.RADAR_DB_PATH)
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
                error=f"{type(exc).__name__}: {exc}",
            )
            log.exception(
                "dry_run.failed",
                profile_id=profile_id, run_id=run_id,
            )
            return run_id
    finally:
        conn.close()


__all__ = ["gather_for_profile", "dry_run_for_draft"]
