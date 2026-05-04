"""Wizard service — Phase 11.

Backs the four-step profile-build wizard's ``/api/profiles/draft/...``
endpoints. A "draft" is a normal ``profiles`` row carrying
``is_draft=1``; the wizard mutates the same row across steps and the
commit endpoint flips the flag, so seeds attached during Step 1 keep
their FK on commit without copy.

  Step 1 — seeds: handled by the existing vault upload endpoint
            (``POST /api/vault/upload`` with ``profile_slug=<draft>``).
            ``profile_seeds`` rows accumulate against the draft's id.
  Step 2 — coherence: ``compute_draft_coherence`` reads the seeds'
            embeddings and returns histogram + median + IQR + bimodality.
  Step 3 — topics: ``aggregate_draft_topics`` aggregates OpenAlex topic
            filters across the draft's seeds and writes the full set
            back to ``topic_filters_json`` so Step 4 sees them.
  Step 4 — calibrate: ``dry_run_draft`` runs ``radar.dry_run`` over the
            last ``days`` of OpenAlex output and returns a sweep + a
            10-card preview of the top hits.
  Commit  — ``commit_draft`` flips ``is_draft`` to 0, persists the
            chosen threshold + the pruned topic_filters, fits and
            stores whichever selector the draft picked (registry
            dispatch — when it's the centroid selector, the fitted
            centroid lands on ``profiles.centroid`` for fast restart),
            and registers a daily schedule.
"""

from __future__ import annotations

import json
import random
import sqlite3
from typing import Any

import numpy as np
import structlog

from ...coherence import coherence as coherence_compute
from ...db import encode_vector
from ...db.repos import (
    embeddings as embeddings_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    schedules as schedules_repo,
)
from ...gatherers.openalex import OpenAlexGatherer
from ...paper import Paper, Topic
from ...profile import Profile
from ...selectors import get_selector
from ..mappers import _bucket_for, _read_minutes
from ..schemas import Card, DraftCoherence, DraftDryRun, SweepRow

log = structlog.get_logger("rag_lib.api.services.wizard")


DEFAULT_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
PREVIEW_CARDS = 10
MAX_DRY_RUN_DAYS = 30
# OpenAlex regularly returns 5–10k+ candidates for a multi-topic 30-day
# query. The dry-run is a calibration UX, not a complete sweep; cap the
# fetch so the wizard returns in seconds. Production gathers (post-
# commit) still fetch unbounded.
DRY_RUN_FETCH_LIMIT = 1000


# ---------------------------------------------------------------------------
# Step 0 — create draft
# ---------------------------------------------------------------------------


def create_draft(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    name: str,
    embedding_model: str = "placeholder-v1",
    selector: str = "centroid",
) -> dict:
    """Insert a draft profiles row. Returns ``{slug, name}``.

    The slug is ``slugify(name)`` with a numeric suffix on collision
    against any existing slug. ``profiles.name`` carries a global UNIQUE
    constraint (0001_initial.sql) so we mirror the same suffix into the
    name when needed; the user sees the suffix and can rename later.

    ``embedding_model`` is stored on the profile so coherence joins and
    selector fits use the same vectors the upload path produced.

    ``selector`` is the registry key (``"centroid"``, ``"max_seed"``, or
    a plugin-registered name). Stored as ``selector_config_json={"type":
    <key>}`` on the draft so the dry-run and commit steps know which
    selector to fit. Both knobs are validated up-front; an unknown key
    raises ``ValueError`` before any DB write.
    """
    from ...embedders import get_embedder
    get_embedder(embedding_model)  # validate; raises ValueError if unknown
    get_selector(selector)         # validate; raises ValueError if unknown

    base_slug = profiles_repo.slugify(name)
    slug, suffix = _unique_slug(conn, user_id, base_slug)
    final_name = name if suffix == 0 else f"{name} ({suffix})"
    profile_id = profiles_repo.create_draft(
        conn,
        user_id=user_id,
        name=final_name,
        slug=slug,
        embedding_model=embedding_model,
    )
    # Stash the chosen selector type so dry-run + commit pick it up.
    # No fitted state yet — that lands on commit.
    conn.execute(
        "UPDATE profiles SET selector_config_json = ? WHERE id = ?",
        (json.dumps({"type": selector}), profile_id),
    )
    conn.commit()
    return {"slug": slug, "name": final_name}


def _unique_slug(
    conn: sqlite3.Connection, user_id: int, base: str  # noqa: ARG001
) -> tuple[str, int]:
    """Return ``(slug, suffix_n)`` where suffix_n is 0 for the bare slug.

    Walks ``base``, ``base-2``, ``base-3``, ... and stops once both
    the slug AND the suffix-n itself are not in the global profiles
    table (slug + name carry separate UNIQUE constraints so we have
    to dodge both).
    """
    n = 1
    while True:
        slug = base if n == 1 else f"{base}-{n}"
        taken = conn.execute(
            "SELECT 1 FROM profiles WHERE slug = ? LIMIT 1", (slug,),
        ).fetchone() is not None
        if not taken:
            return slug, 0 if n == 1 else n
        n += 1


# ---------------------------------------------------------------------------
# Step 2 — coherence
# ---------------------------------------------------------------------------


def compute_draft_coherence(
    conn: sqlite3.Connection, *, user_id: int, slug: str
) -> DraftCoherence:
    row = _require_profile(conn, user_id, slug)
    profile_id = int(row["id"])
    embedding_model = row["embedding_model"]

    vecs = _seed_embeddings(conn, profile_id, embedding_model)
    bins = profiles_repo.coherence_bins(conn, profile_id)
    if not vecs:
        conn.execute(
            """
            UPDATE profiles SET
              coherence_median  = NULL,
              coherence_iqr     = NULL,
              coherence_bimodal = NULL,
              n_seed            = 0,
              updated_at        = datetime('now')
            WHERE id = ?
            """,
            (profile_id,),
        )
        conn.commit()
        return DraftCoherence(bins=bins, median=0.0, iqr=0.0, bimodal=False, n=0)
    metrics = coherence_compute(vecs)
    median = _finite(metrics["median"])
    iqr = _finite(metrics["iqr"])
    bimodal = bool(metrics["bimodal"])
    n = int(metrics["n"])
    # Persist now so the profile row reflects what we just showed
    # instead of waiting for commit_draft. Lets the profile detail page
    # render the histogram + median without re-running the compute.
    conn.execute(
        """
        UPDATE profiles SET
          coherence_median  = ?,
          coherence_iqr     = ?,
          coherence_bimodal = ?,
          n_seed            = ?,
          updated_at        = datetime('now')
        WHERE id = ?
        """,
        (median, iqr, 1 if bimodal else 0, len(vecs), profile_id),
    )
    conn.commit()
    return DraftCoherence(bins=bins, median=median, iqr=iqr, bimodal=bimodal, n=n)


# ---------------------------------------------------------------------------
# Step 3 — aggregated topics
# ---------------------------------------------------------------------------


def aggregate_draft_topics(
    conn: sqlite3.Connection, *, user_id: int, slug: str
) -> dict:
    """Aggregate OpenAlex topics across the profile's seeds.

    Works on drafts AND committed profiles, so the profile detail page
    can re-run topic aggregation after the user uploads more seeds.
    Persists the full aggregated dict back to ``topic_filters_json``.
    """
    row = _require_profile(conn, user_id, slug)
    profile_id = int(row["id"])
    papers = _load_seed_papers(conn, profile_id)
    topic_filters = Profile.aggregate_topic_filters(papers)
    conn.execute(
        "UPDATE profiles SET topic_filters_json = ?, updated_at = datetime('now') WHERE id = ?",
        (json.dumps(topic_filters), profile_id),
    )
    conn.commit()
    return topic_filters


# ---------------------------------------------------------------------------
# Step 4 — dry-run sweep + preview
# ---------------------------------------------------------------------------


def dry_run_draft(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    user_id: int,
    slug: str,
    days: int = MAX_DRY_RUN_DAYS,
    thresholds: list[float] | None = None,
    gatherer: Any | None = None,
    reporter: Any | None = None,
) -> DraftDryRun:
    """Resolve the draft + invoke the dry-run compute.

    Thin wrapper that does the slug→draft lookup and hands off to
    :func:`compute_dry_run_for_profile`. The async dry-run job calls
    the compute function directly (it already has the profile_id from
    the kickoff request), bypassing this slug lookup.
    """
    row = _require_draft(conn, user_id, slug)
    return compute_dry_run_for_profile(
        conn,
        settings,
        profile_row=row,
        slug=slug,
        days=days,
        thresholds=thresholds,
        gatherer=gatherer,
        reporter=reporter,
    )


def compute_dry_run_for_profile(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    profile_row: sqlite3.Row,
    slug: str,
    days: int = MAX_DRY_RUN_DAYS,
    thresholds: list[float] | None = None,
    gatherer: Any | None = None,
    reporter: Any | None = None,
) -> DraftDryRun:
    """Run a dry-run sweep + preview against the last ``days`` of OpenAlex.

    Returns a SweepRow per threshold plus a 10-card preview of the
    top-scoring candidates. Tests inject ``gatherer``; in production
    we build an ``OpenAlexGatherer`` from settings.

    ``reporter`` is an optional ``_ProgressReporter`` (or duck) the
    caller hands in when running under the scheduler — it stages step
    transitions through ``gather_runs.set_step``. ``embed_progress``
    is the caller's responsibility (it has to wrap ``select()``); pass
    ``None`` for the inline / test path which doesn't need progress
    reporting.
    """
    profile_id = int(profile_row["id"])

    days = max(1, min(int(days), MAX_DRY_RUN_DAYS))
    thresholds = sorted(thresholds or DEFAULT_THRESHOLDS)

    if reporter is not None:
        reporter.step("loading_profile", message="Loading draft seeds")
    profile = _load_profile(conn, profile_row)
    if not profile.papers:
        return DraftDryRun(sweep=[], preview=[])

    selector = _new_selector_for_draft(profile_row, profile)
    selector.fit(profile)

    gatherer = gatherer or OpenAlexGatherer(mailto=settings.RADAR_DEFAULT_MAILTO)
    since = _days_ago(days)

    log.info(
        "dry_run.openalex_query",
        slug=slug, days=days, since=since,
        limit=DRY_RUN_FETCH_LIMIT,
    )

    if reporter is not None:
        reporter.step("fetching", message="Querying OpenAlex")
    candidates = gatherer.fetch(profile, since=since, limit=DRY_RUN_FETCH_LIMIT)
    log.info(
        "dry_run.fetched",
        slug=slug, n=len(candidates),
        tier_used=getattr(gatherer, "last_tier_used", None),
        filter=getattr(gatherer, "last_filter_str", None),
        cost=dict(gatherer.cost()),
    )

    sample = random.sample(candidates, min(5, len(candidates)))
    for i, p in enumerate(sample):
        log.info(
            "dry_run.random_sample",
            slug=slug, idx=i,
            title=(p.title or "")[:160],
            abstract=(p.abstract or "")[:240],
        )

    if reporter is not None:
        reporter.step(
            "embedding",
            total=len(candidates),
            message=f"Embedding {len(candidates)} candidates",
        )
    # Caller wraps the select() call in embed_progress() when it wants
    # per-paper ticks — this function stays embedder-agnostic so the
    # inline / test path doesn't need a thread-local hook.
    ranked = selector.select(candidates, profile, threshold=None)

    if ranked:
        raws = [r[2].get("score_raw", r[0]) for r in ranked]
        arr = np.asarray(raws, dtype=float)
        log.info(
            "dry_run.cosine_distribution",
            slug=slug, n=len(arr),
            min=float(arr.min()), p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)), max=float(arr.max()),
            mean=float(arr.mean()), std=float(arr.std()),
        )

        def _log_top(label: str, key: str) -> None:
            top = sorted(ranked, key=lambda r: r[2].get(key, 0.0), reverse=True)[:5]
            for i, entry in enumerate(top):
                paper, b = entry[1], entry[2]
                log.info(
                    f"dry_run.top_by_{label}",
                    slug=slug, rank=i + 1,
                    centroid_cos=round(float(b.get("score_raw", 0.0)), 4),
                    max_seed_cos=round(float(b.get("score_max_seed", 0.0)), 4),
                    median_seed_cos=round(float(b.get("score_median_seed", 0.0)), 4),
                    pct=round(float(b.get("score_pct", 0.0)), 4),
                    title=(paper.title or "")[:160],
                )

        _log_top("centroid", "score_raw")
        _log_top("max_seed", "score_max_seed")
        _log_top("median_seed", "score_median_seed")

    sweep: list[SweepRow] = []
    for thr in thresholds:
        passing = [e for e in ranked if e[0] >= thr]
        top_title = passing[0][1].title if passing else ""
        sweep.append(SweepRow(
            th=round(float(thr), 4), n=len(passing), top=top_title or "",
        ))

    preview = _ranked_to_preview(
        ranked,
        slug=slug,
        active_topic_ids=_topic_ids(profile.topic_filters),
    )

    # Raw selector cosines for every fetched candidate. The wizard
    # renders a slider over these so the user picks θ against a real
    # histogram instead of a coarse grid of bucket cards.
    scores = [round(float(e[0]), 4) for e in ranked]

    return DraftDryRun(sweep=sweep, preview=preview, scores=scores)


# ---------------------------------------------------------------------------
# Commit — flip is_draft and register schedule
# ---------------------------------------------------------------------------


def commit_draft(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    slug: str,
    threshold: float,
    selected_topic_ids: list[str],
    cron: str | None = None,
    tz: str | None = None,
) -> dict:
    """Flip the draft to a live profile.

    1. Prune ``topic_filters_json`` to the user's chosen topic ids
       (other levels — subfields/fields/domains — are preserved as
       gatherer hints; the user toggles topics specifically).
    2. Re-fit the chosen selector. When the fitted state includes a
       centroid (the centroid selector's case), store it as a BLOB on
       ``profiles.centroid`` for fast read-side access; the full
       selector state still goes into ``selector_config_json``.
    3. Update n_seed from the actual ``profile_seeds`` count.
    4. ``schedules_repo.upsert`` so the scheduler picks the profile up
       on the next boot at the chosen cron (default: daily 04:00 UTC).
    """
    row = _require_draft(conn, user_id, slug)
    profile_id = int(row["id"])

    full_topics = profiles_repo.topic_filters(conn, profile_id)
    pruned = _prune_topic_filters(full_topics, set(selected_topic_ids))

    profile = _load_profile(conn, row, topic_filters=pruned)
    selector = _new_selector_for_draft(row, profile, threshold=threshold)
    selector.fit(profile)
    sel_cfg = selector.config()
    diag = sel_cfg.get("diagnostics_snapshot") or {}

    centroid_blob: bytes | None = None
    centroid_list = sel_cfg.get("centroid")
    if centroid_list:
        centroid_blob = encode_vector(centroid_list)

    profiles_repo.commit_draft(
        conn,
        profile_id,
        threshold=threshold,
        topic_filters=pruned,
        centroid=centroid_blob,
        selector_config=sel_cfg,
        coherence_median=diag.get("coherence_median"),
        coherence_iqr=diag.get("coherence_iqr"),
        coherence_bimodal=diag.get("coherence_bimodal"),
        n_seed=len(profile.papers),
    )
    schedules_repo.upsert(conn, profile_id=profile_id, cron=cron, tz=tz)
    return {"slug": slug, "id": profile_id}


# ---------------------------------------------------------------------------
# Abandon
# ---------------------------------------------------------------------------


def delete_draft(
    conn: sqlite3.Connection, *, user_id: int, slug: str
) -> bool:
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None or not row["is_draft"]:
        return False
    profiles_repo.delete(conn, int(row["id"]))
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_draft(
    conn: sqlite3.Connection, user_id: int, slug: str
) -> sqlite3.Row:
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None or not row["is_draft"]:
        raise LookupError(f"draft '{slug}' not found for user {user_id}")
    return row


def _new_selector_for_draft(
    row: sqlite3.Row, profile: Profile, *, threshold: float | None = None,
):
    """Instantiate the selector type the wizard recorded on the draft.

    Reads ``selector_config_json["type"]`` and looks the class up in the
    selectors registry. Drafts created before the selector field landed
    (or rows where the JSON is null/malformed) fall back to the
    centroid default so existing profiles keep working.
    """
    sel_json = row["selector_config_json"]
    sel_type = "centroid"
    if sel_json:
        try:
            cfg = json.loads(sel_json)
            if isinstance(cfg, dict) and isinstance(cfg.get("type"), str):
                sel_type = cfg["type"]
        except (TypeError, ValueError):
            pass
    cls = get_selector(sel_type)
    return cls(
        embedding_model=profile.embedding_model,
        threshold=threshold,
    )


def _require_profile(
    conn: sqlite3.Connection, user_id: int, slug: str
) -> sqlite3.Row:
    """Find a profile by slug — draft OR committed.

    Used by recompute paths that need to work after commit so the
    profile detail page can re-aggregate topics / re-fit coherence
    when the user uploads more seeds.
    """
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None:
        raise LookupError(f"profile '{slug}' not found for user {user_id}")
    return row


def _seed_embeddings(
    conn: sqlite3.Connection, profile_id: int, model: str
) -> list[list[float]]:
    rows = conn.execute(
        """
        SELECT pe.vector
        FROM profile_seeds ps
        JOIN paper_embeddings pe USING (openalex_id)
        WHERE ps.profile_id = ? AND pe.embedding_model = ?
        """,
        (profile_id, model),
    ).fetchall()
    if not rows:
        return []
    from ...db.codec import decode_vector
    return [decode_vector(r["vector"]).tolist() for r in rows]


def _load_seed_papers(
    conn: sqlite3.Connection, profile_id: int
) -> list[Paper]:
    seed_ids = profiles_repo.list_seed_openalex_ids(conn, profile_id)
    out: list[Paper] = []
    for oa_id in seed_ids:
        prow = papers_repo.get_by_openalex_id(conn, oa_id)
        if prow is None:
            continue
        topics = papers_repo.decode_topics(prow)
        out.append(Paper.from_dict({
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
        }))
    return out


def _load_profile(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    topic_filters: dict | None = None,
) -> Profile:
    profile_id = int(row["id"])
    embedding_model = row["embedding_model"]
    papers = _load_seed_papers(conn, profile_id)
    for paper in papers:
        vec = embeddings_repo.get(conn, paper.openalex_id, embedding_model)
        if vec is not None:
            paper.embeddings[embedding_model] = vec.tolist()
    if topic_filters is None:
        topic_filters = profiles_repo.topic_filters(conn, profile_id)
    return Profile(
        name=row["name"],
        papers=papers,
        topic_filters=topic_filters,
        embedding_model=embedding_model,
        threshold=row["threshold"],
    )


def _topic_ids(topic_filters: dict | None) -> list[str]:
    if not topic_filters:
        return []
    return [t.get("id") for t in (topic_filters.get("topics") or []) if t.get("id")]


def _prune_topic_filters(
    full: dict, selected_ids: set[str]
) -> dict:
    """Drop topic entries not in ``selected_ids``; keep other levels intact."""
    out = dict(full or {})
    out["topics"] = [
        t for t in (full.get("topics") or [])
        if t.get("id") in selected_ids
    ]
    return out


def _ranked_to_preview(
    ranked: list[tuple],
    *,
    slug: str,
    active_topic_ids: list[str],
) -> list[Card]:
    """Build preview Cards directly from the selector's ranked output."""
    out: list[Card] = []
    for entry in ranked[:PREVIEW_CARDS]:
        score, paper, _breakdown = _unpack(entry)
        out.append(_paper_to_card(paper, score, slug, active_topic_ids))
    return out


def _unpack(entry) -> tuple[float, Paper, dict]:
    if len(entry) == 3:
        return entry[0], entry[1], entry[2] or {}
    return entry[0], entry[1], {}


def _paper_to_card(
    paper: Paper,
    score: float,
    slug: str,
    active_topic_ids: list[str],
) -> Card:
    """Construct a ``Card`` from an in-memory ``Paper`` (preview only).

    Mirrors ``mappers.candidate_row_to_card`` but reads from a Paper
    dataclass instead of a sqlite row, since wizard previews don't
    persist candidates.
    """
    score = round(float(score), 4)
    terms = _paper_terms(paper)
    matched = [t.display_name for t in _matched_topics(paper, set(active_topic_ids))]
    abstract = paper.abstract or ""
    pub_date = ""
    if paper.year is not None:
        pub_date = f"{int(paper.year)}-01-01"
    return Card(
        id=paper.openalex_id or "",
        title=paper.title or "",
        authors=[],
        venue=paper.venue or "",
        date=pub_date,
        doi=paper.doi,
        openalex=paper.openalex_id or "",
        profile=slug,
        score=score,
        bucket=_bucket_for(score),
        abstract=abstract,
        mesh=list(paper.mesh or []),
        terms=terms,
        matched=matched,
        topicMatch=_topic_match(paper, set(active_topic_ids)),
        centroidCos=score,
        noveltyDelta=0.0,
        mins=_read_minutes(abstract),
    )


def _paper_terms(paper: Paper) -> list[str]:
    out: list[str] = []
    if paper.primary_topic and paper.primary_topic.display_name:
        out.append(paper.primary_topic.display_name)
    for t in paper.topics or []:
        if t.display_name and t.display_name not in out:
            out.append(t.display_name)
    return out


def _matched_topics(paper: Paper, active: set[str]) -> list[Topic]:
    if not active:
        return []
    out: list[Topic] = []
    if paper.primary_topic and paper.primary_topic.id in active:
        out.append(paper.primary_topic)
    for t in paper.topics or []:
        if t.id in active and t not in out:
            out.append(t)
    return out


def _topic_match(paper: Paper, active: set[str]) -> float:
    if not active:
        return 0.0
    ids: set[str] = set()
    if paper.primary_topic and paper.primary_topic.id:
        ids.add(paper.primary_topic.id)
    for t in paper.topics or []:
        if t.id:
            ids.add(t.id)
    if not ids:
        return 0.0
    return round(len(ids & active) / len(ids), 3)


def _finite(v: float) -> float:
    f = float(v)
    if not np.isfinite(f):
        return 0.0
    return f


def _days_ago(days: int) -> str:
    import datetime
    d = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    return d.strftime("%Y-%m-%d")
