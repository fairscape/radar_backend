"""Profiles service.

Builds the rich shapes consumed by ``GET /api/profiles*``:

  - ``list_profiles`` — base ``Profile`` list for the user, with 30d
    save/dismiss counts mixed in for the health derivation.
  - ``get_profile`` — single ``Profile`` lookup by slug.
  - ``get_profile_detail`` — composite that wraps a ``Profile`` with
    seeds, topics, sweep rows, coherence histogram, and the (Phase 8)
    feedback log.

The wizard / refit / dry-run write paths are intentionally not
implemented here: Phase 5 ships the placeholder responses inline in the
router (matching the mock-api shape) so the frontend swap can land
without waiting on Phase 11's wizard backend.
"""

from __future__ import annotations

import sqlite3
import time

from rag_lib.db.repos import (
    candidates as candidates_repo,
    feedback as feedback_repo,
    profiles as profiles_repo,
)
from rag_lib.feedback.log import format_event_line

from ..mappers import (
    profile_row_to_profile,
    seed_row_to_seed,
    topic_filters_to_topics,
)
from ..schemas import (
    CoherenceStats,
    Profile,
    ProfileDetail,
    SweepRow,
)


def _saves_dismisses(
    conn: sqlite3.Connection, profile_id: int
) -> tuple[int, int]:
    return candidates_repo.count_saves_dismisses_30d(conn, profile_id)


def list_profiles(
    conn: sqlite3.Connection, user_id: int
) -> list[Profile]:
    """Return all profiles for the user, drafts included.

    Drafts ride the same row as committed profiles (``is_draft=1``);
    the frontend tags them with a DRAFT badge so users can resume an
    interrupted wizard or recompute coherence/topics after uploading
    more seeds.
    """
    rows = profiles_repo.list_for_user(conn, user_id, include_drafts=True)
    out: list[Profile] = []
    for row in rows:
        saves30, dismisses30 = _saves_dismisses(conn, int(row["id"]))
        out.append(profile_row_to_profile(
            row, saves30=saves30, dismisses30=dismisses30,
        ))
    return out


def get_profile(
    conn: sqlite3.Connection, user_id: int, slug: str
) -> Profile | None:
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None:
        return None
    saves30, dismisses30 = _saves_dismisses(conn, int(row["id"]))
    return profile_row_to_profile(
        row, saves30=saves30, dismisses30=dismisses30,
    )


# Process-local cache for the threshold sweep. dry_run hits OpenAlex on
# Phase 1B's ``radar.dry_run`` — Phase 5 does not actually run it from
# the API yet (wizard / Phase 11 will), so this cache is reserved for
# when the body lands. Keeping the import + TTL constant centralised
# means swapping the body to the real call is a one-line change.
_SWEEP_TTL_SECONDS = 60.0
_sweep_cache: dict[tuple[int, int], tuple[float, list[SweepRow]]] = {}


def _cached_sweep(profile_id: int, days: int) -> list[SweepRow] | None:
    entry = _sweep_cache.get((profile_id, days))
    if entry is None:
        return None
    ts, rows = entry
    if (time.monotonic() - ts) > _SWEEP_TTL_SECONDS:
        return None
    return rows


def _store_sweep(profile_id: int, days: int, rows: list[SweepRow]) -> None:
    _sweep_cache[(profile_id, days)] = (time.monotonic(), rows)


def get_profile_detail(
    conn: sqlite3.Connection, user_id: int, slug: str
) -> ProfileDetail | None:
    """Composite payload for ``GET /api/profiles/{key}/detail``.

    Phase 5 returns the static shape: profile + seeds + topics + an
    empty sweep + the live coherence histogram. ``feedbackLog`` is
    [] / 0 until Phase 8 wires the JSONL reader in.
    """
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None:
        return None
    profile_id = int(row["id"])
    saves30, dismisses30 = _saves_dismisses(conn, profile_id)
    profile = profile_row_to_profile(
        row, saves30=saves30, dismisses30=dismisses30,
    )

    # Seeds — order by added_at; idx is 1-based for UI list keys.
    seed_oa_ids = profiles_repo.list_seed_openalex_ids(conn, profile_id)
    seed_rows = []
    for idx, oa_id in enumerate(seed_oa_ids, start=1):
        paper_row = conn.execute(
            "SELECT openalex_id, title, year FROM papers WHERE openalex_id = ?",
            (oa_id,),
        ).fetchone()
        if paper_row is None:
            continue
        seed_rows.append(seed_row_to_seed(paper_row, idx=idx))

    # Topics from profile_filters; coherence histogram from seed embeddings.
    topic_filters = profiles_repo.topic_filters(conn, profile_id)
    topics = topic_filters_to_topics(topic_filters)
    bins = profiles_repo.coherence_bins(conn, profile_id)

    # CoherenceStats: take the histogram-implied min/max range. With
    # 16 bins over [0, 1] this is just the bracket of populated bins.
    if bins:
        first = next((i for i, c in enumerate(bins) if c > 0), 0)
        last = max(i for i, c in enumerate(bins) if c > 0) if any(bins) else 0
        coh_stats = CoherenceStats(
            min=round(first / len(bins), 3),
            max=round((last + 1) / len(bins), 3),
        )
    else:
        coh_stats = CoherenceStats(min=0.0, max=0.0)

    sweep = _cached_sweep(profile_id, days=30) or []

    # Phase 8: feedback panel — last 20 events (newest first), with the
    # remainder count for the "... N more" hint. Rows come from the DB
    # mirror so the JSONL doesn't need to be parsed on every detail GET.
    fb_limit = 20
    fb_rows = feedback_repo.recent_for_profile(conn, profile_id, limit=fb_limit)
    fb_total = feedback_repo.total_for_profile(conn, profile_id)
    feedback_log = [format_event_line(r) for r in fb_rows]
    feedback_more = max(0, fb_total - len(feedback_log))

    return ProfileDetail(
        profile=profile,
        seeds=seed_rows,
        topics=topics,
        sweep=sweep,
        coherenceBins=bins,
        coherenceStats=coh_stats,
        feedbackLog=feedback_log,
        feedbackMoreCount=feedback_more,
    )
