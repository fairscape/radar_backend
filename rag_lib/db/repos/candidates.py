"""profile_candidates repo.

The radar log: every (profile, paper) pair we have ever surfaced, with
triage state. Primary key ``(profile_id, openalex_id)`` enforces dedup
across gather runs — the user's "shown / dismissed / saved" timestamps
are preserved when a paper resurfaces.
"""

from __future__ import annotations

import sqlite3


def insert_dedup(
    conn: sqlite3.Connection,
    *,
    profile_id: int,
    openalex_id: str,
    score: float,
    tier_used: str | None = None,
    gather_run_id: int | None = None,
    score_raw: float | None = None,
    score_max_seed: float | None = None,
    score_pct: float | None = None,
    score_reranker_raw: float | None = None,
    score_reranker_norm: float | None = None,
    score_blended: float | None = None,
    sourced_by_topic_id: str | None = None,
) -> int:
    """Insert one candidate. Returns 1 if new, 0 if already present.

    On conflict (same profile + paper), scores are refreshed but user
    triage state (``shown_at`` / ``dismissed_at`` / ``saved_at``) is
    preserved so prior decisions survive a re-gather.

    The ``score_raw`` / ``score_max_seed`` / ``score_pct`` columns
    (migration 0003) carry the selector's full breakdown when available.
    The ``score_reranker_*`` / ``score_blended`` columns (migration 0013)
    carry the reranker breakdown when a reranker is active.
    They default to ``NULL`` for callers that only have a single score.

    NOTE: the return value cannot come from ``cursor.rowcount`` — SQLite
    reports 1 for the ``DO UPDATE`` branch just as it does for a fresh
    insert, so ``rowcount`` can no longer distinguish new from resurfaced
    (``INSERT OR IGNORE`` used to return 0 on conflict). We probe the PK
    first instead; it is an indexed lookup on ``(profile_id, openalex_id)``
    so the extra read is cheap.
    """
    existed = conn.execute(
        "SELECT 1 FROM profile_candidates "
        "WHERE profile_id = ? AND openalex_id = ?",
        (profile_id, openalex_id),
    ).fetchone() is not None

    conn.execute(
        """
        INSERT INTO profile_candidates
          (profile_id, openalex_id, score, tier_used, gather_run_id,
           score_raw, score_max_seed, score_pct,
           score_reranker_raw, score_reranker_norm, score_blended,
           sourced_by_topic_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (profile_id, openalex_id) DO UPDATE SET
          score              = excluded.score,
          score_raw          = excluded.score_raw,
          score_max_seed     = excluded.score_max_seed,
          score_pct          = excluded.score_pct,
          score_reranker_raw = excluded.score_reranker_raw,
          score_reranker_norm= excluded.score_reranker_norm,
          score_blended      = excluded.score_blended,
          tier_used          = excluded.tier_used,
          gather_run_id      = excluded.gather_run_id,
          -- Keep the first topic that surfaced this paper; a later
          -- gather where a different topic happened to claim it first
          -- shouldn't rewrite the attribution history the UI reports on.
          sourced_by_topic_id = COALESCE(
              profile_candidates.sourced_by_topic_id,
              excluded.sourced_by_topic_id
          )
        """,
        (
            profile_id, openalex_id, score, tier_used, gather_run_id,
            score_raw, score_max_seed, score_pct,
            score_reranker_raw, score_reranker_norm, score_blended,
            sourced_by_topic_id,
        ),
    )
    conn.commit()
    return 0 if existed else 1


def mark_shown(
    conn: sqlite3.Connection, profile_id: int, openalex_ids: list[str]
) -> int:
    if not openalex_ids:
        return 0
    placeholders = ",".join("?" for _ in openalex_ids)
    cur = conn.execute(
        f"""
        UPDATE profile_candidates
        SET shown_at = datetime('now')
        WHERE profile_id = ? AND openalex_id IN ({placeholders})
              AND shown_at IS NULL
        """,
        (profile_id, *openalex_ids),
    )
    conn.commit()
    return cur.rowcount


def mark_saved(
    conn: sqlite3.Connection, profile_id: int, openalex_id: str
) -> str | None:
    """Toggle save state. Setting ``saved_at`` clears any ``dismissed_at``.

    Returns the resulting state: ``"saved"`` or ``None``.
    """
    row = conn.execute(
        "SELECT saved_at FROM profile_candidates WHERE profile_id = ? AND openalex_id = ?",
        (profile_id, openalex_id),
    ).fetchone()
    if row is None:
        return None
    if row["saved_at"] is None:
        conn.execute(
            """
            UPDATE profile_candidates
            SET saved_at = datetime('now'), dismissed_at = NULL
            WHERE profile_id = ? AND openalex_id = ?
            """,
            (profile_id, openalex_id),
        )
        conn.commit()
        return "saved"
    conn.execute(
        """
        UPDATE profile_candidates
        SET saved_at = NULL
        WHERE profile_id = ? AND openalex_id = ?
        """,
        (profile_id, openalex_id),
    )
    conn.commit()
    return None


def mark_dismissed(
    conn: sqlite3.Connection, profile_id: int, openalex_id: str
) -> str | None:
    """Toggle dismiss state. Setting ``dismissed_at`` clears any ``saved_at``.

    Returns ``"dismissed"`` or ``None``.
    """
    row = conn.execute(
        "SELECT dismissed_at FROM profile_candidates WHERE profile_id = ? AND openalex_id = ?",
        (profile_id, openalex_id),
    ).fetchone()
    if row is None:
        return None
    if row["dismissed_at"] is None:
        conn.execute(
            """
            UPDATE profile_candidates
            SET dismissed_at = datetime('now'), saved_at = NULL
            WHERE profile_id = ? AND openalex_id = ?
            """,
            (profile_id, openalex_id),
        )
        conn.commit()
        return "dismissed"
    conn.execute(
        """
        UPDATE profile_candidates
        SET dismissed_at = NULL
        WHERE profile_id = ? AND openalex_id = ?
        """,
        (profile_id, openalex_id),
    )
    conn.commit()
    return None


def unshown_for_profile(
    conn: sqlite3.Connection, profile_id: int, limit: int = 50
) -> list[sqlite3.Row]:
    """Top-N un-triaged candidates for a profile.

    Uses ``score_blended`` (reranker output) when available, falling
    back to ``score`` (selector-only) for candidates that were gathered
    before the reranker was enabled.
    """
    return conn.execute(
        """
        SELECT pc.*, p.title, p.doi, p.abstract, p.year, p.venue
        FROM profile_candidates pc
        JOIN papers p USING (openalex_id)
        WHERE pc.profile_id = ?
          AND pc.shown_at IS NULL
          AND pc.dismissed_at IS NULL
          AND (pc.snoozed_until IS NULL OR pc.snoozed_until < datetime('now'))
        ORDER BY COALESCE(pc.score_blended, pc.score) DESC
        LIMIT ?
        """,
        (profile_id, limit),
    ).fetchall()


def count_for_profile(conn: sqlite3.Connection, profile_id: int) -> int:
    row = conn.execute(
        "SELECT count(*) AS n FROM profile_candidates WHERE profile_id = ?",
        (profile_id,),
    ).fetchone()
    return int(row["n"])


def scores_for_profile(
    conn: sqlite3.Connection, profile_id: int
) -> list[float]:
    """Raw selector scores for every persisted candidate, score desc.

    Drives the dry-run histogram on the profile detail page: the UI
    bins these and lets the user slide θ to see "X of N would pass".
    """
    rows = conn.execute(
        """
        SELECT score FROM profile_candidates
        WHERE profile_id = ?
        ORDER BY score DESC
        """,
        (profile_id,),
    ).fetchall()
    return [float(r["score"]) for r in rows]


def mark_shown_bulk(
    conn: sqlite3.Connection,
    profile_id: int,
    openalex_ids: list[str],
    commit: bool = True,
) -> int:
    """Stamp ``shown_at = now()`` for every ``openalex_id`` in the batch.

    Like ``mark_shown`` but doesn't gate on ``shown_at IS NULL`` — used
    by ``GET /api/radar/daily`` so re-listing a card refreshes its
    last-shown timestamp instead of silently no-oping. Returns the
    number of rows touched.

    Pass ``commit=False`` to fold several profiles' stamps into one
    transaction; the caller then commits once. The daily radar does that
    — sixteen separate commits meant sixteen write locks per request.
    """
    if not openalex_ids:
        return 0
    placeholders = ",".join("?" for _ in openalex_ids)
    cur = conn.execute(
        f"""
        UPDATE profile_candidates
        SET shown_at = datetime('now')
        WHERE profile_id = ? AND openalex_id IN ({placeholders})
        """,
        (profile_id, *openalex_ids),
    )
    if commit:
        conn.commit()
    return cur.rowcount


def count_saves_dismisses_30d(
    conn: sqlite3.Connection, profile_id: int
) -> tuple[int, int]:
    """``(saves, dismisses)`` recorded in the trailing 30 days for a profile."""
    row = conn.execute(
        """
        SELECT
          SUM(CASE WHEN saved_at     >= datetime('now', '-30 days') THEN 1 ELSE 0 END) AS saves,
          SUM(CASE WHEN dismissed_at >= datetime('now', '-30 days') THEN 1 ELSE 0 END) AS dismisses
        FROM profile_candidates
        WHERE profile_id = ?
        """,
        (profile_id,),
    ).fetchone()
    return int(row["saves"] or 0), int(row["dismisses"] or 0)


def top_for_profile(
    conn: sqlite3.Connection,
    profile_id: int,
    *,
    limit: int = 50,
) -> list[sqlite3.Row]:
    """Top-N candidates for the daily radar (no shown/dismissed filter).

    Joins papers so the API mapper has title / doi / abstract / venue /
    year in one row, and carries the score breakdown so callers can show
    *why* a candidate ranks where it does.

    Ordering mirrors ``unshown_for_profile``: ``score_blended`` when a
    reranker produced one, else ``score``. Plain ``ORDER BY pc.score``
    happens to agree today only because ``insert_dedup`` overwrites
    ``score`` with the blended value — an implementation detail, not a
    contract. Spelling the fallback out keeps this path correct if that
    ever changes, and keeps the two read paths from silently diverging.
    """
    return conn.execute(
        """
        SELECT
          pc.profile_id, pc.openalex_id, pc.score, pc.tier_used,
          pc.fetched_at, pc.shown_at, pc.dismissed_at, pc.saved_at,
          pc.score_raw, pc.score_pct, pc.score_blended,
          pc.score_reranker_raw, pc.score_reranker_norm,
          pc.sourced_by_topic_id,
          p.title, p.doi, p.abstract, p.year, p.venue,
          p.publication_date, p.topics_json
        FROM profile_candidates pc
        JOIN papers p USING (openalex_id)
        WHERE pc.profile_id = ?
        ORDER BY COALESCE(pc.score_blended, pc.score) DESC
        LIMIT ?
        """,
        (profile_id, limit),
    ).fetchall()


def topic_yield_for_profile(
    conn: sqlite3.Connection,
    profile_id: int,
    *,
    since: str | None = None,
) -> list[sqlite3.Row]:
    """Per-topic tally of what each topic's gather quota brought in.

    This is the evidence behind the Step 3 toggles: without it the user
    prunes topics by name alone. ``n_saved`` / ``n_dismissed`` are the
    signal that matters — a topic that keeps yielding papers the user
    throws away is spending quota a better topic could use.

    ``since`` filters ``fetched_at`` (ISO-8601) to scope the tally to a
    recent window. Rows predating migration 0014 carry no attribution
    and are excluded.
    """
    where = ["profile_id = ?", "sourced_by_topic_id IS NOT NULL"]
    params: list = [profile_id]
    if since:
        where.append("fetched_at >= ?")
        params.append(since)

    return conn.execute(
        f"""
        SELECT
          sourced_by_topic_id           AS topic_id,
          COUNT(*)                      AS n_candidates,
          SUM(saved_at IS NOT NULL)     AS n_saved,
          SUM(dismissed_at IS NOT NULL) AS n_dismissed,
          SUM(shown_at IS NOT NULL)     AS n_shown,
          MAX(fetched_at)               AS last_fetched_at
        FROM profile_candidates
        WHERE {' AND '.join(where)}
        GROUP BY sourced_by_topic_id
        ORDER BY n_candidates DESC
        """,
        params,
    ).fetchall()


def get_one(
    conn: sqlite3.Connection,
    profile_id: int,
    openalex_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT pc.*, p.title, p.doi, p.abstract, p.year, p.venue,
               p.publication_date, p.topics_json
        FROM profile_candidates pc
        JOIN papers p USING (openalex_id)
        WHERE pc.profile_id = ? AND pc.openalex_id = ?
        """,
        (profile_id, openalex_id),
    ).fetchone()
