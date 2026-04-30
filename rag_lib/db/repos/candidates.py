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
) -> int:
    """Insert one candidate. Returns 1 if new, 0 if already present.

    Existing rows are left untouched so prior ``shown_at`` / ``dismissed_at``
    / ``saved_at`` survive the resurface.

    The ``score_raw`` / ``score_max_seed`` / ``score_pct`` columns
    (migration 0003) carry the selector's full breakdown when available.
    They default to ``NULL`` for callers that only have a single score.
    """
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO profile_candidates
          (profile_id, openalex_id, score, tier_used, gather_run_id,
           score_raw, score_max_seed, score_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            profile_id, openalex_id, score, tier_used, gather_run_id,
            score_raw, score_max_seed, score_pct,
        ),
    )
    conn.commit()
    return cur.rowcount


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
    """Top-N un-triaged candidates for a profile, score desc."""
    return conn.execute(
        """
        SELECT pc.*, p.title, p.doi, p.abstract, p.year, p.venue
        FROM profile_candidates pc
        JOIN papers p USING (openalex_id)
        WHERE pc.profile_id = ?
          AND pc.shown_at IS NULL
          AND pc.dismissed_at IS NULL
          AND (pc.snoozed_until IS NULL OR pc.snoozed_until < datetime('now'))
        ORDER BY pc.score DESC
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
    conn: sqlite3.Connection, profile_id: int, openalex_ids: list[str]
) -> int:
    """Stamp ``shown_at = now()`` for every ``openalex_id`` in the batch.

    Like ``mark_shown`` but doesn't gate on ``shown_at IS NULL`` — used
    by ``GET /api/radar/daily`` so re-listing a card refreshes its
    last-shown timestamp instead of silently no-oping. Returns the
    number of rows touched.
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
    """Top-N candidates by score (no shown/dismissed filter).

    Joins papers so the API mapper has title / doi / abstract / venue /
    year in one row.
    """
    return conn.execute(
        """
        SELECT
          pc.profile_id, pc.openalex_id, pc.score, pc.tier_used,
          pc.fetched_at, pc.shown_at, pc.dismissed_at, pc.saved_at,
          p.title, p.doi, p.abstract, p.year, p.venue,
          p.publication_date, p.topics_json
        FROM profile_candidates pc
        JOIN papers p USING (openalex_id)
        WHERE pc.profile_id = ?
        ORDER BY pc.score DESC
        LIMIT ?
        """,
        (profile_id, limit),
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
