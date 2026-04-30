"""feedback_events repo (Phase 8).

Append-only DB mirror of the per-user JSONL feedback log. Inserts are
fire-and-forget (the JSONL line is the canonical record); reads back
the recent rows for the profile detail panel and the autotune sweep.
"""

from __future__ import annotations

import sqlite3


def insert(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    profile_id: int,
    openalex_id: str,
    action: str,
    score: float | None = None,
    selector: str | None = None,
    selector_config_hash: str | None = None,
    benchmark_run_id: int | None = None,
    ts: str | None = None,
) -> int:
    """Insert one feedback event. Returns the new row id.

    ``ts`` is optional; when ``None`` SQLite stamps ``datetime('now')``
    via the column default. Pass an explicit ISO-8601 UTC string when
    the caller has already produced one (so JSONL ts and DB ts match).
    """
    if ts is None:
        cur = conn.execute(
            """
            INSERT INTO feedback_events
              (user_id, profile_id, openalex_id, action, score,
               selector, selector_config_hash, benchmark_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, profile_id, openalex_id, action, score,
                selector, selector_config_hash, benchmark_run_id,
            ),
        )
    else:
        cur = conn.execute(
            """
            INSERT INTO feedback_events
              (user_id, profile_id, openalex_id, action, score,
               selector, selector_config_hash, benchmark_run_id, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, profile_id, openalex_id, action, score,
                selector, selector_config_hash, benchmark_run_id, ts,
            ),
        )
    conn.commit()
    return int(cur.lastrowid or 0)


def recent_for_profile(
    conn: sqlite3.Connection,
    profile_id: int,
    *,
    limit: int = 20,
    action: str | None = None,
) -> list[sqlite3.Row]:
    """Most-recent feedback events for ``profile_id`` joined with the paper.

    Joins ``papers`` so the formatter can pull the DOI without a second
    round-trip. ``action`` filters to a single triage action when given.
    """
    if action is None:
        return conn.execute(
            """
            SELECT fe.id, fe.user_id, fe.profile_id, fe.openalex_id,
                   fe.action, fe.score, fe.selector,
                   fe.selector_config_hash, fe.benchmark_run_id, fe.ts,
                   p.doi, p.title
            FROM feedback_events fe
            JOIN papers p USING (openalex_id)
            WHERE fe.profile_id = ?
            ORDER BY fe.ts DESC, fe.id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        ).fetchall()
    return conn.execute(
        """
        SELECT fe.id, fe.user_id, fe.profile_id, fe.openalex_id,
               fe.action, fe.score, fe.selector,
               fe.selector_config_hash, fe.benchmark_run_id, fe.ts,
               p.doi, p.title
        FROM feedback_events fe
        JOIN papers p USING (openalex_id)
        WHERE fe.profile_id = ? AND fe.action = ?
        ORDER BY fe.ts DESC, fe.id DESC
        LIMIT ?
        """,
        (profile_id, action, limit),
    ).fetchall()


def count_by_action(
    conn: sqlite3.Connection, profile_id: int
) -> dict[str, int]:
    """``{action: count}`` for every action seen on this profile."""
    rows = conn.execute(
        """
        SELECT action, count(*) AS n
        FROM feedback_events
        WHERE profile_id = ?
        GROUP BY action
        """,
        (profile_id,),
    ).fetchall()
    return {r["action"]: int(r["n"]) for r in rows}


def total_for_profile(conn: sqlite3.Connection, profile_id: int) -> int:
    row = conn.execute(
        "SELECT count(*) AS n FROM feedback_events WHERE profile_id = ?",
        (profile_id,),
    ).fetchone()
    return int(row["n"])


def scored_save_dismiss_for_profile(
    conn: sqlite3.Connection, profile_id: int, *, limit: int | None = None
) -> list[sqlite3.Row]:
    """Saved/dismissed events with non-NULL score, newest first.

    Used by the autotune sweep — needs ``(action, score)`` only.
    """
    if limit is None:
        return conn.execute(
            """
            SELECT action, score
            FROM feedback_events
            WHERE profile_id = ?
              AND action IN ('saved', 'dismissed')
              AND score IS NOT NULL
            ORDER BY ts DESC, id DESC
            """,
            (profile_id,),
        ).fetchall()
    return conn.execute(
        """
        SELECT action, score
        FROM feedback_events
        WHERE profile_id = ?
          AND action IN ('saved', 'dismissed')
          AND score IS NOT NULL
        ORDER BY ts DESC, id DESC
        LIMIT ?
        """,
        (profile_id, limit),
    ).fetchall()


__all__ = [
    "insert",
    "recent_for_profile",
    "count_by_action",
    "total_for_profile",
    "scored_save_dismiss_for_profile",
]
