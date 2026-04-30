"""gather_runs repo.

One row per invocation of the gatherer (CLI in Phase 2, scheduler in
Phase 7). The row is opened with ``start`` before the work begins,
closed with ``finish`` after — including a non-null ``error`` if the run
failed so the audit trail captures both outcomes.
"""

from __future__ import annotations

import sqlite3


def start(
    conn: sqlite3.Connection,
    *,
    profile_id: int,
    user_id: int,
    since_date: str | None = None,
    filter_string: str | None = None,
    tier_used: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO gather_runs
          (profile_id, user_id, since_date, filter_string, tier_used)
        VALUES (?, ?, ?, ?, ?)
        """,
        (profile_id, user_id, since_date, filter_string, tier_used),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    n_fetched: int,
    n_new: int,
    n_redup: int,
    api_calls: int | None = None,
    error: str | None = None,
    tier_used: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE gather_runs SET
          finished_at = datetime('now'),
          n_fetched   = ?,
          n_new       = ?,
          n_redup     = ?,
          api_calls   = COALESCE(?, api_calls),
          error       = ?,
          tier_used   = COALESCE(?, tier_used)
        WHERE id = ?
        """,
        (n_fetched, n_new, n_redup, api_calls, error, tier_used, run_id),
    )
    conn.commit()


def get(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM gather_runs WHERE id = ?", (run_id,)
    ).fetchone()


def recent_for_profile(
    conn: sqlite3.Connection, profile_id: int, limit: int = 20
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM gather_runs
        WHERE profile_id = ?
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (profile_id, limit),
    ).fetchall()
