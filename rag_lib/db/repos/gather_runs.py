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
    result_json: str | None = None,
) -> None:
    """Close out a gather_runs row.

    ``result_json`` is the serialized payload for jobs whose result
    can't be reconstructed from the regular candidates/papers tables —
    most notably the wizard's dry-run, which returns a sweep + preview
    + raw scores blob. ``COALESCE`` keeps any prior value if the caller
    doesn't supply one (so finishing with an error doesn't blank a
    partial result the job had time to write).
    """
    conn.execute(
        """
        UPDATE gather_runs SET
          finished_at  = datetime('now'),
          n_fetched    = ?,
          n_new        = ?,
          n_redup      = ?,
          api_calls    = COALESCE(?, api_calls),
          error        = ?,
          tier_used    = COALESCE(?, tier_used),
          result_json  = COALESCE(?, result_json),
          current_step = 'done'
        WHERE id = ?
        """,
        (n_fetched, n_new, n_redup, api_calls, error, tier_used, result_json, run_id),
    )
    conn.commit()


def set_step(
    conn: sqlite3.Connection,
    run_id: int,
    step: str,
    *,
    n_total: int | None = None,
    message: str | None = None,
) -> None:
    """Enter a new phase: reset the in-step counter, set the new total
    and message. Used at phase boundaries (loading_profile / fetching /
    embedding / persisting). Always zeroes ``n_processed`` so the UI
    doesn't carry a stale denominator into a new step."""
    conn.execute(
        """
        UPDATE gather_runs SET
          current_step        = ?,
          n_processed         = 0,
          n_total             = ?,
          last_message        = ?,
          progress_updated_at = datetime('now')
        WHERE id = ?
        """,
        (step, n_total, message, run_id),
    )
    conn.commit()


def tick(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    n_processed: int,
    message: str | None = None,
) -> None:
    """Bump the within-step counter (e.g., per-embedding tick). Leaves
    ``current_step`` and ``n_total`` alone; updates ``last_message``
    only when one is supplied."""
    conn.execute(
        """
        UPDATE gather_runs SET
          n_processed         = ?,
          last_message        = COALESCE(?, last_message),
          progress_updated_at = datetime('now')
        WHERE id = ?
        """,
        (n_processed, message, run_id),
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
