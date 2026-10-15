"""gather_runs repo.

One row per invocation of the gatherer (CLI in Phase 2, scheduler in
Phase 7). The row is opened with ``start`` before the work begins,
closed with ``finish`` after — including a non-null ``error`` if the run
failed so the audit trail captures both outcomes.
"""

from __future__ import annotations

import sqlite3

from ...redact import public_error


def start(
    conn: sqlite3.Connection,
    *,
    profile_id: int | None,
    user_id: int,
    since_date: str | None = None,
    filter_string: str | None = None,
    tier_used: str | None = None,
    researcher_id: int | None = None,
) -> int:
    """Open a run row. A researcher import has no profile; a scan has no
    researcher. At least one of the two should be set."""
    cur = conn.execute(
        """
        INSERT INTO gather_runs
          (profile_id, researcher_id, user_id, since_date, filter_string, tier_used)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (profile_id, researcher_id, user_id, since_date, filter_string, tier_used),
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
        (n_fetched, n_new, n_redup, api_calls,
         # Shown to the user as the run's error; see rag_lib/redact.py.
         public_error(error) if error is not None else None,
         tier_used, result_json, run_id),
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


def recent_for_researcher(
    conn: sqlite3.Connection, researcher_id: int, limit: int = 20
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM gather_runs
        WHERE researcher_id = ?
        ORDER BY started_at DESC, id DESC
        LIMIT ?
        """,
        (researcher_id, limit),
    ).fetchall()


#: tier_used values an import writes. Lives here because it is a domain of
#: this module's own column; the three routers that pass them
#: (routers/prosopia.py, routers/researchers.py) each hold one literal.
IMPORT_TIERS = ("orcid_import", "prosopia_import", "researcher_import")

#: How long a run may go without progress before it stops counting as in
#: flight. `finished_at IS NULL` alone is not liveness: the job closes the
#: row on exception, but a killed process cannot -- and this deployment's
#: cron watchdog restarts uvicorn every time it stops answering, mid-import
#: included. Without a cutoff one crashed import would answer every later
#: import of that person with a dead run id, forever, and the drafts list
#: would report `importing` for a job that will never move again.
#:
#: The import job ticks progress per paper (services/prosopia.run_import),
#: so 30 minutes is far longer than any real gap while still bounded.
STALE_IMPORT_MINUTES = 30

_INFLIGHT_PREDICATE = """
        g.finished_at IS NULL
    AND COALESCE(g.progress_updated_at, g.started_at)
          > datetime('now', ?)
"""


def _stale_cutoff() -> str:
    return f"-{STALE_IMPORT_MINUTES} minutes"


def has_unfinished_import(conn: sqlite3.Connection, profile_id: int) -> bool:
    """Is an import still running for this draft?

    The wizard's later steps read ``profile_seeds``, which an import fills
    one paper at a time -- so mid-import a draft holds part of its seeds
    (or none, and looks like an empty one). Asking this is how the
    difference gets told.
    """
    placeholders = ",".join("?" * len(IMPORT_TIERS))
    return conn.execute(
        f"""
        SELECT 1 FROM gather_runs g
         WHERE g.profile_id = ? AND g.tier_used IN ({placeholders})
           AND {_INFLIGHT_PREDICATE}
         LIMIT 1
        """,
        (profile_id, *IMPORT_TIERS, _stale_cutoff()),
    ).fetchone() is not None


def find_inflight_import(
    conn: sqlite3.Connection, user_id: int, *, orcid: str,
) -> sqlite3.Row | None:
    """An import of this person already in flight for this user, or None.

    Keyed on ``researchers.orcid`` rather than on ``(source, key)``,
    because the same person can be in flight under either source: a
    Prosopia profile and a bare ORCID resolve and embed the same papers,
    so letting one through while the other runs is the duplicate work this
    exists to prevent.

    ``LEFT JOIN profiles`` on purpose. A researcher import creates no draft
    (``services/researchers.prepare_researcher_import`` returns a plan with
    no ``profile_id``), so an inner join silently dropped every
    ``researcher_import`` row and the tier in ``IMPORT_TIERS`` was
    unreachable. Ownership then comes from the profile when there is one
    and from the run itself when there is not.

    ``slug`` is NULL for a researcher import; a caller that wants to point
    the user somewhere must handle that.
    """
    placeholders = ",".join("?" * len(IMPORT_TIERS))
    return conn.execute(
        f"""
        SELECT p.slug AS slug, g.id AS run_id, g.tier_used AS tier_used
          FROM gather_runs g
          JOIN researchers r ON r.id = g.researcher_id
          LEFT JOIN profiles p ON p.id = g.profile_id
         WHERE r.orcid = ?
           AND (p.user_id = ? OR (p.id IS NULL AND g.user_id = ?))
           AND g.tier_used IN ({placeholders})
           AND {_INFLIGHT_PREDICATE}
         ORDER BY g.id DESC
         LIMIT 1
        """,
        (orcid, user_id, user_id, *IMPORT_TIERS, _stale_cutoff()),
    ).fetchone()


def find_inflight_researcher_import(
    conn: sqlite3.Connection, user_id: int, *,
    researcher_id: int | None = None, orcid: str | None = None,
) -> sqlite3.Row | None:
    """This user's unfinished import of a researcher (by row or ORCID), or None.

    ``researcher_id, run_id`` of the newest one. Any import tier counts:
    the papers are the same whichever way the person is being imported.
    """
    if researcher_id is None and not orcid:
        return None
    placeholders = ",".join("?" * len(IMPORT_TIERS))
    return conn.execute(
        f"""
        SELECT g.researcher_id AS researcher_id, g.id AS run_id
          FROM gather_runs g
          JOIN researchers r ON r.id = g.researcher_id
          LEFT JOIN profiles p ON p.id = g.profile_id
         WHERE (r.id = ? OR (? IS NOT NULL AND r.orcid = ?))
           AND (p.user_id = ? OR (p.id IS NULL AND g.user_id = ?))
           AND g.tier_used IN ({placeholders})
           AND {_INFLIGHT_PREDICATE}
         ORDER BY g.id DESC
         LIMIT 1
        """,
        (researcher_id, orcid, orcid, user_id, user_id, *IMPORT_TIERS, _stale_cutoff()),
    ).fetchone()


INTERRUPTED = "Interrupted: the backend restarted while this was running."


def close_interrupted(conn: sqlite3.Connection) -> int:
    """Close every run still open, at boot. Returns how many.

    Jobs live in the scheduler's memory, so a restart ends every one of
    them, and nothing else would ever close their rows: a researcher showed
    "Importing" for good (polled every 3 s), a scan or trial scan spun in
    the UI forever, and the wizard stayed blocked behind it. Closed with an
    error, the UI shows the failure and offers to run it again.
    """
    cur = conn.execute(
        "UPDATE gather_runs SET finished_at = datetime('now'), error = ? WHERE finished_at IS NULL",
        (INTERRUPTED,),
    )
    conn.commit()
    return cur.rowcount


# A scan that has not reported progress for this long is taken to be dead
# (the process stopped under it), not running.
SCAN_STALE_MINUTES = 120


def running_scan(conn: sqlite3.Connection, profile_id: int) -> sqlite3.Row | None:
    """The interest's scan that is still running, if any.

    A scan is any run but a wizard trial or a seed import. Two scans of one
    interest at once wrote each other's run id onto the new papers and
    blended against centroids the other was replacing.
    """
    return conn.execute(
        """
        SELECT * FROM gather_runs
        WHERE profile_id = ? AND finished_at IS NULL
          AND (tier_used IS NULL OR (tier_used != 'dry_run' AND tier_used NOT LIKE '%import'))
          AND COALESCE(progress_updated_at, started_at) >= datetime('now', ?)
        ORDER BY id DESC LIMIT 1
        """,
        (profile_id, f"-{SCAN_STALE_MINUTES} minutes"),
    ).fetchone()
