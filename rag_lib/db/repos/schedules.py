"""profile_schedules repo.

One row per profile, holding the cron string + timezone + enabled flag
that the APScheduler boot loop in ``rag_lib.scheduler.runner`` reads to
register jobs. Inserts are upserts keyed on ``profile_id`` so a PATCH
from the API replaces an existing schedule cleanly.
"""

from __future__ import annotations

import sqlite3


DEFAULT_CRON = "0 4 * * *"
DEFAULT_TZ = "UTC"


def list_enabled(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """All schedules with ``enabled = 1``, joined with profile + user.

    The runner needs ``user_id`` alongside the schedule to dispatch
    ``gather_for_profile``; we expose it here so the runner doesn't need
    a second query per profile.
    """
    return conn.execute(
        """
        SELECT
          ps.profile_id  AS profile_id,
          ps.cron        AS cron,
          ps.tz          AS tz,
          ps.enabled     AS enabled,
          ps.updated_at  AS updated_at,
          p.user_id      AS user_id,
          p.slug         AS slug,
          p.name         AS name
        FROM profile_schedules ps
        JOIN profiles p ON p.id = ps.profile_id
        WHERE ps.enabled = 1
        ORDER BY ps.profile_id
        """
    ).fetchall()


def get(conn: sqlite3.Connection, profile_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM profile_schedules WHERE profile_id = ?",
        (profile_id,),
    ).fetchone()


def upsert(
    conn: sqlite3.Connection,
    *,
    profile_id: int,
    cron: str | None = None,
    tz: str | None = None,
    enabled: bool | None = None,
) -> sqlite3.Row:
    """Create-or-replace a schedule for ``profile_id``.

    Unset fields keep their existing value when the row already exists,
    or fall back to defaults on first insert. Returns the resulting row.
    """
    existing = get(conn, profile_id)
    if existing is None:
        conn.execute(
            """
            INSERT INTO profile_schedules (profile_id, cron, tz, enabled, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            (
                profile_id,
                cron or DEFAULT_CRON,
                tz or DEFAULT_TZ,
                1 if (enabled is None or enabled) else 0,
            ),
        )
    else:
        conn.execute(
            """
            UPDATE profile_schedules SET
              cron       = COALESCE(?, cron),
              tz         = COALESCE(?, tz),
              enabled    = COALESCE(?, enabled),
              updated_at = datetime('now')
            WHERE profile_id = ?
            """,
            (
                cron,
                tz,
                None if enabled is None else (1 if enabled else 0),
                profile_id,
            ),
        )
    conn.commit()
    row = get(conn, profile_id)
    assert row is not None
    return row
