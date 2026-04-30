"""Vault-shaped queries over papers.

A "vault doc" is a paper the user has put in their vault. That covers
two sources:

1. Papers the user uploaded as a PDF (``papers.uploaded_by_user_id =
   user_id``). These have a ``local_path`` and were typically attached
   to a profile as a seed via ``profile_seeds``.
2. Papers the user saved from the daily radar
   (``profile_candidates.saved_at IS NOT NULL`` for a profile owned by
   the user). These have no PDF on disk; the vault still tracks them so
   the saved-paper UX is honest about the user's reading list.

Tags = profile slugs the paper is attached to in either way:
``profile_seeds`` (seed membership) ∪ ``profile_candidates`` rows whose
``saved_at`` is set. ``tags_for_paper`` returns the union, dedup'd.

Dismiss state lives on ``profile_candidates.dismissed_at`` and is the
"trash" record — there's no UI for it but the column is the durable
audit. ``profile_candidates`` rows can stay around (``ON CONFLICT DO
NOTHING`` on re-gather preserves prior triage) so the user never gets
re-shown a dismissed paper.
"""

from __future__ import annotations

import sqlite3


_VAULT_COLS = """
  p.openalex_id, p.doi, p.title, p.abstract, p.year, p.venue,
  p.publication_date, p.source, p.local_path, p.body_text,
  p.first_seen_at, p.n_pages, p.authors_json,
  p.uploaded_by_user_id, p.file_hash
"""


def list_for_user(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    tag: str | None = None,
) -> list[sqlite3.Row]:
    """Return the user's vault docs, optionally filtered to a profile slug.

    The returned rows are distinct on ``openalex_id`` even when a paper
    is both an uploaded seed and a saved candidate (or saved in multiple
    profiles). Sort matches the previous behavior: most-recent
    ``first_seen_at`` first, then openalex_id for stability.
    """
    if tag in (None, "", "all"):
        return conn.execute(
            f"""
            SELECT {_VAULT_COLS}
            FROM papers p
            WHERE p.uploaded_by_user_id = ?
               OR EXISTS (
                 SELECT 1
                 FROM profile_candidates pc
                 JOIN profiles pr ON pr.id = pc.profile_id
                 WHERE pc.openalex_id = p.openalex_id
                   AND pr.user_id = ?
                   AND pc.saved_at IS NOT NULL
               )
            ORDER BY p.first_seen_at DESC, p.openalex_id
            """,
            (user_id, user_id),
        ).fetchall()
    return conn.execute(
        f"""
        SELECT {_VAULT_COLS}
        FROM papers p
        WHERE EXISTS (
              SELECT 1
              FROM profile_seeds ps
              JOIN profiles pr ON pr.id = ps.profile_id
              WHERE ps.openalex_id = p.openalex_id
                AND pr.user_id = ?
                AND pr.slug = ?
                AND p.uploaded_by_user_id = ?
            )
           OR EXISTS (
              SELECT 1
              FROM profile_candidates pc
              JOIN profiles pr ON pr.id = pc.profile_id
              WHERE pc.openalex_id = p.openalex_id
                AND pr.user_id = ?
                AND pr.slug = ?
                AND pc.saved_at IS NOT NULL
            )
        ORDER BY p.first_seen_at DESC, p.openalex_id
        """,
        (user_id, tag, user_id, user_id, tag),
    ).fetchall()


def stats_for_user(conn: sqlite3.Connection, user_id: int) -> dict:
    """Aggregate counts + most recent ingest timestamp.

    ``last_ingest`` prefers the most recent of (paper upload time, save
    time) so the "last activity" indicator updates when a user saves a
    candidate that was first_seen_at much earlier.
    """
    row = conn.execute(
        """
        WITH vault_papers AS (
          SELECT p.openalex_id, p.n_pages, p.first_seen_at, NULL AS event_at
          FROM papers p
          WHERE p.uploaded_by_user_id = ?
          UNION
          SELECT p.openalex_id, p.n_pages, p.first_seen_at, pc.saved_at AS event_at
          FROM papers p
          JOIN profile_candidates pc ON pc.openalex_id = p.openalex_id
          JOIN profiles pr           ON pr.id = pc.profile_id
          WHERE pr.user_id = ? AND pc.saved_at IS NOT NULL
        ),
        deduped AS (
          SELECT openalex_id,
                 MAX(n_pages)                            AS n_pages,
                 MAX(COALESCE(event_at, first_seen_at))  AS last_event
          FROM vault_papers
          GROUP BY openalex_id
        )
        SELECT
          count(*)                  AS docs,
          COALESCE(SUM(n_pages), 0) AS pages,
          MAX(last_event)           AS last_ingest
        FROM deduped
        """,
        (user_id, user_id),
    ).fetchone()
    return {
        "docs": int(row["docs"]),
        "pages": int(row["pages"] or 0),
        "last_ingest": row["last_ingest"],
    }


def tag_counts_for_user(
    conn: sqlite3.Connection, user_id: int
) -> dict[str, int]:
    """``{slug: count}`` of vault docs per profile.

    Counts both seed memberships (uploaded PDFs attached as seeds) and
    saved candidates. A doc that is *both* a seed and saved in the same
    profile counts once for that slug.
    """
    rows = conn.execute(
        """
        WITH vault_in_profile AS (
          SELECT pr.slug AS slug, ps.openalex_id AS openalex_id
          FROM profile_seeds ps
          JOIN profiles pr ON pr.id = ps.profile_id
          JOIN papers   p  ON p.openalex_id = ps.openalex_id
          WHERE pr.user_id = ?
            AND p.uploaded_by_user_id = ?
          UNION
          SELECT pr.slug AS slug, pc.openalex_id AS openalex_id
          FROM profile_candidates pc
          JOIN profiles pr ON pr.id = pc.profile_id
          WHERE pr.user_id = ?
            AND pc.saved_at IS NOT NULL
        )
        SELECT slug, count(*) AS n
        FROM vault_in_profile
        GROUP BY slug
        """,
        (user_id, user_id, user_id),
    ).fetchall()
    return {r["slug"]: int(r["n"]) for r in rows}


def tags_for_paper(
    conn: sqlite3.Connection, user_id: int, openalex_id: str
) -> list[str]:
    """Profile slugs this paper is attached to (seeds ∪ saved candidates)."""
    rows = conn.execute(
        """
        SELECT slug FROM (
          SELECT pr.slug AS slug
          FROM profile_seeds ps
          JOIN profiles pr ON pr.id = ps.profile_id
          WHERE pr.user_id = ? AND ps.openalex_id = ?
          UNION
          SELECT pr.slug AS slug
          FROM profile_candidates pc
          JOIN profiles pr ON pr.id = pc.profile_id
          WHERE pr.user_id = ?
            AND pc.openalex_id = ?
            AND pc.saved_at IS NOT NULL
        )
        ORDER BY slug
        """,
        (user_id, openalex_id, user_id, openalex_id),
    ).fetchall()
    return [r["slug"] for r in rows]
