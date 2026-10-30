"""Vault-shaped queries over papers.

A "vault doc" is a paper the user has put in their vault. That covers
two sources:

1. Papers the user uploaded as a PDF (a ``paper_uploads`` row for the
   user; migration 0020). These have a ``local_path`` and were typically attached
   to a profile as a seed via ``profile_seeds``.
2. Papers the user saved from the daily radar
   (``profile_candidates.saved_at IS NOT NULL`` for a profile owned by
   the user). These have no PDF on disk; the vault still tracks them so
   the saved-paper UX is honest about the user's reading list.

3. Every seed of the user's interests, however it got there: a paper
   saved in one interest and made a seed of another, or one imported
   with a researcher the user has since forgotten. Seeds used to count
   only when uploaded or linked to a saved researcher, so such a seed
   vanished from the wizard's seed list (and from the vault) while
   still seeding the interest.

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
  p.uploaded_by_user_id, p.file_hash, p.abstract_source
"""
# When this user added the paper: their own upload, else when Radar first
# saw it. papers.first_seen_at alone is the first uploader's time, which
# told a second user who uploaded the same file that someone had, and when.
_ADDED = "COALESCE((SELECT u.uploaded_at FROM paper_uploads u WHERE u.openalex_id = p.openalex_id AND u.user_id = ?), p.first_seen_at)"


def list_for_user(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    tag: str | None = None,
) -> list[sqlite3.Row]:
    """Return the user's vault docs, optionally filtered to a profile slug.

    The returned rows are distinct on ``openalex_id`` even when a paper
    is both an uploaded seed and a saved candidate (or saved in multiple
    profiles). Most recently added by this user first (``added_at``),
    then openalex_id for stability.
    """
    if tag in (None, "", "all"):
        return conn.execute(
            f"""
            SELECT {_VAULT_COLS}, {_ADDED} AS added_at
            FROM papers p
            WHERE EXISTS (SELECT 1 FROM paper_uploads u WHERE u.openalex_id = p.openalex_id AND u.user_id = ?)
               OR EXISTS (
                 SELECT 1
                 FROM researcher_papers rp
                 JOIN researchers r ON r.id = rp.researcher_id
                 WHERE rp.openalex_id = p.openalex_id
                   AND r.user_id = ?
               )
               OR EXISTS (
                 SELECT 1
                 FROM profile_candidates pc
                 JOIN profiles pr ON pr.id = pc.profile_id
                 WHERE pc.openalex_id = p.openalex_id
                   AND pr.user_id = ?
                   AND pc.saved_at IS NOT NULL
               )
               OR EXISTS (
                 SELECT 1
                 FROM profile_seeds ps
                 JOIN profiles pr ON pr.id = ps.profile_id
                 WHERE ps.openalex_id = p.openalex_id
                   AND pr.user_id = ?
               )
            ORDER BY added_at DESC, p.openalex_id
            """,
            (user_id, user_id, user_id, user_id, user_id),
        ).fetchall()
    return conn.execute(
        f"""
        SELECT {_VAULT_COLS}, {_ADDED} AS added_at
        FROM papers p
        WHERE EXISTS (
              SELECT 1
              FROM profile_seeds ps
              JOIN profiles pr ON pr.id = ps.profile_id
              WHERE ps.openalex_id = p.openalex_id
                AND pr.user_id = ?
                AND pr.slug = ?
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
           -- The user's upload of a work tagged by its public record: the
           -- work is the upload here too (list_docs drops the record).
           OR EXISTS (
              SELECT 1
              FROM paper_versions v
              JOIN profiles pr ON pr.user_id = v.user_id AND pr.slug = ?
              WHERE v.version_id = p.openalex_id AND v.user_id = ?
                AND (EXISTS (SELECT 1 FROM profile_seeds ps
                             WHERE ps.profile_id = pr.id AND ps.openalex_id = v.canonical_openalex_id)
                  OR EXISTS (SELECT 1 FROM profile_candidates pc
                             WHERE pc.profile_id = pr.id AND pc.openalex_id = v.canonical_openalex_id
                               AND pc.saved_at IS NOT NULL))
            )
        ORDER BY added_at DESC, p.openalex_id
        """,
        (user_id, user_id, tag, user_id, tag, tag, user_id),
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
          -- The user's own upload time (first_seen_at is the first uploader's).
          SELECT p.openalex_id, p.n_pages, u.uploaded_at AS first_seen_at, NULL AS event_at
          FROM papers p
          JOIN paper_uploads u ON u.openalex_id = p.openalex_id AND u.user_id = ?
          UNION
          -- A researcher's imported papers, as list_for_user counts them.
          -- Imports used to stamp uploaded_by_user_id, which is how they
          -- got counted; they no longer do (it moved other users' uploads),
          -- and without this the list showed 82 papers under "0 docs".
          SELECT p.openalex_id, p.n_pages, p.first_seen_at, NULL AS event_at
          FROM papers p
          JOIN researcher_papers rp ON rp.openalex_id = p.openalex_id
          JOIN researchers r        ON r.id = rp.researcher_id
          WHERE r.user_id = ?
          UNION
          SELECT p.openalex_id, p.n_pages, p.first_seen_at, pc.saved_at AS event_at
          FROM papers p
          JOIN profile_candidates pc ON pc.openalex_id = p.openalex_id
          JOIN profiles pr           ON pr.id = pc.profile_id
          WHERE pr.user_id = ? AND pc.saved_at IS NOT NULL
          UNION
          -- Every seed of the user's interests (module docstring, 3.).
          SELECT p.openalex_id, p.n_pages, p.first_seen_at, NULL AS event_at
          FROM papers p
          JOIN profile_seeds ps ON ps.openalex_id = p.openalex_id
          JOIN profiles pr      ON pr.id = ps.profile_id
          WHERE pr.user_id = ?
        ),
        deduped AS (
          SELECT openalex_id,
                 MAX(n_pages)                            AS n_pages,
                 MAX(COALESCE(event_at, first_seen_at))  AS last_event
          FROM vault_papers vp
          -- A public work the user has an upload of is that upload, as
          -- the list shows it once.
          WHERE NOT EXISTS (
            SELECT 1 FROM vault_papers mine
            JOIN paper_versions v ON v.version_id = mine.openalex_id
            WHERE v.canonical_openalex_id = vp.openalex_id
          )
          GROUP BY openalex_id
        )
        SELECT
          count(*)                  AS docs,
          COALESCE(SUM(n_pages), 0) AS pages,
          MAX(last_event)           AS last_ingest
        FROM deduped
        """,
        (user_id, user_id, user_id, user_id),
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
        WITH direct AS (
          SELECT pr.slug AS slug, ps.openalex_id AS openalex_id
          FROM profile_seeds ps
          JOIN profiles pr ON pr.id = ps.profile_id
          JOIN papers   p  ON p.openalex_id = ps.openalex_id
          WHERE pr.user_id = ? AND pr.edit_of IS NULL   -- not a copy being edited
          UNION
          SELECT pr.slug AS slug, pc.openalex_id AS openalex_id
          FROM profile_candidates pc
          JOIN profiles pr ON pr.id = pc.profile_id
          WHERE pr.user_id = ?
            AND pc.saved_at IS NOT NULL
        ),
        -- With the user's uploads of the works there, as tags_for_paper.
        tagged AS (
          SELECT slug, openalex_id FROM direct
          UNION
          SELECT d.slug, v.version_id
          FROM direct d
          JOIN paper_versions v ON v.canonical_openalex_id = d.openalex_id AND v.user_id = ?
        )
        SELECT slug, count(*) AS n
        FROM tagged vp
        -- Counted as the slug's list shows it: a public work listed there
        -- with the user's upload of it is the upload, once.
        WHERE NOT EXISTS (
          SELECT 1 FROM tagged mine
          JOIN paper_versions v ON v.version_id = mine.openalex_id
          WHERE mine.slug = vp.slug AND v.canonical_openalex_id = vp.openalex_id
        )
        GROUP BY slug
        """,
        (user_id, user_id, user_id),
    ).fetchall()
    return {r["slug"]: int(r["n"]) for r in rows}


def tags_for_paper(
    conn: sqlite3.Connection, user_id: int, openalex_id: str
) -> list[str]:
    """Profile slugs this paper is attached to (seeds ∪ saved candidates).

    For the user's upload of a work, also its public record's: the vault
    lists the work once, as the upload."""
    from .versions import canonical_id

    canonical = canonical_id(conn, openalex_id)
    if canonical is not None:
        return sorted(set(_tags(conn, user_id, openalex_id)) | set(_tags(conn, user_id, canonical)))
    return _tags(conn, user_id, openalex_id)


def _tags(conn: sqlite3.Connection, user_id: int, openalex_id: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT slug FROM (
          SELECT pr.slug AS slug
          FROM profile_seeds ps
          JOIN profiles pr ON pr.id = ps.profile_id
          WHERE pr.user_id = ? AND ps.openalex_id = ? AND pr.edit_of IS NULL   -- not a copy being edited
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


def added_at(conn: sqlite3.Connection, user_id: int, openalex_id: str, first_seen_at: str | None) -> str | None:
    """``_ADDED`` for one paper."""
    row = conn.execute(
        "SELECT uploaded_at FROM paper_uploads WHERE openalex_id = ? AND user_id = ?",
        (openalex_id, user_id),
    ).fetchone()
    return row["uploaded_at"] if row is not None else first_seen_at
