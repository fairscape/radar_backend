"""researchers repo.

A researcher is a stored person: a Prosopia profile or an ORCID, the
metadata that came with them, and the papers the import resolved. The
papers themselves live in ``papers`` (shared, content-addressed); this
module owns the two tables that say *whose* they are.

Identity is ``(user_id, source, key)`` — the same profile imported twice
by one user is one row, refreshed; imported by two users it is two rows,
because ownership is per user and the vault listing follows it.
"""

from __future__ import annotations

import sqlite3
from typing import Any


def get(conn: sqlite3.Connection, researcher_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM researchers WHERE id = ?", (researcher_id,),
    ).fetchone()


def get_for_user(
    conn: sqlite3.Connection, user_id: int, researcher_id: int
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM researchers WHERE id = ? AND user_id = ?",
        (researcher_id, user_id),
    ).fetchone()


def get_by_key(
    conn: sqlite3.Connection, user_id: int, source: str, key: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM researchers WHERE user_id = ? AND source = ? AND key = ?",
        (user_id, source, key),
    ).fetchone()


MARKS = ("self", "followed")

# The user's arrangement first (``position``), then the researchers never
# placed: the user themself, the ones they follow, the rest, by name.
_ORDER = (
    "ORDER BY position IS NULL, position, "
    "CASE mark WHEN 'self' THEN 0 WHEN 'followed' THEN 1 ELSE 2 END, name, id"
)


def list_for_user(conn: sqlite3.Connection, user_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT * FROM researchers WHERE user_id = ? {_ORDER}", (user_id,),
    ).fetchall()


def set_mark(
    conn: sqlite3.Connection, user_id: int, researcher_id: int, mark: str | None
) -> None:
    """Mark a researcher as the user themself (``self``), someone they follow
    (``followed``) or neither, and move them to the top of that group.

    A mark is meant to bring the person forward in the list, so the row
    moves: before the first researcher whose group ranks the same or lower
    (``self`` to the front; ``followed`` ahead of the followed and unmarked;
    unmarked to the head of the unmarked). Every row then gets a position,
    so the arrangement is explicit from here on and drag-and-drop
    (:func:`reorder`) works over the whole list. Raises ``ValueError`` for
    an unknown mark and ``LookupError`` when the researcher is not the user's.
    """
    if mark is not None and mark not in MARKS:
        raise ValueError(f"mark must be one of {', '.join(MARKS)}, or null")
    rows = list_for_user(conn, user_id)
    ids = [int(r["id"]) for r in rows]
    if researcher_id not in ids:
        raise LookupError(f"researcher {researcher_id} not found")
    marks = {int(r["id"]): r["mark"] for r in rows}
    ids.remove(researcher_id)
    marks[researcher_id] = mark
    rank = {"self": 0, "followed": 1, None: 2}
    at = next((i for i, oid in enumerate(ids) if rank[marks[oid]] >= rank[mark]), len(ids))
    ids.insert(at, researcher_id)
    conn.execute("UPDATE researchers SET mark = ? WHERE id = ?", (mark, researcher_id))
    _write_positions(conn, ids)


def reorder(conn: sqlite3.Connection, user_id: int, ids: list[int]) -> None:
    """Store ``ids`` as the order of the user's researchers.

    ``ids`` must name each of the user's researchers exactly once (a
    partial list would leave the rest unplaced and the drag the user just
    made would not stick); ``ValueError`` otherwise.
    """
    have = sorted(int(r["id"]) for r in list_for_user(conn, user_id))
    if sorted(ids) != have or len(set(ids)) != len(ids):
        raise ValueError("ids must list each of your researchers exactly once")
    _write_positions(conn, ids)


def _write_positions(conn: sqlite3.Connection, ids: list[int]) -> None:
    conn.executemany(
        "UPDATE researchers SET position = ? WHERE id = ?",
        [(i, rid) for i, rid in enumerate(ids)],
    )
    conn.commit()


def upsert(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    source: str,
    key: str,
    name: str,
    base_url: str | None = None,
    orcid: str | None = None,
    affiliation: str | None = None,
    url: str | None = None,
    expertise: str | None = None,
    soul: str | None = None,
    grants_json: str | None = None,
    document_json: str | None = None,
) -> int:
    """Insert or refresh a researcher row; returns its id.

    A refresh overwrites the descriptive fields with what the source
    says now, but keeps a value the new read did not supply (an ORCID
    re-import has no affiliation to offer, and must not blank the one a
    Prosopia read filled in).
    """
    existing = get_by_key(conn, user_id, source, key)
    if existing is None:
        cur = conn.execute(
            """
            INSERT INTO researchers (
              user_id, source, key, base_url, orcid, name, affiliation, url,
              expertise, soul, grants_json, document_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, source, key, base_url, orcid, name, affiliation, url,
                expertise, soul, grants_json, document_json,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    conn.execute(
        """
        UPDATE researchers SET
          base_url      = COALESCE(?, base_url),
          orcid         = COALESCE(?, orcid),
          name          = ?,
          affiliation   = COALESCE(?, affiliation),
          url           = COALESCE(?, url),
          expertise     = COALESCE(?, expertise),
          soul          = COALESCE(?, soul),
          grants_json   = COALESCE(?, grants_json),
          document_json = COALESCE(?, document_json),
          updated_at    = datetime('now')
        WHERE id = ?
        """,
        (
            base_url, orcid, name, affiliation, url, expertise, soul,
            grants_json, document_json, int(existing["id"]),
        ),
    )
    conn.commit()
    return int(existing["id"])


def mark_imported(
    conn: sqlite3.Connection, researcher_id: int, *, run_id: int | None
) -> None:
    conn.execute(
        """
        UPDATE researchers SET
          n_papers    = (SELECT COUNT(*) FROM researcher_papers WHERE researcher_id = ?),
          imported_at = datetime('now'),
          last_run_id = COALESCE(?, last_run_id),
          updated_at  = datetime('now')
        WHERE id = ?
        """,
        (researcher_id, run_id, researcher_id),
    )
    conn.commit()


def set_last_run(conn: sqlite3.Connection, researcher_id: int, run_id: int) -> None:
    """The run to watch, at kick-off. ``imported_at`` waits for ``mark_imported``:
    stamped here it read "imported just now" for an import still running."""
    conn.execute(
        "UPDATE researchers SET last_run_id = ?, updated_at = datetime('now') WHERE id = ?",
        (run_id, researcher_id),
    )
    conn.commit()


def delete(conn: sqlite3.Connection, researcher_id: int) -> None:
    """Drop the researcher and its paper links. Papers and interests stay.

    Runs that imported into an interest are that interest's history too
    (its Scans list): unlinked here, since ``gather_runs.researcher_id``
    cascades and deleted them. Runs of the researcher alone go with it.
    """
    conn.execute(
        "UPDATE gather_runs SET researcher_id = NULL WHERE researcher_id = ? AND profile_id IS NOT NULL",
        (researcher_id,),
    )
    conn.execute("DELETE FROM researchers WHERE id = ?", (researcher_id,))
    conn.commit()


def attach_paper(
    conn: sqlite3.Connection,
    researcher_id: int,
    openalex_id: str,
    *,
    paper_id: str | None = None,
    resolved_by: str | None = None,
    summary: str | None = None,
) -> int:
    """Idempotent. A re-import refreshes the rung and summary in place."""
    cur = conn.execute(
        """
        INSERT INTO researcher_papers
          (researcher_id, openalex_id, paper_id, resolved_by, summary)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (researcher_id, openalex_id) DO UPDATE SET
          paper_id    = COALESCE(excluded.paper_id, paper_id),
          resolved_by = COALESCE(excluded.resolved_by, resolved_by),
          summary     = COALESCE(excluded.summary, summary)
        """,
        (researcher_id, openalex_id, paper_id, resolved_by, summary),
    )
    conn.commit()
    return cur.rowcount


def list_papers(
    conn: sqlite3.Connection, researcher_id: int
) -> list[sqlite3.Row]:
    """The researcher's papers with the paper row joined in, newest first."""
    return conn.execute(
        """
        SELECT
          rp.researcher_id, rp.paper_id, rp.resolved_by, rp.summary, rp.added_at,
          p.openalex_id, p.doi, p.title, p.abstract, p.year, p.venue,
          p.publication_date, p.authors_json, p.source, p.pdf_url, p.oa_status,
          p.topics_json
        FROM researcher_papers rp
        JOIN papers p ON p.openalex_id = rp.openalex_id
        WHERE rp.researcher_id = ?
        ORDER BY p.year DESC, p.title
        """,
        (researcher_id,),
    ).fetchall()


def paper_ids(conn: sqlite3.Connection, researcher_id: int) -> set[str]:
    rows = conn.execute(
        "SELECT openalex_id FROM researcher_papers WHERE researcher_id = ?",
        (researcher_id,),
    ).fetchall()
    return {r["openalex_id"] for r in rows}


def user_owns_papers(
    conn: sqlite3.Connection, user_id: int, openalex_ids: list[str]
) -> set[str]:
    """The subset of ``openalex_ids`` the user may use as seeds.

    A paper is the user's if they uploaded it, if it came in with one of
    their researchers, if it is already a seed of one of their
    interests, or if it is a candidate in one of their interests -- a
    paper Radar put in their feed, saved or not. That last one is what
    lets a feed paper become a seed, and what stops a paper saved from
    the feed (listed in the vault) being refused when picked as a seed.
    Anything else is another user's business.
    """
    if not openalex_ids:
        return set()
    marks = ",".join("?" for _ in openalex_ids)
    rows = conn.execute(
        f"""
        SELECT p.openalex_id
        FROM papers p
        WHERE p.openalex_id IN ({marks})
          AND (
            EXISTS (SELECT 1 FROM paper_uploads u WHERE u.openalex_id = p.openalex_id AND u.user_id = ?)
            OR EXISTS (
              SELECT 1 FROM researcher_papers rp
              JOIN researchers r ON r.id = rp.researcher_id
              WHERE rp.openalex_id = p.openalex_id AND r.user_id = ?
            )
            OR EXISTS (
              SELECT 1 FROM profile_seeds ps
              JOIN profiles pr ON pr.id = ps.profile_id
              WHERE ps.openalex_id = p.openalex_id AND pr.user_id = ?
            )
            OR EXISTS (
              SELECT 1 FROM profile_candidates pc
              JOIN profiles pr ON pr.id = pc.profile_id
              WHERE pc.openalex_id = p.openalex_id AND pr.user_id = ?
            )
          )
        """,
        (*openalex_ids, user_id, user_id, user_id, user_id),
    ).fetchall()
    return {r["openalex_id"] for r in rows}


def interests_for(
    conn: sqlite3.Connection, researcher_id: int
) -> list[sqlite3.Row]:
    """Profiles (drafts included) built from this researcher."""
    return conn.execute(
        "SELECT * FROM profiles WHERE researcher_id = ? ORDER BY id",
        (researcher_id,),
    ).fetchall()


def set_profile_researcher(
    conn: sqlite3.Connection, profile_id: int, researcher_id: int | None
) -> None:
    conn.execute(
        "UPDATE profiles SET researcher_id = ?, updated_at = datetime('now') WHERE id = ?",
        (researcher_id, profile_id),
    )
    conn.commit()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}
