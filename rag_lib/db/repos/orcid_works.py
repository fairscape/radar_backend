"""``orcid_works`` — the fetched works of an ORCID / profile draft.

Written once by the import's fetch phase, read by the seed picker, and
updated when the user confirms a selection. See migration 0017.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable


def replace_all(conn: sqlite3.Connection, profile_id: int, rows: Iterable[dict[str, Any]]) -> int:
    """Replace the profile's work rows. Each row: the column names of the table
    minus ``profile_id``; missing keys default. Returns the row count."""
    conn.execute("DELETE FROM orcid_works WHERE profile_id = ?", (profile_id,))
    n = 0
    for r in rows:
        conn.execute(
            """
            INSERT INTO orcid_works (
              profile_id, openalex_id, position, is_corresponding, author_index,
              total_authors, work_type, cited_by_count, claimed, seed_eligible,
              dup_of, default_selected, selected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id, r["openalex_id"], r.get("position"),
                1 if r.get("is_corresponding") else 0,
                r.get("author_index"), r.get("total_authors"), r.get("work_type"),
                int(r.get("cited_by_count") or 0),
                None if r.get("claimed") is None else (1 if r["claimed"] else 0),
                0 if r.get("seed_eligible") is False else 1,
                r.get("dup_of"),
                1 if r.get("default_selected") else 0,
                None if r.get("selected") is None else (1 if r["selected"] else 0),
            ),
        )
        n += 1
    conn.commit()
    return n


def list_for_profile(conn: sqlite3.Connection, profile_id: int) -> list[sqlite3.Row]:
    """Work rows joined with the paper (title, year, venue, abstract presence,
    authors) and whether the work is currently a seed. Newest first."""
    return conn.execute(
        """
        SELECT w.*, p.title, p.year, p.venue, p.doi, p.authors_json,
               (p.abstract IS NOT NULL AND p.abstract <> '') AS has_abstract,
               EXISTS (
                 SELECT 1 FROM profile_seeds s
                 WHERE s.profile_id = w.profile_id AND s.openalex_id = w.openalex_id
               ) AS is_seed
        FROM orcid_works w
        JOIN papers p ON p.openalex_id = w.openalex_id
        WHERE w.profile_id = ?
        ORDER BY COALESCE(p.year, 0) DESC, w.cited_by_count DESC, p.title
        """,
        (profile_id,),
    ).fetchall()


def count_for_profile(conn: sqlite3.Connection, profile_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM orcid_works WHERE profile_id = ?", (profile_id,)
    ).fetchone()
    return int(row[0]) if row else 0


def set_selected(conn: sqlite3.Connection, profile_id: int, openalex_ids: Iterable[str]) -> None:
    """Record the user's choice: the given ids become ``selected=1``, every other row ``0``."""
    ids = list(dict.fromkeys(openalex_ids))
    conn.execute("UPDATE orcid_works SET selected = 0 WHERE profile_id = ?", (profile_id,))
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        q = ",".join("?" * len(chunk))
        conn.execute(
            f"UPDATE orcid_works SET selected = 1 WHERE profile_id = ? AND openalex_id IN ({q})",
            (profile_id, *chunk),
        )
    conn.commit()


def selected_ids(conn: sqlite3.Connection, profile_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT openalex_id FROM orcid_works WHERE profile_id = ? AND selected = 1",
        (profile_id,),
    ).fetchall()
    return [r["openalex_id"] for r in rows]


def default_ids(conn: sqlite3.Connection, profile_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT openalex_id FROM orcid_works WHERE profile_id = ? AND default_selected = 1",
        (profile_id,),
    ).fetchall()
    return [r["openalex_id"] for r in rows]
