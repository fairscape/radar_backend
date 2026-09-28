"""``researchers`` — the user's library of imported Researcher Profiles (migration 0018)."""

from __future__ import annotations

import sqlite3
from typing import Any

SUMMARY_COLS = (
    "id, user_id, rid, orcid, name, affiliation, field, level, provenance, date_modified, "
    "source_kind, source_url, imported_at, updated_at"
)


def upsert(conn: sqlite3.Connection, user_id: int, row: dict[str, Any]) -> tuple[int, bool]:
    """Insert or overwrite the researcher identified by ``(user_id, rid)``.

    Returns ``(id, created)``. On overwrite every content column is replaced
    and ``updated_at`` is stamped; ``imported_at`` keeps the first import.
    """
    existing = conn.execute(
        "SELECT id FROM researchers WHERE user_id = ? AND rid = ?", (user_id, row["rid"])
    ).fetchone()
    if existing is None:
        cur = conn.execute(
            """
            INSERT INTO researchers (
              user_id, rid, orcid, name, affiliation, field, level, provenance,
              date_modified, source_kind, source_url, doc_json, parsed_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, row["rid"], row.get("orcid"), row["name"], row.get("affiliation"),
                row.get("field"), row.get("level"), row.get("provenance"),
                row.get("date_modified"), row.get("source_kind") or "paste", row.get("source_url"),
                row["doc_json"], row["parsed_json"],
            ),
        )
        conn.commit()
        return int(cur.lastrowid), True
    conn.execute(
        """
        UPDATE researchers
           SET orcid = ?, name = ?, affiliation = ?, field = ?, level = ?, provenance = ?,
               date_modified = ?, source_kind = ?, source_url = ?, doc_json = ?, parsed_json = ?,
               updated_at = datetime('now')
         WHERE id = ?
        """,
        (
            row.get("orcid"), row["name"], row.get("affiliation"), row.get("field"),
            row.get("level"), row.get("provenance"), row.get("date_modified"),
            row.get("source_kind") or "paste", row.get("source_url"),
            row["doc_json"], row["parsed_json"], int(existing["id"]),
        ),
    )
    conn.commit()
    return int(existing["id"]), False


def list_for_user(conn: sqlite3.Connection, user_id: int) -> list[sqlite3.Row]:
    """Summary rows (no documents), most recently imported first."""
    return conn.execute(
        f"""
        SELECT {SUMMARY_COLS},
               json_extract(parsed_json, '$.n_expertise')     AS n_expertise,
               json_extract(parsed_json, '$.n_not_interests') AS n_not_interests,
               json_extract(parsed_json, '$.n_papers')        AS n_papers
        FROM researchers
        WHERE user_id = ?
        ORDER BY COALESCE(updated_at, imported_at) DESC, id DESC
        """,
        (user_id,),
    ).fetchall()


def get(conn: sqlite3.Connection, user_id: int, researcher_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM researchers WHERE user_id = ? AND id = ?", (user_id, researcher_id)
    ).fetchone()


def delete(conn: sqlite3.Connection, user_id: int, researcher_id: int) -> bool:
    cur = conn.execute(
        "DELETE FROM researchers WHERE user_id = ? AND id = ?", (user_id, researcher_id)
    )
    conn.commit()
    return cur.rowcount > 0
