"""Whether a ``papers`` row may be deleted, and deleting it only then
(FUNCTIONAL_SPEC CR.1).

A paper row goes only when nothing names it any more, for any user. What
names it is read from the schema, not listed by hand: every foreign key to
``papers(openalex_id)`` that does not cascade is a use (a seed, a feed
row, a researcher's paper, a version's public work, a paper kept in
someone's Vault). One that cascades is part of the paper (its vectors,
its upload row, its version record) and goes with it. A hand-kept list
missed ``vault_kept`` and took papers out of other users' Vaults; a table
added later is a use from the start.

The database is the last word: ``vault_kept`` does not cascade (migration
0040), so a delete that slipped past the check fails on the foreign key,
and the row stays. ``delete_if_unused`` is the only ``DELETE FROM papers``
outside the migrations (``tests/test_paper_rows.py`` holds it to that).
"""

from __future__ import annotations

import sqlite3

import structlog

log = structlog.get_logger("rag_lib.db.repos.paper_rows")

# Not foreign keys: another draft that brought the same paper in.
_OTHER_USES = ("SELECT 1 FROM draft_owned WHERE kind = 'paper' AND item = :id",)


def _uses(conn: sqlite3.Connection) -> list[str]:
    out = []
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
        for fk in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            # (id, seq, table, from, to, on_update, on_delete, match)
            if fk[2] == "papers" and fk[4] in (None, "openalex_id") and str(fk[6]).upper() != "CASCADE":
                out.append(f'SELECT 1 FROM "{table}" WHERE "{fk[3]}" = :id')
    return out + list(_OTHER_USES)


def in_use(conn: sqlite3.Connection, openalex_id: str) -> bool:
    """Something in the database, any user's, still names paper ``openalex_id``."""
    return any(conn.execute(sql, {"id": openalex_id}).fetchone() for sql in _uses(conn))


def delete_if_unused(conn: sqlite3.Connection, openalex_id: str) -> bool:
    """Delete paper ``openalex_id`` (and what cascades with it) if nothing
    names it. Returns whether it was deleted. Not committed; inside the
    caller's transaction."""
    if in_use(conn, openalex_id):
        return False
    conn.execute("SAVEPOINT paper_rows_delete")
    try:
        conn.execute("DELETE FROM papers WHERE openalex_id = ?", (openalex_id,))
    except sqlite3.IntegrityError as exc:
        # Named by something the check does not know: it stays.
        conn.execute("ROLLBACK TO paper_rows_delete")
        conn.execute("RELEASE paper_rows_delete")
        log.warning("paper_rows.kept_by_foreign_key", openalex_id=openalex_id, reason=str(exc)[:160])
        return False
    conn.execute("RELEASE paper_rows_delete")
    return True
