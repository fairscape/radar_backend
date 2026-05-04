"""chat_turns repo.

One row per turn (user prompt or assistant reply). The chat service
writes the user row first and then the assistant row after Ollama
returns; both share the same ``scope_json`` so a turn pair can be
reassembled. ``sources_json`` is set on assistant rows when the prompt
was answered from retrieved chunks.
"""

from __future__ import annotations

import json
import sqlite3


def append_turn(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    role: str,
    body: str,
    sources: list[dict] | None = None,
    scope: list[str] | None = None,
) -> sqlite3.Row:
    """Insert a turn and return the resulting row.

    ``sources`` / ``scope`` are JSON-serialized at the boundary so the
    callers stay in plain-Python dicts.
    """
    sources_json = json.dumps(sources) if sources is not None else None
    scope_json = json.dumps(scope) if scope is not None else None
    cursor = conn.execute(
        """
        INSERT INTO chat_turns (user_id, role, body, sources_json, scope_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, role, body, sources_json, scope_json),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM chat_turns WHERE id = ?", (cursor.lastrowid,)
    ).fetchone()
    assert row is not None
    return row


def recent(
    conn: sqlite3.Connection, user_id: int, limit: int = 50
) -> list[sqlite3.Row]:
    """Most-recent ``limit`` turns for a user, oldest-first.

    The history endpoint shows turns chronologically, but the index is
    descending so we slice the tail of the most-recent ``limit`` and
    re-sort ascending here.
    """
    rows = conn.execute(
        """
        SELECT * FROM chat_turns
        WHERE user_id = ?
        ORDER BY ts DESC, id DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    return list(reversed(rows))


def delete_all_for_user(conn: sqlite3.Connection, user_id: int) -> int:
    """Delete every chat turn for a user. Returns rows affected."""
    cursor = conn.execute(
        "DELETE FROM chat_turns WHERE user_id = ?", (user_id,)
    )
    conn.commit()
    return cursor.rowcount or 0


def decode_sources(row: sqlite3.Row) -> list[dict]:
    raw = row["sources_json"] if row is not None else None
    if not raw:
        return []
    return list(json.loads(raw))


def decode_scope(row: sqlite3.Row) -> list[str]:
    raw = row["scope_json"] if row is not None else None
    if not raw:
        return []
    return list(json.loads(raw))
