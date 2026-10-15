"""users repo.

Identity is email-only at the demo stage. Phase 12 fills in real auth on
top of the same table; the repo doesn't change.
"""

from __future__ import annotations

import sqlite3


def get(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, email, mailto, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()


def get_by_email(conn: sqlite3.Connection, email: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, email, mailto, created_at FROM users WHERE email = ?",
        (email.strip().lower(),),
    ).fetchone()


def upsert(
    conn: sqlite3.Connection,
    email: str,
    mailto: str | None = None,
) -> sqlite3.Row:
    """Insert or update a user row by email. Returns the resulting row.

    On insert: ``mailto`` defaults to ``email`` when not given.
    On update: only overwrites ``mailto`` when explicitly passed.
    """
    email_norm = email.strip().lower()
    existing = get_by_email(conn, email_norm)
    if existing is None:
        conn.execute(
            "INSERT INTO users (email, mailto) VALUES (?, ?)",
            (email_norm, mailto if mailto is not None else email_norm),
        )
        conn.commit()
        return get_by_email(conn, email_norm)  # type: ignore[return-value]
    if mailto is not None and mailto != existing["mailto"]:
        conn.execute(
            "UPDATE users SET mailto = ? WHERE id = ?",
            (mailto, existing["id"]),
        )
        conn.commit()
        return get_by_email(conn, email_norm)  # type: ignore[return-value]
    return existing


def list_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, email, mailto, created_at FROM users ORDER BY id"
    ).fetchall()


def contact_mailto(conn: sqlite3.Connection, user_id: int | None, default: str) -> str:
    """The address sent to OpenAlex as ``mailto`` for this user's requests.

    The one rule for every OpenAlex call made for a user: the contact email
    they set in Settings, else their login email, else the deployment's
    default. Only the scheduled scans used to apply it; trial scans,
    imports, uploads and researcher refreshes always sent the default,
    though Settings says the address goes with every request.
    """
    return mailto_from_row(get(conn, int(user_id)) if user_id is not None else None, default)


def mailto_from_row(row: sqlite3.Row | None, default: str) -> str:
    """``contact_mailto`` for a ``users`` row already in hand."""
    if row is not None and row["mailto"]:
        return row["mailto"]
    if row is not None and row["email"]:
        return row["email"]
    return default
