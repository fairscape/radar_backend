"""SQLite connection helper.

Single entry point that every reader/writer goes through so foreign-key
enforcement, WAL journaling, and the row factory are consistent. Callers
pass a path; we handle parent-directory creation, pragmas, and row-typing.

WAL is set to keep the CLI and the FastAPI service from blocking each
other when both touch the same file. Foreign keys are off by default in
SQLite — turn them on every connection.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (or create) the SQLite database at ``path``.

    Creates the parent directory if missing. Sets:
      - ``foreign_keys = ON`` (per-connection in SQLite)
      - ``journal_mode = WAL`` (persists across connections)
      - ``row_factory = sqlite3.Row`` so callers can access columns by name
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(p),
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
        timeout=30.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    # journal_mode persists in the file header, so setting it is a one-time
    # job -- but `PRAGMA journal_mode = WAL` grabs an exclusive lock even
    # when the mode is already WAL. Doing that on every connection meant a
    # single in-flight write stalled every other request, readers included,
    # for the full busy timeout. Read the mode first (lock-free) and only
    # write it when the database is genuinely not in WAL yet.
    if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn
