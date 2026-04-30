"""Hand-rolled migration runner.

Append-only convention: each migration is a SQL file under
``rag_lib/db/migrations/<NNNN>_<topic>.sql``. They run in lexicographic
order, each wrapped in BEGIN…COMMIT so a failure mid-script rolls back
both the schema change and the row that records it. ``apply_migrations``
is idempotent — already-applied versions are skipped.

We deliberately use ``executescript`` rather than executing statements
one at a time so a migration can use multiple statements without the
runner needing a SQL splitter. The trade-off is that ``executescript``
auto-commits the current transaction first; we re-establish a
transaction by emitting BEGIN/COMMIT inside the wrapped script.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _ensure_bootstrap(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _migrations_applied (
            version    TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def apply_migrations(
    conn: sqlite3.Connection,
    migrations_dir: Path | str | None = None,
) -> list[str]:
    """Apply every migration not yet recorded in ``_migrations_applied``.

    Returns the list of versions applied during this call (empty if the
    DB was already up to date). Safe to call repeatedly.
    """
    mdir = Path(migrations_dir) if migrations_dir is not None else MIGRATIONS_DIR
    _ensure_bootstrap(conn)

    applied = {
        row["version"]
        for row in conn.execute("SELECT version FROM _migrations_applied")
    }

    sql_files = sorted(mdir.glob("*.sql"))
    new_versions: list[str] = []
    for sql_path in sql_files:
        version = sql_path.stem
        if version in applied:
            continue
        body = sql_path.read_text()
        # Wrap the migration plus its bookkeeping insert in one transaction
        # so a failed migration doesn't leave a half-applied schema or a
        # bookkeeping row without its schema change.
        script = (
            "BEGIN;\n"
            f"{body}\n"
            "INSERT INTO _migrations_applied (version) VALUES (?);\n"
            "COMMIT;\n"
        )
        # executescript doesn't support parameter binding, so inline the
        # version after escaping single quotes (versions come from filenames
        # we control, but defense in depth).
        safe_version = version.replace("'", "''")
        script = script.replace("(?)", f"('{safe_version}')")
        try:
            conn.executescript(script)
        except sqlite3.Error:
            conn.execute("ROLLBACK")
            raise
        new_versions.append(version)

    return new_versions


def applied_versions(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return ``[(version, applied_at), ...]`` ordered by version."""
    rows = conn.execute(
        "SELECT version, applied_at FROM _migrations_applied ORDER BY version"
    ).fetchall()
    return [(r["version"], r["applied_at"]) for r in rows]
