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


# A migration that rebuilds a table other tables reference runs with
# foreign keys off: with them on, DROP TABLE on the old copy deletes from it
# first, and every ON DELETE CASCADE child row goes with it. SQLite ignores
# the pragma inside a transaction, so the runner sets it around one, and
# checks the keys before committing. Such a migration is a ``.py`` file with
# ``migrate(conn)`` (and ``NO_FOREIGN_KEYS = True``), since a rebuild has to
# read the table it replaces.


def _apply_python(conn: sqlite3.Connection, version: str, path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_radar_migration_{version}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    no_fk = bool(getattr(module, "NO_FOREIGN_KEYS", False))
    fk_was_on = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    if no_fk:
        conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN")
        module.migrate(conn)
        bad = conn.execute("PRAGMA foreign_key_check").fetchall()
        if bad:
            raise sqlite3.IntegrityError(
                f"{version}: foreign key check failed: {[tuple(r) for r in bad[:5]]}")
        conn.execute("INSERT INTO _migrations_applied (version) VALUES (?)", (version,))
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        if no_fk:
            conn.execute(f"PRAGMA foreign_keys = {'ON' if fk_was_on else 'OFF'}")


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

    sql_files = sorted(
        [*mdir.glob("*.sql"), *(p for p in mdir.glob("*.py") if p.stem[:4].isdigit())],
        key=lambda p: p.stem,
    )
    new_versions: list[str] = []
    for sql_path in sql_files:
        version = sql_path.stem
        if version in applied:
            continue
        if sql_path.suffix == ".py":
            _apply_python(conn, version, sql_path)
            new_versions.append(version)
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
