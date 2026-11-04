"""0040 -- FUNCTIONAL_SPEC CR.1 and CR.10.

CR.1: a paper kept in a user's Vault is never deleted from under it. 0039
made ``vault_kept`` follow its paper with ON DELETE CASCADE, so a draft's
discard that judged the paper unused took it out of whoever's Vault kept
it. Without the cascade the delete fails on the foreign key instead, and
the paper stays (``rag_lib.db.repos.paper_rows``).

CR.10: every upload is a paper of its own, so one user may hold several
versions of the same bytes (the same file in two interests). 0029 kept one
per (user, file hash).

Both tables are rebuilt. The triggers 0029 put on other tables name
``paper_versions``, and SQLite refuses the rename while they do, so they
are taken off for the rebuild and put back as they were.
"""

from __future__ import annotations

import sqlite3

NO_FOREIGN_KEYS = True


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE vault_kept_new (
          user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          openalex_id  TEXT    NOT NULL REFERENCES papers(openalex_id),
          kept_at      TEXT    NOT NULL DEFAULT (datetime('now')),
          PRIMARY KEY (user_id, openalex_id)
        )
        """
    )
    conn.execute("INSERT INTO vault_kept_new (user_id, openalex_id, kept_at) "
                 "SELECT user_id, openalex_id, kept_at FROM vault_kept")
    conn.execute("DROP TABLE vault_kept")
    conn.execute("ALTER TABLE vault_kept_new RENAME TO vault_kept")

    triggers = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND sql LIKE '%paper_versions%' ORDER BY name"
    ).fetchall()
    for name, _sql in triggers:
        conn.execute(f'DROP TRIGGER "{name}"')
    conn.execute(
        """
        CREATE TABLE paper_versions_new (
          version_id TEXT PRIMARY KEY REFERENCES papers(openalex_id) ON DELETE CASCADE,
          user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          canonical_openalex_id TEXT REFERENCES papers(openalex_id),
          file_hash TEXT NOT NULL,
          -- 1: holds this file's text; 0: waiting for it; -1: file unreadable
          parsed INTEGER NOT NULL DEFAULT 1,
          -- 1: the PDF had no title, the title is its file's name
          title_from_file INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    # rowid kept: two files uploaded within one second are numbered in the
    # order they were registered (versions.version_numbers).
    conn.execute(
        "INSERT INTO paper_versions_new (rowid, version_id, user_id, canonical_openalex_id, file_hash, parsed, "
        "title_from_file) SELECT rowid, version_id, user_id, canonical_openalex_id, file_hash, parsed, "
        "title_from_file FROM paper_versions"
    )
    conn.execute("DROP TABLE paper_versions")
    conn.execute("ALTER TABLE paper_versions_new RENAME TO paper_versions")
    conn.execute("CREATE INDEX idx_paper_versions_canonical ON paper_versions(canonical_openalex_id)")
    conn.execute("CREATE INDEX idx_paper_versions_user_hash ON paper_versions(user_id, file_hash)")
    for _name, sql in triggers:
        conn.execute(sql)
