"""0021_profile_names_per_user — interest names and slugs unique per user.

profiles.name was UNIQUE across every user (0001) and so was slug (0002's
index), so naming an interest what another user had named one got it
renamed "<name> (2)" and its URL "<slug>-2": a user could probe which
interest names other users have. Both are now unique per user.

SQLite cannot drop a column constraint, so the table is rebuilt (the
documented 12-step procedure, with foreign keys off -- see migrate.py).
It is a Python migration because deployed databases carry columns a fresh
one does not (orcid, researcher_name, rp_meta_json from an earlier 0016):
the new table is the old one's own definition minus the UNIQUE, so every
column, whichever they are, is kept.
"""

from __future__ import annotations

import re
import sqlite3

NO_FOREIGN_KEYS = True


def migrate(conn: sqlite3.Connection) -> None:
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'profiles'"
    ).fetchone()[0]
    new_sql, n = re.subn(r'^CREATE TABLE\s+"?profiles"?', "CREATE TABLE profiles_new", sql)
    assert n == 1, sql[:80]
    new_sql, n = re.subn(r"(\bname\s+TEXT\s+NOT\s+NULL)\s+UNIQUE", r"\1", new_sql, count=1)
    assert n == 1, "profiles.name is not 'TEXT NOT NULL UNIQUE'"
    cols = ", ".join(f'"{r[1]}"' for r in conn.execute("PRAGMA table_info(profiles)"))
    indexes = [
        r[0] for r in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'profiles'"
            " AND sql IS NOT NULL AND name != 'idx_profiles_slug'"
        )
    ]
    seq = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'profiles'").fetchone()

    conn.execute(new_sql)
    conn.execute(f"INSERT INTO profiles_new ({cols}) SELECT {cols} FROM profiles")
    conn.execute("DROP TABLE profiles")
    conn.execute("ALTER TABLE profiles_new RENAME TO profiles")
    for index_sql in indexes:
        conn.execute(index_sql)
    conn.execute("CREATE UNIQUE INDEX idx_profiles_user_name ON profiles(user_id, name)")
    conn.execute("CREATE UNIQUE INDEX idx_profiles_user_slug ON profiles(user_id, slug)")
    # Keep AUTOINCREMENT's high-water mark: a deleted interest's id must
    # not come back as a new one.
    if seq is not None:
        conn.execute(
            "UPDATE sqlite_sequence SET seq = MAX(seq, ?) WHERE name = 'profiles'", (int(seq[0]),),
        )
