"""0024_reserve_all_slug -- no interest keeps the slug "all".

"all" is what the feed's interest filter and the Vault's tag filter send
for "every interest", so an interest whose slug was "all" (one named "ALL")
could not be filtered to, listed the whole Vault as its seeds, and had its
tag count overwritten by the Vault total. ``slugify`` now suffixes it like
the other reserved slugs ("all-profile"); this renames the rows already
stored, per user, to the first free "all-profile[-n]". A bookmark to the
old URL then opens the not-found page, which is what it did in effect.
"""

from __future__ import annotations

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, user_id FROM profiles WHERE slug = 'all'").fetchall()
    for pid, uid in rows:
        taken = {r[0] for r in conn.execute(
            "SELECT slug FROM profiles WHERE user_id IS ? AND slug LIKE 'all-profile%'", (uid,))}
        slug, n = "all-profile", 2
        while slug in taken:
            slug, n = f"all-profile-{n}", n + 1
        conn.execute("UPDATE profiles SET slug = ? WHERE id = ?", (slug, pid))
