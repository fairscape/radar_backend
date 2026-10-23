"""0027_implausible_years -- forget publication years that cannot be right yet.

``papers_repo.plausible_year`` keeps them out of new writes (2029, 2050:
OpenAlex placeholders); this clears the ones already stored, so an
interest whose years run "to now" no longer lists them first. An unknown
year passes every year filter.
"""

from __future__ import annotations

import sqlite3

from rag_lib.db.repos.papers import plausible_year


def migrate(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT openalex_id, year FROM papers WHERE year IS NOT NULL").fetchall()
    for oa_id, year in rows:
        if plausible_year(year) is None:
            conn.execute("UPDATE papers SET year = NULL WHERE openalex_id = ?", (oa_id,))
