"""0028_merge_zenodo_versions -- one card per Zenodo record per interest.

0026 merged copies sharing a DOI or a (long) title. A short-titled Zenodo
record's concept and version DOIs share neither, so both stayed ("The Trust
Dial" twice). Scans now key them together (``paper.paper_keys``); this
merges the copies already stored, as 0026 did: the triaged copy is kept
(saved, then dismissed), else the highest scored, and only an untriaged
copy is removed.
"""

from __future__ import annotations

import json
import sqlite3

from rag_lib.paper import paper_keys


def migrate(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT pc.profile_id, pc.openalex_id, pc.saved_at, pc.dismissed_at,
               p.doi, p.title, p.year, p.authors_json
        FROM profile_candidates pc JOIN papers p USING (openalex_id)
        ORDER BY pc.profile_id,
                 pc.saved_at IS NULL, pc.dismissed_at IS NULL,
                 COALESCE(pc.score_raw, pc.score) DESC, pc.fetched_at, pc.openalex_id
        """
    ).fetchall()
    owner: dict[tuple[int, str], str] = {}
    drop: list[tuple[int, str]] = []
    for pid, oid, saved, dismissed, doi, title, year, authors_json in rows:
        try:
            authors = json.loads(authors_json) if authors_json else []
        except ValueError:
            authors = []
        keys = [(pid, k) for k in paper_keys(doi, title, year, authors if isinstance(authors, list) else [])]
        if any(k in owner for k in keys) and saved is None and dismissed is None:
            drop.append((pid, oid))
            continue
        for k in keys:
            owner.setdefault(k, oid)
    conn.executemany("DELETE FROM profile_candidates WHERE profile_id = ? AND openalex_id = ?", drop)
