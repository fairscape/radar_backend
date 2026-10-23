"""0026_merge_duplicate_candidates -- one candidate per paper per interest.

Scans deduped a paper only within one batch, so a later scan that brought
another OpenAlex id of the same paper (a versioned Zenodo DOI, a preprint
and its article) stored it again: two cards, and one still to review after
the other was saved. Scans now check what is stored
(``db_store.dedup_and_insert_candidates``); this removes the copies
already there. Per interest, papers with the same DOI or title are one
paper; the copy kept is the triaged one (saved, then dismissed), else the
highest scored. A copy is removed only while it is untriaged.
"""

from __future__ import annotations

import sqlite3

from rag_lib.paper import title_key


def _keys(doi, title):
    keys = []
    if doi:
        keys.append("doi:" + doi.strip().lower())
    tk = title_key(title)
    if tk:
        keys.append("title:" + tk)
    return keys


def migrate(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT pc.profile_id, pc.openalex_id, pc.saved_at, pc.dismissed_at, pc.score, p.doi, p.title
        FROM profile_candidates pc JOIN papers p USING (openalex_id)
        ORDER BY pc.profile_id,
                 pc.saved_at IS NULL, pc.dismissed_at IS NULL,
                 COALESCE(pc.score_raw, pc.score) DESC, pc.fetched_at, pc.openalex_id
        """
    ).fetchall()
    owner: dict[tuple[int, str], str] = {}
    drop: list[tuple[int, str]] = []
    for pid, oid, saved, dismissed, _score, doi, title in rows:
        keys = [(pid, k) for k in _keys(doi, title)]
        if any(k in owner for k in keys) and saved is None and dismissed is None:
            drop.append((pid, oid))
            continue
        for k in keys:
            owner.setdefault(k, oid)
    conn.executemany("DELETE FROM profile_candidates WHERE profile_id = ? AND openalex_id = ?", drop)
