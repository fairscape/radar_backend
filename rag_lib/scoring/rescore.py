"""Re-score an interest's stored candidates against its current seeds.

A refit (seeds added or removed) moves the centroid, but the candidates
already stored kept the similarity they got against the old one, so adding
a seed changed nothing in the feed until the next scan -- the morning after.
This recomputes the selector scores of every stored candidate that has a
vector, then puts the pool back on one scale (``jobs.rescore_pool``).

Candidates without a stored vector are left as they are: gathers keep the
vectors they compute from 2026-10-01 on (selectors attach them to the
paper, db_store persists them); older ones are filled by
``scripts/backfill_candidate_embeddings.py``.

The reranker's raw score is not recomputed. Its queries come from the
seeds too, but it is a cross-encoder pass per paper -- too slow for a
click. The next scan that fetches the paper again refreshes it.
"""

from __future__ import annotations

import sqlite3

from ..db.repos import embeddings as embeddings_repo
from ..paper import Paper
from .pool import KEEP_ALL


def rescore_candidates(conn: sqlite3.Connection, profile_id: int) -> dict:
    """Rewrite ``score_raw`` / ``score_max_seed`` / ``score`` of one interest's
    stored candidates from its fitted selector, then rescore the pool.

    Returns ``{"n_rescored": int, "n_skipped": int}`` (skipped = no vector).
    """
    from ..scheduler import jobs  # jobs imports scoring; import late

    row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    if row is None or row["is_draft"]:
        return {"n_rescored": 0, "n_skipped": 0}
    profile = jobs._load_profile(conn, row)
    model = profile.embedding_model
    selector = jobs._build_selector(row, profile)

    papers: list[Paper] = []
    n_skipped = 0
    for (oa,) in conn.execute(
        "SELECT openalex_id FROM profile_candidates WHERE profile_id = ?", (profile_id,),
    ).fetchall():
        vec = embeddings_repo.get(conn, oa, model)
        if vec is None:
            n_skipped += 1
            continue
        p = Paper(doi=None, openalex_id=oa, title="")
        p.embeddings[model] = vec.tolist()
        papers.append(p)

    if papers:
        ranked = selector.select(papers, profile, threshold=KEEP_ALL)
        conn.executemany(
            """
            UPDATE profile_candidates
            SET score_raw = ?, score_max_seed = ?, score = ?
            WHERE profile_id = ? AND openalex_id = ?
            """,
            [
                (float(bd.get("score_raw", s)), bd.get("score_max_seed"),
                 float(bd.get("score_raw", s)), profile_id, p.openalex_id)
                for s, p, bd in ranked
            ],
        )
        conn.commit()
    jobs.rescore_pool(conn, profile_id)
    return {"n_rescored": len(papers), "n_skipped": n_skipped}
