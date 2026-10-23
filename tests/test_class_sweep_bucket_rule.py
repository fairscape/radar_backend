"""Class sweep: a card's bucket is decided in one rule.

The card's bucket comes from ``mappers.bucket_for_similarity``; the feed's
bucket filter and its "not shown" counts are SQL. They had drifted: the SQL
used the raw cosine alone, the card falls back to the rank percentile when
there is no raw cosine or no threshold (round 19). ``BUCKET_SQL`` must give
the card's answer for every combination.
"""

from __future__ import annotations

import itertools
import sqlite3

from rag_lib.api.mappers import bucket_for_similarity
from rag_lib.db.repos.candidates import BUCKET_SQL

VALUES = [None, 0.0, 0.5, 0.66, 0.67, 0.8, 0.85, 0.899, 0.9, 0.95, 1.0]


def test_sql_bucket_matches_the_card():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE pc (score REAL, score_pct REAL, score_raw REAL)")
    db.execute("CREATE TABLE pr (threshold REAL, seed_sim_min REAL)")
    bad = []
    for raw, pct, thr, band in itertools.product(VALUES, VALUES, VALUES, VALUES):
        score = 0.5  # the legacy figure, used only when there is no percentile
        db.execute("DELETE FROM pc"); db.execute("DELETE FROM pr")
        db.execute("INSERT INTO pc VALUES (?, ?, ?)", (score, pct, raw))
        db.execute("INSERT INTO pr VALUES (?, ?)", (thr, band))
        got = db.execute(f"SELECT {BUCKET_SQL} FROM pc, pr").fetchone()[0]
        want = bucket_for_similarity(raw, threshold=thr, seed_sim_min=band,
                                     percentile=pct if pct is not None else score)
        if got != want:
            bad.append((raw, pct, thr, band, got, want))
    assert not bad, bad[:10]
