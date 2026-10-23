"""Review 18 (b): the feed's per-bucket "N+" marker against the papers it hid.

Built straight in SQLite (no OpenAlex, no embedder): one live interest with
threshold 0.80 and least-typical seed 0.90, so a paper at >= 0.90 is "Like
your seeds" (high) and one in [0.80, 0.90) is "Above the bar" (medium).
"""

from __future__ import annotations

import pytest

from rag_lib.api.services import radar as radar_service
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import papers as papers_repo, users as users_repo
from rag_lib.scoring.pool import rescore_profile_pool


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "radar.db")
    apply_migrations(c)
    yield c
    c.close()


def _interest(conn, user_id, slug="feed", threshold=0.80, seed_sim_min=0.90):
    cur = conn.execute(
        "INSERT INTO profiles (user_id, name, slug, embedding_model, topic_filters_json,"
        " n_seed, is_draft, threshold, seed_sim_min) VALUES (?, ?, ?, 'placeholder-v1', '{}', 0, 0, ?, ?)",
        (user_id, slug, slug, threshold, seed_sim_min),
    )
    conn.commit()
    return int(cur.lastrowid)


def _paper(conn, pid, i, raw):
    oa = f"https://openalex.org/W{pid}{i:05d}"
    papers_repo.upsert(conn, {"openalex_id": oa, "title": f"Paper {pid}-{i}", "source": "test", "year": 2024})
    conn.execute(
        "INSERT INTO profile_candidates (profile_id, openalex_id, score, score_raw) VALUES (?, ?, ?, ?)",
        (pid, oa, raw, raw),
    )
    return oa


def test_hidden_medium_papers_are_marked_when_the_rows_read_were_all_high(conn, monkeypatch):
    """With no reranker the pool ranks by cosine, so every "high" paper
    comes before every "medium" one. Past the cap, the one extra row the
    feed reads is "high" too, and "Above the bar" is reported as an exact 0
    -- picking it then lists all of them."""
    monkeypatch.setattr(radar_service, "_DAILY_LIMIT", 10)
    uid = int(users_repo.upsert(conn, "u@example.com")["id"])
    pid = _interest(conn, uid)
    for i in range(15):
        _paper(conn, pid, i, 0.95 - i * 0.001)          # high
    for i in range(15, 25):
        _paper(conn, pid, i, 0.85 - i * 0.001)          # medium
    conn.commit()
    rescore_profile_pool(conn, pid, alpha=0.4, beta=0.6)

    view = radar_service.daily(conn, uid)
    shown = [c for c in view.cards if c.state is None]
    assert {c.bucket for c in shown} == {"high"}       # the cap held only high papers
    n_medium_listed = sum(1 for c in shown if c.bucket == "medium")
    assert n_medium_listed == 0

    medium_view = radar_service.daily(conn, uid, bucket="medium")
    n_medium_exist = sum(1 for c in medium_view.cards if c.state is None)
    assert n_medium_exist == 10

    by = view.untriagedNotShownByBucket
    # The frontend shows "Above the bar 0" (no "+") unless by["medium"] is set.
    assert by.get("medium"), (
        f"'Above the bar' shown as exactly {n_medium_listed} with {n_medium_exist} hidden; by-bucket={by}"
    )
