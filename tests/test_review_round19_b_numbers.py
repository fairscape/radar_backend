"""Review 19 (b): numbers and data that must agree across endpoints.

Built in SQLite with stubs; no OpenAlex, no embedder, no Chroma.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rag_lib.api.services import radar as radar_service
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import users as users_repo
from rag_lib.paper import Paper
from rag_lib.persistence import dedup_and_insert_candidates
from rag_lib.scoring.pool import rescore_profile_pool


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "radar.db")
    apply_migrations(c)
    yield c
    c.close()


def _interest(conn, uid, slug="feed", threshold=0.80, seed_sim_min=0.90):
    cur = conn.execute(
        "INSERT INTO profiles (user_id, name, slug, embedding_model, topic_filters_json,"
        " n_seed, is_draft, threshold, seed_sim_min) VALUES (?, ?, ?, 'm', '{}', 0, 0, ?, ?)",
        (uid, slug, slug, threshold, seed_sim_min),
    )
    conn.commit()
    return int(cur.lastrowid)


def _run(conn, pid, uid):
    cur = conn.execute("INSERT INTO gather_runs (profile_id, user_id, tier_used) VALUES (?, ?, 'manual')", (pid, uid))
    conn.commit()
    return int(cur.lastrowid)


def _paper(oa, doi, title):
    return Paper(doi=doi, openalex_id=oa, title=title, abstract="a", year=2026,
                 venue="Zenodo", source="openalex_gatherer")


# ---------------------------------------------------------------------------
# Dedup: the same paper (same DOI) under a second OpenAlex id in a later scan
# ---------------------------------------------------------------------------

def test_same_doi_from_a_later_scan_is_not_a_second_card(conn):
    """Real on the sandbox copy: 9 (interest, DOI) pairs listed twice, e.g.
    interest 74 has W7160248259 and W7160319602, both DOI
    10.5281/zenodo.20044865, both 0.919 over a 0.899 bar, from runs 1201 and
    1314. The in-batch title dedup never sees the earlier batch."""
    uid = int(users_repo.upsert(conn, "u@example.com")["id"])
    pid = _interest(conn, uid)
    doi = "10.5281/zenodo.20044865"
    title = "Towards Early and Accurate Disease Detection Through Multimodal Predictive Modeling"
    r1 = _run(conn, pid, uid)
    dedup_and_insert_candidates(conn, profile_id=pid, gather_run_id=r1,
                                ranked=[(0.919, _paper("https://openalex.org/W1", doi, title), {"score_raw": 0.919})])
    r2 = _run(conn, pid, uid)
    n_new, _ = dedup_and_insert_candidates(conn, profile_id=pid, gather_run_id=r2,
                                           ranked=[(0.919, _paper("https://openalex.org/W2", doi, title), {"score_raw": 0.919})])
    rescore_profile_pool(conn, pid, alpha=0.4, beta=0.6)
    view = radar_service.daily(conn, uid)
    ids = [c.id for c in view.cards]
    assert len(ids) == 1, f"one paper, {len(ids)} cards: {ids}; second scan reported n_new={n_new}"


def test_a_saved_paper_does_not_come_back_untriaged_under_its_other_id(conn):
    uid = int(users_repo.upsert(conn, "u@example.com")["id"])
    pid = _interest(conn, uid)
    doi = "10.5281/zenodo.18364883"
    title = "Fragmentation and Continuity: A Systems Analysis of Multi-morbid NCD Care"
    r1 = _run(conn, pid, uid)
    dedup_and_insert_candidates(conn, profile_id=pid, gather_run_id=r1,
                                ranked=[(0.95, _paper("https://openalex.org/W1", doi, title), {"score_raw": 0.95})])
    radar_service.save(conn, uid, "https://openalex.org/W1", profile_slug="feed", on=True)
    r2 = _run(conn, pid, uid)
    dedup_and_insert_candidates(conn, profile_id=pid, gather_run_id=r2,
                                ranked=[(0.95, _paper("https://openalex.org/W2", doi, title), {"score_raw": 0.95})])
    rescore_profile_pool(conn, pid, alpha=0.4, beta=0.6)
    view = radar_service.daily(conn, uid)
    to_review = [c.id for c in view.cards if c.state is None]
    assert to_review == [], f"saved paper back as new to review: {to_review}"


# ---------------------------------------------------------------------------
# 30-day windows: topic-yield vs saves30
# ---------------------------------------------------------------------------

def test_topic_yield_window_is_the_same_30_days_as_everywhere_else(conn, tmp_path, monkeypatch):
    """topic-yield cuts on a date ("YYYY-MM-DD"), so ``days=30`` reaches back
    to 00:00 UTC thirty days ago: up to 31 days. saves30 and every other
    window use datetime('now', '-30 days')."""
    from rag_lib.api.routers import profiles as profiles_router

    uid = int(users_repo.upsert(conn, "u@example.com")["id"])
    pid = _interest(conn, uid)
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?",
                 ('{"topics": [{"id": "T1", "display_name": "T1"}]}', pid))
    from rag_lib.db.repos import papers as papers_repo
    papers_repo.upsert(conn, {"openalex_id": "https://openalex.org/W9", "title": "Old", "source": "t", "year": 2026})
    # Fetched 30 days and 12 hours ago: outside a 30-day window.
    old = (datetime.now(timezone.utc) - timedelta(days=30, hours=12))
    # Keep it on the same calendar day as "30 days ago" so the date cut lets it in.
    cut_day = (datetime.now(timezone.utc) - timedelta(days=30)).date()
    if old.date() != cut_day:
        old = datetime.combine(cut_day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(minutes=1)
        if datetime.now(timezone.utc) - old <= timedelta(days=30):
            pytest.skip("too close to midnight UTC to build the case")
    conn.execute(
        "INSERT INTO profile_candidates (profile_id, openalex_id, score, score_raw, sourced_by_topic_id, fetched_at, saved_at)"
        " VALUES (?, ?, 0.95, 0.95, 'T1', ?, ?)",
        (pid, "https://openalex.org/W9", old.strftime("%Y-%m-%d %H:%M:%S"), old.strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    out = profiles_router.topic_yield("feed", user, conn, days=30)
    from rag_lib.db.repos import candidates as candidates_repo
    saves30, _ = candidates_repo.count_saves_dismisses_30d(conn, pid)
    t1 = next(t for t in out.topics if t.topic_id == "T1")
    assert saves30 == 0
    assert t1.n_candidates == 0, (
        f"a paper fetched {datetime.now(timezone.utc) - old} ago counted in the 30-day topic yield"
    )


# ---------------------------------------------------------------------------
# Parsing: one odd Prosopia work sinks the whole profile
# ---------------------------------------------------------------------------

def _storage_with(entries):
    from rag_lib import prosopia_client as pc

    class S(pc.PagedApiStorage):
        def _request(self, method, path, params=None, **kw):  # noqa: ARG002
            if path.endswith("papers"):
                return {"items": entries, "has_more": False}
            return {"fields": {"name": "Someone"}}

    return S


@pytest.mark.parametrize("bad", [
    {"paper_id": "p2", "title": None, "year": 2021},                       # no title
    {"paper_id": "p2", "title": "Second", "year": "2021-05-03"},           # a date where a year goes
    {"paper_id": "p2", "title": "Second", "year": 2021, "authors": ["A", None]},  # a null author
])
def test_one_malformed_prosopia_work_does_not_fail_the_whole_profile(monkeypatch, bad):
    from rag_lib import prosopia_client as pc

    good = {"paper_id": "p1", "title": "First", "year": 2020, "doi": "10.1/x"}
    monkeypatch.setattr(pc, "PagedApiStorage", _storage_with([good, bad]))
    client = pc.ProsopiaClient("https://prosopia.example")
    try:
        prof = client.fetch_profile("someone")
    except pc.ProsopiaError as exc:
        pytest.fail(f"81 good works lost to one odd one: {exc}"[:400])
    assert any(p.paper_id == "p1" for p in prof.papers)


def test_0026_merges_copies_already_stored(tmp_path):
    """Copies stored before scans checked what was stored: the untriaged
    copy goes, a triaged one stays, unrelated papers are untouched."""
    import importlib.util
    import pathlib
    import sqlite3

    import rag_lib

    path = pathlib.Path(rag_lib.__file__).parent / "db" / "migrations" / "0026_merge_duplicate_candidates.py"
    spec = importlib.util.spec_from_file_location("m0026", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE papers (openalex_id TEXT PRIMARY KEY, doi TEXT, title TEXT);
        CREATE TABLE profile_candidates (profile_id INT, openalex_id TEXT, score REAL, score_raw REAL,
          fetched_at TEXT, saved_at TEXT, dismissed_at TEXT, PRIMARY KEY (profile_id, openalex_id));
        INSERT INTO papers VALUES ('W1', '10.5281/zenodo.1', 'A long enough title about reef ecology'),
                                  ('W2', '10.5281/ZENODO.1', 'A long enough title about reef ecology v2'),
                                  ('W3', NULL, 'Another paper with a perfectly long title'),
                                  ('W4', NULL, 'Another paper with a perfectly long title.'),
                                  ('W5', NULL, 'Unrelated paper about volcanic ash plumes');
        INSERT INTO profile_candidates VALUES (1, 'W1', 0.9, 0.9, 't1', NULL, NULL),
                                              (1, 'W2', 0.95, 0.95, 't2', '2026-01-01', NULL),
                                              (1, 'W3', 0.8, 0.8, 't1', NULL, NULL),
                                              (1, 'W4', 0.7, 0.7, 't2', NULL, NULL),
                                              (1, 'W5', 0.6, 0.6, 't1', NULL, NULL),
                                              (2, 'W1', 0.9, 0.9, 't1', NULL, NULL);
    """)
    mod.migrate(conn)
    left = conn.execute("SELECT profile_id, openalex_id FROM profile_candidates ORDER BY 1, 2").fetchall()
    assert left == [(1, "W2"), (1, "W3"), (1, "W5"), (2, "W1")]
