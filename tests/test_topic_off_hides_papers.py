"""Rule B (user 2026-11-06): a topic switched off takes its papers out of the feed.

A paper a topic brought in stays stored, but is in the feed only while one
of its own OpenAlex topics (primary included), or the topic that brought it
in, is switched on in the interest. Switching the topic on again brings it
back without a scan. ``candidates.TOPIC_ON`` (SQL) and ``scoring.pool.topic_on``
(Python) are the one rule; every "the feed" read applies it.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from rag_lib.db import connect
from rag_lib.db.repos import candidates as candidates_repo
from rag_lib.scoring.pool import topic_on
from tests.test_api_radar import _request, app  # noqa: F401 -- fixture

A, B = "T11431", "T10123"           # the seeded interest's two topics (tests/test_api_profiles.py)


def _tf(**on):
    return {"topics": [{"id": tid, "display_name": tid, **({} if v is ... else {"on": v})} for tid, v in on.items()]}


def _paper(primary=None, *others):
    return json.dumps({"primary_topic": {"id": primary} if primary else None,
                       "topics": [{"id": t} for t in others]})


# --- the rule ------------------------------------------------------------------

def test_a_paper_is_in_while_one_of_its_topics_or_its_source_is_on():
    full = "https://openalex.org/" + A
    assert topic_on(None, _paper("T9"), _tf(T11431=False))            # no topic brought it in
    assert topic_on(A, _paper("T9"), None)                            # no topic list
    assert topic_on(A, _paper("T9"), {"topics": []})
    assert topic_on(A, _paper("T9"), _tf(T11431=True))                # its source is on
    assert topic_on(A, _paper("T9"), _tf(T11431=...))                 # absent "on" means on
    assert not topic_on(A, _paper("T9"), _tf(T11431=False))           # off, and nothing else of it on
    assert not topic_on(A, _paper("T9"), _tf(T11431=None))            # null is off (is_enabled)
    assert topic_on(A, _paper(B), _tf(T11431=False, T10123=True))     # its primary topic is on
    assert topic_on(A, _paper("T9", B), _tf(T11431=False, T10123=True))  # another of its topics is on
    assert not topic_on(A, _paper("T9", B), _tf(T11431=False, T10123=False))  # all off
    assert topic_on(full, _paper("T9"), _tf(T11431=True))             # full and bare ids are one topic
    assert topic_on(A, _paper("T9"), {"topics": [{"id": full, "on": True}]})
    assert not topic_on(A, "not json", _tf(T11431=False))
    assert not topic_on(A, None, _tf(T11431=False))


def test_the_sql_is_the_same_rule(tmp_path):
    """Random interests and papers: TOPIC_ON in SQL agrees with topic_on."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE papers (openalex_id TEXT PRIMARY KEY, topics_json TEXT);
        CREATE TABLE profiles (id INTEGER PRIMARY KEY, topic_filters_json TEXT);
        CREATE TABLE profile_candidates (profile_id INT, openalex_id TEXT, sourced_by_topic_id TEXT);
    """)
    rng = random.Random(7)
    ids = ["T1", "T2", "T3", "https://openalex.org/T4", "T5"]
    cases = []
    for pid in range(1, 201):
        shape = rng.random()
        if shape < 0.05:
            tf = None
        elif shape < 0.1:
            tf = {"topics": []}
        elif shape < 0.13:
            tf = {"subfields": []}
        else:
            ons = [{}, {"on": True}, {"on": False}, {"on": None}, {"on": 0}, {"on": 1},
                   {"on": "false"}, {"on": ""}, {"on": [1]}, {"on": []}, {"on": 2.5}]
            tf = {"topics": [{"id": t, **rng.choice(ons)} for t in rng.sample(ids, rng.randint(1, 5))]}
            if rng.random() < 0.1:
                tf["topics"].append(rng.choice(["T1", 7, None, ["T2"]]))     # not an object: skipped
        conn.execute("INSERT INTO profiles VALUES (?, ?)", (pid, None if tf is None else json.dumps(tf)))
        for j in range(6):
            oa = f"W{pid}-{j}"
            r = rng.random()
            odd = ['{"topics": {"id": "T1"}}', '{"topics": "T2"}', '{"topics": ["T2", 3]}',
                   '["T1"]', '{"primary_topic": "T1"}', '{"primary_topic": {"id": 5}}']
            tj = None if r < 0.08 else "{bad" if r < 0.12 else rng.choice(odd) if r < 0.25 else _paper(
                rng.choice([None, *ids]), *rng.sample(ids, rng.randint(0, 3)))
            src = None if rng.random() < 0.15 else rng.choice(ids)
            conn.execute("INSERT INTO papers VALUES (?, ?)", (oa, tj))
            conn.execute("INSERT INTO profile_candidates VALUES (?, ?, ?)", (pid, oa, src))
            cases.append((pid, oa, src, tj, tf))
    sql = f"""SELECT {candidates_repo.TOPIC_ON} FROM profile_candidates pc
              JOIN papers p USING (openalex_id) JOIN profiles pr ON pr.id = pc.profile_id
              WHERE pc.profile_id = ? AND pc.openalex_id = ?"""
    shown = 0
    for pid, oa, src, tj, tf in cases:
        want = topic_on(src, tj, tf)
        assert bool(conn.execute(sql, (pid, oa)).fetchone()[0]) is want, (src, tj, tf)
        shown += want
    assert 0 < shown < len(cases)                 # both answers were exercised


# --- every "the feed" read applies it -------------------------------------------

def test_every_read_with_the_year_rule_has_the_topic_rule_too():
    """The class: TOPIC_ON goes wherever IN_YEARS goes (each is a "the feed"
    read), except the per-topic yield, which counts what an off topic brought."""
    root = Path(__file__).resolve().parents[1] / "rag_lib"
    missing = []
    for f in root.rglob("*.py"):
        text = f.read_text()
        for m in re.finditer(r"\{(?:candidates_repo\.)?IN_YEARS\}", text):
            line = text[: m.start()].count("\n") + 1
            tail = text[m.end(): m.end() + 60]
            if "TOPIC_ON" not in tail:
                missing.append(f"{f.relative_to(root)}:{line}")
    assert missing == []
    # Wider: every SQL statement reading candidates "as the feed" (it drops
    # the seeds, NOT_SEED) has the topic rule -- the threshold list beside
    # the slider has no year rule (it tries other ranges) and was missed.
    # The per-topic yield builds its WHERE as a list and is the one exception.
    for f in root.rglob("*.py"):
        text = f.read_text()
        for m in re.finditer(r'f"""(.*?)"""', text, re.S):
            sql = m.group(1)
            # (The pool rescore reads NOT_SEED as a column and applies topic_on in Python.)
            if (re.search(r"\{(?:candidates_repo\.)?NOT_SEED\}", sql) and "TOPIC_ON" not in sql
                    and "AS not_seed" not in sql):
                missing.append(f"{f.relative_to(root)}:{text[: m.start()].count(chr(10)) + 1}")
    assert missing == []
    # The pool the feed ranks (Python) applies it too.
    assert "topic_on(" in (root / "scoring/pool.py").read_text().split("def rescore_profile_pool")[1]


def _set_sources_and_topics(db_path, *, on: dict) -> None:
    """Even papers came in through A, odd ones through B; then switch ``on``."""
    conn = connect(db_path)
    conn.execute("UPDATE profile_candidates SET sourced_by_topic_id = CASE"
                 " WHEN CAST(substr(openalex_id, 2) AS INT) % 2 = 0 THEN ? ELSE ? END", (A, B))
    tf = json.loads(conn.execute("SELECT topic_filters_json FROM profiles").fetchone()[0])
    for t in tf["topics"]:
        t["on"] = on[t["id"]]
    conn.execute("UPDATE profiles SET topic_filters_json = ?", (json.dumps(tf),))
    conn.commit()
    from rag_lib.scheduler.jobs import rescore_pool
    rescore_pool(conn, conn.execute("SELECT id FROM profiles").fetchone()[0])
    conn.close()


def _feed(app):
    r = _request(app, "GET", "/api/radar/daily")
    assert r.status_code == 200, r.text
    return {c["id"] for c in r.json()["cards"]}


def _n(i):
    return f"W{2000000 + i}"


def test_the_feed_drops_a_switched_off_topics_papers_and_takes_them_back(app, tmp_path):
    """Seeded papers: even ones have primary topic A, every fourth also B;
    odd ones have neither (primary T99999). The first 11 are on the first page."""
    db = tmp_path / "radar.db"
    _set_sources_and_topics(db, on={A: True, B: True})
    everything = _feed(app)
    assert {_n(i) for i in range(11)} <= everything

    _set_sources_and_topics(db, on={A: True, B: False})
    feed = _feed(app)
    assert all(_n(i) not in feed for i in range(1, 11, 2))         # B brought them, B is off
    assert {_n(i) for i in range(0, 11, 2)} <= feed                # A's stay

    _set_sources_and_topics(db, on={A: False, B: True})
    feed = _feed(app)
    assert {_n(i) for i in range(1, 11, 2)} <= feed                # back, without a scan
    assert {_n(0), _n(4), _n(8)} <= feed                           # A brought them, but B is theirs too
    assert all(_n(i) not in feed for i in (2, 6, 10))              # A's alone: out

    _set_sources_and_topics(db, on={A: True, B: True})
    assert _feed(app) == everything                                # nothing was deleted
    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) FROM profile_candidates").fetchone()[0] == 50
    conn.close()


def test_the_ranking_pool_and_the_threshold_counts_follow_the_feed(app, tmp_path):
    db = tmp_path / "radar.db"
    _set_sources_and_topics(db, on={A: True, B: False})
    conn = connect(db)
    pid = conn.execute("SELECT id FROM profiles").fetchone()[0]
    pct = dict(conn.execute("SELECT openalex_id, score_pct FROM profile_candidates").fetchall())
    assert all(pct[_n(i)] is None for i in range(1, 11, 2))        # out of the pool the feed ranks
    assert all(pct[_n(i)] is not None for i in range(0, 11, 2))
    assert max(v for v in pct.values() if v is not None) == pytest.approx(1.0)
    # The threshold step's histogram and counts read the same papers.
    assert len(candidates_repo.scores_for_profile(conn, pid)) == 25
    assert len(candidates_repo.scored_papers_for_profile(conn, pid)) == 25    # the list beside the slider
    n, _ = candidates_repo.count_untriaged_for_feed(conn, pid)
    assert n == 25                                                 # the even ones (none has a raw score: all pass the bar)
    # The per-topic yield still says what the off topic brought.
    assert {r["topic_id"] for r in candidates_repo.topic_yield_for_profile(conn, pid)} == {A, B}
    conn.close()


def test_saved_papers_stay_listed_with_their_topic_off(app, tmp_path):
    db = tmp_path / "radar.db"
    conn = connect(db)
    conn.execute("UPDATE profile_candidates SET saved_at = datetime('now') WHERE openalex_id = ?", (_n(1),))
    conn.commit()
    conn.close()
    _set_sources_and_topics(db, on={A: True, B: False})
    conn = connect(db)
    pid = conn.execute("SELECT id FROM profiles").fetchone()[0]
    assert _n(1) in {r["openalex_id"] for r in candidates_repo.saved_for_profile(conn, pid)}
    conn.close()
