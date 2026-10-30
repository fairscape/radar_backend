"""round 13 review (b): evidence for the findings in FINDINGS.md.

Each test states the behaviour as it IS (so the suite passes on the code
under review); the asserts marked ``# finding`` are the ones that document
a bug.
"""

from __future__ import annotations

import sqlite3

import pytest

from rag_lib.db import connect
from rag_lib.db.repos import profiles as profiles_repo
from rag_lib.db.repos import users as users_repo
from tests.test_review_round13_b_probe import A, SLUG, _edit, _ok, _request, app, env  # noqa: F401


def test_b1_profiles_upsert_keeps_the_slug_of_an_existing_interest(env):
    """rag_lib/db/repos/profiles.py upsert(): the UPDATE branch used to write
    ``slug = slugify(name)`` over whatever slug the row already had.

    The wizard suffixes a slug on collision ("my-topic-2"), so an interest
    whose name slugifies like another's cannot be re-stored through the CLI
    (``cli.ingest --db`` / ``cli.gather --db`` -> store_profile_from_object):
    IntegrityError on idx_profiles_user_slug. When there is no collision the
    same line silently renames the interest's URL key.
    """
    conn = connect(env)
    uid = int(users_repo.get(conn, 1)["id"])
    p1 = profiles_repo.upsert(conn, user_id=uid, name="My Topic", embedding_model="placeholder-v1", n_seed=0)
    p2 = profiles_repo.upsert(conn, user_id=uid, name="my topic", embedding_model="placeholder-v1", n_seed=0)
    slugs = {r[0]: r[1] for r in conn.execute("SELECT id, slug FROM profiles WHERE id IN (?, ?)", (p1, p2))}
    assert slugs == {p1: "my-topic", p2: "my-topic-2"}
    assert profiles_repo.upsert(conn, user_id=uid, name="my topic", embedding_model="placeholder-v1", n_seed=1) == p2
    assert tuple(conn.execute("SELECT slug, n_seed FROM profiles WHERE id = ?", (p2,)).fetchone()) == ("my-topic-2", 1)
    # No collision either: the key stays what bookmarks and open tabs hold.
    conn.execute("DELETE FROM profiles WHERE id = ?", (p1,))
    conn.commit()
    profiles_repo.upsert(conn, user_id=uid, name="my topic", embedding_model="placeholder-v1", n_seed=1)
    assert conn.execute("SELECT slug FROM profiles WHERE id = ?", (p2,)).fetchone()[0] == "my-topic-2"
    # A caller naming a slug still gets it.
    profiles_repo.upsert(conn, user_id=uid, name="my topic", embedding_model="placeholder-v1", n_seed=1, slug="renamed")
    assert conn.execute("SELECT slug FROM profiles WHERE id = ?", (p2,)).fetchone()[0] == "renamed"
    conn.close()


def test_b2_feedback_log_records_the_unsave(app, env):
    """Un-saving used to leave the SAVED event alone in the Activity log while
    ``saves30`` and the threshold suggestion read the row's current state;
    the undo is an event of its own now."""
    daily = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))
    cid = daily["cards"][0]["id"]
    _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": cid, "profile": SLUG, "on": True}, headers=A))
    _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": cid, "profile": SLUG, "on": False}, headers=A))
    det = _ok(_request(app, "GET", f"/api/profiles/{SLUG}/detail", headers=A))
    assert det["profile"]["saves30"] == 0
    assert len(det["feedbackLog"]) == 2
    assert "UNSAVED" in det["feedbackLog"][0] and "SAVED" in det["feedbackLog"][1]
    ft = _ok(_request(app, "GET", f"/api/profiles/{SLUG}/feedback-threshold", headers=A))
    assert ft["n_saved"] == 0
    # The same for dismiss and its undo; a repeated undo logs nothing.
    _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": cid, "profile": SLUG, "on": True}, headers=A))
    _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": cid, "profile": SLUG, "on": False}, headers=A))
    _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": cid, "profile": SLUG, "on": False}, headers=A))
    det = _ok(_request(app, "GET", f"/api/profiles/{SLUG}/detail", headers=A))
    assert [l.split()[1] for l in det["feedbackLog"]] == ["UNDISMISSED", "DISMISSED", "UNSAVED", "SAVED"]


def test_b3_reranker_comparison_ranks_the_whole_pool(app, env):
    """The comparison's rank_after=1 is the feed's first card even when the
    pool holds rows without a reranker logit (blended at the median)."""
    conn = connect(env)
    ids = [r[0] for r in conn.execute(
        "SELECT openalex_id FROM profile_candidates WHERE profile_id = 1 ORDER BY score DESC").fetchall()]
    for i, oa in enumerate(ids):
        conn.execute("UPDATE profile_candidates SET score_raw = ? WHERE openalex_id = ?", (0.95 - i * 0.01, oa))
        if i % 2 == 0:
            conn.execute("UPDATE profile_candidates SET score_reranker_raw = ? WHERE openalex_id = ?", (float(i), oa))
    conn.commit()
    from rag_lib.scoring.pool import rescore_profile_pool
    rescore_profile_pool(conn, 1, alpha=0.4, beta=0.6)
    conn.close()
    cmp_ = _ok(_request(app, "GET", f"/api/profiles/{SLUG}/reranker-comparison?limit=5", headers=A))
    feed = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))
    # ``n`` is what the feed ranks: every row over the threshold, reranked
    # or not -- not the whole pool and not only the reranked rows.
    assert 0 < cmp_["n"] < len(ids)
    assert cmp_["n"] == len(feed["cards"])
    assert cmp_["candidates"][0]["rank_after"] == 1
    assert cmp_["candidates"][0]["openalex_id"] == feed["cards"][0]["id"]


def test_b4_checked_ok_feed_years_vault_users(app, env):
    """Behaviours verified as correct (see FINDINGS.md 'checked and found OK')."""
    # Cross-user scoping of feed / triage / vault / seeds.
    B = {"X-User-Email": "other@example.com"}
    daily = _ok(_request(app, "GET", "/api/radar/daily", headers=A))
    cid = daily["cards"][0]["id"]
    assert _ok(_request(app, "GET", "/api/radar/daily", headers=B))["cards"] == []
    assert _request(app, "POST", "/api/radar/cards/save", json={"card_id": cid, "profile": SLUG}, headers=B).status_code == 404
    assert _request(app, "POST", f"/api/profiles/{SLUG}/seeds", json={"openalex_ids": [cid]}, headers=B).status_code == 404
    assert _ok(_request(app, "GET", "/api/vault/docs?tag=" + SLUG, headers=B)) == []
    # B cannot open A's interest in Edit interest either.
    assert _request(app, "POST", f"/api/profiles/{SLUG}/edit", headers=B).status_code == 404
    # Years (in Edit interest, saved -- spec E): validation, to-only, feed +
    # threshold list restricted, clear.
    assert _request(app, "PATCH", f"/api/profiles/{SLUG}", json={"years": {"year_from": 2020}}, headers=A).status_code == 409
    with _edit(app) as copy:
        assert _request(app, "PATCH", f"/api/profiles/{copy}", json={"years": {"year_from": 2030}}, headers=A).status_code == 400
        assert _request(app, "PATCH", f"/api/profiles/{copy}", json={"years": {"year_from": 2020, "year_to": 2010}}, headers=A).status_code == 400
        _ok(_request(app, "PATCH", f"/api/profiles/{copy}", json={"years": {"year_from": None, "year_to": 2020}}, headers=A))
    assert _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))["cards"] == []
    # The histogram follows the saved years as the feed does; the paper list
    # beside the slider keeps every stored year, since the Threshold tab
    # filters it by the range being tried out (which may be wider).
    dr = _ok(_request(app, "POST", f"/api/profiles/{SLUG}/dry-run", headers=A))
    assert dr["scores"] == [] and len(dr["papers"]) == 50
    with _edit(app) as copy:
        _ok(_request(app, "PATCH", f"/api/profiles/{copy}", json={"years": {"year_from": None, "year_to": None}}, headers=A))
    assert len(_ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))["cards"]) == 50
    # Backfill: validation, run accounting, out-of-range rows get no percentile.
    assert _request(app, "POST", f"/api/profiles/{SLUG}/backfill", json={"year_from": None}, headers=A).status_code == 400
    run = _ok(_request(app, "POST", f"/api/profiles/{SLUG}/backfill", json={"year_from": 2000, "year_to": 2003}, headers=A))
    runs = _ok(_request(app, "GET", f"/api/profiles/{SLUG}/runs", headers=A))
    assert runs[0]["id"] == run["run_id"] and runs[0]["n_fetched"] == 10 and runs[0]["since_date"] == "2000-01-01"
    with _edit(app) as copy:
        _ok(_request(app, "PATCH", f"/api/profiles/{copy}", json={"years": {"year_from": 2001, "year_to": 2002}}, headers=A))
    conn = connect(env)
    outside = conn.execute(
        "SELECT count(*) FROM profile_candidates pc JOIN papers p USING (openalex_id)"
        " WHERE pc.profile_id = 1 AND p.year NOT BETWEEN 2001 AND 2002 AND pc.score_pct IS NOT NULL").fetchone()[0]
    conn.close()
    assert outside == 0
    # Users.
    me = _ok(_request(app, "PATCH", "/api/users/me", json={"mailto": " x@y.org "}, headers=A))
    assert me["mailto"] == "x@y.org"
    assert _request(app, "PATCH", "/api/users/me", json={"mailto": "nope"}, headers=A).status_code == 400
    assert _request(app, "GET", "/api/users/me", headers={"X-User-Email": "not an email"}).status_code == 400
