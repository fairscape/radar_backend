"""Adding seeds from the feed in Edit interest (FUNCTIONAL_SPEC E.11).

The Seeds step of an Edit interest copy lists the interest's own feed
papers -- above its saved threshold, in its years, not a seed, not
dismissed, not a seed of the copy already -- saved ones first, then by
score, searchable by title. Adding one is the same as any other seed of
the copy: its new topics start off, nothing is rescored until the scan,
and leaving the edit leaves the paper in the feed.
"""

from __future__ import annotations

import pytest

from rag_lib.api.services import edit_copy, wizard
from rag_lib.db import connect
from rag_lib.db.repos import candidates as candidates_repo, embeddings as embeddings_repo, profiles as profiles_repo
from tests.test_edit_copy import api  # noqa: F401 -- fixture
from tests.test_researchers import _request, env  # noqa: F401 -- fixture
from tests.test_seed_removal_topics import T1, T2, T3, T4, W1, W2, W3, _paper

W = {n: f"https://openalex.org/W90{n}" for n in range(1, 10)}
T5 = "https://openalex.org/T10005"


def _candidate(conn, pid, oa_id, score, *, saved=False, dismissed=False):
    conn.execute(
        "INSERT INTO profile_candidates (profile_id, openalex_id, score, score_raw, tier_used, fetched_at,"
        " saved_at, dismissed_at) VALUES (?, ?, ?, ?, 'scan', datetime('now'), ?, ?)",
        (pid, oa_id, score, score, "2026-10-01" if saved else None, "2026-10-01" if dismissed else None),
    )


@pytest.fixture()
def feed(api):
    """``api``'s interest 'live' (threshold 0.5, years from 2010) with a feed:
    W901 0.80, W902 0.70 saved, W903 0.40 (below), W904 0.90 dismissed,
    W905 0.60 "100% dna", W906 0.95 but from 2001, W1 (a seed) as a
    candidate; W907 is in the other interest's feed only."""
    app, slug, env = api
    conn = connect(env)
    pid = int(profiles_repo.get_by_slug(conn, 1, slug)["id"])
    other = int(conn.execute("SELECT id FROM profiles WHERE name = 'Other'").fetchone()[0])
    conn.execute("UPDATE profiles SET year_from = 2010 WHERE id = ?", (pid,))
    titles = {1: "Alpha cells", 2: "Beta saved", 3: "Gamma low", 4: "Delta dismissed",
              5: "100% DNA_repair", 6: "Old paper", 7: "Other feed"}
    for n, t in titles.items():
        _paper(conn, W[n], (T4 if n != 2 else T5, "Four" if n != 2 else "Five"))
        conn.execute("UPDATE papers SET title = ?, year = ? WHERE openalex_id = ?",
                     (t, 2001 if n == 6 else 2024, W[n]))
        embeddings_repo.upsert(conn, W[n], "placeholder-v1", [0.1] * 8)
    for n, score, kw in ((1, 0.80, {}), (2, 0.70, {"saved": True}), (3, 0.40, {}),
                         (4, 0.90, {"dismissed": True}), (5, 0.60, {}), (6, 0.95, {})):
        _candidate(conn, pid, W[n], score, **kw)
    _candidate(conn, pid, W1, 0.99)
    _candidate(conn, other, W[7], 0.99)
    conn.commit()
    conn.close()
    copy = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    return app, slug, copy, env, pid


def _list(app, copy, **params):
    r = _request(app, "GET", f"/api/profiles/draft/{copy}/feed-papers", params=params)
    assert r.status_code == 200, r.text
    return [p["id"] for p in r.json()]


def test_x33_the_interests_feed_saved_first_then_by_score(feed):
    app, _, copy, *_ = feed
    r = _request(app, "GET", f"/api/profiles/draft/{copy}/feed-papers").json()
    # Not W903 (below the bar), W904 (dismissed), W906 (out of its years),
    # W1 (a seed), W907 (another interest's feed).
    assert [p["id"] for p in r] == [W[2], W[1], W[5]]
    assert [p["saved"] for p in r] == [True, False, False]
    assert r[0]["title"] == "Beta saved" and r[0]["year"] == 2024
    assert r[1]["score"] == pytest.approx(0.80)


def test_x34_search_by_title_case_insensitive_and_literal(feed):
    app, _, copy, *_ = feed
    assert _list(app, copy, q="ALPHA") == [W[1]]
    assert _list(app, copy, q="100%") == [W[5]]          # % is a character, not a wildcard
    assert _list(app, copy, q="a_r") == [W[5]]           # nor is _
    assert _list(app, copy, q="0%") == [W[5]]
    assert _list(app, copy, q="l%s") == []               # as a wildcard it would find "Alpha cells"
    assert _list(app, copy, q="l_h") == []               # and "_" "Alpha"
    assert _list(app, copy, q="nothing like it") == []
    assert _list(app, copy, limit=1) == [W[2]]


def test_x35_x37_x38_added_to_the_copy_only_then_discarded(feed):
    app, slug, copy, env, pid = feed
    r = _request(app, "POST", f"/api/profiles/{copy}/seeds",
                 json={"openalex_ids": [W[1], W[2]], "source": "feed"})
    assert r.status_code == 200 and r.json()["attached"] == 2, r.text
    seeds = {s["id"]: s for s in _request(app, "GET", f"/api/profiles/{copy}/detail").json()["seeds"]}
    assert seeds[W[1]]["source"] == "feed" and seeds[W[2]]["source"] == "feed"
    # Not offered again; the interest itself unchanged.
    assert _list(app, copy) == [W[5]]
    assert {s["id"] for s in _request(app, "GET", f"/api/profiles/{slug}/detail").json()["seeds"]} == {W1, W2}
    # X-36: the topic W902 brings is new and off.
    conn = connect(env)
    try:
        cslug_row = profiles_repo.get_by_slug(conn, 1, copy)
        filters, new = wizard.preview_topics(conn, user_id=1, slug=copy)
        five = [t for t in filters["topics"] if t["id"] == T5][0]
        assert T5 in new and five["on"] is False
        # X-37: the seeds differ from the last scan; nothing rescored.
        assert wizard.scan_outdated(conn, user_id=1, slug=copy)["seeds"] is True
        scores = dict(conn.execute("SELECT openalex_id, score FROM profile_candidates WHERE profile_id = ?",
                                   (pid,)).fetchall())
        assert scores[W[1]] == pytest.approx(0.80) and scores[W[2]] == pytest.approx(0.70)
        assert cslug_row is not None
    finally:
        conn.close()
    # X-38: discarded -- the interest's seeds as they were, the papers still in its feed.
    assert _request(app, "DELETE", f"/api/profiles/draft/{copy}").status_code == 200
    assert {s["id"] for s in _request(app, "GET", f"/api/profiles/{slug}/detail").json()["seeds"]} == {W1, W2}
    copy2 = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    assert _list(app, copy2) == [W[2], W[1], W[5]]


def test_x39_saved_they_are_seeds_and_out_of_the_feed(feed):
    app, slug, copy, env, pid = feed
    _request(app, "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [W[1]], "source": "feed"})
    assert _request(app, "POST", f"/api/profiles/draft/{copy}/save-edit").status_code == 200
    seeds = {s["id"]: s for s in _request(app, "GET", f"/api/profiles/{slug}/detail").json()["seeds"]}
    assert W[1] in seeds and seeds[W[1]]["source"] == "feed"
    conn = connect(env)
    try:
        feed_ids = [r["openalex_id"] for r in candidates_repo.top_for_profile(conn, pid, limit=50)]
    finally:
        conn.close()
    assert W[1] not in feed_ids and W[2] in feed_ids


def test_x40_only_the_users_own_open_copy(feed):
    app, slug, copy, *_ = feed
    other = {"X-User-Email": "other@example.com"}
    assert _request(app, "GET", f"/api/profiles/draft/{copy}/feed-papers", headers=other).status_code == 404
    # The saved interest itself, or a new interest's draft, is not a copy.
    assert _request(app, "GET", f"/api/profiles/draft/{slug}/feed-papers").status_code == 404
    draft = _request(app, "POST", "/api/profiles/draft", json={"name": "New one"})
    assert draft.status_code == 200, draft.text
    assert _request(app, "GET", f"/api/profiles/draft/{draft.json()['slug']}/feed-papers").status_code == 404


def test_x40_another_users_feed_paper_is_rejected(feed):
    app, slug, copy, env, _ = feed
    # The other user's interest has a feed paper the demo user does not own.
    _request(app, "GET", "/api/profiles", headers={"X-User-Email": "other@example.com"})
    conn = connect(env)
    try:
        ouid = int(conn.execute("SELECT id FROM users WHERE email = 'other@example.com'").fetchone()[0])
        opid = profiles_repo.upsert(conn, user_id=ouid, name="Theirs", embedding_model="placeholder-v1",
                                    n_seed=1, topic_filters={"topics": []}, threshold=0.5)
        _paper(conn, W[8], (T4, "Four"))
        _candidate(conn, opid, W[8], 0.9)
        conn.commit()
    finally:
        conn.close()
    r = _request(app, "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [W[8]], "source": "feed"})
    assert r.status_code == 200 and r.json()["attached"] == 0 and r.json()["rejected"] == [W[8]], r.text


def test_a_seed_removed_in_the_copy_is_still_the_interests_and_not_offered(feed):
    """W2 is a seed of the interest (so not in its feed) even once the copy drops it."""
    app, _, copy, env, pid = feed
    conn = connect(env)
    try:
        _candidate(conn, pid, W2, 0.85)
        conn.commit()
    finally:
        conn.close()
    assert _request(app, "DELETE", f"/api/profiles/{copy}/seeds", params={"openalex_id": W2}).status_code == 200
    assert W2 not in _list(app, copy)


def test_feed_as_the_source_takes_only_what_the_list_offers(feed):
    """Review E.11: "Added from the feed" is only true of the listed papers.
    Below the bar, dismissed, out of the years, another interest's feed, or
    a Vault paper are refused under that source; a seed of the copy again
    is the usual no-op, not a refusal."""
    app, _, copy, *_ = feed
    for oa in (W[3], W[4], W[6], W[7], W3):
        r = _request(app, "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [oa], "source": "feed"})
        assert r.status_code == 200 and r.json()["rejected"] == [oa], (oa, r.text)
    r = _request(app, "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [W[1], W1], "source": "feed"})
    assert r.json() == {"attached": 1, "rejected": [], "rescored": None}
    r = _request(app, "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [W[1]], "source": "feed"})
    assert r.json()["attached"] == 0 and r.json()["rejected"] == []
    # W3 the user owns: as a Vault pick it is taken.
    r = _request(app, "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [W3], "source": "vault"})
    assert r.json()["attached"] == 1


def test_search_ignores_case_beyond_ascii(feed):
    app, _, copy, env, pid = feed
    conn = connect(env)
    try:
        _paper(conn, W[9], (T4, "Four"))
        conn.execute("UPDATE papers SET title = 'Über ΔΒ-Zellen', year = 2024 WHERE openalex_id = ?", (W[9],))
        _candidate(conn, pid, W[9], 0.65)
        conn.commit()
    finally:
        conn.close()
    assert _list(app, copy, q="über") == [W[9]]
    assert _list(app, copy, q="ÜBER δβ") == [W[9]]


def test_the_copys_threshold_does_not_change_the_list(feed):
    """Judged as the interest is saved, not as the copy is being changed."""
    app, _, copy, *_ = feed
    assert _request(app, "PATCH", f"/api/profiles/{copy}", json={"threshold": 0.75}).status_code == 200
    assert _list(app, copy) == [W[2], W[1], W[5]]


def test_the_service_refuses_a_slug_that_is_not_a_copy(feed):
    _, slug, _, env, _ = feed
    conn = connect(env)
    try:
        with pytest.raises(LookupError):
            edit_copy.feed_seed_choices(conn, user_id=1, slug=slug)
    finally:
        conn.close()


_ = (T1, T2, T3, W2, W3)
