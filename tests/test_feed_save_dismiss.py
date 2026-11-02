"""Feed Save and Dismiss (FUNCTIONAL_SPEC 5, F-1 to F-9).

Save puts a paper on the interest's own Saved list and in the Vault; it
changes no seed, fit, score or threshold. Un-saving takes it off the
Saved list only: it stays in the Vault (only the Vault's delete takes it
out), stays a seed if it is one, and goes back to the feed after every
other paper -- if it still passes the bar and the years (5.5, Q4 (b)).
"""

from __future__ import annotations

import importlib

import pytest

from rag_lib.db import connect
from tests.edit_helpers import edit_saved
from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    B, _feed, _go_live, _ok, _one, _q, _request, _scan, app, env,
)


def _live(app, env, name="Triage", **kw):
    slug, _ = _go_live(app, env, name=name, threshold=0.0, **kw)
    _scan(app, slug)
    return slug


def _save(app, slug, cid, on=True, headers=None):
    return _request(app, "POST", "/api/radar/cards/save", json={"card_id": cid, "profile": slug, "on": on},
                    headers=headers or {})


def _dismiss(app, slug, cid, on=True):
    return _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": cid, "profile": slug, "on": on}))


def _saved(app, slug, headers=None):
    return _ok(_request(app, "GET", f"/api/profiles/{slug}/saved", headers=headers or {}))


def _vault(app, slug):
    return {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs", params={"tag": slug}))}


def _fresh(app, slug, **params):
    """The feed's papers still to look at, in the feed's order."""
    return [c["id"] for c in _feed(app, slug, **params)["cards"] if c.get("state") is None]


def _exec(env, sql, *params):
    conn = connect(env)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _pid(env, slug):
    return _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]


def _fit(env, slug):
    """What a save must not move: seeds, the fit, the threshold, the scores."""
    pid = _pid(env, slug)
    prof = _one(env, "SELECT threshold, selector_config_json, centroid, seed_sim_min FROM profiles WHERE id = ?", pid)
    seeds = sorted(r["openalex_id"] for r in _q(env, "SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", pid))
    scores = {r["openalex_id"]: (r["score_raw"], r["score"])
              for r in _q(env, "SELECT openalex_id, score_raw, score FROM profile_candidates WHERE profile_id = ?", pid)}
    return dict(prof), seeds, scores


# --- F-1 / Q1: save -----------------------------------------------------------------------


def test_f1_save_leaves_the_feed_and_is_on_saved_and_in_the_vault(app, env):
    slug = _live(app, env)
    cid = _fresh(app, slug)[0]
    assert _ok(_save(app, slug, cid))["state"] == "saved"
    assert cid not in _fresh(app, slug)
    assert [p["id"] for p in _saved(app, slug)] == [cid]
    assert cid in _vault(app, slug)


def test_saved_lists_last_saved_first_whatever_the_bar_and_years_say(app, env):
    slug = _live(app, env)
    a, b = _fresh(app, slug)[:2]
    _ok(_save(app, slug, a))
    # Saved a second later than the first, whatever the clock's resolution.
    _exec(env, "UPDATE profile_candidates SET saved_at = datetime('now', '-1 minute') WHERE openalex_id = ?", a)
    _ok(_save(app, slug, b))
    assert [p["id"] for p in _saved(app, slug)] == [b, a]
    # Q1: a bar and years that keep both out of the feed keep them on Saved.
    _exec(env, "UPDATE profiles SET threshold = 0.999, year_from = 1900, year_to = 1901 WHERE id = ?", _pid(env, slug))
    assert [p["id"] for p in _saved(app, slug)] == [b, a]


# --- F-2: a save changes nothing a scan reads ---------------------------------------------------


def test_f2_a_save_then_a_scan_changes_no_seed_fit_score_or_threshold(app, env):
    slug = _live(app, env)
    _scan(app, slug)
    before = _fit(env, slug)
    cid = _fresh(app, slug)[0]
    _ok(_save(app, slug, cid))
    _scan(app, slug)
    assert _fit(env, slug) == before
    assert [p["id"] for p in _saved(app, slug)] == [cid]


# --- F-3 / F-9: un-save -------------------------------------------------------------------------


@pytest.mark.parametrize("on", [False, None], ids=["saved-tab", "feed-toggle"])
def test_f3_f9_unsave_leaves_saved_only_and_goes_to_the_end_of_the_feed(app, env, on):
    slug = _live(app, env)
    fresh = _fresh(app, slug)
    assert len(fresh) >= 3
    top = fresh[0]
    _ok(_save(app, slug, top))
    assert _ok(_save(app, slug, top, on=on))["state"] is None
    assert _saved(app, slug) == []
    assert top in _vault(app, slug)                  # only the Vault's delete takes it out
    after = _fresh(app, slug)
    assert after[-1] == top and sorted(after) == sorted(fresh)
    # Last across the whole feed too, saved and dismissed cards included.
    assert _feed(app, slug)["cards"][-1]["id"] == top


def test_f3_unsave_keeps_a_seed_a_seed(app, env):
    slug = _live(app, env)
    cid = _fresh(app, slug)[0]
    _ok(_save(app, slug, cid))
    with edit_saved(lambda m, p, **k: _request(app, m, p, **k), slug) as copy:
        _ok(_request(app, "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [cid], "source": "feed"}))
    assert [(p["id"], p["is_seed"]) for p in _saved(app, slug)] == [(cid, True)]      # Q2
    _ok(_save(app, slug, cid, on=False))
    assert _saved(app, slug) == []
    assert _one(env, "SELECT COUNT(*) n FROM profile_seeds WHERE profile_id = ? AND openalex_id = ?",
                _pid(env, slug), cid)["n"] == 1
    assert cid in _vault(app, slug)


def test_q4_an_unsaved_paper_under_the_bar_is_not_in_the_feed(app, env):
    slug = _live(app, env)
    cid = _fresh(app, slug)[0]
    _ok(_save(app, slug, cid))
    _ok(_save(app, slug, cid, on=False))
    _exec(env, "UPDATE profile_candidates SET score_raw = -1, score = -1 WHERE openalex_id = ?", cid)
    assert cid not in _fresh(app, slug)
    assert cid in _fresh(app, slug, bucket="low")    # below the bar, as any paper
    assert cid in _vault(app, slug)


def test_saving_again_lifts_the_end_of_the_feed(app, env):
    slug = _live(app, env)
    fresh = _fresh(app, slug)
    top = fresh[0]
    _ok(_save(app, slug, top))
    _ok(_save(app, slug, top, on=False))
    _ok(_save(app, slug, top))
    _ok(_save(app, slug, top, on=False))
    assert _fresh(app, slug)[-1] == top              # un-saved again: last again
    _ok(_save(app, slug, top))
    assert _one(env, "SELECT unsaved_at FROM profile_candidates WHERE openalex_id = ?", top)["unsaved_at"] is None
    assert [p["id"] for p in _saved(app, slug)] == [top]


def test_the_end_of_the_feed_is_after_every_interest(app, env):
    one = _live(app, env, name="One")
    two = _live(app, env, name="Two", seeds_from_researcher=False)
    cid = _fresh(app, one)[0]
    _ok(_save(app, one, cid))
    _ok(_save(app, one, cid, on=False))
    cards = [c for c in _feed(app)["cards"] if c.get("state") is None]
    assert {c["profile"] for c in cards} == {one, two}
    assert (cards[-1]["id"], cards[-1]["profile"]) == (cid, one)


# --- F-4 / Q5: dismiss ---------------------------------------------------------------------------


def test_f4_dismissed_stays_out_of_the_feed_saved_and_add_from_feed(app, env):
    slug = _live(app, env)
    cid = _fresh(app, slug)[0]
    _dismiss(app, slug, cid)
    _scan(app, slug)
    assert cid not in _fresh(app, slug)
    assert _saved(app, slug) == []
    with edit_saved(lambda m, p, **k: _request(app, m, p, **k), slug, save=False) as copy:
        listed = [p["id"] for p in _ok(_request(app, "GET", f"/api/profiles/draft/{copy}/feed-papers"))]
    assert cid not in listed


def test_q5_dismissing_a_saved_paper_keeps_it_in_the_vault(app, env):
    slug = _live(app, env)
    cid = _fresh(app, slug)[0]
    _ok(_save(app, slug, cid))
    assert _dismiss(app, slug, cid)["state"] == "dismissed"
    assert _saved(app, slug) == []
    assert cid not in _fresh(app, slug)
    assert cid in _vault(app, slug)


# --- F-5 / Q6: Add from feed ---------------------------------------------------------------------


def test_f5_add_from_feed_saved_first_unsaved_as_any_and_a_seed_leaves_the_feed(app, env):
    slug = _live(app, env)
    fresh = _fresh(app, slug)
    low, top = fresh[-1], fresh[0]
    _ok(_save(app, slug, low))
    _ok(_save(app, slug, top))
    _ok(_save(app, slug, top, on=False))             # Q6: un-saved, listed as any paper
    with edit_saved(lambda m, p, **k: _request(app, m, p, **k), slug) as copy:
        listed = _ok(_request(app, "GET", f"/api/profiles/draft/{copy}/feed-papers"))
        assert listed[0]["id"] == low and listed[0]["saved"] is True
        assert listed[1]["id"] == top and listed[1]["saved"] is False
        _ok(_request(app, "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [low], "source": "feed"}))
    assert low not in [c["id"] for c in _feed(app, slug)["cards"]]


# --- F-6: the threshold from feedback is gone ---------------------------------------------------


def test_f6_no_threshold_from_feedback(app, env):
    slug = _live(app, env)
    assert _request(app, "GET", f"/api/profiles/{slug}/feedback-threshold").status_code == 404
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rag_lib.feedback.autotune")
    # Thirty saves and dismisses, as the suggestion once asked for: the bar stays.
    thr = _one(env, "SELECT threshold FROM profiles WHERE slug = ?", slug)["threshold"]
    for i, cid in enumerate(_fresh(app, slug)):
        _ok(_save(app, slug, cid)) if i % 2 else _dismiss(app, slug, cid)
    with edit_saved(lambda m, p, **k: _request(app, m, p, **k), slug) as copy:
        r = _request(app, "POST", f"/api/profiles/{copy}/refit?autotune=1")
        assert r.status_code in (200, 400), r.text
    assert _one(env, "SELECT threshold FROM profiles WHERE slug = ?", slug)["threshold"] == thr


# --- F-7: another user ---------------------------------------------------------------------------


def test_f7_another_user_cannot_read_or_unsave_the_saved_list(app, env):
    slug = _live(app, env)
    cid = _fresh(app, slug)[0]
    _ok(_save(app, slug, cid))
    assert _request(app, "GET", f"/api/profiles/{slug}/saved", headers=B).status_code == 404
    assert _save(app, slug, cid, on=False, headers=B).status_code == 404
    assert [p["id"] for p in _saved(app, slug)] == [cid]


# --- F-8: a save while the interest is being edited ---------------------------------------------


def test_f8_a_save_during_edit_is_on_the_interest_and_survives_the_save(app, env):
    slug = _live(app, env)
    cid = _fresh(app, slug)[0]
    with edit_saved(lambda m, p, **k: _request(app, m, p, **k), slug) as copy:
        _ok(_save(app, slug, cid))
        assert _saved(app, copy) == []               # the copy has no papers of its own
    assert [p["id"] for p in _saved(app, slug)] == [cid]
    assert cid in _vault(app, slug)


def test_the_query_itself_puts_unsaved_last_so_the_feed_limit_cuts_it_first(app, env):
    """The feed reads each interest up to a limit: an un-saved paper sorted
    last only after the read would still take a top paper's place."""
    from rag_lib.db.repos import candidates as candidates_repo

    slug = _live(app, env)
    top = _fresh(app, slug)[0]
    _ok(_save(app, slug, top))
    _ok(_save(app, slug, top, on=False))
    conn = connect(env)
    try:
        pid = _pid(env, slug)
        n = len(candidates_repo.top_for_profile(conn, pid, limit=500, triaged=False))
        cut = candidates_repo.top_for_profile(conn, pid, limit=n - 1, triaged=False)
    finally:
        conn.close()
    assert top not in [r["openalex_id"] for r in cut]
