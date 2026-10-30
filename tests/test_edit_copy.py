"""Editing a saved interest on a copy (FUNCTIONAL_SPEC E).

Edit interest copies the interest; every step changes the copy; Save
changes writes it back in one go and deletes it; leaving without saving
discards it with what it brought in that nothing else uses. Saving does
not scan, refit or rescore (except a new threshold or year range, which
re-decides the feed); the scan refits. While a copy is open the interest
cannot be changed or scanned anywhere else.
"""

from __future__ import annotations

import json

import pytest

from rag_lib.api.services import edit_copy, wizard
from rag_lib.db.repos import (
    draft_owned as draft_owned_repo,
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
)
from rag_lib.openalex_tiers import enabled_topic_ids
from tests.test_seed_removal_topics import (  # noqa: F401 -- fixture
    T1, T2, T3, T4, T9, W1, W2, W3, _by_id, _paper, _stored, live,
)


@pytest.fixture()
def edit(live, monkeypatch):
    """``live`` with vectors for its seeds, and the refit counted (it must
    not run on a copy, nor on save)."""
    conn, uid, pid, slug = live
    _paper(conn, W3, (T4, "Four"))                       # in the Vault, not a seed
    for w in (W1, W2, W3):
        embeddings_repo.upsert(conn, w, "placeholder-v1", [0.1] * 8)
    conn.commit()
    monkeypatch.setattr(wizard, "embed_missing", lambda *a, **k: None)
    refits: list = []
    monkeypatch.setattr(wizard, "refit_profile", lambda conn, **kw: refits.append(kw))
    rescored: list = []
    from rag_lib.scheduler import jobs
    monkeypatch.setattr(jobs, "rescore_pool", lambda conn, pid: rescored.append(pid))
    return conn, uid, pid, slug, refits, rescored


def _row(conn, pid):
    return conn.execute("SELECT * FROM profiles WHERE id = ?", (pid,)).fetchone()


def _seeds(conn, pid):
    return sorted(profiles_repo.list_seed_openalex_ids(conn, pid))


def _snapshot(conn, pid):
    r = _row(conn, pid)
    return {k: r[k] for k in r.keys() if k not in ("updated_at",)}, _seeds(conn, pid)


def _open(conn, uid, slug):
    c = edit_copy.open_copy(conn, user_id=uid, slug=slug)
    return int(c["id"]), c["slug"]


def _scan(conn, pid, *, topics, seeds):
    uid = int(conn.execute("SELECT user_id FROM profiles WHERE id = ?", (pid,)).fetchone()[0])
    run = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, since_date=None, filter_string=None, tier_used="scan")
    gather_runs_repo.finish(conn, run, n_fetched=0, n_new=0, n_redup=0, api_calls=0,
                            topic_ids=topics, seed_ids=seeds)
    conn.commit()


# --- opening -------------------------------------------------------------------


def test_e1_the_copy_is_the_interest_unlisted(edit):
    conn, uid, pid, slug, *_ = edit
    cid, cslug = _open(conn, uid, slug)
    copy, orig = _row(conn, cid), _row(conn, pid)
    assert copy["is_draft"] == 1 and copy["edit_of"] == pid and cslug != slug
    for k in edit_copy._SAVED:
        assert copy[k] == orig[k], k
    assert _seeds(conn, cid) == _seeds(conn, pid)
    # Not listed anywhere drafts or interests are.
    assert cid not in [int(r["id"]) for r in profiles_repo.list_for_user(conn, uid, include_drafts=True)]
    assert cid not in [int(r["id"]) for r in profiles_repo.list_drafts_for_user(conn, uid)]


def test_opening_again_discards_the_copy_left_behind(edit):
    conn, uid, pid, slug, *_ = edit
    first, _ = _open(conn, uid, slug)
    second, _ = _open(conn, uid, slug)
    assert _row(conn, first) is None and _row(conn, second) is not None
    assert edit_copy.copy_of(conn, pid)["id"] == second


def test_a_draft_cannot_be_opened_as_a_copy(edit):
    conn, uid, pid, slug, *_ = edit
    _, cslug = _open(conn, uid, slug)
    with pytest.raises(LookupError):
        edit_copy.open_copy(conn, user_id=uid, slug=cslug)


# --- removing a seed on the copy ---------------------------------------------------


def test_e2_removing_a_seed_changes_only_the_copy_and_leaves_its_topics_on(edit):
    """E.2-3 as revised: the topics stay as they were; only marked."""
    conn, uid, pid, slug, refits, _ = edit
    before = _snapshot(conn, pid)
    cid, cslug = _open(conn, uid, slug)
    topics_before = _stored(conn, cid)
    assert wizard.remove_seed(conn, user_id=uid, slug=cslug, openalex_id=W2)
    assert _seeds(conn, cid) == [W1]
    assert _stored(conn, cid) == topics_before        # no switch moved
    prev, _ = wizard.preview_topics(conn, user_id=uid, slug=cslug)
    got = _by_id(prev)
    assert got[T3]["on"] is True and got[T3].get("unmatched") is True   # only W2 brought it
    assert not got[T1].get("unmatched") and not got[T2].get("unmatched")
    assert not got[T9].get("unmatched")              # Claude's: not the seeds'
    assert refits == []                               # a copy is not refitted
    assert _snapshot(conn, pid) == before             # the interest is untouched


def test_a_marked_topic_stays_as_the_user_sets_it_and_re_adding_the_seed_clears_the_mark(edit):
    conn, uid, pid, slug, *_ = edit
    cid, cslug = _open(conn, uid, slug)
    wizard.remove_seed(conn, user_id=uid, slug=cslug, openalex_id=W2)
    wizard.save_topics(conn, user_id=uid, slug=cslug, selected_topic_ids=[T1, T2, T9])
    prev, _ = wizard.preview_topics(conn, user_id=uid, slug=cslug)
    assert _by_id(prev)[T3]["on"] is False and _by_id(prev)[T3].get("unmatched") is True
    profiles_repo.attach_seed(conn, cid, W2)
    conn.commit()
    prev, _ = wizard.preview_topics(conn, user_id=uid, slug=cslug)
    assert _by_id(prev)[T3]["on"] is False            # the user's switch, kept
    assert not _by_id(prev)[T3].get("unmatched")      # a seed brings it again
    wizard.save_topics(conn, user_id=uid, slug=cslug, selected_topic_ids=[T1, T2, T3, T9])
    assert _by_id(_stored(conn, cid))[T3]["on"] is True


def test_a_topic_another_seed_still_brings_stays_on(edit):
    conn, uid, pid, slug, *_ = edit
    _paper(conn, W3, (T3, "Three"))
    profiles_repo.attach_seed(conn, pid, W3)
    conn.commit()
    cid, cslug = _open(conn, uid, slug)
    wizard.remove_seed(conn, user_id=uid, slug=cslug, openalex_id=W2)
    assert _by_id(_stored(conn, cid))[T3]["on"] is True


# --- saving ---------------------------------------------------------------------


def test_e3_save_writes_the_copy_back_without_scan_refit_or_rescore(edit):
    conn, uid, pid, slug, refits, rescored = edit
    centroid = _row(conn, pid)["centroid"]
    _scan(conn, pid, topics=sorted(enabled_topic_ids(_stored(conn, pid))), seeds=_seeds(conn, pid))
    n_runs = conn.execute("SELECT COUNT(*) FROM gather_runs").fetchone()[0]
    cid, cslug = _open(conn, uid, slug)
    wizard.remove_seed(conn, user_id=uid, slug=cslug, openalex_id=W2)
    copy_topics = _stored(conn, cid)
    assert edit_copy.save_copy(conn, user_id=uid, slug=cslug) == slug
    assert _row(conn, cid) is None                                     # the copy goes
    assert _seeds(conn, pid) == [W1]
    assert _stored(conn, pid) == copy_topics
    assert _row(conn, pid)["n_seed"] == 1
    assert refits == [] and rescored == []                             # E.4 Q5
    assert _row(conn, pid)["centroid"] == centroid                     # the fit waits for the scan
    assert conn.execute("SELECT COUNT(*) FROM gather_runs").fetchone()[0] == n_runs
    # ... and step 4 says the recommendations no longer match the seeds;
    # the topics searched are as they were (E.8: removing a seed moves no switch).
    assert wizard.outdated(conn, pid) == {"topics": False, "seeds": True}


def test_save_rescored_only_when_threshold_or_years_changed(edit):
    conn, uid, pid, slug, _, rescored = edit
    _, cslug = _open(conn, uid, slug)
    edit_copy.save_copy(conn, user_id=uid, slug=cslug)
    assert rescored == []
    cid, cslug = _open(conn, uid, slug)
    profiles_repo.update_threshold(conn, cid, 0.7)
    edit_copy.save_copy(conn, user_id=uid, slug=cslug)
    assert rescored == [pid] and _row(conn, pid)["threshold"] == pytest.approx(0.7)
    cid, cslug = _open(conn, uid, slug)
    profiles_repo.update_year_range(conn, cid, 2020, None)
    edit_copy.save_copy(conn, user_id=uid, slug=cslug)
    assert rescored == [pid, pid] and _row(conn, pid)["year_from"] == 2020


def test_save_refused_without_a_topic_on_or_a_placeable_seed(edit):
    conn, uid, pid, slug, *_ = edit
    before = _snapshot(conn, pid)
    cid, cslug = _open(conn, uid, slug)
    f = _stored(conn, cid)
    for t in f["topics"]:
        t["on"] = False
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(f), cid))
    conn.commit()
    with pytest.raises(wizard.NoTopicOn):
        edit_copy.save_copy(conn, user_id=uid, slug=cslug)
    conn.execute("DELETE FROM paper_embeddings WHERE openalex_id IN (?, ?)", (W1, W2))
    conn.commit()
    with pytest.raises(ValueError):
        edit_copy.save_copy(conn, user_id=uid, slug=cslug)
    assert _snapshot(conn, pid) == before and _row(conn, cid) is not None


def test_save_refused_while_the_interest_scans(edit):
    conn, uid, pid, slug, *_ = edit
    _, cslug = _open(conn, uid, slug)
    gather_runs_repo.start(conn, profile_id=pid, user_id=uid, since_date=None, filter_string=None, tier_used="scan")
    conn.commit()
    with pytest.raises(wizard.ScanRunning):
        edit_copy.save_copy(conn, user_id=uid, slug=cslug)


def test_what_the_copy_brought_in_is_the_interests_after_save(edit):
    conn, uid, pid, slug, *_ = edit
    cid, cslug = _open(conn, uid, slug)
    _paper(conn, "local:upload:new", (T4, "Four"))
    profiles_repo.attach_seed(conn, cid, "local:upload:new")
    draft_owned_repo.own(conn, cid, "paper", "local:upload:new")
    conn.commit()
    edit_copy.save_copy(conn, user_id=uid, slug=cslug)
    assert "local:upload:new" in _seeds(conn, pid)
    assert conn.execute("SELECT COUNT(*) FROM draft_owned").fetchone()[0] == 0
    assert papers_repo.get_by_openalex_id(conn, "local:upload:new") is not None


# --- discarding -------------------------------------------------------------------


def test_e4_discard_leaves_the_interest_and_removes_only_new_papers(edit):
    """E-4, E-7: a seed C uploaded in the edit goes; one already in the
    Vault stays; neither joins the interest."""
    conn, uid, pid, slug, *_ = edit
    before = _snapshot(conn, pid)
    cid, cslug = _open(conn, uid, slug)
    wizard.remove_seed(conn, user_id=uid, slug=cslug, openalex_id=W2)
    _paper(conn, "local:upload:c", (T4, "Four"))
    for w in ("local:upload:c", W3):
        profiles_repo.attach_seed(conn, cid, w)
    draft_owned_repo.own(conn, cid, "paper", "local:upload:c")
    conn.commit()
    edit_copy.discard_copy(conn, user_id=uid, slug=cslug)
    assert _row(conn, cid) is None
    assert _snapshot(conn, pid) == before
    assert papers_repo.get_by_openalex_id(conn, "local:upload:c") is None
    assert papers_repo.get_by_openalex_id(conn, W3) is not None


def test_leftovers_are_discarded_but_not_one_an_import_still_fills(edit):
    conn, uid, pid, slug, *_ = edit
    cid, cslug = _open(conn, uid, slug)
    run = gather_runs_repo.start(conn, profile_id=cid, user_id=uid, since_date=None, filter_string=None,
                                 tier_used="orcid_import")
    conn.commit()
    assert gather_runs_repo.has_unfinished_import(conn, cid)
    assert edit_copy.discard_leftovers(conn, uid) == []
    assert _row(conn, cid) is not None
    with pytest.raises(wizard.ImportInProgress):                 # nor saved, nor discarded
        edit_copy.save_copy(conn, user_id=uid, slug=cslug)
    gather_runs_repo.finish(conn, run, n_fetched=0, n_new=0, n_redup=0)
    conn.commit()
    assert edit_copy.discard_leftovers(conn, uid) == [cslug]
    assert _row(conn, cid) is None


def test_changed_compares_the_copy_with_the_interest(edit):
    conn, uid, pid, slug, *_ = edit
    cid, cslug = _open(conn, uid, slug)
    assert edit_copy.changed(conn, user_id=uid, slug=cslug) is False
    wizard.remove_seed(conn, user_id=uid, slug=cslug, openalex_id=W2)
    assert edit_copy.changed(conn, user_id=uid, slug=cslug) is True
    cid, cslug = _open(conn, uid, slug)
    profiles_repo.update_threshold(conn, cid, 0.66)
    assert edit_copy.changed(conn, user_id=uid, slug=cslug) is True


# --- the interest during the edit --------------------------------------------------


def test_e9_a_copy_with_no_recorded_scan_compares_with_the_interest_as_saved(edit):
    """E.9: the interest's scans ran before 0032 recorded seeds and topics
    (the test site's): a seed removed in the edit is still said."""
    conn, uid, pid, slug, *_ = edit
    _scan(conn, pid, topics=None, seeds=None)                     # recorded nothing
    cid, cslug = _open(conn, uid, slug)
    assert wizard.outdated(conn, cid) == {"topics": False, "seeds": False}
    wizard.remove_seed(conn, user_id=uid, slug=cslug, openalex_id=W2)
    assert wizard.outdated(conn, cid) == {"topics": False, "seeds": True}
    on = sorted(enabled_topic_ids(_stored(conn, cid)))
    wizard.save_topics(conn, user_id=uid, slug=cslug, selected_topic_ids=on[:1])
    assert wizard.outdated(conn, cid) == {"topics": True, "seeds": True}
    assert wizard.outdated(conn, pid) == {"topics": False, "seeds": False}   # the interest: nothing to compare


def test_e9_a_scan_that_recorded_only_topics_takes_the_seeds_as_saved(edit):
    conn, uid, pid, slug, *_ = edit
    _scan(conn, pid, topics=sorted(enabled_topic_ids(_stored(conn, pid))), seeds=None)
    cid, cslug = _open(conn, uid, slug)
    wizard.remove_seed(conn, user_id=uid, slug=cslug, openalex_id=W2)
    assert wizard.outdated(conn, cid) == {"topics": False, "seeds": True}


def test_outdated_on_the_copy_compares_with_the_interests_last_scan(edit):
    conn, uid, pid, slug, *_ = edit
    _scan(conn, pid, topics=sorted(enabled_topic_ids(_stored(conn, pid))), seeds=_seeds(conn, pid))
    cid, cslug = _open(conn, uid, slug)
    assert wizard.outdated(conn, cid) == {"topics": False, "seeds": False}
    profiles_repo.attach_seed(conn, cid, W3)
    conn.commit()
    assert wizard.outdated(conn, cid) == {"topics": False, "seeds": True}
    assert wizard.outdated(conn, pid) == {"topics": False, "seeds": False}


def test_deleting_the_interest_takes_its_copy(edit):
    conn, uid, pid, slug, *_ = edit
    cid, _ = _open(conn, uid, slug)
    from rag_lib.api.services import profiles as profiles_service
    profiles_service.delete_profile(conn, uid, slug)
    assert _row(conn, pid) is None and _row(conn, cid) is None


# --- over the API ---------------------------------------------------------------

from tests.test_researchers import _app, _request, env  # noqa: E402,F401 -- fixture


@pytest.fixture()
def api(env, monkeypatch):
    """A saved interest 'live' of the demo user with two seeds and topics,
    and the app."""
    from rag_lib.db import connect
    monkeypatch.setenv("RADAR_CONTENT_TOPICS_ENABLED", "false")
    conn = connect(env)
    uid = 1
    pid = profiles_repo.upsert(conn, user_id=uid, name="Live", embedding_model="placeholder-v1",
                               n_seed=2, topic_filters={"topics": []}, threshold=0.5)
    conn.execute("UPDATE profiles SET is_draft = 0 WHERE id = ?", (pid,))
    _paper(conn, W1, (T1, "One"))
    _paper(conn, W2, (T2, "Two"))
    for w in (W1, W2):
        profiles_repo.attach_seed(conn, pid, w)
        embeddings_repo.upsert(conn, w, "placeholder-v1", [0.1] * 8)
    slug = profiles_repo.get(conn, pid)["slug"]
    wizard.aggregate_draft_topics(conn, user_id=uid, slug=slug)
    # W3: the user's (a seed of another interest), not one of this one's --
    # what a route adding seeds would add.
    other = profiles_repo.upsert(conn, user_id=uid, name="Other", embedding_model="placeholder-v1",
                                 n_seed=1, topic_filters={"topics": []}, threshold=0.5)
    _paper(conn, W3, (T3, "Three"))
    profiles_repo.attach_seed(conn, other, W3)
    embeddings_repo.upsert(conn, W3, "placeholder-v1", [0.1] * 8)
    conn.commit()
    conn.close()
    app = _app()
    return app, slug, env


def test_x_a_saved_interest_changes_only_through_a_copy(api):
    """E.6 R12: every route that changes an interest refuses a saved one."""
    app, slug, _ = api
    calls = [
        ("POST", f"/api/profiles/{slug}/seeds", {"json": {"openalex_ids": [W1]}}),
        ("DELETE", f"/api/profiles/{slug}/seeds", {"params": {"openalex_id": W1}}),
        ("PATCH", f"/api/profiles/{slug}", {"json": {"threshold": 0.6}}),
        ("PUT", f"/api/profiles/{slug}/topics", {"json": {"selected_topic_ids": [T1]}}),
        ("POST", f"/api/profiles/{slug}/recompute-topics", {}),
        ("POST", f"/api/profiles/{slug}/recompute-coherence", {}),
        ("POST", f"/api/profiles/{slug}/refit", {}),
        ("PUT", f"/api/profiles/{slug}/researcher-topics", {"json": {"enabled": True}}),
    ]
    for method, path, kw in calls:
        r = _request(app, method, path, **kw)
        assert r.status_code == 409, (method, path, r.text)
    r = _request(app, "POST", "/api/vault/upload", data={"profile_slug": slug},
                 files={"file": ("x.pdf", b"%PDF-1.4 x", "application/pdf")})
    assert r.status_code == 409, r.text


def test_x_while_a_copy_is_open_the_interest_is_not_scanned_and_the_copy_not_listed(api):
    app, slug, _ = api
    from rag_lib.api.deps import get_scheduler

    queued = []

    class Recorder:
        def add_job(self, fn, **kw):
            queued.append(fn.__name__)

    app.dependency_overrides[get_scheduler] = lambda: Recorder()
    copy = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    assert _request(app, "POST", f"/api/profiles/{slug}/gather-now").status_code == 409
    assert _request(app, "POST", f"/api/profiles/{slug}/backfill",
                    json={"year_from": 2020, "year_to": 2021}).status_code == 409
    keys = [p["key"] for p in _request(app, "GET", "/api/profiles").json()]
    assert slug in keys and copy not in keys
    assert copy not in [d["slug"] for d in _request(app, "GET", "/api/profiles/drafts").json()]
    assert queued == []
    # Once saved, the scan starts.
    assert _request(app, "POST", f"/api/profiles/draft/{copy}/save-edit").status_code == 200
    assert _request(app, "POST", f"/api/profiles/{slug}/gather-now").status_code == 200
    assert queued == ["gather_for_profile"]


def test_x_the_full_edit_over_the_api(api):
    """Open, change on the copy, the edit page's state, save; then discard."""
    app, slug, env = api
    copy = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    assert _request(app, "GET", f"/api/profiles/draft/{copy}/edit-state").json() == {"changed": False}
    r = _request(app, "DELETE", f"/api/profiles/{copy}/seeds", params={"openalex_id": W2})
    assert r.status_code == 200, r.text
    assert _request(app, "PATCH", f"/api/profiles/{copy}", json={"threshold": 0.61}).status_code == 200
    assert _request(app, "GET", f"/api/profiles/draft/{copy}/edit-state").json() == {"changed": True}
    st = _request(app, "GET", f"/api/profiles/{copy}/scan-topics").json()
    assert set(st) == {"changed", "topics", "seeds"}
    # The interest is as it was until saved.
    detail = _request(app, "GET", f"/api/profiles/{slug}/detail").json()
    assert len(detail["seeds"]) == 2 and detail["profile"]["threshold"] == pytest.approx(0.5)
    r = _request(app, "POST", f"/api/profiles/draft/{copy}/save-edit")
    assert r.status_code == 200, r.text
    assert r.json()["key"] == slug and r.json()["threshold"] == pytest.approx(0.61)
    detail = _request(app, "GET", f"/api/profiles/{slug}/detail").json()
    assert [s["id"] for s in detail["seeds"]] == [W1]
    assert _request(app, "POST", f"/api/profiles/draft/{copy}/save-edit").status_code == 404
    # Discarded: the interest stays as saved.
    copy = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    _request(app, "PATCH", f"/api/profiles/{copy}", json={"threshold": 0.9})
    assert _request(app, "DELETE", f"/api/profiles/draft/{copy}").status_code == 200
    assert _request(app, "GET", f"/api/profiles/{slug}").json()["threshold"] == pytest.approx(0.61)


def test_x_leftover_copies_go_when_the_site_opens(api):
    app, slug, env = api
    _request(app, "POST", f"/api/profiles/{slug}/edit")
    assert _request(app, "POST", "/api/profiles/drafts/discard-edits").json() == {"discarded": 1}
    from rag_lib.db import connect
    conn = connect(env)
    try:
        assert conn.execute("SELECT COUNT(*) FROM profiles WHERE edit_of IS NOT NULL").fetchone()[0] == 0
    finally:
        conn.close()


def test_x_another_user_cannot_open_save_or_discard_the_copy(api):
    app, slug, _ = api
    copy = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    other = {"X-User-Email": "other@example.com"}
    assert _request(app, "POST", f"/api/profiles/{slug}/edit", headers=other).status_code == 404
    assert _request(app, "POST", f"/api/profiles/draft/{copy}/save-edit", headers=other).status_code == 404
    assert _request(app, "GET", f"/api/profiles/draft/{copy}/edit-state", headers=other).status_code == 404
    _request(app, "DELETE", f"/api/profiles/draft/{copy}", headers=other)
    assert _request(app, "GET", f"/api/profiles/draft/{copy}/edit-state").status_code == 200


def test_the_copys_last_seed_cannot_be_removed(edit):
    conn, uid, pid, slug, *_ = edit
    cid, cslug = _open(conn, uid, slug)
    wizard.remove_seed(conn, user_id=uid, slug=cslug, openalex_id=W2)
    with pytest.raises(ValueError):
        wizard.remove_seed(conn, user_id=uid, slug=cslug, openalex_id=W1)
    assert _seeds(conn, cid) == [W1]


# --- review round 1 (2026-10-29) ---------------------------------------------------


def _saved_state(env, slug):
    from rag_lib.db import connect
    conn = connect(env)
    try:
        row = conn.execute("SELECT * FROM profiles WHERE slug = ? AND user_id = 1", (slug,)).fetchone()
        state = {k: row[k] for k in row.keys() if k != "updated_at"}
        seeds = sorted(r[0] for r in conn.execute("SELECT openalex_id FROM profile_seeds WHERE profile_id = ?",
                                                  (row["id"],)))
        conn.execute("DELETE FROM profiles WHERE edit_of IS NOT NULL")     # an /edit call's copy
        conn.commit()
        return state, seeds
    finally:
        conn.close()


def test_x_no_route_changes_a_saved_interest(api):
    """The class behind review H1 (``POST /draft/{slug}/seeds`` added seeds
    to a saved interest): every route that is not a read, called on a
    saved interest with the bodies its siblings take, leaves it as it was.
    Deleting it is the one route that may."""
    from rag_lib.api.routers.profiles import router
    app, slug, env = api
    before = _saved_state(env, slug)
    bodies = [{}, {"openalex_ids": [W1, W2, "https://openalex.org/W3"]}, {"threshold": 0.9},
              {"years": {"year_from": 2001, "year_to": 2002}}, {"selected_topic_ids": [T1]},
              {"enabled": True}, {"enabled": False}, {"terms": [{"label": "x"}], "removed": [], "not_interests": []},
              {"slug": slug, "threshold": 0.9, "selected_topic_ids": [T1]}, {"topic_id": T1, "choice": "filter"}]
    tried = []
    for route in router.routes:
        if "{key}" not in route.path and "{slug}" not in route.path and route.path != "":
            continue
        for method in route.methods - {"GET", "HEAD"}:
            if method == "DELETE" and route.path == "/{key}":
                continue
            if route.path == "/{key}/terms/generate":
                # A preview, written by nothing until PUT terms (swept); it
                # runs the LLM pipeline, which needs its local cache.
                continue
            path = "/api/profiles" + route.path.replace("{key}", slug).replace("{slug}", slug)
            path = path.replace("{openalex_id:path}", W1).replace("{run_id}", "1")
            for body in bodies:
                r = _request(app, method, path, json=body, params={"openalex_id": W1})
                # 503: the scheduler is off here (a scan writes the feed, not the interest).
                assert r.status_code < 500 or r.status_code == 503, (method, path, body, r.status_code, r.text)
                tried.append((method, route.path))
                assert _saved_state(env, slug) == before, (method, path, body, r.status_code, r.text)
    assert ("POST", "/draft/{slug}/seeds") in tried and ("POST", "") in tried and len(set(tried)) >= 24


def test_x_the_wizards_draft_routes_refuse_a_copy(api):
    """Review M2: committing a copy made it a second, live "X (editing)"."""
    app, slug, env = api
    copy = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    r = _request(app, "POST", "/api/profiles", json={"slug": copy, "threshold": 0.5, "selected_topic_ids": [T1]})
    assert r.status_code == 404, r.text
    for method, path in (("POST", f"/api/profiles/draft/{copy}/seeds"), ("POST", f"/api/profiles/draft/{copy}/coherence"),
                         ("GET", f"/api/profiles/draft/{copy}/topics")):
        r = _request(app, method, path, json={"openalex_ids": [W1]})
        assert r.status_code == 404, (path, r.text)
    keys = [p["key"] for p in _request(app, "GET", "/api/profiles").json()]
    assert slug in keys and copy not in keys and not any(k.endswith("(editing)") for k in keys)
    assert _request(app, "GET", f"/api/profiles/draft/{copy}/edit-state").status_code == 200


def test_x_each_copy_has_its_own_slug(api):
    """Review M1: a tab left on an earlier copy, or a late discard of it,
    must not reach the copy opened since."""
    app, slug, _ = api
    first = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    second = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    assert first != second
    assert _request(app, "PATCH", f"/api/profiles/{first}", json={"threshold": 0.9}).status_code == 404
    _request(app, "DELETE", f"/api/profiles/draft/{first}")
    assert _request(app, "POST", f"/api/profiles/draft/{second}/save-edit").status_code == 200


def test_x_a_copy_is_not_one_of_the_researchers_interests(edit):
    """Review L2."""
    conn, uid, pid, slug, *_ = edit
    from rag_lib.db.repos import researchers as researchers_repo
    rid = researchers_repo.upsert(conn, user_id=uid, source="orcid", key="0000-0001", name="R")
    conn.execute("UPDATE profiles SET researcher_id = ? WHERE id = ?", (rid, pid))
    conn.commit()
    _open(conn, uid, slug)
    assert [int(r["id"]) for r in researchers_repo.interests_for(conn, rid)] == [pid]


def test_seeds_of_a_saved_interest_are_refused_in_the_services(edit):
    """Review H1, below the routes."""
    conn, uid, pid, slug, *_ = edit
    from rag_lib.api.services import researchers as researchers_service
    with pytest.raises(wizard.SavedInterest):
        researchers_service.attach_seeds(conn, user_id=uid, slug=slug, openalex_ids=[W3])
    with pytest.raises(wizard.SavedInterest):
        wizard.remove_seed(conn, user_id=uid, slug=slug, openalex_id=W2)
    assert _seeds(conn, pid) == [W1, W2]


def test_a_failed_rescore_after_saving_does_not_say_not_saved(edit, monkeypatch):
    """Review round 2: the save had committed when the rescore failed, and
    the user was told it was not saved (a retry then found no copy)."""
    conn, uid, pid, slug, *_ = edit
    from rag_lib.scheduler import jobs

    def broken(conn, pid):
        raise RuntimeError("pool busy")
    monkeypatch.setattr(jobs, "rescore_pool", broken)
    cid, cslug = _open(conn, uid, slug)
    profiles_repo.update_threshold(conn, cid, 0.7)
    assert edit_copy.save_copy(conn, user_id=uid, slug=cslug) == slug
    assert _row(conn, pid)["threshold"] == pytest.approx(0.7) and _row(conn, cid) is None


def test_x_a_tab_left_on_a_closed_copy_changes_nothing(api):
    """A page left on a copy since closed (the same tab reopened it, or it
    went stale and another tab took the interest): every route it can
    still call with that copy's slug leaves the interest and the open copy
    as they were."""
    from rag_lib.api.routers.profiles import router
    from rag_lib.db import connect
    app, slug, env = api
    first = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    second = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]

    def state():
        conn = connect(env)
        try:
            out = []
            for s in (slug, second):
                row = conn.execute("SELECT * FROM profiles WHERE slug = ? AND user_id = 1", (s,)).fetchone()
                assert row is not None, s
                out.append(({k: row[k] for k in row.keys() if k != "updated_at"},
                            sorted(r[0] for r in conn.execute(
                                "SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", (row["id"],)))))
            out.append(conn.execute("SELECT COUNT(*) FROM profiles WHERE edit_of IS NOT NULL").fetchone()[0])
            return out
        finally:
            conn.close()

    before = state()
    bodies = [{}, {"openalex_ids": [W1, W2, "https://openalex.org/W3"]}, {"threshold": 0.9},
              {"years": {"year_from": 2001, "year_to": 2002}}, {"selected_topic_ids": [T1]},
              {"enabled": True}, {"terms": [{"label": "x"}], "removed": [], "not_interests": []},
              {"slug": first, "threshold": 0.9, "selected_topic_ids": [T1]}, {"topic_id": T1, "choice": "filter"}]
    tried = set()
    for route in router.routes:
        if "{key}" not in route.path and "{slug}" not in route.path and route.path != "":
            continue
        for method in route.methods - {"GET", "HEAD"}:
            if route.path == "/{key}/terms/generate":
                continue                                    # see the sweep above
            path = "/api/profiles" + route.path.replace("{key}", first).replace("{slug}", first)
            path = path.replace("{openalex_id:path}", W1).replace("{run_id}", "1")
            for body in bodies:
                r = _request(app, method, path, json=body, params={"openalex_id": W1})
                assert r.status_code < 500 or r.status_code == 503, (method, path, body, r.status_code, r.text)
                assert r.status_code >= 400, (method, path, body, r.status_code, r.text)
                tried.add((method, route.path))
                assert state() == before, (method, path, body, r.status_code, r.text)
    assert ("POST", "/draft/{slug}/save-edit") in tried and ("DELETE", "/{key}/seeds") in tried and len(tried) >= 24
    r = _request(app, "POST", "/api/vault/upload", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")},
                 data={"profile_slug": first})
    assert r.status_code == 404, r.text
    assert state() == before
    assert _request(app, "POST", f"/api/profiles/draft/{second}/save-edit").status_code == 200


# --- first come, first served (E.7 R5, the user's rule of 2026-10-29) -----------


def _age(conn, cid, seconds):
    conn.execute("UPDATE profiles SET edit_seen_at = datetime('now', ?) WHERE id = ?", (f"-{seconds} seconds", cid))
    conn.commit()


def test_lock_a_second_tab_is_refused_and_the_first_keeps_its_edit(edit):
    conn, uid, pid, slug, *_ = edit
    c = edit_copy.open_copy(conn, user_id=uid, slug=slug, tab="A")
    conn.execute("UPDATE profiles SET threshold = 0.77 WHERE id = ?", (int(c["id"]),))
    conn.commit()
    with pytest.raises(edit_copy.BeingEdited):
        edit_copy.open_copy(conn, user_id=uid, slug=slug, tab="B")
    # The site opening in tab B leaves A's edit alone.
    assert edit_copy.discard_leftovers(conn, uid, "B") == []
    assert _row(conn, int(c["id"]))["threshold"] == pytest.approx(0.77)
    edit_copy.save_copy(conn, user_id=uid, slug=c["slug"])
    assert _row(conn, pid)["threshold"] == pytest.approx(0.77)
    # Saved: the interest is free for the next tab.
    assert edit_copy.open_copy(conn, user_id=uid, slug=slug, tab="B")["edit_tab"] == "B"


def test_lock_the_same_tab_reopens_a_reload(edit):
    conn, uid, pid, slug, *_ = edit
    first = edit_copy.open_copy(conn, user_id=uid, slug=slug, tab="A")
    second = edit_copy.open_copy(conn, user_id=uid, slug=slug, tab="A")
    assert first["slug"] != second["slug"] and _row(conn, int(first["id"])) is None
    # The site opened in the same tab (not on the edit page): its own copy goes.
    assert edit_copy.discard_leftovers(conn, uid, "A") == [second["slug"]]


def test_lock_a_copy_not_heard_from_for_2_minutes_is_left_behind(edit):
    conn, uid, pid, slug, *_ = edit
    a = edit_copy.open_copy(conn, user_id=uid, slug=slug, tab="A")
    _age(conn, int(a["id"]), 110)                                    # still within 2 minutes
    with pytest.raises(edit_copy.BeingEdited):
        edit_copy.open_copy(conn, user_id=uid, slug=slug, tab="B")
    _age(conn, int(a["id"]), 130)
    assert edit_copy.is_stale(conn, _row(conn, int(a["id"])))
    edit_copy.alive(conn, user_id=uid, slug=a["slug"], tab="A")     # heard from again
    assert not edit_copy.is_stale(conn, _row(conn, int(a["id"])))
    _age(conn, int(a["id"]), 130)
    b = edit_copy.open_copy(conn, user_id=uid, slug=slug, tab="B")
    assert _row(conn, int(a["id"])) is None and b["edit_tab"] == "B"
    with pytest.raises(LookupError):                                 # tab A, back: told it was closed
        edit_copy.alive(conn, user_id=uid, slug=a["slug"], tab="A")
    # Another tab's call on B's copy (a duplicated tab) is not B.
    with pytest.raises(edit_copy.BeingEdited):
        edit_copy.alive(conn, user_id=uid, slug=b["slug"], tab="A")
    # And the site opening anywhere discards a left-behind copy.
    _age(conn, int(b["id"]), 130)
    assert edit_copy.discard_leftovers(conn, uid, "C") == [b["slug"]]


def test_lock_a_copy_an_import_fills_is_never_left_behind(edit):
    conn, uid, pid, slug, *_ = edit
    a = edit_copy.open_copy(conn, user_id=uid, slug=slug, tab="A")
    gather_runs_repo.start(conn, profile_id=int(a["id"]), user_id=uid, since_date=None, filter_string=None,
                           tier_used="orcid_import")
    conn.commit()
    _age(conn, int(a["id"]), 600)
    assert edit_copy.is_stale(conn, _row(conn, int(a["id"]))) is False
    with pytest.raises(edit_copy.BeingEdited):                     # said as another tab's
        edit_copy.open_copy(conn, user_id=uid, slug=slug, tab="B")
    with pytest.raises(wizard.ImportInProgress):                   # its own tab: the import
        edit_copy.open_copy(conn, user_id=uid, slug=slug, tab="A")
    assert edit_copy.open_copy_of(conn, pid)["id"] == a["id"]


def test_lock_two_tabs_opening_at_once_one_gets_it(edit, monkeypatch):
    """Tab B's open lands between tab A clearing the leftovers and taking
    the lock: A is refused, B keeps its copy."""
    conn, uid, pid, slug, *_ = edit
    real = edit_copy.discard_leftovers
    got = {}

    def racing(conn_, user_id, tab=""):
        out = real(conn_, user_id, tab)
        monkeypatch.setattr(edit_copy, "discard_leftovers", real)
        got["b"] = edit_copy.open_copy(conn_, user_id=user_id, slug=slug, tab="B")
        return out

    monkeypatch.setattr(edit_copy, "discard_leftovers", racing)
    with pytest.raises(edit_copy.BeingEdited):
        edit_copy.open_copy(conn, user_id=uid, slug=slug, tab="A")
    assert _row(conn, int(got["b"]["id"])) is not None
    assert conn.execute("SELECT COUNT(*) FROM profiles WHERE edit_of = ?", (pid,)).fetchone()[0] == 1


def test_x_lock_over_the_api(api):
    app, slug, env = api
    from rag_lib.api.deps import get_scheduler
    from rag_lib.db import connect

    class Recorder:
        def add_job(self, fn, **kw):
            pass

    app.dependency_overrides[get_scheduler] = lambda: Recorder()
    a = _request(app, "POST", f"/api/profiles/{slug}/edit", json={"tab": "A"}).json()["slug"]
    r = _request(app, "POST", f"/api/profiles/{slug}/edit", json={"tab": "B"})
    assert r.status_code == 409 and "another tab" in r.json()["detail"], r.text
    assert _request(app, "POST", "/api/profiles/drafts/discard-edits", json={"tab": "B"}).json() == {"discarded": 0}
    assert _request(app, "POST", f"/api/profiles/draft/{a}/edit-alive", json={"tab": "A"}).status_code == 200
    assert _request(app, "POST", f"/api/profiles/draft/{a}/edit-alive", json={"tab": "B"}).status_code == 409
    assert _request(app, "POST", f"/api/profiles/{slug}/gather-now").status_code == 409
    # Tab A closes: its beacon frees the interest at once.
    assert _request(app, "POST", f"/api/profiles/draft/{a}/discard-edit").status_code == 200
    assert _request(app, "POST", f"/api/profiles/draft/{a}/edit-alive", json={"tab": "A"}).status_code == 404
    b = _request(app, "POST", f"/api/profiles/{slug}/edit", json={"tab": "B"}).json()["slug"]
    # Tab B is then gone without a word (a crash): after 2 minutes the
    # interest scans again and the copy is discarded.
    conn = connect(env)
    try:
        conn.execute("UPDATE profiles SET edit_seen_at = datetime('now', '-130 seconds') WHERE slug = ?", (b,))
        conn.commit()
    finally:
        conn.close()
    assert _request(app, "POST", f"/api/profiles/{slug}/gather-now").status_code == 200
    assert _request(app, "POST", f"/api/profiles/draft/{b}/edit-alive", json={"tab": "B"}).status_code == 404
