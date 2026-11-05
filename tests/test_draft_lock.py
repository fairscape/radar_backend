"""A new interest's draft is open in one tab at a time (FUNCTIONAL_SPEC W.L,
user 2026-11-04): the tab that opened it holds it; every other tab is
refused anything that would change it, Resume included, until the holder
leaves or is not heard from for 2 minutes. The draft itself stays."""

from __future__ import annotations

import io

import pytest

from rag_lib.api import draft_lock, write_lease
from rag_lib.db import connect
from tests.test_researchers import _request, env  # noqa: F401  (fixture)
from tests.test_review_round12_isolation import A, world  # noqa: F401  (fixture)



@pytest.fixture(autouse=True)
def _draft_lock_alone(monkeypatch):
    """The draft lock on its own. Since W.A one tab holds the whole account,
    so another tab is refused before the draft lock is reached
    (test_write_lease.py); the draft lock stays as the inner layer and is
    tested here with the account lease out of the way."""
    monkeypatch.setattr(write_lease, "check", lambda *a, **k: None)


T1 = {**A, "X-Radar-Tab": "tab-one"}
T2 = {**A, "X-Radar-Tab": "tab-two"}


def _q(env, sql, *a):
    c = connect(env)
    try:
        return [tuple(r) for r in c.execute(sql, a).fetchall()]
    finally:
        c.close()


def _state(env, slug):
    """What a change to the draft would touch: its row (but who holds it),
    its seeds, its uploads, its owned rows."""
    (pid,) = _q(env, "SELECT id FROM profiles WHERE slug = ?", slug)[0]
    row = _q(env, "SELECT * FROM profiles WHERE id = ?", pid)[0]
    cols = [r[1] for r in _q(env, "PRAGMA table_info(profiles)")]
    keep = tuple(v for c, v in zip(cols, row) if c not in ("edit_tab", "edit_seen_at"))
    return {
        "row": keep,
        "seeds": _q(env, "SELECT openalex_id FROM profile_seeds WHERE profile_id = ? ORDER BY 1", pid),
        "owned": _q(env, "SELECT kind, item FROM draft_owned WHERE profile_id = ? ORDER BY 1, 2", pid),
        "runs": _q(env, "SELECT COUNT(*) FROM gather_runs WHERE profile_id = ?", pid),
    }


def _hold(app, slug, h):
    return _request(app, "POST", f"/api/profiles/draft/{slug}/hold", headers=h)


def _refused(r):
    assert r.status_code == 409, (r.status_code, r.text)
    body = r.json()
    assert body.get("openElsewhere") is True and "open in another tab" in body["detail"], body


def test_the_first_tab_holds_it_and_resume_elsewhere_is_refused(world):
    app, draft = world["app"], world["plain"]
    assert _hold(app, draft, T1).status_code == 200
    _refused(_hold(app, draft, T2))
    _refused(_hold(app, draft, A))              # a request with no tab is another tab
    assert _hold(app, draft, T1).status_code == 200   # the holder, every 30 s


def test_every_change_from_another_tab_is_refused_and_changes_nothing(world):
    app, env, draft, rdraft, rid = world["app"], world["env"], world["plain"], world["draft"], world["rid"]
    seed = world["seed"]
    # The holder puts a seed and an upload in, so there is something to take out.
    assert _hold(app, draft, T1).status_code == 200
    r = _request(app, "POST", f"/api/profiles/draft/{draft}/seeds", json={"openalex_ids": [seed]}, headers=T1)
    assert r.status_code == 200, r.text
    files = {"file": ("mine.pdf", io.BytesIO(b"%PDF-1.4\nheld-draft-bytes"), "application/pdf")}
    r = _request(app, "POST", "/api/vault/upload", files=files, data={"profile_slug": draft}, headers=T1)
    assert r.status_code == 200, r.text
    upload = r.json()["id"]
    assert _hold(app, rdraft, T1).status_code == 200
    before = {s: _state(env, s) for s in (draft, rdraft)}

    tries = [
        ("DELETE", f"/api/profiles/draft/{draft}", {}),
        ("POST", f"/api/profiles/draft/{draft}/seeds", {"json": {"openalex_ids": [seed]}}),
        ("DELETE", f"/api/profiles/draft/{draft}/seeds/{seed}", {}),
        ("POST", f"/api/profiles/draft/{draft}/coherence", {}),
        ("POST", f"/api/profiles/draft/{draft}/dry-run", {"json": {}}),
        ("PUT", f"/api/profiles/{draft}/topics", {"json": {"on": []}}),
        ("POST", f"/api/profiles/{draft}/seeds", {"json": {"openalex_ids": [seed]}}),
        ("PATCH", f"/api/profiles/{draft}", {"json": {"threshold": 0.3}}),
        ("POST", "/api/profiles", {"json": {"slug": draft, "threshold": 0.5}}),
        ("POST", "/api/vault/upload", {"files": {"file": ("x.pdf", io.BytesIO(b"%PDF-1.4\nother"), "application/pdf")},
                                       "data": {"profile_slug": draft}}),
        ("DELETE", f"/api/vault/docs/{upload}", {}),
        ("POST", "/api/import/orcid", {"json": {"orcid": "0000-0001-5643-4068", "openalex_ids": [seed], "into": draft}}),
        ("POST", "/api/import/prosopia", {"json": {"ref": "sheffield-nathan", "into": draft}}),
        ("DELETE", f"/api/researchers/{rid}", {}),       # takes its topics out of rdraft
    ]
    for method, path, kw in tries:
        if "files" in kw:
            kw = {**kw, "files": {"file": ("x.pdf", io.BytesIO(b"%PDF-1.4\nother"), "application/pdf")}}
        r = _request(app, method, path, headers=T2, **kw)
        _refused(r)
        assert {s: _state(env, s) for s in (draft, rdraft)} == before, (method, path)


def test_the_holder_itself_is_not_refused(world):
    app, draft, seed = world["app"], world["plain"], world["seed"]
    assert _hold(app, draft, T1).status_code == 200
    r = _request(app, "POST", f"/api/profiles/draft/{draft}/seeds", json={"openalex_ids": [seed]}, headers=T1)
    assert r.status_code == 200, r.text
    r = _request(app, "DELETE", f"/api/profiles/draft/{draft}/seeds/{seed}", headers=T1)
    assert r.status_code == 200, r.text


def test_leaving_frees_it_at_once_and_the_draft_stays(world):
    app, env, draft = world["app"], world["env"], world["plain"]
    assert _hold(app, draft, T1).status_code == 200
    _refused(_hold(app, draft, T2))
    # Only the holder's leaving counts.
    assert _request(app, "POST", f"/api/profiles/draft/{draft}/release", headers=T1).status_code == 200
    assert _q(env, "SELECT is_draft FROM profiles WHERE slug = ?", draft) == [(1,)]
    assert _hold(app, draft, T2).status_code == 200
    _refused(_hold(app, draft, T1))


def test_a_holder_unheard_for_two_minutes_has_left_it(world):
    app, env, draft = world["app"], world["env"], world["plain"]
    assert _hold(app, draft, T1).status_code == 200
    c = connect(env)
    c.execute("UPDATE profiles SET edit_seen_at = datetime('now', '-90 seconds') WHERE slug = ?", (draft,))
    c.commit()
    c.close()
    _refused(_hold(app, draft, T2))             # 90 s: still there
    c = connect(env)
    c.execute("UPDATE profiles SET edit_seen_at = datetime('now', '-121 seconds') WHERE slug = ?", (draft,))
    c.commit()
    c.close()
    assert _hold(app, draft, T2).status_code == 200
    # The old tab, back from sleep: refused, and it shows that (its next 30 s check).
    _refused(_hold(app, draft, T1))


def test_a_saved_or_deleted_draft_answers_404_to_hold(world):
    app, draft = world["app"], world["plain"]
    assert _hold(app, draft, T1).status_code == 200
    assert _request(app, "DELETE", f"/api/profiles/draft/{draft}", headers=T1).status_code == 200
    assert _hold(app, draft, T1).status_code == 404
    assert _hold(app, world["live"], T1).status_code == 404     # a saved interest is no draft to hold
    for slug in (draft, world["live"]):                          # nor to let go of
        assert _request(app, "POST", f"/api/profiles/draft/{slug}/release", headers=T1).status_code == 404


def test_work_outside_a_request_is_never_refused(world):
    """An import or a scan job fills the draft for the tab that started it."""
    app, env, draft = world["app"], world["env"], world["plain"]
    assert _hold(app, draft, T1).status_code == 200
    c = connect(env)
    try:
        row = c.execute("SELECT * FROM profiles WHERE slug = ?", (draft,)).fetchone()
        draft_lock.check(c, row)          # no request: no tab, no refusal
    finally:
        c.close()


def test_edit_interest_copies_and_saved_interests_are_not_this_lock(world):
    """Edit interest keeps its own lock (E.7 R5); a saved interest has none here."""
    app, live = world["app"], world["live"]
    r = _request(app, "PATCH", f"/api/profiles/{live}", json={"threshold": 0.4}, headers=T2)
    # Refused by Edit interest's own rule (E.6 R12), not as a draft open elsewhere.
    assert r.status_code == 409 and not r.json().get("openElsewhere"), r.text


def test_every_route_that_changes_a_profile_by_path_is_guarded():
    """A route added later under /api/profiles that changes a draft named in
    its path is guarded with the rest (the router's dependency): fails on a
    route the guard does not cover. Walks the routes as served (BL-1)."""
    from rag_lib.api.app import create_app
    from rag_lib.api.routers.profiles import _guard_draft_writes
    from tests.route_walk import served_routes

    app = create_app()
    checked = [r for r in served_routes(app)
               if r.path.startswith("/api/profiles") and (r.methods - {"GET", "HEAD", "OPTIONS"})
               and ("{slug}" in r.path or "{key}" in r.path or r.path == "/api/profiles")]
    assert len(checked) >= 20, len(checked)
    missing = [f"{sorted(r.methods)} {r.path}" for r in checked if _guard_draft_writes not in r.calls]
    assert not missing, missing
    assert all(draft_lock.request_tab in r.calls for r in served_routes(app) if r.path.startswith("/api/"))
    assert draft_lock.request_tab in {d.dependency for d in app.router.dependencies}


def _as_tab(tab):
    """Run ``draft_lock`` as a request from ``tab`` would, without the
    router's guard in front (which refuses another tab first)."""
    return draft_lock._tab.set(tab)


def test_hold_itself_refuses_another_tab(world):
    """Two tabs opening at once both pass the router's guard (it reads
    before either holds); ``hold`` under BEGIN IMMEDIATE must refuse the
    second on its own."""
    env, draft = world["env"], world["plain"]
    c = connect(env)
    try:
        uid = c.execute("SELECT user_id FROM profiles WHERE slug = ?", (draft,)).fetchone()[0]
        tok = _as_tab("tab-one")
        draft_lock.hold(c, user_id=uid, slug=draft)
        draft_lock._tab.reset(tok)
        tok = _as_tab("tab-two")
        with pytest.raises(draft_lock.DraftOpenElsewhere):
            draft_lock.hold(c, user_id=uid, slug=draft)
        draft_lock._tab.reset(tok)
    finally:
        c.close()
    assert _q(env, "SELECT edit_tab FROM profiles WHERE slug = ?", draft) == [("tab-one",)]


def test_release_itself_frees_only_for_the_holder(world):
    env, draft = world["env"], world["plain"]
    c = connect(env)
    try:
        uid = c.execute("SELECT user_id FROM profiles WHERE slug = ?", (draft,)).fetchone()[0]
        tok = _as_tab("tab-one")
        draft_lock.hold(c, user_id=uid, slug=draft)
        draft_lock._tab.reset(tok)
        tok = _as_tab("tab-two")
        draft_lock.release(c, user_id=uid, slug=draft)
        draft_lock._tab.reset(tok)
        assert _q(env, "SELECT edit_tab FROM profiles WHERE slug = ?", draft) == [("tab-one",)]
        tok = _as_tab("tab-one")
        draft_lock.release(c, user_id=uid, slug=draft)
        draft_lock._tab.reset(tok)
    finally:
        c.close()
    assert _q(env, "SELECT edit_tab, edit_seen_at FROM profiles WHERE slug = ?", draft) == [(None, None)]
