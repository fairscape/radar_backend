"""One tab at a time changes an account (FUNCTIONAL_SPEC W.A, user 2026-11-04):
the first tab holds the right to change things; every other tab -- any
browser, any device -- is view-only until the holder lets go or is not heard
from for 2 minutes. No take-over."""

from __future__ import annotations

import io

from rag_lib.api import draft_lock, write_lease
from rag_lib.db import connect
from tests.test_researchers import _request, env  # noqa: F401  (fixture)
from tests.test_review_round12_isolation import A, B, world  # noqa: F401  (fixture)

T1 = {**A, "X-Radar-Tab": "tab-one"}
T2 = {**A, "X-Radar-Tab": "tab-two"}


def _q(env, sql, *a):
    c = connect(env)
    try:
        return [tuple(r) for r in c.execute(sql, a).fetchall()]
    finally:
        c.close()


import pytest


@pytest.fixture
def w(world):
    """``world``, built by requests without a tab (which took the lease as it
    went): nobody holds it as a test starts."""
    c = connect(world["env"])
    try:
        c.execute("DELETE FROM write_leases")
        c.commit()
    finally:
        c.close()
    return world


def _claim(app, h):
    r = _request(app, "POST", "/api/users/me/lease", headers=h)
    assert r.status_code == 200, r.text
    return r.json()["editing"]


def _refused(r):
    assert r.status_code == 409, (r.status_code, r.text)
    body = r.json()
    assert body.get("editingElsewhere") is True and "another tab" in body["detail"], body


def _snapshot(env):
    """Everything a refused change could have touched."""
    # Every table but the lease's own and the migration log: a table added
    # later is checked too, and a misspelt name cannot be skipped quietly.
    tables = [r[0] for r in _q(env, "SELECT name FROM sqlite_master WHERE type='table' AND name NOT IN"
                                    " ('write_leases', '_migrations_applied', 'sqlite_sequence') ORDER BY name")]
    assert {"profiles", "profile_candidates", "feedback_events", "paper_uploads", "researchers"} <= set(tables)
    snap = {}
    for t in tables:
        cols = [r[1] for r in _q(env, f"PRAGMA table_info({t})")]
        # Who holds a draft or edit copy is the lock's own bookkeeping, not data.
        keep = [c for c in cols if c not in ("edit_tab", "edit_seen_at")]
        snap[t] = _q(env, f"SELECT {', '.join(keep)} FROM {t} ORDER BY 1")
    return snap


def test_the_first_tab_edits_and_the_next_one_is_view_only(w):
    world = w
    app = world["app"]
    assert _claim(app, T1) is True
    assert _claim(app, T2) is False
    assert _claim(app, A) is False              # no tab header: a tab of its own
    assert _claim(app, T1) is True              # the holder, every 30 s


def test_every_change_from_a_view_only_tab_is_refused_and_changes_nothing(w):
    world = w
    app, env = world["app"], world["env"]
    live, plain, rid, seed = world["live"], world["plain"], world["rid"], world["seed"]
    card = world["cards"][0]
    assert _claim(app, T1) is True
    before = _snapshot(env)
    pdf = {"file": ("x.pdf", io.BytesIO(b"%PDF-1.4\nview-only-bytes"), "application/pdf")}
    tries = [
        ("PATCH", "/api/users/me", {"json": {"mailto": "other@example.com"}}),
        ("POST", "/api/radar/cards/save", {"json": {"profile_key": live, "openalex_id": card}}),
        ("POST", "/api/radar/cards/dismiss", {"json": {"profile_key": live, "openalex_id": card}}),
        ("POST", "/api/vault/upload", {"files": pdf, "data": {"profile_slug": plain}}),
        ("DELETE", f"/api/vault/docs/{world['upload']}", {}),
        ("POST", "/api/profiles/draft", {"json": {"name": "a second draft"}}),
        ("POST", f"/api/profiles/draft/{plain}/hold", {}),
        ("POST", f"/api/profiles/draft/{plain}/seeds", {"json": {"openalex_ids": [seed]}}),
        ("POST", f"/api/profiles/draft/{plain}/dry-run", {"json": {}}),
        ("DELETE", f"/api/profiles/draft/{plain}", {}),
        ("POST", f"/api/profiles/{live}/edit", {"json": {"tab": "tab-two"}}),
        ("PATCH", f"/api/profiles/{live}", {"json": {"threshold": 0.5}}),
        ("PUT", f"/api/profiles/{live}/topics", {"json": {"topics": []}}),
        ("POST", f"/api/profiles/{live}/gather-now", {"json": {"days": 7}}),
        ("DELETE", f"/api/profiles/{live}", {}),
        ("POST", "/api/researchers/import", {"json": {"source": "orcid", "ref": "0000-0003-4060-7360"}}),
        ("PATCH", f"/api/researchers/{rid}", {"json": {"mark": "followed"}}),
        ("PUT", "/api/researchers/order", {"json": {"ids": [rid]}}),
        ("DELETE", f"/api/researchers/{rid}", {}),
        ("POST", "/api/import/orcid", {"json": {"ref": "0000-0003-4060-7360"}}),
    ]
    for method, path, kw in tries:
        if "files" in kw:
            kw = {**kw, "files": {"file": ("x.pdf", io.BytesIO(b"%PDF-1.4\nview-only-bytes"), "application/pdf")}}
        _refused(_request(app, method, path, headers=T2, **kw))
    assert _snapshot(env) == before


def test_a_request_without_a_tab_is_refused_like_any_other(w):
    """No X-Radar-Tab (an old page, a script with the cookie): a tab of its
    own, view-only while a tab holds the account."""
    app = w["app"]
    assert _claim(app, T1) is True
    _refused(_request(app, "PATCH", "/api/users/me", json={"mailto": "nohdr@example.com"}, headers=A))


def test_a_view_only_tab_still_reads(w):
    world = w
    app, live, plain = world["app"], world["live"], world["plain"]
    assert _claim(app, T1) is True
    for path in ("/api/users/me", "/api/profiles", f"/api/profiles/{live}", "/api/vault/docs",
                 "/api/researchers", "/api/radar/daily"):
        r = _request(app, "GET", path, headers=T2)
        assert r.status_code == 200, (path, r.status_code, r.text)
    # The three POSTs that only read or tidy up (W.A R3).
    r = _request(app, "POST", f"/api/profiles/{live}/dry-run", headers=T2)
    assert r.status_code == 200, r.text
    r = _request(app, "POST", "/api/profiles/drafts/discard-edits", json={"tab": "tab-two"}, headers=T2)
    assert r.status_code == 200, r.text
    r = _request(app, "POST", f"/api/profiles/draft/{plain}/release", headers=T2)
    assert r.status_code == 200, r.text


def test_leaving_hands_it_to_the_next_tab(w):
    world = w
    app = world["app"]
    assert _claim(app, T1) is True
    assert _claim(app, T2) is False
    r = _request(app, "POST", "/api/users/me/lease/release", headers=T2)   # not the holder: nothing
    assert r.status_code == 200
    assert _claim(app, T2) is False
    r = _request(app, "POST", "/api/users/me/lease/release", headers=T1)
    assert r.status_code == 200
    assert _claim(app, T2) is True
    r = _request(app, "PATCH", "/api/users/me", json={"mailto": "two@example.com"}, headers=T2)
    assert r.status_code == 200, r.text
    _refused(_request(app, "PATCH", "/api/users/me", json={"mailto": "one@example.com"}, headers=T1))
    assert _claim(app, T1) is False


def test_a_holder_not_heard_from_for_two_minutes_has_left(w):
    world = w
    app, env = world["app"], world["env"]
    assert _claim(app, T1) is True
    c = connect(env)
    try:
        c.execute("UPDATE write_leases SET seen_at = datetime('now', '-119 seconds')")
        c.commit()
    finally:
        c.close()
    assert _claim(app, T2) is False             # 1 s short: still there
    c = connect(env)
    try:
        c.execute("UPDATE write_leases SET seen_at = datetime('now', '-121 seconds')")
        c.commit()
    finally:
        c.close()
    assert _claim(app, T2) is True
    assert _claim(app, T1) is False             # back from sleep: view-only now


def test_a_change_never_takes_the_lease(w):
    """Only the pages' own asking decides who holds it, in the order they
    opened (user 11-04): a change while nobody holds it goes through but
    takes nothing."""
    app = w["app"]
    r = _request(app, "PATCH", "/api/users/me", json={"mailto": "first@example.com"}, headers=T2)
    assert r.status_code == 200, r.text
    assert _claim(app, T1) is True              # opened first: T1's, not T2's
    assert _claim(app, T2) is False


def test_a_closing_page_s_last_request_does_not_take_it_back(w):
    """Edit interest closing: the release, then its discard-edit after it
    (review WA-R1). The next tab gets the right at once."""
    app = w["app"]
    assert _claim(app, T1) is True
    assert _claim(app, T2) is False
    assert _request(app, "POST", "/api/users/me/lease/release", headers=T1).status_code == 200
    r = _request(app, "PATCH", "/api/users/me", json={"mailto": "late@example.com"}, headers=T1)
    assert r.status_code == 200, r.text
    assert _claim(app, T2) is True


def test_a_request_without_a_tab_never_holds_it(w):
    """No X-Radar-Tab (a script with the cookie): it cannot lock the
    account's pages out for 2 minutes (review WA-R5)."""
    app = w["app"]
    r = _request(app, "PATCH", "/api/users/me", json={"mailto": "script@example.com"}, headers=A)
    assert r.status_code == 200, r.text
    assert _claim(app, T1) is True


def test_a_change_does_not_renew_the_lease_only_a_claim_does(w):
    """Only ``claim`` keeps the right (FC-1): a change inside the release
    grace (a closing page's last request) must not hold it 2 more minutes."""
    app, env = w["app"], w["env"]
    assert _claim(app, T1) is True
    c = connect(env)
    try:
        c.execute("UPDATE write_leases SET seen_at = datetime('now', '-110 seconds')")
        c.commit()
    finally:
        c.close()
    r = _request(app, "PATCH", "/api/users/me", json={"mailto": "keep@example.com"}, headers=T1)
    assert r.status_code == 200, r.text
    assert _q(env, "SELECT seen_at <= datetime('now', '-100 seconds') FROM write_leases") == [(1,)]


def test_a_reload_keeps_the_right_and_a_close_hands_it_on_after_the_grace(w):
    """Release with ``?grace=1`` (pagehide): for RELEASE_GRACE_S the same tab
    takes it straight back and another tab is still refused; then the other
    tab gets it (user 2026-11-05: "刷新不算离开", FC-1). Without grace (Sign
    out) it is free at once."""
    app, env = w["app"], w["env"]
    assert _claim(app, T1) is True
    r = _request(app, "POST", "/api/users/me/lease/release", params={"grace": "1"}, headers=T1)
    assert r.status_code == 200
    assert _claim(app, T2) is False                       # inside the grace
    # The closing page's own last change still lands (BL-2).
    r = _request(app, "PATCH", "/api/users/me", json={"mailto": "a2@example.com"}, headers=T1)
    assert r.status_code == 200, r.text
    assert _claim(app, T1) is True                        # the reloaded tab takes it back
    r = _request(app, "POST", "/api/users/me/lease/release", params={"grace": "1"}, headers=T1)
    c = connect(env)
    try:   # the grace runs out
        c.execute(f"UPDATE write_leases SET seen_at = datetime('now', '-{write_lease.STALE_AFTER_S + 1} seconds')")
        c.commit()
    finally:
        c.close()
    assert _claim(app, T2) is True
    r = _request(app, "POST", "/api/users/me/lease/release", headers=T2)
    assert _claim(app, T1) is True                        # no grace: free at once


def test_an_unused_tab_keeps_the_right(w):
    """However long a tab is left unused (X-Radar-Idle: 1), it keeps the right
    (user 2026-11-05, second round): only closing it or signing out lets go."""
    app = w["app"]
    idle = {"X-Radar-Idle": "1"}
    assert _claim(app, T1) is True
    r = _request(app, "POST", "/api/users/me/lease", headers={**T1, **idle})
    assert r.status_code == 200 and r.json() == {"editing": True}
    assert _claim(app, T2) is False


def test_a_tab_without_the_right_keeps_and_drops_its_own_edit_copy_and_no_other(w):
    """FC-2 / BL-2: the tab lost the right (asleep, unused, or closing after
    its release) and leaves Edit interest: its own copy still goes, so the
    tab that has the right can edit that interest at once. Another tab's
    copy cannot be dropped by a view-only tab."""
    app, env = w["app"], w["env"]
    live = w["live"]
    assert _claim(app, T1) is True
    r = _request(app, "POST", f"/api/profiles/{live}/edit", json={"tab": "tab-one"}, headers=T1)
    assert r.status_code == 200, r.text
    mine = r.json()["slug"]
    # T2 takes the right (T1 went quiet), T1 is view-only now.
    c = connect(env)
    try:
        c.execute(f"UPDATE write_leases SET seen_at = datetime('now', '-{write_lease.STALE_AFTER_S + 1} seconds')")
        c.commit()
    finally:
        c.close()
    assert _claim(app, T2) is True
    # Its own copy's "still here" passes too, so the copy is kept (R1);
    # another tab's does not.
    r = _request(app, "POST", f"/api/profiles/draft/{mine}/edit-alive", json={"tab": "tab-one"}, headers=T1)
    assert r.status_code == 200, r.text
    _refused(_request(app, "POST", f"/api/profiles/draft/{mine}/edit-alive", json={"tab": "tab-three"},
                      headers={**A, "X-Radar-Tab": "tab-three"}))
    # Another tab's copy, by POST or DELETE: refused.
    _refused(_request(app, "POST", f"/api/profiles/draft/{mine}/discard-edit", headers={**A, "X-Radar-Tab": "tab-three"}))
    _refused(_request(app, "DELETE", f"/api/profiles/draft/{mine}", headers={**A, "X-Radar-Tab": "tab-three"}))
    # Its own: goes.
    r = _request(app, "DELETE", f"/api/profiles/draft/{mine}", headers=T1)
    assert r.status_code == 200, r.text
    assert _q(env, "SELECT COUNT(*) FROM profiles WHERE slug = ?", mine) == [(0,)]
    # The holder edits that interest at once.
    r = _request(app, "POST", f"/api/profiles/{live}/edit", json={"tab": "tab-two"}, headers=T2)
    assert r.status_code == 200, r.text
    # A view-only tab still cannot drop a draft of its own that is not an edit copy.
    r = _request(app, "POST", "/api/profiles/draft", json={"name": "Not a copy"}, headers=T2)
    assert r.status_code == 200, r.text
    _refused(_request(app, "DELETE", f"/api/profiles/draft/{r.json()['slug']}", headers=T1))


def test_a_tab_without_the_right_keeps_holding_its_own_draft(w):
    """R2 / user 2026-11-05: while a page has a new interest's draft open, no
    other page resumes it -- even once that page has lost the right (asleep
    past 2 minutes and back). Its hold passes; another tab's does not."""
    app, env = w["app"], w["env"]
    assert _claim(app, T1) is True
    r = _request(app, "POST", "/api/profiles/draft", json={"name": "Held here"}, headers=T1)
    assert r.status_code == 200, r.text
    d = r.json()["slug"]
    assert _request(app, "POST", f"/api/profiles/draft/{d}/hold", headers=T1).status_code == 200
    c = connect(env)
    try:
        c.execute(f"UPDATE write_leases SET seen_at = datetime('now', '-{write_lease.STALE_AFTER_S + 1} seconds')")
        c.commit()
    finally:
        c.close()
    assert _claim(app, T2) is True                    # T1 slept; T2 has the right
    r = _request(app, "POST", f"/api/profiles/draft/{d}/hold", headers=T1)
    assert r.status_code == 200, r.text               # T1 still holds its draft
    r = _request(app, "POST", f"/api/profiles/draft/{d}/hold", headers=T2)
    assert r.status_code == 409 and r.json().get("openElsewhere"), r.text   # no Resume elsewhere


def test_accounts_are_separate(w):
    world = w
    app = world["app"]
    assert _claim(app, T1) is True
    assert _claim(app, {**B, "X-Radar-Tab": "tab-two"}) is True
    r = _request(app, "PATCH", "/api/users/me", json={"mailto": "bob2@example.com"},
                 headers={**B, "X-Radar-Tab": "tab-two"})
    assert r.status_code == 200, r.text
    # B's release frees B's, never A's -- even from a tab named as A's holder.
    r = _request(app, "POST", "/api/users/me/lease/release", headers={**B, "X-Radar-Tab": "tab-one"})
    assert r.status_code == 200
    assert _claim(app, T2) is False
    assert _claim(app, T1) is True


def test_work_outside_a_request_is_never_refused(w):
    world = w
    env = world["env"]
    c = connect(env)
    try:
        uid = int(c.execute("SELECT id FROM users WHERE email = 'alice@example.com'").fetchone()[0])
        assert write_lease.claim(c, uid, "tab-one") is True
        token = draft_lock._tab.set(None)       # a scan or an import: no request, no tab
        try:
            write_lease.check(c, uid)
        finally:
            draft_lock._tab.reset(token)
        token = draft_lock._tab.set("tab-two")
        try:
            try:
                write_lease.check(c, uid)
            except write_lease.EditingElsewhere:
                pass
            else:
                raise AssertionError("another tab passed")
        finally:
            draft_lock._tab.reset(token)
    finally:
        c.close()


def test_the_exemptions_are_exactly_the_three_reads_and_the_lease():
    yes = ["/api/users/me/lease", "/api/users/me/lease/release", "/api/profiles/my-interest/dry-run",
           "/api/profiles/drafts/discard-edits", "/api/profiles/draft/x/release"]
    no = ["/api/profiles/draft/x/dry-run", "/api/profiles/draft/dry-run", "/api/profiles/drafts/dry-run",
          "/api/profiles/draft/x/hold", "/api/profiles/x/gather-now", "/api/users/me",
          "/api/users/me/lease/x", "/api/profiles/a/b/dry-run", "/api/profiles/draft/x/release/y"]
    assert all(any(p.match(u) for p in write_lease.EXEMPT) for u in yes)
    assert not any(any(p.match(u) for p in write_lease.EXEMPT) for u in no)


def test_every_route_that_can_change_something_is_guarded(w):
    """A router added later without the guard would let a view-only tab
    change things: found here, not by a reviewer. Walks the routes as they
    are served (tests/route_walk.py): the old filter on ``app.routes``
    checked none of them (BL-1)."""
    from tests.route_walk import served_routes

    changing = [r for r in served_routes(w["app"])
                if r.path.startswith("/api/") and (r.methods - write_lease.SAFE_METHODS)]
    # Today there are 48; far fewer means the walk lost them, not that they went.
    assert len(changing) >= 40, len(changing)
    missing = [f"{sorted(r.methods)} {r.path}" for r in changing if write_lease.guard not in r.calls]
    assert missing == []
