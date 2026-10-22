"""Deleting a saved interest (DELETE /api/profiles/{key}): everything the
interest owns goes, what the user owns besides it stays, another user's
interest of the same slug is untouched, and a running scan blocks it."""

from __future__ import annotations

from rag_lib.db import connect
from rag_lib.db.repos import gather_runs as gather_runs_repo
from rag_lib.scheduler import jobs
from tests.test_review_round12_journeys import (  # noqa: F401
    B, env, app, _request, _ok, _go_live, _scan, _feed, _one, _q,
)

_OWNED = ("profile_seeds", "profile_candidates", "feedback_events", "profile_schedules", "gather_runs")


def _counts(env, pid):
    return {t: _one(env, f"SELECT COUNT(*) AS n FROM {t} WHERE profile_id = ?", pid)["n"] for t in _OWNED}


def test_delete_a_saved_interest(app, env):
    slug, rid = _go_live(app, env, name="Gone")
    keep, _ = _go_live(app, env, name="Kept", seeds_from_researcher=False)
    other, _ = _go_live(app, env, name="Gone", headers=B)
    _scan(app, slug)
    card = _feed(app, slug)["cards"][0]["id"]
    _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": card, "profile": slug, "on": True}))
    pid = _one(env, "SELECT id FROM profiles WHERE slug = ? AND user_id = 1", slug)["id"]
    before = _counts(env, pid)
    assert before["profile_seeds"] and before["profile_candidates"] and before["profile_schedules"] and before["feedback_events"]
    n_papers = _one(env, "SELECT COUNT(*) AS n FROM papers")["n"]

    _ok(_request(app, "DELETE", f"/api/profiles/{slug}"))

    assert _one(env, "SELECT 1 FROM profiles WHERE id = ?", pid) is None
    left = _counts(env, pid)
    assert left == {**{t: 0 for t in _OWNED}, "gather_runs": left["gather_runs"]}
    # The researcher's import history stays, detached; the papers stay.
    assert all(r["researcher_id"] for r in _q(env, "SELECT researcher_id FROM gather_runs WHERE profile_id = ?", pid))
    assert _one(env, "SELECT COUNT(*) AS n FROM papers")["n"] == n_papers
    assert _ok(_request(app, "GET", f"/api/researchers/{rid}"))["papers"]
    assert [p["key"] for p in _ok(_request(app, "GET", "/api/profiles"))] == [keep]
    assert _request(app, "GET", f"/api/profiles/{slug}/detail").json() is None
    # B's interest of the same slug is untouched; A can make the name again.
    assert [p["key"] for p in _ok(_request(app, "GET", "/api/profiles", headers=B))] == [other]
    assert _go_live(app, env, name="Gone")[0] == slug


def test_delete_refuses_others_drafts_and_missing(app, env):
    slug, _ = _go_live(app, env, name="Mine")
    assert _request(app, "DELETE", f"/api/profiles/{slug}", headers=B).status_code == 404
    assert _request(app, "DELETE", "/api/profiles/nope").status_code == 404
    d = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Draft"}))["slug"]
    assert _request(app, "DELETE", f"/api/profiles/{d}").status_code == 404
    assert {p["key"] for p in _ok(_request(app, "GET", "/api/profiles"))} == {slug, d}


def test_delete_waits_for_a_running_scan(app, env):
    slug, _ = _go_live(app, env, name="Busy")
    pid = _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]
    # An open scan, then an open import (seeds still arriving).
    for tier in ("manual", "orcid_import"):
        conn = connect(env)
        try:
            run = gather_runs_repo.start(conn, profile_id=pid, user_id=1, tier_used=tier)
        finally:
            conn.close()
        assert _request(app, "DELETE", f"/api/profiles/{slug}").status_code == 409, tier
        conn = connect(env)
        try:
            gather_runs_repo.finish(conn, run, n_fetched=0, n_new=0, n_redup=0)
        finally:
            conn.close()
    # A scan holding the lock in this process, before it has a run row.
    lock = jobs._scan_locks.setdefault(pid, jobs.threading.Lock())
    lock.acquire()
    try:
        assert _request(app, "DELETE", f"/api/profiles/{slug}").status_code == 409
    finally:
        lock.release()
    # Seed UMLS extraction (paper-level data only) does not block it.
    jobs.SEED_UMLS_RUNNING.add(pid)
    try:
        _ok(_request(app, "DELETE", f"/api/profiles/{slug}"))
    finally:
        jobs.SEED_UMLS_RUNNING.discard(pid)
    assert pid not in jobs._scan_locks
