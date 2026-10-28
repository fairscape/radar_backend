"""Review 16 (a): scan/job lifecycle, destructive operations, two-user isolation."""

from __future__ import annotations

import threading

from rag_lib.db import connect
from rag_lib.db.repos import profiles as profiles_repo
from rag_lib.scheduler import jobs as scheduler_jobs
from tests.fixture_gatherer import FixtureGatherer
from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    B, _feed, _go_live, _ok, _one, _q, _request, _scan, _upload, app, env,
)


def _delete_now(env, pid):
    conn = connect(env)
    try:
        profiles_repo.delete(conn, pid)
    finally:
        conn.close()


def test_add_seeds_racing_a_delete_is_not_a_500(app, env, monkeypatch):
    """POST /{key}/seeds reads the interest, then DELETE /{key} commits on
    another request thread, then the seed insert hits the vanished id."""
    from rag_lib.db.repos import researchers as researchers_repo

    slug, rid = _go_live(app, env, name="Seed Race")
    pid = _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]
    papers = _ok(_request(app, "GET", f"/api/researchers/{rid}"))["papers"]
    seeds = {r["openalex_id"] for r in _q(env, "SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", pid)}
    # A paper the user owns (a researcher's) but not yet a seed of this interest.
    _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": papers[-1]["id"]}))
    target = papers[-1]["id"]
    assert target not in {r["openalex_id"] for r in _q(env, "SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", pid)}
    del seeds

    real = researchers_repo.user_owns_papers

    def _owns_then_deleted(conn, user_id, ids):
        out = real(conn, user_id, ids)
        _delete_now(env, pid)
        return out

    monkeypatch.setattr(researchers_repo, "user_owns_papers", _owns_then_deleted)
    try:
        res = _request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [target]})
        code = res.status_code
    except Exception as exc:  # noqa: BLE001 -- ASGITransport re-raises app errors
        code = f"unhandled {type(exc).__name__}: {exc}"
    assert code in (404, 409), code


class _BlockingGatherer(FixtureGatherer):
    def __init__(self, started, release, **kw):
        super().__init__(**kw)
        self._started, self._release = started, release

    def fetch(self, profile, since, *, limit=None):
        self._started.set()
        assert self._release.wait(20)
        return super().fetch(profile, since, limit=limit)


def test_delete_while_scan_thread_runs_is_refused_then_allowed(app, env, monkeypatch):
    from rag_lib.api.settings import get_settings

    slug, _ = _go_live(app, env, name="Busy", threshold=0.0)
    pid = _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]
    uid = _one(env, "SELECT user_id FROM profiles WHERE id = ?", pid)["user_id"]
    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer",
                        lambda **kw: _BlockingGatherer(started, release, seed=3, n_candidates=8))
    settings = get_settings()
    # The scheduled-scan path: no pre-created row, the job opens its own.
    t = threading.Thread(target=scheduler_jobs.gather_for_profile, args=(uid, pid),
                         kwargs={"settings": settings, "days": 7, "tier": "manual"})
    t.start()
    try:
        assert started.wait(20)
        res = _request(app, "DELETE", f"/api/profiles/{slug}")
        assert res.status_code == 409, res.text
        # Scan now is refused too.
        assert _request(app, "POST", f"/api/profiles/{slug}/gather-now?days=7").status_code == 409
    finally:
        release.set()
        t.join(30)
    run = _one(env, "SELECT finished_at, error, n_new FROM gather_runs WHERE profile_id = ? ORDER BY id DESC", pid)
    assert run["finished_at"] is not None and run["error"] is None, dict(run)
    _ok(_request(app, "DELETE", f"/api/profiles/{slug}"))
    assert _one(env, "SELECT COUNT(*) AS n FROM profile_candidates WHERE profile_id = ?", pid)["n"] == 0
    assert _one(env, "SELECT COUNT(*) AS n FROM gather_runs WHERE profile_id = ?", pid)["n"] == 0


def test_failed_scan_releases_lock_and_closes_row(app, env, monkeypatch):
    slug, _ = _go_live(app, env, name="Boom")

    class _Boom(FixtureGatherer):
        def fetch(self, *a, **k):
            raise RuntimeError("openalex down")

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: _Boom(seed=1, n_candidates=3))
    run_id = _scan(app, slug)["run_id"]
    row = _one(env, "SELECT finished_at, error FROM gather_runs WHERE id = ?", run_id)
    assert row["finished_at"] is not None and "openalex down" in (row["error"] or "")
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: FixtureGatherer(seed=7, n_candidates=10))
    run2 = _scan(app, slug)["run_id"]
    assert _one(env, "SELECT error FROM gather_runs WHERE id = ?", run2)["error"] is None
    _ok(_request(app, "DELETE", f"/api/profiles/{slug}"))


def test_slug_reuse_after_delete_starts_clean(app, env):
    slug, _ = _go_live(app, env, name="Again", threshold=0.0)
    _scan(app, slug)
    old_pid = _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]
    card = _feed(app, slug)["cards"][0]["id"]
    _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": card, "profile": slug, "on": True}))
    _ok(_request(app, "DELETE", f"/api/profiles/{slug}"))
    slug2, _ = _go_live(app, env, name="Again", threshold=0.0)
    assert slug2 == slug
    new_pid = _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]
    assert new_pid != old_pid
    # Only its own first scan, which saving it ran (FUNCTIONAL_SPEC D.2-5).
    runs = [r for r in _ok(_request(app, "GET", f"/api/profiles/{slug}/runs")) if r["tier_used"] != "dry_run"]
    assert [r["profile_id"] for r in runs] == [new_pid]
    assert _ok(_request(app, "GET", f"/api/profiles/{slug}/feedback")) == []
    # Its feed is its own first scan's: nothing the old interest saved.
    assert not any(c.get("state") for c in _feed(app, slug)["cards"])
    tags = _ok(_request(app, "GET", "/api/vault/tags"))
    print("tags", tags)


def test_second_user_cannot_touch_first_users_interest(app, env):
    slug, _ = _go_live(app, env, name="Private", threshold=0.0)
    run_id = _scan(app, slug)["run_id"]
    pid = _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]
    before = _one(env, "SELECT threshold, year_from, year_to FROM profiles WHERE id = ?", pid)
    calls = [
        ("GET", f"/api/profiles/{slug}", {}),
        ("GET", f"/api/profiles/{slug}/detail", {}),
        ("GET", f"/api/profiles/{slug}/runs", {}),
        ("GET", f"/api/profiles/{slug}/feedback", {}),
        ("GET", f"/api/profiles/{slug}/topic-yield", {}),
        ("GET", f"/api/profiles/{slug}/reranker-comparison", {}),
        ("GET", f"/api/profiles/{slug}/feedback-threshold", {}),
        ("GET", f"/api/profiles/{slug}/umls", {}),
        ("GET", f"/api/profiles/{slug}/terms", {}),
        ("POST", f"/api/profiles/{slug}/dry-run", {}),
        ("PATCH", f"/api/profiles/{slug}", {"json": {"threshold": 0.9}}),
        ("PATCH", f"/api/profiles/{slug}", {"json": {"years": {"year_from": 2001, "year_to": 2002}}}),
        ("POST", f"/api/profiles/{slug}/gather-now?days=7", {}),
        ("POST", f"/api/profiles/{slug}/backfill", {"json": {"year_from": 2020, "year_to": 2021}}),
        ("POST", f"/api/profiles/{slug}/refit", {}),
        ("DELETE", f"/api/profiles/{slug}/seeds", {"params": {"openalex_id": "W1"}}),
        ("DELETE", f"/api/profiles/{slug}", {}),
        ("GET", f"/api/import/prosopia/{run_id}", {}),
    ]
    leaks = []
    for method, path, kw in calls:
        res = _request(app, method, path, headers=B, **kw)
        body = res.json() if res.content else None
        if res.status_code < 400 and body not in (None, [], {}) and not (method == "GET" and path.endswith(f"/{slug}") and body is None):
            leaks.append((method, path, res.status_code, str(body)[:200]))
    after = _one(env, "SELECT threshold, year_from, year_to FROM profiles WHERE id = ?", pid)
    assert dict(after) == dict(before)
    n_runs = _one(env, "SELECT COUNT(*) AS n FROM gather_runs WHERE profile_id = ? AND tier_used != 'dry_run'", pid)["n"]
    assert n_runs == 2, n_runs        # the save's first scan and A's scan; none of B's
    assert not leaks, leaks
    # B's upload to A's slug is refused.
    assert _upload(app, b"b-file", "b.pdf", slug, headers=B).status_code == 404


def test_upload_seed_then_interest_deleted_before_refit_is_not_a_500(app, env, monkeypatch):
    """The Seeds tab's PDF drop: the seed row lands, then DELETE /{key} commits
    (no scan runs, so it is allowed), then the upload's refit looks the slug up."""
    from rag_lib.api.services import wizard as wizard_service

    slug, _ = _go_live(app, env, name="Late Delete", seeds_from_researcher=False)
    pid = _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]
    real = wizard_service.embed_missing

    def _embed_then_deleted(conn, model, ids):
        out = real(conn, model, ids)
        _delete_now(env, pid)
        return out

    monkeypatch.setattr(wizard_service, "embed_missing", _embed_then_deleted)
    try:
        res = _upload(app, b"late-delete-body", "late.pdf", slug)
        code = res.status_code
    except Exception as exc:  # noqa: BLE001
        code = f"unhandled {type(exc).__name__}: {exc}"
    assert code in (200, 404), code
