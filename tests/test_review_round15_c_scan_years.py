"""Review 6 (c): scan counts vs the interest's year range."""

from __future__ import annotations

from rag_lib.db import connect
from rag_lib.scheduler import jobs as scheduler_jobs
from tests.edit_helpers import edit_saved
from tests.fixture_gatherer import FixtureGatherer
from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    _feed, _go_live, _ok, _one, _q, _request, _scan, app, env,
)


def _set_years(app, slug, year_from, year_to):
    """In Edit interest, saved (FUNCTIONAL_SPEC E): a saved interest's own
    PATCH is refused; saving new years re-decides the feed."""
    with edit_saved(lambda m, p, **k: _request(app, m, p, **k), slug) as copy:
        _ok(_request(app, "PATCH", f"/api/profiles/{copy}", json={"years": {"year_from": year_from, "year_to": year_to}}))


class _RecentGatherer(FixtureGatherer):
    """Every paper a scan finds is from this year, as a real 'last 7 days' scan."""

    def fetch(self, profile, since, *, limit=None):
        out = super().fetch(profile, since, limit=limit)
        for p in out:
            p.year = 2026
        return out


def test_scan_new_count_ignores_interest_years(app, env, monkeypatch):
    slug, _ = _go_live(app, env, threshold=0.0)
    # The interest only recommends 2015-2020.
    _set_years(app, slug, 2015, 2020)
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer",
                        lambda **kw: _RecentGatherer(seed=9, n_candidates=10))
    run_id = _scan(app, slug)["run_id"]
    run = _one(env, "SELECT n_new, n_fetched, error FROM gather_runs WHERE id = ?", run_id)
    feed = _feed(app, slug)
    print("run", dict(run), "feed cards", len(feed["cards"]))
    # The run (and the toast "Scan finished: N new papers") says N new...
    assert run["error"] is None
    # ...but none of them can reach the feed: all are outside 2015-2020.
    assert all(not c["date"].startswith("2026") for c in feed["cards"])
    assert run["n_new"] == 0, f"n_new={run['n_new']} but the feed can show none of them"


def test_scan_out_of_years_papers_are_kept_but_not_counted_or_credited(app, env, monkeypatch):
    slug, _ = _go_live(app, env, threshold=0.0)
    _set_years(app, slug, 2015, 2020)
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer",
                        lambda **kw: _RecentGatherer(seed=9, n_candidates=10))
    run_id = _scan(app, slug)["run_id"]
    # Stored (so widening the years later shows them), like below-threshold papers.
    assert _one(env, "SELECT COUNT(*) AS n FROM profile_candidates WHERE gather_run_id = ?", run_id)["n"] > 0
    # Credit them to a topic, as an OpenAlex scan would.
    conn = connect(env)
    try:
        conn.execute("UPDATE profile_candidates SET sourced_by_topic_id = 'T10001' WHERE gather_run_id = ?", (run_id,))
        conn.commit()
    finally:
        conn.close()

    def credited():
        ty = _ok(_request(app, "GET", f"/api/profiles/{slug}/topic-yield"))
        rows = ty["topics"] if isinstance(ty, dict) and "topics" in ty else ty
        return sum((r.get("n_candidates") or 0) for r in rows)

    assert credited() == 0
    # Widening the years brings them into the feed, and into the topic's yield.
    _set_years(app, slug, None, None)
    assert any(c["date"].startswith("2026") for c in _feed(app, slug)["cards"])
    assert credited() > 0
