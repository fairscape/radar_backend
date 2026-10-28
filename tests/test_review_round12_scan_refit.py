"""Review round 12 (reviewer a): seeds attached while a scan runs are fitted only when
*that* scan ends cleanly and nothing else has opened a run by then."""

from __future__ import annotations

import pytest

from rag_lib.api.services import wizard
from rag_lib.db import connect
from rag_lib.db.repos import embeddings as embeddings_repo, gather_runs as gather_runs_repo, papers as papers_repo, profiles as profiles_repo
from rag_lib.scheduler import jobs as scheduler_jobs
from tests.test_scheduler_jobs import _StubGatherer, _candidate_paper, db_path, profile_id, settings  # noqa: F401  (fixtures)


def _make_live(db_path, profile_id, *, legacy=False):
    """A live interest whose fit was made from W1 alone: recorded as
    ``fit_seed_ids`` (every fit since round 12), or -- ``legacy`` -- as a
    one-row ``seed_matrix`` only."""
    conn = connect(db_path)
    try:
        conn.execute("UPDATE profiles SET is_draft = 0 WHERE id = ?", (profile_id,))
        if legacy:
            conn.execute("UPDATE profiles SET selector_config_json = json_set(selector_config_json, '$.seed_matrix', json('[[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]')) WHERE id = ?", (profile_id,))
        else:
            conn.execute("UPDATE profiles SET selector_config_json = json_set(selector_config_json, '$.fit_seed_ids', json('[\"W1\"]')) WHERE id = ?", (profile_id,))
        conn.commit()
    finally:
        conn.close()


def _attach_w2(db_path, profile_id):
    """What an upload / "add as seeds" does while the scan is running: the
    seed lands, the refit is refused (ScanRunning) and swallowed."""
    c = connect(db_path)
    try:
        papers_repo.upsert(c, {"openalex_id": "W2", "title": "Seed W2", "abstract": "words " * 20, "source": "openalex"})
        embeddings_repo.upsert(c, "W2", "specter2", [0.2] * 8)
        profiles_repo.attach_seed(c, profile_id, "W2")
        c.commit()
        with pytest.raises(wizard.ScanRunning):
            wizard.refit_profile(c, user_id=1, slug=profiles_repo.get(c, profile_id)["slug"])
    finally:
        c.close()


def _count_refits(monkeypatch):
    calls: list[str] = []
    real = wizard.refit_profile

    def counting(conn, *, user_id, slug, **kw):
        out = real(conn, user_id=user_id, slug=slug, **kw)
        calls.append(slug)
        return out

    monkeypatch.setattr(wizard, "refit_profile", counting)
    return calls


def test_a_seed_added_during_a_scan_that_fails_is_fitted_by_the_next_scan(monkeypatch, db_path, settings, profile_id):
    _make_live(db_path, profile_id)
    calls = _count_refits(monkeypatch)

    class _AttachThenBoom:
        def __init__(self, **_):
            pass

        def fetch(self, *a, **kw):
            _attach_w2(db_path, profile_id)
            raise RuntimeError("HTTP 429 from upstream")

        def cost(self):
            return {"api_calls": 0}

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", _AttachThenBoom)
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings)
    # The failed scan fits W2 itself once its row is closed as failed: the
    # feed is scored against the seeds the user has, not after the next
    # scan happens to run.
    assert calls, "the failed scan closed without fitting the seed that arrived under it"
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: _StubGatherer([_candidate_paper("W100")]))
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings)
    conn = connect(db_path)
    try:
        assert "W2" in profiles_repo.list_seed_openalex_ids(conn, profile_id)
        assert profiles_repo.get(conn, profile_id)["n_seed"] == 2
    finally:
        conn.close()
    assert calls, "W2 was attached under a scan and no scan since has fitted it: the centroid is stale for good"


def test_a_seed_added_during_a_scan_is_fitted_even_when_scan_now_lands_as_it_closes(monkeypatch, db_path, settings, profile_id):
    """The scan's row is closed, then the refit checks for a running scan: a
    "Scan now" accepted in between (the router refuses only while a row is
    open) makes the refit refuse, and that scan starts with W2 already among
    its seeds-at-start, so it never refits either."""
    _make_live(db_path, profile_id)
    calls = _count_refits(monkeypatch)
    real_finish = gather_runs_repo.finish
    opened: list[int] = []

    def finish_then_scan_now(conn, run_id, **kw):
        real_finish(conn, run_id, **kw)
        if not opened:   # the user's "Scan now", accepted the moment the row closed
            opened.append(gather_runs_repo.start(conn, profile_id=profile_id, user_id=1, tier_used="manual"))

    monkeypatch.setattr(scheduler_jobs.gather_runs_repo, "finish", finish_then_scan_now)

    class _AttachThenFetch(_StubGatherer):
        def fetch(self, *a, **kw):
            if not opened:
                _attach_w2(db_path, profile_id)
            return super().fetch(*a, **kw)

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: _AttachThenFetch([_candidate_paper("W100")]))
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings)
    assert opened
    # ...and the "Scan now" job runs with the row it was given.
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, run_id=opened[0], tier="manual")
    conn = connect(db_path)
    try:
        assert gather_runs_repo.running_scan(conn, profile_id) is None
    finally:
        conn.close()
    assert calls, "two clean scans ran after W2 was attached and neither fitted it"


def test_a_seed_added_after_scan_now_was_accepted_but_before_the_job_started_is_fitted(monkeypatch, db_path, settings, profile_id):
    """"Scan now" opens the run row in the request; the job reads its
    seeds-at-start only once it runs. An upload in between is refused its
    refit (the row is open) and is already among the seeds the scan compares
    against, so the scan does not fit it either."""
    _make_live(db_path, profile_id)
    calls = _count_refits(monkeypatch)
    conn = connect(db_path)
    try:
        run_id = gather_runs_repo.start(conn, profile_id=profile_id, user_id=1, tier_used="manual")
    finally:
        conn.close()
    _attach_w2(db_path, profile_id)            # the upload lands before the scheduler picks the job up

    class _NotingGatherer(_StubGatherer):
        def fetch(self, *a, **kw):
            calls.append("fetch")
            return super().fetch(*a, **kw)

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: _NotingGatherer([_candidate_paper("W100")]))
    assert scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, run_id=run_id, tier="manual") == run_id
    # Fitted before the fetch: this scan scores its candidates against W1+W2.
    assert calls and calls[0] != "fetch", f"the scan fetched before fitting the seed that arrived: {calls}"


def test_a_legacy_fit_with_a_different_seed_count_is_refitted(monkeypatch, db_path, settings, profile_id):
    """Fits from before the seed ids were recorded carry only the embedded
    seed rows; two seeds against a one-row matrix is stale."""
    _make_live(db_path, profile_id, legacy=True)
    calls = _count_refits(monkeypatch)
    conn = connect(db_path)
    try:
        run_id = gather_runs_repo.start(conn, profile_id=profile_id, user_id=1, tier_used="manual")
    finally:
        conn.close()
    _attach_w2(db_path, profile_id)
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: _StubGatherer([_candidate_paper("W100")]))
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, run_id=run_id, tier="manual")
    assert calls == [profiles_repo.get(connect(db_path), profile_id)["slug"]]
    conn = connect(db_path)
    try:
        assert wizard.fit_seed_ids(profiles_repo.get(conn, profile_id)) == ["W1", "W2"]
    finally:
        conn.close()


def test_a_fit_on_the_current_seeds_is_left_alone(monkeypatch, db_path, settings, profile_id):
    _make_live(db_path, profile_id)
    calls = _count_refits(monkeypatch)
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: _StubGatherer([_candidate_paper("W100")]))
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings)
    assert calls == []


def test_commit_and_refit_record_the_fitted_seeds(db_path, profile_id):
    conn = connect(db_path)
    try:
        conn.execute("UPDATE profiles SET is_draft = 1 WHERE id = ?", (profile_id,))
        conn.commit()
        row = profiles_repo.get(conn, profile_id)
        ids = [t["id"] for t in (profiles_repo.topic_filters(conn, profile_id) or {}).get("topics") or []]
        wizard.commit_draft(conn, user_id=1, slug=row["slug"], threshold=0.5, selected_topic_ids=ids)
        assert wizard.fit_seed_ids(profiles_repo.get(conn, profile_id)) == ["W1"]
        wizard.go_live(conn, profile_id)          # its first scan succeeded
        papers_repo.upsert(conn, {"openalex_id": "W2", "title": "Seed W2", "abstract": "words " * 20, "source": "openalex"})
        embeddings_repo.upsert(conn, "W2", "specter2", [0.2] * 8)
        profiles_repo.attach_seed(conn, profile_id, "W2")
        conn.commit()
        wizard.refit_profile(conn, user_id=1, slug=row["slug"])
        assert wizard.fit_seed_ids(profiles_repo.get(conn, profile_id)) == ["W1", "W2"]
    finally:
        conn.close()
