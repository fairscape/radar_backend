"""Scans after the 2026-11-02 review (FUNCTIONAL_SPEC CR.5, CR.6, CR.11
L7 L8 L9)."""

from __future__ import annotations

from rag_lib.api.services import wizard
from rag_lib.db import connect
from rag_lib.db.repos import embeddings as embeddings_repo, gather_runs as gather_runs_repo, papers as papers_repo, profiles as profiles_repo
from rag_lib.openalex_status import OpenAlexUnavailable
from rag_lib.scheduler import jobs as scheduler_jobs
from tests.test_review_round12_scan_refit import _make_live
from tests.test_scheduler_jobs import _StubGatherer, _candidate_paper, db_path, profile_id, settings  # noqa: F401  (fixtures)


def _state(db_path, profile_id):
    """What a scan may change: the fit, every score, the deleted marks."""
    c = connect(db_path)
    try:
        row = profiles_repo.get(c, profile_id)
        return {
            "fit": row["selector_config_json"],
            "scores": [tuple(r) for r in c.execute(
                "SELECT openalex_id, score_raw, score_pct, score_blended FROM profile_candidates "
                "WHERE profile_id = ? ORDER BY 1", (profile_id,))],
            "deleted": [tuple(r) for r in c.execute(
                "SELECT openalex_id FROM deleted_seeds WHERE profile_id = ? ORDER BY 1", (profile_id,))],
        }
    finally:
        c.close()


class _Refused:
    """OpenAlex says no: its daily quota is used up."""

    def __init__(self, **_):
        pass

    def fetch(self, *a, **kw):
        raise OpenAlexUnavailable("quota", detail="OpenAlex 429")

    def fetch_years(self, *a, **kw):
        raise OpenAlexUnavailable("quota", detail="OpenAlex 429")

    def cost(self):
        return {"api_calls": 1}


def test_cr5_a_scan_openalex_refuses_changes_no_fit_and_no_score(monkeypatch, db_path, settings, profile_id):
    """Review SF-01: a seed added since the fit, then a scan OpenAlex
    refuses. The fit, the scores and the deleted marks are as they were,
    as the run says ("Nothing was changed."); the next scan that gets its
    papers fits the new seed."""
    _make_live(db_path, profile_id)
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer",
                        lambda **kw: _StubGatherer([_candidate_paper("W100"), _candidate_paper("W101")]))
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings)

    c = connect(db_path)
    papers_repo.upsert(c, {"openalex_id": "W2", "title": "Seed W2", "abstract": "words " * 20, "source": "openalex"})
    embeddings_repo.upsert(c, "W2", "specter2", [0.9, 0, 0, 0, 0, 0, 0, 0.9])
    profiles_repo.attach_seed(c, profile_id, "W2")
    c.execute("INSERT INTO deleted_seeds (profile_id, openalex_id, title) VALUES (?, 'W9', 'gone')", (profile_id,))
    # A row off the pool's scale: a rescore would put it back on it. A scan
    # that got nothing touches no score, this one's either.
    c.execute("UPDATE profile_candidates SET score_pct = 0.123, score_blended = 0.123 "
              "WHERE profile_id = ? AND openalex_id = 'W101'", (profile_id,))
    c.commit()
    c.close()
    before = _state(db_path, profile_id)
    assert before["scores"]

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", _Refused)
    run_id = scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, tier="manual")
    c = connect(db_path)
    error = gather_runs_repo.get(c, run_id)["error"]
    c.close()
    assert error and "Nothing was changed." in error, error
    assert _state(db_path, profile_id) == before

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: _StubGatherer([_candidate_paper("W100")]))
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings)
    c = connect(db_path)
    try:
        assert wizard.fit_seed_ids(profiles_repo.get(c, profile_id)) == ["W1", "W2"]
    finally:
        c.close()
    assert _state(db_path, profile_id)["scores"] != before["scores"]


def _miss(scheduler, job_id):
    """Hand ``job_id`` to the real executor as a worker would pick it up
    after 11 minutes in the queue: past the 600 s grace, APScheduler skips
    it and dispatches EVENT_JOB_MISSED (review SF-02)."""
    import time
    from datetime import datetime, timedelta, timezone

    job = scheduler.get_job(job_id)
    late = datetime.now(timezone.utc) - timedelta(minutes=11)
    scheduler._lookup_executor("default").submit_job(job, [late])
    for _ in range(100):
        if job_id not in __import__("rag_lib.scheduler.runner", fromlist=["_on_missed"])._on_missed:
            return
        time.sleep(0.05)


def _scheduler(monkeypatch, db_path):
    from rag_lib.api import settings as settings_module
    from rag_lib.scheduler import runner

    monkeypatch.setenv("RADAR_DB_PATH", str(db_path))
    settings_module.get_settings.cache_clear()
    sched = runner.build_scheduler(None)
    sched.start(paused=True)
    return sched


def test_cr6_a_scan_dropped_unrun_closes_its_run(monkeypatch, db_path, settings, profile_id):
    from datetime import datetime, timezone

    from rag_lib.api.routers.profiles import _schedule_run
    from rag_lib.api import settings as settings_module

    ran = []
    monkeypatch.setattr(scheduler_jobs, "gather_for_profile", lambda *a, **k: ran.append(1))
    sched = _scheduler(monkeypatch, db_path)
    try:
        c = connect(db_path)
        run_id = gather_runs_repo.start(c, profile_id=profile_id, user_id=1, tier_used="manual")
        job_id = f"gather-now:{profile_id}:{run_id}"
        _schedule_run(sched, c, run_id, "manual", scheduler_jobs.gather_for_profile,
                      args=[1, profile_id], kwargs={"run_id": run_id}, id=job_id)
        _miss(sched, job_id)
        row = gather_runs_repo.get(c, run_id)
        assert not ran
        assert row["finished_at"] is not None and "did not start" in row["error"] and "try again" in row["error"]
        assert gather_runs_repo.running_scan(c, profile_id) is None
        c.close()
    finally:
        sched.shutdown(wait=False)
        settings_module.get_settings.cache_clear()


def test_cr6_a_job_that_runs_leaves_its_run_to_itself(monkeypatch, db_path, settings, profile_id):
    """Ran -> the missed callback is dropped: nothing closes the run behind it."""
    import time

    from rag_lib.api.routers.profiles import _schedule_run
    from rag_lib.api import settings as settings_module
    from rag_lib.scheduler import runner

    sched = _scheduler(monkeypatch, db_path)
    try:
        c = connect(db_path)
        run_id = gather_runs_repo.start(c, profile_id=profile_id, user_id=1, tier_used="manual")
        job_id = f"gather-now:{profile_id}:{run_id}"
        _schedule_run(sched, c, run_id, "manual", lambda *a, **k: None, args=[], kwargs={}, id=job_id)
        sched.resume()
        for _ in range(100):
            if job_id not in runner._on_missed:
                break
            time.sleep(0.05)
        assert job_id not in runner._on_missed
        assert gather_runs_repo.get(c, run_id)["finished_at"] is None   # the job's own to close
        c.close()
    finally:
        sched.shutdown(wait=False)
        settings_module.get_settings.cache_clear()


def test_cr6_the_seed_umls_flag_goes_when_its_job_is_dropped(monkeypatch, db_path):
    from rag_lib.api import settings as settings_module
    from rag_lib.api.routers.profiles import _start_seed_umls
    from rag_lib.scheduler.jobs import SEED_UMLS_RUNNING

    sched = _scheduler(monkeypatch, db_path)
    try:
        _start_seed_umls(sched, 77)
        assert 77 in SEED_UMLS_RUNNING
        _miss(sched, "seed-umls:77")
        assert 77 not in SEED_UMLS_RUNNING
    finally:
        SEED_UMLS_RUNNING.discard(77)
        sched.shutdown(wait=False)
        settings_module.get_settings.cache_clear()


def test_cr6_an_import_dropped_unrun_closes_its_run_and_takes_back_its_draft(monkeypatch, db_path):
    from types import SimpleNamespace

    from rag_lib.api import settings as settings_module
    from rag_lib.api.routers.prosopia import _schedule_import

    c = connect(db_path)
    pid = profiles_repo.create_draft(c, user_id=1, name="Imported", slug="imported")
    run_id = gather_runs_repo.start(c, profile_id=pid, user_id=1, tier_used="orcid")
    c.commit()
    plan = SimpleNamespace(profile_id=pid, researcher_id=None, new_researcher=False, new_draft=True,
                           draft_slug="imported", embedding_model="specter2")
    sched = _scheduler(monkeypatch, db_path)
    try:
        job_id = _schedule_import(sched, plan, run_id=run_id, tier="orcid")
        _miss(sched, job_id)
        row = gather_runs_repo.get(c, run_id)
        assert row["finished_at"] is not None and "import did not start" in row["error"]
        assert profiles_repo.get(c, pid) is None          # the draft it opened went
        c.close()
    finally:
        sched.shutdown(wait=False)
        settings_module.get_settings.cache_clear()


def test_l7_a_second_save_skipped_as_busy_keeps_the_first_saves_recommendations(db_path, settings, profile_id):
    """Review SF-03: the first save's scan holds the interest and has stored
    its candidates; a second save's job is skipped (SCAN_BUSY) and leaves
    them."""
    from rag_lib.persistence.db_store import dedup_and_insert_candidates

    c = connect(db_path)
    c.execute("UPDATE profiles SET is_draft = 1 WHERE id = ?", (profile_id,))
    c.commit()
    run1 = gather_runs_repo.start(c, profile_id=profile_id, user_id=1, tier_used="first_scan")
    dedup_and_insert_candidates(c, profile_id=profile_id, gather_run_id=run1, tier_used="first_scan",
                                ranked=[(0.9, _candidate_paper(w), {"score_raw": 0.9}) for w in ("W100", "W101")])
    lock = scheduler_jobs._scan_locks.setdefault(profile_id, scheduler_jobs.threading.Lock())
    lock.acquire()
    try:
        run2 = gather_runs_repo.start(c, profile_id=profile_id, user_id=1, tier_used="first_scan")
        scheduler_jobs.save_draft(1, profile_id, run_id=run2, settings=settings)
    finally:
        lock.release()
    assert gather_runs_repo.get(c, run2)["error"] == scheduler_jobs.SCAN_BUSY
    n = c.execute("SELECT COUNT(*) FROM profile_candidates WHERE profile_id = ?", (profile_id,)).fetchone()[0]
    c.close()
    assert n == 2


def test_l7_a_failed_first_scan_still_leaves_the_draft_empty(monkeypatch, db_path, settings, profile_id):
    """The other side: the save's own scan failed -> what it stored goes (D)."""
    c = connect(db_path)
    c.execute("UPDATE profiles SET is_draft = 1 WHERE id = ?", (profile_id,))
    c.commit()
    run = gather_runs_repo.start(c, profile_id=profile_id, user_id=1, tier_used="first_scan")
    c.close()

    class StoresThenFails(_StubGatherer):
        def fetch(self, *a, **kw):
            raise RuntimeError("boom after a chunk")

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: StoresThenFails([]))
    c = connect(db_path)
    from rag_lib.persistence.db_store import dedup_and_insert_candidates
    dedup_and_insert_candidates(c, profile_id=profile_id, gather_run_id=run, tier_used="first_scan",
                                ranked=[(0.9, _candidate_paper("W100"), {"score_raw": 0.9})])
    c.close()
    scheduler_jobs.save_draft(1, profile_id, run_id=run, settings=settings)
    c = connect(db_path)
    n = c.execute("SELECT COUNT(*) FROM profile_candidates WHERE profile_id = ?", (profile_id,)).fetchone()[0]
    c.close()
    assert n == 0



def test_l9_a_burst_429_is_told_as_busy_not_as_the_daily_limit():
    """Review SF-05: a 429 with requests left is a burst. A minute's pause,
    said as such; the day's limit only when none are left."""
    import time

    import pytest

    from rag_lib import openalex_status
    from rag_lib.openalex_client import OpenAlexClient

    class Resp:
        def __init__(self, headers):
            self.status_code, self.headers, self.text = 429, headers, "slow down"

    class Session:
        headers: dict = {}

        def __init__(self, answer):
            self.answer = answer

        def get(self, url, params=None, timeout=None):
            return self.answer

    def ask(headers):
        openalex_status.reset()
        with pytest.raises(OpenAlexUnavailable) as e:
            OpenAlexClient(mailto="t@example.com", session=Session(Resp(headers)), rate_limit_sleep=0).get_work("W1")
        return e.value

    try:
        busy = ask({"X-RateLimit-Remaining": "93512", "X-RateLimit-Reset": "1"})
        assert busy.kind == "busy"
        assert busy.user_message == "OpenAlex is busy (too many requests at once). Try again in a minute. Nothing was changed."
        st = openalex_status.status()
        assert st["kind"] == "busy" and "daily" not in st["message"]
        assert 55 <= openalex_status.current()["until"] - time.time() <= 60

        quota = ask({"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "3600"})
        assert quota.kind == "quota" and "daily request limit" in quota.user_message
        assert "Paused until" in quota.user_message
    finally:
        openalex_status.reset()


def test_b2_1_a_scan_with_every_topic_off_still_fits_changed_seeds(monkeypatch, db_path, settings, profile_id):
    """Review B2-1: no OpenAlex call, a clean run -- the seeds that changed
    since the fit are fitted, and the run records them."""
    import json

    _make_live(db_path, profile_id)
    c = connect(db_path)
    papers_repo.upsert(c, {"openalex_id": "W2", "title": "Seed W2", "abstract": "words " * 20, "source": "openalex"})
    embeddings_repo.upsert(c, "W2", "specter2", [0.9, 0, 0, 0, 0, 0, 0, 0.9])
    profiles_repo.attach_seed(c, profile_id, "W2")
    tf = json.loads(profiles_repo.get(c, profile_id)["topic_filters_json"])
    for t in tf["topics"]:
        t["on"] = False
    c.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(tf), profile_id))
    c.commit()
    c.close()
    asked = []
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: asked.append(1) or _StubGatherer([_candidate_paper("W100")]))
    run_id = scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, tier="manual")
    c = connect(db_path)
    run = gather_runs_repo.get(c, run_id)
    fit = wizard.fit_seed_ids(profiles_repo.get(c, profile_id))
    c.close()
    assert not asked
    assert run["error"] is None and run["tier_used"] == "no-enabled-topics"
    assert fit == ["W1", "W2"]
    assert json.loads(run["seed_ids_json"]) == ["W1", "W2"]
