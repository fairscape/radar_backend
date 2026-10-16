"""Round 8: a fifth independent review's findings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_lib.api.services import researchers as researchers_service
from rag_lib.db import connect
from rag_lib.db.repos import gather_runs as gather_runs_repo, researchers as researchers_repo
from rag_lib.scheduler import jobs
from tests.test_review_round3 import _requests, app  # noqa: F401  (fixture)
from tests.test_review_round4 import _db
from tests.test_review_round7 import _researcher


def test_no_interest_from_a_researcher_whose_import_is_running(tmp_path):
    conn, uid = _db(tmp_path)
    rid = _researcher(conn, uid)
    run = gather_runs_repo.start(conn, profile_id=None, researcher_id=rid, user_id=uid, tier_used="researcher_import")
    conn.execute("UPDATE researchers SET last_run_id = ? WHERE id = ?", (run, rid))
    conn.commit()
    settings = SimpleNamespace(RADAR_DEFAULT_EMBEDDING_MODEL="placeholder-v1", RADAR_DEFAULT_SELECTOR="centroid")
    with pytest.raises(researchers_service.ResearcherImportRunning):
        researchers_service.create_interest(conn, settings, user_id=uid, researcher_id=rid, name="Too soon")
    assert conn.execute("SELECT COUNT(*) FROM profiles WHERE is_draft = 1").fetchone()[0] == 0
    gather_runs_repo.finish(conn, run, n_fetched=2, n_new=2, n_redup=0)
    draft, n = researchers_service.create_interest(conn, settings, user_id=uid, researcher_id=rid, name="Now")
    assert n == 2


def test_the_route_answers_409_mid_import(app, tmp_path):
    conn = connect(tmp_path / "radar.db")
    rid = researchers_repo.upsert(conn, user_id=1, source="orcid", key="0000-0001-5643-4068", name="P",
                                  orcid="0000-0001-5643-4068", url="x")
    researchers_repo.attach_paper(conn, rid, "https://openalex.org/W1")
    run = gather_runs_repo.start(conn, profile_id=None, researcher_id=rid, user_id=1, tier_used="researcher_import")
    conn.execute("UPDATE researchers SET last_run_id = ? WHERE id = ?", (run, rid))
    conn.commit()
    conn.close()
    res = _requests(app, [("POST", f"/api/researchers/{rid}/interests", {"json": {"name": "Too soon"}})])[0]
    assert res.status_code == 409 and "still being imported" in res.json()["detail"]


def test_a_refresh_request_during_a_refresh_makes_it_go_round_again(monkeypatch):
    added = []

    class _Sched:
        def add_job(self, *a, **k):
            added.append(k.get("id"))
    monkeypatch.setattr(jobs, "_SCHEDULER", _Sched())
    jobs._refresh_again.discard(7)
    jobs.request_rerank_refresh(7)
    assert added == ["rerank-refresh:7"]
    # The flag the running loop checks between passes is set as well, so a
    # request APScheduler skips (one with its id mid-execution) is not lost.
    assert 7 in jobs._refresh_again
    jobs._refresh_again.discard(7)


def test_no_temporary_prints_in_the_topic_merge():
    import inspect
    from rag_lib.api.services import wizard
    assert "print(" not in inspect.getsource(wizard._merge_umls_topics)
