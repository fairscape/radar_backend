"""Review round 13 (reviewer a): regressions from the latest fix commits."""

from __future__ import annotations

import json

from rag_lib.api.mappers import topic_filters_to_topics
from rag_lib.db import connect
from rag_lib.db.repos import profiles as profiles_repo
from rag_lib.scheduler import jobs as scheduler_jobs
from tests.test_review_round11 import _AttachingGatherer
from tests.test_scheduler_jobs import _candidate_paper, db_path, profile_id, settings  # noqa: F401  (fixtures)
from tests.test_review_round12_scan_refit import _count_refits


def test_detail_topics_carry_the_papers_count():
    """``papers`` was added to the Topic schema and to the three topic
    endpoints of routers/profiles.py, but the profile-detail mapper (the
    Topics tab of a saved interest) still builds Topic without it, so that
    tab keeps showing the assignment count ("8 seeds" for 4)."""
    stored = {"topics": [{"id": "T1", "display_name": "One", "count": 8, "papers": 4, "on": True}]}
    out = topic_filters_to_topics(stored)
    assert out[0].count == 8
    assert out[0].papers == 4, f"detail mapper dropped papers: {out[0]}"


def test_a_seed_added_under_a_scan_of_a_fit_without_a_seed_matrix_is_still_refitted(monkeypatch, db_path, settings, profile_id):
    """Before round 12 the scan compared its seeds at start and end and refit
    on any difference, whatever the stored fit looked like. Now a fit that
    recorded neither ``fit_seed_ids`` nor a ``seed_matrix`` (this fixture's
    config: a centroid only; any full_text / plugin config) is "assumed
    current", so a seed attached while the scan runs is not fitted by it
    nor by the next scan."""
    conn = connect(db_path)
    try:
        conn.execute("UPDATE profiles SET is_draft = 0 WHERE id = ?", (profile_id,))
        conn.commit()
        cfg = json.loads(profiles_repo.get(conn, profile_id)["selector_config_json"])
        assert "seed_matrix" not in cfg and "fit_seed_ids" not in cfg
    finally:
        conn.close()
    calls = _count_refits(monkeypatch)
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer",
                        lambda **kw: _AttachingGatherer([_candidate_paper("W100")], db_path, profile_id))
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings)
    conn = connect(db_path)
    try:
        assert "https://openalex.org/W900" in profiles_repo.list_seed_openalex_ids(conn, profile_id)
    finally:
        conn.close()
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer",
                        lambda **kw: _AttachingGatherer([_candidate_paper("W101")], db_path, profile_id))
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings)
    assert calls, "W900 was attached under a scan and neither that scan nor the next fitted it"
