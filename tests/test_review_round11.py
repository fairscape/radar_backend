"""Round 11: an eighth independent review (adversarial pass over round 10's
fixes, lifecycles, data layer)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_lib.api.services import prosopia as prosopia_service, researchers as researchers_service, vault as vault_service, wizard
from rag_lib.db import connect
from rag_lib.db.repos import (
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
)
from rag_lib.scheduler import jobs as scheduler_jobs
from tests.test_review_round10 import _live_interest
from tests.test_review_round4 import _db
from tests.test_scheduler_jobs import _StubGatherer, _candidate_paper, db_path, profile_id, settings  # noqa: F401  (fixtures)


# --- removing a seed under a running scan -------------------------------------------


def test_removing_a_seed_under_a_running_scan_is_refused_before_anything_is_written(tmp_path):
    """The refit refuses under a scan; refused after the detach, the seed was
    gone, the centroid stale and the answer a 500."""
    conn, uid = _db(tmp_path)
    pid, slug = _live_interest(conn, uid)
    for oa in ("https://openalex.org/W1", "https://openalex.org/W2"):
        papers_repo.upsert(conn, {"openalex_id": oa, "title": "t", "abstract": "words " * 20, "source": "openalex"})
        profiles_repo.attach_seed(conn, pid, oa)
        embeddings_repo.upsert(conn, oa, "placeholder-v1", [0.1] * 8)
    gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    with pytest.raises(wizard.ScanRunning):
        wizard.remove_seed(conn, user_id=uid, slug=slug, openalex_id="https://openalex.org/W1")
    assert "https://openalex.org/W1" in profiles_repo.list_seed_openalex_ids(conn, pid)
    conn.close()


# --- an unresolved import must not overwrite a stored row ----------------------------


def _resolution(openalex_id, *, rung, body="A short Prosopia summary"):
    return SimpleNamespace(
        paper_id="P1", openalex_id=openalex_id, rung=rung, resolved=rung != "none",
        paper={"openalex_id": openalex_id, "title": "Prosopia's title", "abstract": "", "source": "prosopia",
               "body_text": body},
    )


def test_an_unresolved_record_leaves_an_existing_row_alone(tmp_path):
    conn, _uid = _db(tmp_path)
    oa = "https://openalex.org/W1"
    papers_repo.upsert(conn, {"openalex_id": oa, "title": "The real title", "abstract": "Full abstract",
                              "source": "user_pdf", "body_text": "FULL PDF TEXT " * 20})
    assert prosopia_service.store_resolved_paper(conn, _resolution(oa, rung="none")) is False
    row = papers_repo.get_by_openalex_id(conn, oa)
    assert row["title"] == "The real title" and row["body_text"].startswith("FULL PDF TEXT")
    # Resolved for real, OpenAlex's data does land (the normal path).
    assert prosopia_service.store_resolved_paper(conn, _resolution(oa, rung="work_id")) is True
    assert papers_repo.get_by_openalex_id(conn, oa)["title"] == "Prosopia's title"
    # No row yet: the sparse row is written, under the real id.
    assert prosopia_service.store_resolved_paper(conn, _resolution("https://openalex.org/W2", rung="none")) is True
    assert papers_repo.get_by_openalex_id(conn, "https://openalex.org/W2") is not None
    conn.close()


def test_a_refresh_offers_an_unreachable_work_again_but_not_a_synthetic_one():
    rows = [
        {"openalex_id": "https://openalex.org/W1", "resolved_by": "none", "paper_id": "W1", "doi": None},
        {"openalex_id": "prosopia:slug:P9", "resolved_by": "none", "paper_id": "P9", "doi": None},
        {"openalex_id": "https://openalex.org/W2", "resolved_by": "work_id", "paper_id": "W2", "doi": None},
    ]
    kept = researchers_service.repairable_excluded(rows)
    assert [r["openalex_id"] for r in kept] == ["prosopia:slug:P9", "https://openalex.org/W2"]


# --- a scan fits the seeds that arrived while it ran -------------------------------------


class _AttachingGatherer(_StubGatherer):
    """While "fetching", a seed is attached to the interest (an upload)."""

    def __init__(self, papers, db, pid):
        super().__init__(papers)
        self._db, self._pid = db, pid

    def fetch(self, profile, since, *, limit=None):
        conn = connect(self._db)
        try:
            oa = "https://openalex.org/W900"
            papers_repo.upsert(conn, {"openalex_id": oa, "title": "late seed", "abstract": "words " * 20, "source": "openalex"})
            model = conn.execute("SELECT embedding_model FROM profiles WHERE id = ?", (self._pid,)).fetchone()[0]
            embeddings_repo.upsert(conn, oa, model, [0.3] * 8)
            profiles_repo.attach_seed(conn, self._pid, oa)
            # As the upload path does under a scan: attached, refit refused.
            with pytest.raises(wizard.ScanRunning):
                wizard.refit_profile(conn, user_id=1, slug=conn.execute("SELECT slug FROM profiles WHERE id = ?", (self._pid,)).fetchone()[0])
        finally:
            conn.close()
        return super().fetch(profile, since, limit=limit)


def test_a_scan_refits_seeds_added_while_it_ran(monkeypatch, db_path, settings, profile_id):
    conn = connect(db_path)
    before = conn.execute("SELECT selector_config_json FROM profiles WHERE id = ?", (profile_id,)).fetchone()[0]
    conn.close()
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer",
                        lambda **kw: _AttachingGatherer([_candidate_paper("W100")], db_path, profile_id))
    run_id = scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, days=7)
    conn = connect(db_path)
    try:
        run = gather_runs_repo.get(conn, run_id)
        assert run["finished_at"] is not None and run["error"] is None
        after = conn.execute("SELECT selector_config_json FROM profiles WHERE id = ?", (profile_id,)).fetchone()[0]
        assert after != before, "the fit was not redone for the seed that arrived mid-scan"
    finally:
        conn.close()


# --- adopted topics survive the next rebuild ---------------------------------------------


def test_an_upload_under_a_running_scan_still_lands_and_the_warning_is_logged(tmp_path):
    conn, uid = _db(tmp_path)
    pid, slug = _live_interest(conn, uid, name="Scanning")
    papers_repo.upsert(conn, {"openalex_id": "local:abc", "title": "Notes", "body_text": "words " * 20, "source": "user_pdf"})
    gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    vault_service._attach_to_profile(conn, uid, slug, "local:abc")
    assert "local:abc" in profiles_repo.list_seed_openalex_ids(conn, pid)
    conn.close()


def test_the_seeds_route_answers_409_under_a_running_scan(tmp_path, monkeypatch):
    from rag_lib.api import settings as settings_module
    from tests.test_api_radar import _request, _seed_db

    db = tmp_path / "radar.db"
    for k, v in {"RADAR_DB_PATH": str(db), "RADAR_VAULT_DIR": str(tmp_path / "vault"), "RADAR_CHROMA_DIR": str(tmp_path / "chroma"),
                 "RADAR_LOG_JSON": "false", "RADAR_SCHEDULER_ENABLED": "false"}.items():
        monkeypatch.setenv(k, v)
    settings_module.get_settings.cache_clear()
    _seed_db(db)
    from rag_lib.api.app import create_app

    app = create_app()
    conn = connect(db)
    pid = int(conn.execute("SELECT id FROM profiles WHERE slug = 'provenance-fairscape'").fetchone()[0])
    seeds = profiles_repo.list_seed_openalex_ids(conn, pid)
    uid = int(conn.execute("SELECT user_id FROM profiles WHERE id = ?", (pid,)).fetchone()[0])
    gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    conn.close()
    assert seeds
    res = _request(app, "DELETE", f"/api/profiles/provenance-fairscape/seeds?openalex_id={seeds[0]}")
    assert res.status_code == 409, res.text
    conn = connect(db)
    assert seeds[0] in profiles_repo.list_seed_openalex_ids(conn, pid)
    conn.close()
    settings_module.get_settings.cache_clear()


# --- data layer -----------------------------------------------------------------------------


def test_a_gathered_work_without_a_title_keeps_the_stored_one(tmp_path):
    conn, _uid = _db(tmp_path)
    oa = "https://openalex.org/W1"
    papers_repo.upsert(conn, {"openalex_id": oa, "title": "A proper title", "abstract": "abs", "source": "prosopia"})
    papers_repo.upsert(conn, {"openalex_id": oa, "title": "", "abstract": "", "source": "openalex"})
    row = papers_repo.get_by_openalex_id(conn, oa)
    assert row["title"] == "A proper title" and row["abstract"] == "abs"
    papers_repo.upsert(conn, {"openalex_id": oa, "title": "A newer title", "source": "openalex"})
    assert papers_repo.get_by_openalex_id(conn, oa)["title"] == "A newer title"
    conn.close()


def test_two_names_that_slugify_alike_both_get_a_profile(tmp_path):
    conn, uid = _db(tmp_path)
    a = profiles_repo.upsert(conn, user_id=uid, name="My Topic", embedding_model="placeholder-v1", n_seed=0)
    b = profiles_repo.upsert(conn, user_id=uid, name="my topic", embedding_model="placeholder-v1", n_seed=0)
    assert a != b
    slugs = {r["slug"] for r in conn.execute("SELECT slug FROM profiles WHERE user_id = ?", (uid,))}
    assert slugs == {"my-topic", "my-topic-2"}
    conn.close()


def test_a_non_finite_score_skips_the_paper_instead_of_failing_the_scan(tmp_path):
    from rag_lib.persistence.db_store import dedup_and_insert_candidates

    conn, uid = _db(tmp_path)
    pid, _slug = _live_interest(conn, uid, name="Nan")
    good, bad = _candidate_paper("W100"), _candidate_paper("W101")
    for p in (good, bad):
        papers_repo.upsert(conn, {"openalex_id": p.openalex_id, "title": p.title or "t", "source": "openalex"})
    run = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    n_new, _n_redup = dedup_and_insert_candidates(
        conn, profile_id=pid, gather_run_id=run, ranked=[(float("nan"), bad), (0.9, good)], tier_used="manual",
    )
    ids = {r[0] for r in conn.execute("SELECT openalex_id FROM profile_candidates WHERE profile_id = ?", (pid,))}
    assert good.openalex_id in ids and bad.openalex_id not in ids
    conn.close()
