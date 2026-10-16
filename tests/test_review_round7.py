"""Round 7: a fourth independent review's findings (workflows end to end)."""

from __future__ import annotations

from types import SimpleNamespace

from rag_lib.api.services import researchers as researchers_service, wizard
from rag_lib.db.repos import (
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    researchers as researchers_repo,
)
from rag_lib.scheduler import jobs
from tests.test_review_round4 import _db, _live

ORCID = "0000-0001-5643-4068"


def _researcher(conn, uid, papers=("https://openalex.org/W1", "https://openalex.org/W2")):
    rid = researchers_repo.upsert(conn, user_id=uid, source="orcid", key=ORCID, name="P", orcid=ORCID, url="x")
    for oa in papers:
        papers_repo.upsert(conn, {"openalex_id": oa, "title": f"Paper {oa[-2:]}", "abstract": "about things", "source": "openalex"})
        researchers_repo.attach_paper(conn, rid, oa)
    return rid


# --- F1: a half-done import's papers count --------------------------------------


def test_a_researchers_paper_count_is_what_is_stored_not_what_the_last_clean_import_wrote(tmp_path):
    conn, uid = _db(tmp_path)
    rid = _researcher(conn, uid)
    run = gather_runs_repo.start(conn, profile_id=None, researcher_id=rid, user_id=uid, tier_used="researcher_import")
    conn.execute("UPDATE researchers SET last_run_id = ?, n_papers = 0 WHERE id = ?", (run, rid))
    gather_runs_repo.finish(conn, run, n_fetched=3, n_new=2, n_redup=0, error="RuntimeError: GPU fell over")
    r = researchers_service.get_researcher(conn, uid, rid).researcher
    assert r.n_papers == 2 and r.importing is False and r.last_error == "RuntimeError: GPU fell over"


# --- F2: "failed" is the latest import, as the error is ----------------------


def _draft(conn, uid, name="D"):
    d = wizard.create_draft(conn, user_id=uid, name=name, embedding_model="placeholder-v1")
    return d["slug"], int(profiles_repo.get_by_slug(conn, uid, d["slug"])["id"])


def _info(conn, uid, slug):
    return next(r for r in profiles_repo.list_drafts_for_user(conn, uid) if r["slug"] == slug)


def test_a_failed_import_followed_by_a_clean_one_is_not_failed(tmp_path):
    conn, uid = _db(tmp_path)
    slug, pid = _draft(conn, uid)
    r1 = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="orcid_import")
    gather_runs_repo.finish(conn, r1, n_fetched=0, n_new=0, n_redup=0, error="boom")
    assert (_info(conn, uid, slug)["phase"], _info(conn, uid, slug)["import_error"]) == ("failed", "boom")
    r2 = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="orcid_import")
    gather_runs_repo.finish(conn, r2, n_fetched=0, n_new=0, n_redup=0)
    assert (_info(conn, uid, slug)["phase"], _info(conn, uid, slug)["import_error"]) == ("empty", None)


def test_a_run_left_open_by_a_dead_thread_is_a_failure_for_every_import_tier(tmp_path):
    """Every tier, so the cutoff and tier placeholders are bound in the right order."""
    conn, uid = _db(tmp_path)
    for tier in gather_runs_repo.IMPORT_TIERS:
        slug, pid = _draft(conn, uid, name=f"D {tier}")
        run = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used=tier)
        conn.execute("UPDATE gather_runs SET started_at = datetime('now', '-2 hours'), progress_updated_at = NULL WHERE id = ?", (run,))
        conn.commit()
        info = _info(conn, uid, slug)
        assert info["phase"] == "failed", tier
        assert info["import_error"] == "The import stopped without finishing.", tier
        assert info["import_run_id"] is None


# --- F3: "importing" has the staleness cutoff --------------------------------


def test_a_researcher_whose_import_run_went_stale_is_not_importing(tmp_path):
    conn, uid = _db(tmp_path)
    rid = _researcher(conn, uid)
    run = gather_runs_repo.start(conn, profile_id=None, researcher_id=rid, user_id=uid, tier_used="researcher_import")
    conn.execute("UPDATE researchers SET last_run_id = ? WHERE id = ?", (run, rid))
    conn.commit()
    assert researchers_service.get_researcher(conn, uid, rid).researcher.importing is True
    conn.execute("UPDATE gather_runs SET started_at = datetime('now', '-2 hours') WHERE id = ?", (run,))
    conn.commit()
    assert researchers_service.get_researcher(conn, uid, rid).researcher.importing is False


# --- F4: a scan that fails mid-persist leaves the pool on one scale -----------


def test_a_failed_scan_rescores_what_it_had_stored(tmp_path):
    conn, uid = _db(tmp_path)
    pid = _live(conn, uid, "I", threshold=0.5)
    for j, (oa, raw, pct) in enumerate((("W-a", 0.9, 0.2), ("W-b", 0.8, 1.0), ("W-c", 0.7, 0.6))):
        conn.execute("INSERT INTO papers (openalex_id, title, source) VALUES (?, 't', 'openalex')", (oa,))
        # score_pct here is the batch's own, not the pool's: out of order.
        conn.execute("INSERT INTO profile_candidates (profile_id, openalex_id, score, score_raw, score_pct, fetched_at)"
                     " VALUES (?, ?, ?, ?, ?, datetime('now'))", (pid, oa, raw, raw, pct))
    conn.commit()
    run = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    jobs._close_failed_scan(conn, pid, run, None, RuntimeError("disk full"))
    row = gather_runs_repo.get(conn, run)
    assert row["finished_at"] and row["error"] == "RuntimeError: disk full"
    pcts = dict(conn.execute("SELECT openalex_id, score_pct FROM profile_candidates WHERE profile_id = ?", (pid,)).fetchall())
    assert pcts["W-a"] > pcts["W-b"] > pcts["W-c"]          # by the pool's own ranking now


# --- F7: an interest from a profile embeds its seeds under the draft's model --


def test_an_interest_from_a_profile_embeds_the_seeds(tmp_path):
    conn, uid = _db(tmp_path)
    rid = _researcher(conn, uid)
    assert embeddings_repo.get(conn, "https://openalex.org/W1", "placeholder-v1") is None
    settings = SimpleNamespace(RADAR_DEFAULT_EMBEDDING_MODEL="placeholder-v1", RADAR_DEFAULT_SELECTOR="centroid")
    draft, n = researchers_service.create_interest(conn, settings, user_id=uid, researcher_id=rid, name="From P")
    assert n == 2
    for oa in ("https://openalex.org/W1", "https://openalex.org/W2"):
        assert embeddings_repo.get(conn, oa, "placeholder-v1") is not None
