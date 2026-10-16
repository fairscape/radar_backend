"""Round 6: the seams a third independent review found loose."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_lib.api.services import vault as vault_service, wizard
from rag_lib.db import connect
from rag_lib.db.repos import (
    candidates as candidates_repo,
    gather_runs as gather_runs_repo,
    profiles as profiles_repo,
    researchers as researchers_repo,
)
from tests.test_review_round3 import _requests, app  # noqa: F401  (fixture)
from tests.test_review_round4 import _db, _draft_mid_import, _live

ORCID = "0000-0001-5643-4068"


def _drafts(conn, uid):
    return {r["slug"]: r for r in profiles_repo.list_drafts_for_user(conn, uid)}


# --- 1. an import that died half-way says so ---------------------------------


def test_a_half_done_import_reports_its_error_beside_the_seeds(tmp_path):
    conn, uid = _db(tmp_path)
    slug, pid, run = _draft_mid_import(conn, uid)          # 2 of 4 seeds attached
    assert _drafts(conn, uid)[slug]["phase"] == "importing"
    assert _drafts(conn, uid)[slug]["import_error"] is None
    gather_runs_repo.finish(conn, run, n_fetched=4, n_new=2, n_redup=0, error="RuntimeError: boom")
    d = _drafts(conn, uid)[slug]
    assert d["phase"] == "seeded" and d["n_seeds"] == 2
    assert d["import_error"] == "RuntimeError: boom"
    # A later import that finishes clears it: only the latest run counts.
    run2 = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="prosopia_import")
    gather_runs_repo.finish(conn, run2, n_fetched=2, n_new=2, n_redup=0)
    assert _drafts(conn, uid)[slug]["import_error"] is None


def test_the_drafts_route_passes_the_error_whatever_the_phase(app, tmp_path):
    slug = _requests(app, [("POST", "/api/profiles/draft", {"json": {"name": "Half done"}})])[0].json()["slug"]
    conn = connect(tmp_path / "radar.db")
    pid = int(conn.execute("SELECT id FROM profiles WHERE slug = ?", (slug,)).fetchone()[0])
    profiles_repo.attach_seed(conn, pid, "https://openalex.org/W1")
    run = gather_runs_repo.start(conn, profile_id=pid, user_id=1, tier_used="orcid_import")
    gather_runs_repo.finish(conn, run, n_fetched=3, n_new=1, n_redup=0, error="OpenAlex 429")
    conn.close()
    d = next(x for x in _requests(app, [("GET", "/api/profiles/drafts", {})])[0].json() if x["slug"] == slug)
    assert (d["phase"], d["n_seeds"], d["import_error"]) == ("seeded", 1, "OpenAlex 429")


# --- 2. runs started in the same second keep a stable order -------------------


def test_runs_started_in_the_same_second_list_newest_first(tmp_path):
    conn, uid = _db(tmp_path)
    pid = _live(conn, uid, "I")
    a = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="dry_run")
    b = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    conn.execute("UPDATE gather_runs SET started_at = '2026-10-15 12:00:00' WHERE id IN (?, ?)", (a, b))
    conn.commit()
    assert [r["id"] for r in gather_runs_repo.recent_for_profile(conn, pid)] == [b, a]


# --- 4. deleting a draft keeps the researcher's import history ---------------


def test_deleting_a_draft_keeps_the_researchers_import_run(tmp_path):
    conn, uid = _db(tmp_path)
    d = wizard.create_draft(conn, user_id=uid, name="P's papers", embedding_model="placeholder-v1")
    pid = int(profiles_repo.get_by_slug(conn, uid, d["slug"])["id"])
    rid = researchers_repo.upsert(conn, user_id=uid, source="orcid", key=ORCID, name="P", orcid=ORCID, url="x")
    run = gather_runs_repo.start(conn, profile_id=pid, researcher_id=rid, user_id=uid, tier_used="orcid_import")
    gather_runs_repo.finish(conn, run, n_fetched=5, n_new=5, n_redup=0)
    researchers_repo.mark_imported(conn, rid, run_id=run)
    assert wizard.delete_draft(conn, user_id=uid, slug=d["slug"]) is True
    row = conn.execute("SELECT profile_id, researcher_id FROM gather_runs WHERE id = ?", (run,)).fetchone()
    assert row is not None and row["profile_id"] is None and row["researcher_id"] == rid
    assert researchers_repo.get(conn, rid)["last_run_id"] == run


# --- 5. no scan on a draft ----------------------------------------------------


def test_gather_now_refuses_a_draft(app):
    slug = _requests(app, [("POST", "/api/profiles/draft", {"json": {"name": "Just a draft"}})])[0].json()["slug"]
    res = _requests(app, [("POST", f"/api/profiles/{slug}/gather-now?days=7&limit=10", {})])[0]
    assert res.status_code == 404


# --- 6. seed edits wait for the import on every path -------------------------


def test_seed_edits_answer_409_mid_import(app, tmp_path):
    slug = _requests(app, [("POST", "/api/profiles/draft", {"json": {"name": "Mid import"}})])[0].json()["slug"]
    conn = connect(tmp_path / "radar.db")
    pid = int(conn.execute("SELECT id FROM profiles WHERE slug = ?", (slug,)).fetchone()[0])
    profiles_repo.attach_seed(conn, pid, "https://openalex.org/W1")
    gather_runs_repo.start(conn, profile_id=pid, user_id=1, tier_used="orcid_import")
    conn.close()
    res = _requests(app, [
        ("POST", f"/api/profiles/{slug}/seeds", {"json": {"openalex_ids": ["https://openalex.org/W1"]}}),
        ("POST", f"/api/profiles/draft/{slug}/seeds", {"json": {"openalex_ids": ["https://openalex.org/W1"]}}),
        ("DELETE", f"/api/profiles/{slug}/seeds?openalex_id=https://openalex.org/W1", {}),
    ])
    assert [r.status_code for r in res] == [409, 409, 409]


def test_an_upload_into_an_importing_draft_is_refused_before_anything_is_stored(tmp_path):
    conn, uid = _db(tmp_path)
    slug, pid, run = _draft_mid_import(conn, uid)
    with pytest.raises(vault_service.UploadRejected) as e:
        vault_service.upload(conn, SimpleNamespace(), user_id=uid, filename="x.pdf",
                             data=b"%PDF-1.4\nhello", profile_slug=slug)
    assert e.value.status_code == 409
    assert conn.execute("SELECT COUNT(*) FROM paper_uploads").fetchone()[0] == 0


# --- 7. the threshold tab's scores respect the interest's years --------------


def test_scores_and_suggestion_stay_within_the_years(tmp_path):
    conn, uid = _db(tmp_path)
    pid = _live(conn, uid, "I")
    profiles_repo.update_year_range(conn, pid, 2020, None)
    for oa, year, s in (("W-old", 2015, 0.99), ("W-new", 2023, 0.80), ("W-none", None, 0.70)):
        conn.execute("INSERT INTO papers (openalex_id, title, year, source) VALUES (?, 't', ?, 'openalex')", (oa, year))
        conn.execute("INSERT INTO profile_candidates (profile_id, openalex_id, score, score_raw, fetched_at)"
                     " VALUES (?, ?, ?, ?, datetime('now'))", (pid, oa, s, s))
    conn.commit()
    assert candidates_repo.scores_for_profile(conn, pid) == [0.80, 0.70]        # no year passes, as the feed
    assert [r["openalex_id"] for r in candidates_repo.scored_papers_for_profile(conn, pid)] == ["W-new", "W-none"]


# --- 9. the wizard's year-range trial, through the app -------------------------


def test_the_year_range_trial_samples_the_years_asked_for():
    from tests.test_api_wizard import _attach_seeds, _request, app as wizard_app  # noqa: F401
    import tests.test_api_wizard as tw
    import tempfile, pathlib, pytest as _pt
    # Drive the fixture the way that module does.
    tmp = pathlib.Path(tempfile.mkdtemp())
    gen = tw.app.__wrapped__(tmp, _pt.MonkeyPatch()) if hasattr(tw.app, "__wrapped__") else None
    assert gen is not None, "fixture shape changed"
    application = next(gen)
    try:
        slug = _request(application, "POST", "/api/profiles/draft",
                        json={"name": "Years", "embedding_model": "placeholder-v1"}).json()["slug"]
        _attach_seeds(str(tmp / "radar.db"), user_id=1, slug=slug)
        start = _request(application, "POST", f"/api/profiles/draft/{slug}/dry-run",
                         json={"year_from": 2019, "year_to": 2021})
        assert start.status_code == 200, start.text
        status = _request(application, "GET", f"/api/profiles/draft/{slug}/dry-run/{start.json()['run_id']}").json()
        years = {p["year"] for p in status["result"]["papers"]}
        assert years and years <= {2019, 2020, 2021}
    finally:
        try:
            next(gen)
        except StopIteration:
            pass
