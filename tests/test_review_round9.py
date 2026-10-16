"""Round 9: a sixth independent review's findings (legacy data, adversarial input)."""

from __future__ import annotations

import inspect
import shutil

import pytest
from pydantic import ValidationError

from rag_lib.api.schemas import ResearcherInterestRequest
from rag_lib.api.services import vault as vault_service, wizard
from rag_lib.db import apply_migrations, connect
from rag_lib.db.migrate import MIGRATIONS_DIR
from rag_lib.db.repos import (
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    researchers as researchers_repo,
)
from tests.test_review_round3 import _requests, app  # noqa: F401  (fixture)
from tests.test_review_round4 import _db, _live

ORCID = "0000-0001-5643-4068"


def test_a_trial_status_poll_with_an_import_run_id_is_a_404(app, tmp_path):
    slug = _requests(app, [("POST", "/api/profiles/draft", {"json": {"name": "D"}})])[0].json()["slug"]
    conn = connect(tmp_path / "radar.db")
    rid = researchers_repo.upsert(conn, user_id=1, source="orcid", key=ORCID, name="P", orcid=ORCID, url="x")
    run = gather_runs_repo.start(conn, profile_id=None, researcher_id=rid, user_id=1, tier_used="researcher_import")
    conn.close()
    res = _requests(app, [("GET", f"/api/profiles/draft/{slug}/dry-run/{run}", {})])[0]
    assert res.status_code == 404


def test_an_interest_name_from_a_profile_is_bounded():
    with pytest.raises(ValidationError):
        ResearcherInterestRequest(name="x" * 121)
    with pytest.raises(ValidationError):
        ResearcherInterestRequest(name="")
    assert ResearcherInterestRequest(name="x" * 120).name


def test_an_upload_into_a_draft_is_embedded_under_the_drafts_model(tmp_path):
    conn, uid = _db(tmp_path)
    d = wizard.create_draft(conn, user_id=uid, name="D", embedding_model="placeholder-v1")
    papers_repo.upsert(conn, {"openalex_id": "local:abc", "title": "Notes", "body_text": "words " * 20, "source": "user_pdf"})
    vault_service._attach_to_profile(conn, uid, d["slug"], "local:abc")
    assert embeddings_repo.get(conn, "local:abc", "placeholder-v1") is not None


def test_coherence_over_seeds_without_vectors_keeps_the_seed_count(tmp_path):
    conn, uid = _db(tmp_path)
    d = wizard.create_draft(conn, user_id=uid, name="D", embedding_model="placeholder-v1")
    pid = int(profiles_repo.get_by_slug(conn, uid, d["slug"])["id"])
    for oa in ("https://openalex.org/W1", "https://openalex.org/W2"):
        papers_repo.upsert(conn, {"openalex_id": oa, "title": "t", "source": "openalex"})
        profiles_repo.attach_seed(conn, pid, oa)
    conn.execute("UPDATE profiles SET n_seed = 2 WHERE id = ?", (pid,))
    conn.commit()
    wizard.compute_draft_coherence(conn, user_id=uid, slug=d["slug"])
    assert profiles_repo.get_by_slug(conn, uid, d["slug"])["n_seed"] == 2


def test_the_draft_routes_refuse_a_live_interest(app):
    res = _requests(app, [
        ("POST", "/api/profiles/draft/chromatin/coherence", {}),
        ("GET", "/api/profiles/draft/chromatin/topics", {}),
    ])
    assert [r.status_code for r in res] == [404, 404]


def test_no_temporary_prints_in_the_reranker():
    from rag_lib.rerankers.medcpt import MedCPTReranker
    assert "print(" not in inspect.getsource(MedCPTReranker.rerank)


def test_a_taken_slug_does_not_rename_a_free_name(tmp_path):
    conn, uid = _db(tmp_path)
    first = wizard.create_draft(conn, user_id=uid, name="a" * 120, embedding_model="placeholder-v1")
    second = wizard.create_draft(conn, user_id=uid, name="a" * 119 + "b", embedding_model="placeholder-v1")
    assert first["slug"] == "a" * 80 and second["slug"] == "a" * 80 + "-2"
    assert second["name"] == "a" * 119 + "b"
    # The same name again is still renamed.
    third = wizard.create_draft(conn, user_id=uid, name="a" * 120, embedding_model="placeholder-v1")
    assert third["name"] == "a" * 120 + " (2)"


def test_imported_at_waits_for_the_import_to_finish(tmp_path):
    conn, uid = _db(tmp_path)
    rid = researchers_repo.upsert(conn, user_id=uid, source="orcid", key=ORCID, name="P", orcid=ORCID, url="x")
    run = gather_runs_repo.start(conn, profile_id=None, researcher_id=rid, user_id=uid, tier_used="researcher_import")
    researchers_repo.set_last_run(conn, rid, run)
    row = researchers_repo.get(conn, rid)
    assert row["last_run_id"] == run and row["imported_at"] is None
    researchers_repo.mark_imported(conn, rid, run_id=run)
    assert researchers_repo.get(conn, rid)["imported_at"] is not None


def test_open_scan_profiles_lists_scans_only(tmp_path):
    conn, uid = _db(tmp_path)
    a, b, c = (_live(conn, uid, n) for n in ("A", "B", "C"))
    gather_runs_repo.start(conn, profile_id=a, user_id=uid, tier_used="manual")
    gather_runs_repo.start(conn, profile_id=b, user_id=uid, tier_used="dry_run")
    gather_runs_repo.start(conn, profile_id=c, user_id=uid, tier_used="orcid_import")
    done = gather_runs_repo.start(conn, profile_id=c, user_id=uid, tier_used="manual")
    gather_runs_repo.finish(conn, done, n_fetched=0, n_new=0, n_redup=0)
    assert gather_runs_repo.open_scan_profiles(conn) == [a]


def test_legacy_candidates_get_a_raw_score(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    for f in MIGRATIONS_DIR.glob("*.sql"):
        if f.stem < "0022":
            shutil.copy(f, old / f.name)
    shutil.copy(MIGRATIONS_DIR / "0021_profile_names_per_user.py", old / "0021_profile_names_per_user.py")
    conn = connect(tmp_path / "l.db")
    apply_migrations(conn, old)
    conn.execute("INSERT INTO papers (openalex_id, title, source) VALUES ('W1', 't', 'openalex')")
    conn.execute("INSERT INTO profiles (user_id, name, slug, embedding_model, n_seed, topic_filters_json) VALUES (1, 'P', 'p', 'm', 1, '{}')")
    conn.execute("INSERT INTO profile_candidates (profile_id, openalex_id, score, score_raw, fetched_at)"
                 " VALUES (1, 'W1', 0.83, NULL, datetime('now'))")
    conn.commit()
    assert "0022_score_raw_backfill" in apply_migrations(conn)
    assert conn.execute("SELECT score_raw FROM profile_candidates").fetchone()[0] == 0.83


def test_an_import_stamps_imported_at_once_when_it_finishes(monkeypatch):
    """The kick-off used to call mark_imported too, so the researcher read
    "imported just now" while the import ran (or after it failed)."""
    from rag_lib.db.repos import researchers as repo
    from tests.test_researchers import StubProsopia, _app, _import, _stub_openalex, _stub_profile

    calls = []
    real = repo.mark_imported
    monkeypatch.setattr(repo, "mark_imported", lambda *a, **k: calls.append(k.get("run_id")) or real(*a, **k))
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    resp = _import(app, source="prosopia", ref="sheffield-nathan")
    assert resp.status_code == 200, resp.text
    assert len(calls) == 1          # the job, at the end; not the kick-off as well
