"""Round 5: the second independent review's findings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_lib.api.services import orcid as orcid_service, radar, vault, wizard
from rag_lib.db.repos import (
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    researchers as researchers_repo,
)
from tests.test_review_round3 import _requests, app  # noqa: F401  (fixture)
from tests.test_review_round4 import _BODY, _SELF_CITE, _db, _draft_mid_import, _live

ORCID = "0000-0001-5643-4068"


class _DoiClient:
    def __init__(self, *a, **k):
        pass

    def lookup_by_doi(self, doi):
        return dict(_SELF_CITE, doi="https://doi.org/" + doi)

    def search_by_title(self, title):
        return []


def test_a_rejected_doi_does_not_stay_on_the_unmatched_paper(tmp_path, monkeypatch):
    """The DOI check rejects a reference's DOI, but it stayed on the local:
    paper; seeds match by DOI, so the cited paper was kept out of the feed
    as if it were a seed."""
    monkeypatch.setattr(vault, "OpenAlexClient", _DoiClient)
    record = SimpleNamespace(title="Fast Sequence Collections for Genome Assemblies", body_text=_BODY,
                             doi="10.1093/bioinformatics/btab524", doi_from_body=True)
    oa, paper = vault._enrich_via_openalex(record, mailto="a@b.c", filename="f.pdf", file_hash="h1")
    assert oa == "local:h1" and paper["doi"] is None

    # A DOI printed on page 1 that OpenAlex resolves to some other paper
    # (neither title nor first author fit) is dropped on that ground alone.
    class _Other(_DoiClient):
        def lookup_by_doi(self, doi):
            return {"title": "Something else entirely", "doi": "https://doi.org/" + doi,
                    "authorships": [{"author": {"display_name": "Zed Zog"}}]}
    monkeypatch.setattr(vault, "OpenAlexClient", _Other)
    front = SimpleNamespace(title=record.title, body_text=_BODY, doi="10.1/other", doi_from_body=False)
    assert vault._enrich_via_openalex(front, mailto="a@b.c", filename="f.pdf", file_hash="h4")[1]["doi"] is None
    monkeypatch.setattr(vault, "OpenAlexClient", _DoiClient)

    conn, uid = _db(tmp_path)
    pid = _live(conn, uid, "I", threshold=0.1)
    papers_repo.upsert(conn, dict(paper, openalex_id=oa))
    profiles_repo.attach_seed(conn, pid, oa)
    cited = "https://openalex.org/W111"
    conn.execute("INSERT INTO papers (openalex_id, title, doi, source) VALUES (?, 'Refget', ?, 'openalex')",
                 (cited, "10.1093/bioinformatics/btab524"))
    conn.execute("INSERT INTO profile_candidates (profile_id, openalex_id, score, score_raw, fetched_at)"
                 " VALUES (?, ?, 0.9, 0.9, datetime('now'))", (pid, cited))
    conn.commit()
    assert [c.id for c in radar.daily(conn, uid).cards] == [cited]


def test_a_body_doi_openalex_does_not_know_is_dropped_too(monkeypatch):
    class _Unknown(_DoiClient):
        def lookup_by_doi(self, doi):
            return None
    monkeypatch.setattr(vault, "OpenAlexClient", _Unknown)
    record = SimpleNamespace(title="A note", body_text="x", doi="10.1/ref", doi_from_body=True)
    assert vault._enrich_via_openalex(record, mailto="a@b.c", filename="f.pdf", file_hash="h2")[1]["doi"] is None
    # On page 1 it is the paper's own, and stays.
    record.doi_from_body = False
    assert vault._enrich_via_openalex(record, mailto="a@b.c", filename="f.pdf", file_hash="h3")[1]["doi"] == "10.1/ref"


def _running_import(conn, uid, draft_slug, *, draft_for_this_person: bool, orcid=ORCID):
    pid = int(profiles_repo.get_by_slug(conn, uid, draft_slug)["id"])
    rid = researchers_repo.upsert(conn, user_id=uid, source="orcid", key=orcid, name="P", orcid=orcid, url="x")
    if draft_for_this_person:
        researchers_repo.set_profile_researcher(conn, pid, rid)
    gather_runs_repo.start(conn, profile_id=pid, researcher_id=rid, user_id=uid, tier_used="orcid_import")
    return rid


def test_a_new_draft_request_does_not_join_an_add_more_seeds_import(tmp_path):
    """P's works are being added into draft A (an upload draft). A new
    interest from P must not be answered with draft A."""
    conn, uid = _db(tmp_path)
    a = wizard.create_draft(conn, user_id=uid, name="Draft A", embedding_model="placeholder-v1")
    _running_import(conn, uid, a["slug"], draft_for_this_person=False)
    with pytest.raises(orcid_service.ImportAlreadyRunning) as e:
        orcid_service.refuse_if_importing(conn, uid, ORCID, None)
    assert e.value.slug is None
    # Into A itself: that run.
    with pytest.raises(orcid_service.ImportAlreadyRunning) as e:
        orcid_service.refuse_if_importing(conn, uid, ORCID, a["slug"])
    assert e.value.slug == a["slug"]


def test_a_new_draft_request_joins_the_draft_opened_for_that_person(tmp_path):
    conn, uid = _db(tmp_path)
    a = wizard.create_draft(conn, user_id=uid, name="P's papers", embedding_model="placeholder-v1")
    _running_import(conn, uid, a["slug"], draft_for_this_person=True)
    with pytest.raises(orcid_service.ImportAlreadyRunning) as e:
        orcid_service.refuse_if_importing(conn, uid, ORCID, None)
    assert e.value.slug == a["slug"]


def test_a_person_without_an_orcid_is_found_by_their_row(tmp_path):
    conn, uid = _db(tmp_path)
    a = wizard.create_draft(conn, user_id=uid, name="Q", embedding_model="placeholder-v1")
    pid = int(profiles_repo.get_by_slug(conn, uid, a["slug"])["id"])
    rid = researchers_repo.upsert(conn, user_id=uid, source="prosopia", key="q-no-orcid", name="Q", orcid=None, url="x")
    researchers_repo.set_profile_researcher(conn, pid, rid)
    gather_runs_repo.start(conn, profile_id=pid, researcher_id=rid, user_id=uid, tier_used="prosopia_import")
    orcid_service.refuse_if_importing(conn, uid, None, None)      # nothing to go on: no error
    with pytest.raises(orcid_service.ImportAlreadyRunning) as e:
        orcid_service.refuse_if_importing(conn, uid, None, None, researcher_id=rid)
    assert e.value.slug == a["slug"]


def test_import_start_lock_is_per_user():
    assert orcid_service.import_start_lock(1) is orcid_service.import_start_lock(1)
    assert orcid_service.import_start_lock(1) is not orcid_service.import_start_lock(2)


def test_mid_import_the_draft_cannot_be_deleted_or_its_topics_read(tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")
    from rag_lib.api import settings as settings_module
    settings_module.get_settings.cache_clear()
    conn, uid = _db(tmp_path)
    slug, pid, run = _draft_mid_import(conn, uid)
    with pytest.raises(wizard.ImportInProgress):
        wizard.delete_draft(conn, user_id=uid, slug=slug)
    with pytest.raises(wizard.ImportInProgress):
        wizard.aggregate_draft_topics(conn, user_id=uid, slug=slug)
    assert profiles_repo.get_by_slug(conn, uid, slug) is not None
    gather_runs_repo.finish(conn, run, n_fetched=4, n_new=4, n_redup=0)
    wizard.aggregate_draft_topics(conn, user_id=uid, slug=slug)
    assert wizard.delete_draft(conn, user_id=uid, slug=slug) is True
    settings_module.get_settings.cache_clear()


def test_the_routes_answer_409_mid_import(app, tmp_path):
    slug = _requests(app, [("POST", "/api/profiles/draft", {"json": {"name": "Mid import"}})])[0].json()["slug"]
    from rag_lib.db import connect
    conn = connect(tmp_path / "radar.db")
    pid = int(conn.execute("SELECT id FROM profiles WHERE slug = ?", (slug,)).fetchone()[0])
    rid = researchers_repo.upsert(conn, user_id=1, source="orcid", key=ORCID, name="P", orcid=ORCID, url="x")
    gather_runs_repo.start(conn, profile_id=pid, researcher_id=rid, user_id=1, tier_used="orcid_import")
    conn.close()
    res = _requests(app, [
        ("POST", f"/api/profiles/{slug}/recompute-coherence", {}),
        ("GET", f"/api/profiles/draft/{slug}/topics", {}),
        ("DELETE", f"/api/profiles/draft/{slug}", {}),
        ("DELETE", f"/api/researchers/{rid}", {}),
    ])
    assert [r.status_code for r in res] == [409, 409, 409, 409]


def test_a_prosopia_profile_without_an_orcid_is_guarded_too(tmp_path):
    from rag_lib.api import settings as settings_module
    from rag_lib.api.services import prosopia as prosopia_service
    from rag_lib.db import apply_migrations, connect
    from tests.test_review_round4 import _Prof

    class _NoRid(_Prof):
        def __init__(self):
            super().__init__()
            self.rid = ""

    class _Client:
        def resolve_orcid(self, o):
            return "doe-jane"

        def fetch_profile(self, slug):
            return _NoRid()

    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    s = settings_module.get_settings()
    plan = prosopia_service.prepare_import(conn, s, user_id=1, ref="doe-jane", prosopia_client=_Client())
    run = gather_runs_repo.start(conn, profile_id=plan.profile_id, researcher_id=plan.researcher_id,
                                 user_id=1, tier_used="prosopia_import")
    with pytest.raises(orcid_service.ImportAlreadyRunning) as e:
        prosopia_service.prepare_import(conn, s, user_id=1, ref="doe-jane", prosopia_client=_Client())
    assert (e.value.slug, e.value.run_id) == (plan.draft_slug, run)
