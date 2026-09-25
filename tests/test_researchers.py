"""Researchers: a stored person, their papers, and interests built from them.

Offline like the two import suites — the same Prosopia and OpenAlex
stubs, so a paper that came in on the ``doi`` rung there comes in on
the ``doi`` rung here. What this file adds is the bookkeeping: the
person row exists whichever route imported them, the papers are
attached to the person and embedded once, a second interest from the
same person costs no embedding, and the wizard's own imports leave the
researcher behind too.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import users as users_repo
from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex
from tests.test_prosopia_import import (
    StubProsopia,
    _stub_openalex,
    _stub_profile,
)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    monkeypatch.setenv("RADAR_DB_PATH", str(db))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("RADAR_DEFAULT_EMBEDDING_MODEL", "placeholder-v1")
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")
    settings_module.get_settings.cache_clear()

    conn = connect(db)
    apply_migrations(conn)
    users_repo.upsert(conn, "demo@example.com")
    conn.close()

    yield db
    settings_module.get_settings.cache_clear()


def _app(prosopia=None, openalex=None):
    from rag_lib.api.app import create_app
    from rag_lib.api.routers.prosopia import (
        get_import_openalex_client,
        get_prosopia_client,
    )

    app = create_app()
    if prosopia is not None:
        app.dependency_overrides[get_prosopia_client] = lambda: prosopia
    if openalex is not None:
        app.dependency_overrides[get_import_openalex_client] = lambda: openalex
    return app


def _request(app, method, path, **kwargs) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test",
            ) as client:
                return await client.request(method, path, **kwargs)

    return asyncio.run(_run())


def _import(app, **body) -> httpx.Response:
    return _request(app, "POST", "/api/researchers/import", json=body)


def _count(env, sql, *params) -> int:
    conn = connect(env)
    try:
        return int(conn.execute(sql, params).fetchone()[0])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Importing a person
# ---------------------------------------------------------------------------


def test_prosopia_import_stores_the_person_and_their_papers(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    resp = _import(app, source="prosopia", ref="sheffield-nathan")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"researcher_id", "run_id"}

    # No draft: a researcher import is not an interest.
    assert _count(env, "SELECT COUNT(*) FROM profiles") == 0

    detail = _request(app, "GET", f"/api/researchers/{body['researcher_id']}").json()
    r = detail["researcher"]
    assert r["source"] == "prosopia"
    assert r["key"] == "sheffield-nathan"
    assert r["name"] == "Nathan C. Sheffield"
    assert r["n_papers"] == 5
    assert r["n_interests"] == 0
    assert r["importing"] is False
    assert r["last_run_id"] == body["run_id"]
    assert r["imported_at"]
    assert detail["interests"] == []

    papers = detail["papers"]
    assert len(papers) == 5
    by_id = {p["id"]: p for p in papers}
    assert by_id["https://openalex.org/W4415881950"]["resolved_by"] == "work_id"
    assert by_id["https://openalex.org/W2184563578"]["resolved_by"] == "pmcid"
    ghost = by_id["prosopia:sheffield-nathan:ghost2020nowhere"]
    assert ghost["resolved_by"] == "none"
    assert ghost["summary"] == "Only the Prosopia summary describes this work."

    # Every paper is embedded under the configured model, so an
    # interest built from it needs no embedding.
    n_vecs = _count(
        env,
        """
        SELECT COUNT(*) FROM researcher_papers rp
        JOIN paper_embeddings pe USING (openalex_id)
        WHERE rp.researcher_id = ? AND pe.embedding_model = 'placeholder-v1'
        """,
        body["researcher_id"],
    )
    assert n_vecs == 5

    # The run is readable through the shared status route and says
    # which rung each paper came in on.
    status = _request(app, "GET", f"/api/import/prosopia/{body['run_id']}").json()
    assert status["run"]["profile_id"] is None
    assert status["run"]["researcher_id"] == body["researcher_id"]
    assert status["run"]["tier_used"] == "researcher_import"
    assert status["result"]["draft_slug"] is None
    assert status["result"]["researcher_id"] == body["researcher_id"]
    assert status["result"]["drafted"] == 5


def test_orcid_import_stores_the_person_with_every_work(env):
    app = _app(openalex=StubOrcidOpenAlex(WORKS))
    resp = _import(app, source="orcid", ref=f"https://orcid.org/{ORCID}")
    assert resp.status_code == 200, resp.text
    rid = resp.json()["researcher_id"]

    listing = _request(app, "GET", "/api/researchers").json()
    assert [r["id"] for r in listing] == [rid]
    r = listing[0]
    assert r["source"] == "orcid"
    assert r["key"] == ORCID
    assert r["orcid"] == ORCID
    assert r["name"] == "Justin C. Niestroy"
    assert r["url"] == f"https://orcid.org/{ORCID}"
    assert r["n_papers"] == 3


def test_orcid_import_keeps_a_selection(env):
    app = _app(openalex=StubOrcidOpenAlex(WORKS))
    resp = _import(app, source="orcid", ref=ORCID, paper_ids=["W1", "https://openalex.org/W3"])
    assert resp.status_code == 200, resp.text
    detail = _request(app, "GET", f"/api/researchers/{resp.json()['researcher_id']}").json()
    assert {p["id"] for p in detail["papers"]} == {
        "https://openalex.org/W1", "https://openalex.org/W3",
    }


def test_reimport_refreshes_the_same_row(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    first = _import(app, source="prosopia", ref="sheffield-nathan").json()
    second = _import(app, source="prosopia", ref="https://prosopia.databio.org/sheffield-nathan").json()
    assert first["researcher_id"] == second["researcher_id"]
    assert second["run_id"] != first["run_id"]
    assert _count(env, "SELECT COUNT(*) FROM researchers") == 1
    assert _count(env, "SELECT COUNT(*) FROM researcher_papers") == 5
    listing = _request(app, "GET", "/api/researchers").json()
    assert listing[0]["last_run_id"] == second["run_id"]


def test_import_rejects_bad_input(env):
    app = _app(StubProsopia(_stub_profile()), StubOrcidOpenAlex(WORKS))
    assert _import(app, source="scholar", ref="x").status_code == 400
    assert _import(app, source="prosopia", ref="  ").status_code == 400
    assert _import(app, source="orcid", ref="not-an-orcid").status_code == 400
    assert _import(app, source="orcid", ref=ORCID, paper_ids=[]).status_code == 400
    assert _import(app, source="orcid", ref=ORCID, paper_ids=["W999"]).status_code == 404
    assert _import(app, source="prosopia", ref="sheffield-nathan", paper_ids=["nope"]).status_code == 404
    assert _count(env, "SELECT COUNT(*) FROM researchers") == 0


def test_unknown_profile_is_404_and_stores_nothing(env):
    from rag_lib.prosopia_client import ProfileNotFound

    app = _app(StubProsopia(error=ProfileNotFound("nope")), _stub_openalex())
    resp = _import(app, source="prosopia", ref="nobody")
    assert resp.status_code == 404
    assert "nobody" in resp.json()["detail"]
    assert _count(env, "SELECT COUNT(*) FROM researchers") == 0


def test_researchers_are_per_user(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    rid = _import(app, source="prosopia", ref="sheffield-nathan").json()["researcher_id"]
    other = {"X-User-Email": "someone-else@example.com"}
    assert _request(app, "GET", "/api/researchers", headers=other).json() == []
    assert _request(app, "GET", f"/api/researchers/{rid}", headers=other).status_code == 404
    assert _request(app, "DELETE", f"/api/researchers/{rid}", headers=other).status_code == 404
    assert _request(
        app, "POST", f"/api/researchers/{rid}/interests", json={"name": "Mine"}, headers=other,
    ).status_code == 404


# ---------------------------------------------------------------------------
# Building interests
# ---------------------------------------------------------------------------


def test_an_interest_from_a_researcher_is_seeded_without_embedding(env):
    openalex = _stub_openalex()
    app = _app(StubProsopia(_stub_profile()), openalex)
    rid = _import(app, source="prosopia", ref="sheffield-nathan").json()["researcher_id"]
    calls_after_import = openalex.api_calls

    resp = _request(
        app, "POST", f"/api/researchers/{rid}/interests",
        json={"name": "Region sets", "openalex_ids": [
            "https://openalex.org/W4415881950",
            "https://openalex.org/W2184563578",
            "not-theirs",
        ]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["draft"]["slug"] == "region-sets"
    assert body["n_seeds"] == 2
    # Synchronous and offline: nothing was resolved or embedded again.
    assert openalex.api_calls == calls_after_import

    conn = connect(env)
    try:
        row = conn.execute(
            "SELECT id, is_draft, researcher_id, n_seed FROM profiles WHERE slug = 'region-sets'",
        ).fetchone()
        assert row["is_draft"] == 1
        assert row["researcher_id"] == rid
        assert row["n_seed"] == 2
        seeds = {
            r["openalex_id"] for r in conn.execute(
                "SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", (row["id"],),
            )
        }
        assert seeds == {"https://openalex.org/W4415881950", "https://openalex.org/W2184563578"}
    finally:
        conn.close()

    # The wizard's next step works on it straight away.
    coh = _request(app, "POST", "/api/profiles/draft/region-sets/coherence")
    assert coh.status_code == 200, coh.text
    assert coh.json()["n"] == 2

    # The draft's seeds are listed under its tag in the vault, which is
    # what the wizard's seed step shows.
    docs = _request(app, "GET", "/api/vault/docs", params={"tag": "region-sets"}).json()
    assert len(docs) == 2

    # The interest points back at the person, and the person counts it.
    profiles = _request(app, "GET", "/api/profiles").json()
    assert [(p["key"], p["isDraft"], p["researcherId"]) for p in profiles] == [("region-sets", True, rid)]
    detail = _request(app, "GET", f"/api/researchers/{rid}").json()
    assert detail["researcher"]["n_interests"] == 1
    assert [i["key"] for i in detail["interests"]] == ["region-sets"]
    assert detail["interests"][0]["researcherId"] == rid
    assert detail["interests"][0]["isDraft"] is True


def test_several_interests_from_one_researcher(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    rid = _import(app, source="prosopia", ref="sheffield-nathan").json()["researcher_id"]
    a = _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "All of it"}).json()
    b = _request(
        app, "POST", f"/api/researchers/{rid}/interests",
        json={"name": "Just one", "openalex_ids": ["https://openalex.org/W4415881950"]},
    ).json()
    assert a["n_seeds"] == 5
    assert b["n_seeds"] == 1
    detail = _request(app, "GET", f"/api/researchers/{rid}").json()
    assert {i["key"] for i in detail["interests"]} == {a["draft"]["slug"], b["draft"]["slug"]}
    assert detail["researcher"]["n_interests"] == 2


def test_interest_creation_rejects_bad_input(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    rid = _import(app, source="prosopia", ref="sheffield-nathan").json()["researcher_id"]
    post = lambda body, r=rid: _request(app, "POST", f"/api/researchers/{r}/interests", json=body)
    assert post({"name": "  "}).status_code == 400
    assert post({"name": "x", "openalex_ids": []}).status_code == 400
    assert post({"name": "x", "openalex_ids": ["not-theirs"]}).status_code == 400
    assert post({"name": "x"}, r=999).status_code == 404
    assert _count(env, "SELECT COUNT(*) FROM profiles") == 0


def test_deleting_a_researcher_keeps_papers_and_interests(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    rid = _import(app, source="prosopia", ref="sheffield-nathan").json()["researcher_id"]
    slug = _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "Keep"}).json()["draft"]["slug"]

    assert _request(app, "DELETE", f"/api/researchers/{rid}").status_code == 200
    assert _request(app, "GET", f"/api/researchers/{rid}").status_code == 404
    assert _request(app, "GET", "/api/researchers").json() == []
    assert _count(env, "SELECT COUNT(*) FROM researcher_papers") == 0
    assert _count(env, "SELECT COUNT(*) FROM papers") == 5
    assert _count(env, "SELECT COUNT(*) FROM profile_seeds") == 5
    conn = connect(env)
    try:
        row = conn.execute("SELECT researcher_id FROM profiles WHERE slug = ?", (slug,)).fetchone()
        assert row["researcher_id"] is None
    finally:
        conn.close()
    # Its run rows went with it; the status route says so.
    assert _count(env, "SELECT COUNT(*) FROM gather_runs") == 0


# ---------------------------------------------------------------------------
# The wizard's own imports leave a researcher behind
# ---------------------------------------------------------------------------


def test_wizard_prosopia_import_records_the_researcher(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    resp = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan"})
    assert resp.status_code == 200, resp.text
    draft_slug = resp.json()["draft_slug"]

    listing = _request(app, "GET", "/api/researchers").json()
    assert len(listing) == 1
    rid = listing[0]["id"]
    assert listing[0]["n_papers"] == 5
    assert listing[0]["n_interests"] == 1
    detail = _request(app, "GET", f"/api/researchers/{rid}").json()
    assert [i["key"] for i in detail["interests"]] == [draft_slug]

    # A second interest from the stored person, this time without a job.
    second = _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "Second"})
    assert second.status_code == 200, second.text
    assert second.json()["n_seeds"] == 5


def test_wizard_orcid_import_records_the_researcher(env):
    app = _app(openalex=StubOrcidOpenAlex(WORKS))
    resp = _request(
        app, "POST", "/api/import/orcid",
        json={"orcid": ORCID, "openalex_ids": ["W1", "W2"]},
    )
    assert resp.status_code == 200, resp.text
    listing = _request(app, "GET", "/api/researchers").json()
    assert len(listing) == 1
    assert listing[0]["source"] == "orcid"
    assert listing[0]["n_papers"] == 2
    assert listing[0]["n_interests"] == 1


# ---------------------------------------------------------------------------
# Attaching stored papers to any draft
# ---------------------------------------------------------------------------


def test_stored_papers_can_be_attached_to_a_plain_draft(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    _import(app, source="prosopia", ref="sheffield-nathan")
    slug = _request(app, "POST", "/api/profiles/draft", json={"name": "Mixed"}).json()["slug"]

    resp = _request(
        app, "POST", f"/api/profiles/draft/{slug}/seeds",
        json={"openalex_ids": [
            "https://openalex.org/W4415881950", "https://openalex.org/W4415881950",
            "prosopia:sheffield-nathan:ghost2020nowhere", "https://openalex.org/W-not-mine",
        ]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"attached": 2, "rejected": ["https://openalex.org/W-not-mine"]}
    assert _count(env, "SELECT n_seed FROM profiles WHERE slug = ?", slug) == 2

    assert _request(app, "POST", f"/api/profiles/draft/{slug}/seeds", json={"openalex_ids": []}).status_code == 400
    assert _request(app, "POST", "/api/profiles/draft/none/seeds", json={"openalex_ids": ["x"]}).status_code == 404
