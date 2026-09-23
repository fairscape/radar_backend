"""Prosopia pick list: list a profile's papers, import only the chosen ones."""

from __future__ import annotations

from rag_lib.db import connect
from tests.test_prosopia_import import (  # noqa: F401 — ``env`` is a fixture
    StubProsopia,
    _app,
    _request,
    _stub_openalex,
    _stub_profile,
    env,
)


def _works(app, ref: str):
    return _request(app, "GET", "/api/import/prosopia/works", params={"ref": ref})


def test_works_lists_the_profile_papers(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    resp = _works(app, "sheffield-nathan")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["slug"] == "sheffield-nathan"
    assert body["name"] == "Nathan C. Sheffield"
    ids = {w["id"] for w in body["works"]}
    assert "leroy2025atacformer" in ids
    assert len(body["works"]) == 5
    full = next(w for w in body["works"] if w["id"] == "leroy2025atacformer")
    assert full["doi"] == "10.1101/2025.11.03.685753"
    assert full["openalex_id"] == "W4415881950"
    assert full["year"] == 2025


def test_works_404s_like_the_kickoff(env):
    from rag_lib.prosopia_client import ProfileNotFound

    app = _app(StubProsopia(error=ProfileNotFound("nope")), _stub_openalex())
    resp = _works(app, "nobody-here")
    assert resp.status_code == 404
    assert "nobody-here" in resp.json()["detail"]
    # Read-only: nothing was created.
    conn = connect(env)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM profiles").fetchone()["n"] == 0
    finally:
        conn.close()


def test_import_keeps_only_the_chosen_papers(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    resp = _request(app, "POST", "/api/import/prosopia",
                    json={"ref": "sheffield-nathan", "paper_ids": ["leroy2025atacformer", "smith2023protein"]})
    assert resp.status_code == 200, resp.text
    status = _request(app, "GET", f"/api/import/prosopia/{resp.json()['run_id']}").json()
    assert status["run"]["error"] is None
    assert status["result"]["drafted"] == 2


def test_import_without_paper_ids_takes_everything(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    resp = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan"})
    assert resp.status_code == 200, resp.text
    status = _request(app, "GET", f"/api/import/prosopia/{resp.json()['run_id']}").json()
    assert status["result"]["drafted"] == 5


def test_import_rejects_an_empty_or_foreign_selection(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    resp = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "paper_ids": []})
    assert resp.status_code == 400
    resp = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "paper_ids": ["not-there"]})
    assert resp.status_code == 404
    assert "sheffield-nathan" in resp.json()["detail"]
