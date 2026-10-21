"""Review round 12 (reviewer b): one "You" at a time, an upload into an
unknown interest is refused, a failing embedder leaves no half-built draft."""

from __future__ import annotations

import io

from rag_lib.api.services import researchers as researchers_service, wizard as wizard_service
from rag_lib.db import connect
from tests.test_api_vault import _upload, app  # noqa: F401 -- fixture
from tests.test_prosopia_import import StubProsopia, _stub_openalex, _stub_profile
from tests.test_researcher_marks import _both_stubs, _order, _two
from tests.test_researchers import _app, _request, env  # noqa: F401 -- fixture


def test_marking_a_second_person_as_you_unmarks_the_first(env):
    app_ = _app(StubProsopia(_stub_profile()), _both_stubs())
    nathan, justin = _two(app_)
    _request(app_, "PATCH", f"/api/researchers/{nathan}", json={"mark": "self"})
    assert _order(app_) == [(nathan, "self", 0), (justin, None, 1)]
    resp = _request(app_, "PATCH", f"/api/researchers/{justin}", json={"mark": "self"})
    assert resp.status_code == 200, resp.text
    assert _order(app_) == [(justin, "self", 0), (nathan, None, 1)]
    assert sum(1 for _, m, _ in _order(app_) if m == "self") == 1


def test_an_upload_into_an_unknown_interest_is_a_404_and_stores_nothing(app):  # noqa: F811
    resp = _upload(app, b"some-bytes", profile_slug="not-mine")
    assert resp.status_code == 404, resp.text
    assert "no interest" in resp.json()["detail"]
    docs = _request(app, "GET", "/api/vault/docs").json()
    assert not any(d["title"] and "some-bytes" in str(d) for d in docs)
    assert _request(app, "GET", "/api/vault/stats").json()["docs"] == 0


def test_a_failing_embedder_leaves_no_draft_behind(env, monkeypatch):
    app_ = _app(StubProsopia(_stub_profile()), _both_stubs())
    nathan, _ = _two(app_)

    def boom(conn, model, ids):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(wizard_service, "embed_missing", boom)
    resp = _request(app_, "POST", f"/api/researchers/{nathan}/interests", json={"name": "Mine"})
    assert resp.status_code >= 500, resp.text
    conn = connect(env)
    try:
        assert conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 0
    finally:
        conn.close()
