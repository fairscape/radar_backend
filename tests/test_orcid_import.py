"""ORCID import: list an author's OpenAlex works, import the chosen ones.

Offline like the Prosopia suite — the OpenAlex client is a stub that
answers ``author.orcid:`` and ``doi:`` filters from canned works.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.api.services.orcid import list_works, normalize_orcid
from rag_lib.api.services.prosopia import normalize_ref
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import users as users_repo
from tests.fake_openalex_client import canned_openalex_work
from tests.test_prosopia_import import StubOpenAlex


ORCID = "0000-0002-1103-3882"
OTHER = "0000-0001-5643-4068"


def _work(oid: str, doi: str, title: str, year: int, *, cited: int = 0, orcid: str = ORCID, position: str = "first"):
    w = canned_openalex_work(
        doi=doi, openalex_id=f"https://openalex.org/{oid}", title=title,
        year=year, venue="A Journal", abstract_words=["an", "abstract"],
    )
    w["cited_by_count"] = cited
    w["authorships"] = [
        {"author": {"display_name": "Justin C. Niestroy", "orcid": f"https://orcid.org/{orcid}"},
         "author_position": position},
        {"author": {"display_name": "Someone Else", "orcid": None}, "author_position": "last"},
    ]
    return w


WORKS = [
    _work("W1", "10.1/one", "Vital signs of low-birth-weight infants", 2024, cited=5),
    _work("W2", "10.1/two", "FAIRSCAPE server", 2026, cited=1),
    _work("W3", "10.1/three", "AI-readiness grader", 2026, cited=9),
]


class StubOrcidOpenAlex(StubOpenAlex):
    """The Prosopia stub plus an ``author.orcid:`` answer."""

    def __init__(self, works):
        super().__init__(
            by_id={w["id"].rsplit("/", 1)[-1]: w for w in works},
            by_doi={w["doi"].replace("https://doi.org/", ""): w for w in works},
        )
        self.works = list(works)

    def paginate_filter(self, filter_str, *, limit=None, per_page=200):
        if filter_str.startswith("author.orcid:"):
            self.api_calls += 1
            self.calls.append(("paginate_filter", filter_str))
            orcid = filter_str[len("author.orcid:"):]
            return [
                w for w in self.works
                if any(((a.get("author") or {}).get("orcid") or "").endswith(orcid)
                       for a in w["authorships"])
            ][: limit or None]
        return super().paginate_filter(filter_str, limit=limit, per_page=per_page)


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


def _app(openalex):
    from rag_lib.api.app import create_app
    from rag_lib.api.routers.prosopia import get_import_openalex_client

    app = create_app()
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


# --- pure helpers ----------------------------------------------------------


def test_normalize_orcid_accepts_ids_and_urls():
    assert normalize_orcid("0000-0002-1103-3882") == ORCID
    assert normalize_orcid("https://orcid.org/0000-0002-1103-3882/") == ORCID
    assert normalize_orcid("0000-0002-1103-388x") == "0000-0002-1103-388X"
    assert normalize_orcid("sheffield-nathan") is None
    assert normalize_orcid("") is None


def test_normalize_ref_takes_the_slug_from_a_deeper_prosopia_url():
    assert normalize_ref(
        "https://prosopia.databio.org/api/v1/profiles/sheffield-nathan/content/profile.jsonld"
    ) == "sheffield-nathan"
    assert normalize_ref("https://prosopia.databio.org/sheffield-nathan/content/") == "sheffield-nathan"
    assert normalize_ref("https://prosopia.databio.org/api/v1/profiles/sheffield-nathan/") == "sheffield-nathan"


def test_list_works_reads_the_name_off_the_authorships_and_sorts_newest_first():
    listing = list_works(ORCID, StubOrcidOpenAlex(WORKS))
    assert listing["name"] == "Justin C. Niestroy"
    assert [w["openalex_id"] for w in listing["works"]] == ["W3", "W2", "W1"]
    top = listing["works"][0]
    assert top["doi"] == "10.1/three"
    assert top["authors"] == ["Justin C. Niestroy", "Someone Else"]
    assert top["author_position"] == "first"


def test_list_works_for_an_orcid_with_nothing_has_no_name():
    listing = list_works(OTHER, StubOrcidOpenAlex(WORKS))
    assert listing == {"orcid": OTHER, "name": None, "works": []}


# --- routes ----------------------------------------------------------------


def test_works_route_lists_and_rejects_non_orcids(env):
    app = _app(StubOrcidOpenAlex(WORKS))
    resp = _request(app, "GET", f"/api/import/orcid/{ORCID}/works")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "Justin C. Niestroy"
    assert len(body["works"]) == 3

    resp = _request(app, "GET", "/api/import/orcid/sheffield-nathan/works")
    assert resp.status_code == 400
    assert "ORCID" in resp.json()["detail"]


def test_import_keeps_only_the_chosen_works(env):
    stub = StubOrcidOpenAlex(WORKS)
    app = _app(stub)
    resp = _request(app, "POST", "/api/import/orcid",
                    json={"orcid": f"https://orcid.org/{ORCID}", "openalex_ids": ["W1", "W3"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["draft_slug"] == "justin-c-niestroy"
    run_id = body["run_id"]

    status = _request(app, "GET", f"/api/import/prosopia/{run_id}").json()
    assert status["run"]["tier_used"] == "orcid_import"
    assert status["run"]["error"] is None
    assert status["result"]["drafted"] == 2
    assert status["result"]["resolved_by"]["work_id"] == 2
    assert status["result"]["name"] == "Justin C. Niestroy"

    conn = connect(env)
    try:
        seeds = conn.execute(
            "SELECT openalex_id FROM profile_seeds ORDER BY openalex_id"
        ).fetchall()
        # Seeds are keyed however the client's ``paper_from_work`` keys
        # them (the stub keeps the URL form); the ids are what matter.
        assert sorted(r["openalex_id"].rsplit("/", 1)[-1] for r in seeds) == ["W1", "W3"]
    finally:
        conn.close()


def test_import_names_the_draft_as_asked(env):
    app = _app(StubOrcidOpenAlex(WORKS))
    resp = _request(app, "POST", "/api/import/orcid",
                    json={"orcid": ORCID, "openalex_ids": ["W2"], "name": "Infra papers"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["draft_slug"] == "infra-papers"


def test_import_rejects_bad_input(env):
    app = _app(StubOrcidOpenAlex(WORKS))
    resp = _request(app, "POST", "/api/import/orcid", json={"orcid": "nope", "openalex_ids": ["W1"]})
    assert resp.status_code == 400
    resp = _request(app, "POST", "/api/import/orcid", json={"orcid": ORCID, "openalex_ids": []})
    assert resp.status_code == 400
    resp = _request(app, "POST", "/api/import/orcid", json={"orcid": ORCID, "openalex_ids": ["W999"]})
    assert resp.status_code == 404
    assert ORCID in resp.json()["detail"]

    # Nothing was created for any of those.
    conn = connect(env)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM profiles").fetchone()["n"] == 0
    finally:
        conn.close()
