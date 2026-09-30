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

#: The user the db fixture seeds; the identity these tests authenticate as.
SEEDED_USER = "demo@example.com"


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
    users_repo.upsert(conn, SEEDED_USER)
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
                # Every route here is behind get_current_user, which reads
                # the identity Caddy injects. Sent as a client default so a
                # single test can still override or drop it; without it the
                # whole file 401s before reaching what it means to assert.
                headers={"X-User-Email": SEEDED_USER},
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


#: A preprint and its published version: the same paper, a year apart, so
#: the newest-first sort would put them in different year blocks. The title
#: has to survive ``title_key``, which returns "" below MIN_TITLE_KEY_LEN
#: (30 alphanumerics) so that every paper called "Editorial" does not
#: collapse into one -- a short title never dedupes by title at all.
#: The case differs on purpose: that is the kind of difference between two
#: copies that normalizing away is the point of.
PREPRINT_PAIR = WORKS + [
    _work("W4", "10.1/four",
          "Fast, memory-efficient genomic interval tokenizers", 2025, cited=40),
    _work("W5", None,
          "Fast, Memory-Efficient Genomic Interval Tokenizers", 2024, cited=2),
]


def test_list_works_keeps_one_copy_of_a_paper_and_points_the_rest_at_it():
    """A preprint and its published version must not both become seeds.

    Both stay in the list -- the rule is a heuristic and the user is
    looking at it -- but only one is left for the picker to tick, so the
    centroid does not count the paper twice.
    """
    works = list_works(ORCID, StubOrcidOpenAlex(PREPRINT_PAIR))["works"]
    by_id = {w["openalex_id"]: w for w in works}
    assert len(works) == 5, "no row may be dropped"

    # W4 is kept over W5: it has a DOI and more citations.
    assert by_id["W4"]["duplicate_of"] is None
    assert by_id["W5"]["duplicate_of"] == "W4"
    assert [w["openalex_id"] for w in works if w["duplicate_of"] is None] == [
        "W3", "W2", "W4", "W1",
    ]


def test_list_works_puts_a_duplicate_directly_under_the_copy_it_duplicates():
    """Adjacency, not just a label, is what makes the pair legible.

    Sorted by year alone, W5 (2024) would land next to W1 (2024) and a
    whole year block away from the W4 (2025) it duplicates. The user
    scrolling past it meets a title they already saw and reads the list as
    full of duplicates -- which is exactly the report this guards.
    """
    works = list_works(ORCID, StubOrcidOpenAlex(PREPRINT_PAIR))["works"]
    order = [w["openalex_id"] for w in works]
    assert order == ["W3", "W2", "W4", "W5", "W1"]
    assert order.index("W5") == order.index("W4") + 1


def test_grouping_keeps_a_duplicate_whose_original_is_missing():
    """The orphan branch: an unreachable parent must not swallow the row.

    _mark_duplicates cannot produce this -- it only ever points at a copy
    it just chose from the same list -- so this calls the reordering
    directly. Without the trailing loop the row is silently dropped, and a
    paper vanishing from the picker is far worse than one sitting in an odd
    place.
    """
    from rag_lib.api.services.orcid import _group_duplicates

    rows = [
        {"openalex_id": "W1", "duplicate_of": None},
        {"openalex_id": "W2", "duplicate_of": "W_NOT_HERE"},
        {"openalex_id": "W3", "duplicate_of": None},
    ]
    out = _group_duplicates(rows)
    assert [r["openalex_id"] for r in out] == ["W1", "W3", "W2"]


def _typed(oid, doi, title, year, typ, *, authors=("Sadnan Al Manir", "Clark, Timothy")):
    w = _work(oid, doi, title, year)
    w["type"] = typ
    w["authorships"] = [
        {"author": {"display_name": a, "orcid": f"https://orcid.org/{ORCID}" if i == 0 else None},
         "author_position": "first" if i == 0 else "last"}
        for i, a in enumerate(authors)
    ]
    return w


def test_a_short_title_merges_when_year_and_authors_agree():
    """A Zenodo concept DOI and version DOI: title too short to trust alone.

    "EVI: Evidence Graph Ontology v1.0" is 27 alphanumerics, under
    MIN_TITLE_KEY_LEN, so title_key refuses it and the two copies -- DOIs
    ...527 and ...528 -- used to be listed twice.
    """
    works = list_works(ORCID, StubOrcidOpenAlex([
        _typed("W1", "10.5281/zenodo.7903528", "EVI: Evidence Graph Ontology v1.0", 2023, "dataset"),
        _typed("W2", "10.5281/zenodo.7903527", "EVI: Evidence Graph Ontology v1.0", 2023, "dataset"),
    ]))["works"]
    assert sorted(w["duplicate_of"] is None for w in works) == [False, True]


def test_a_short_title_alone_is_not_enough():
    """Same short title and a versioned type, but a different year or author."""
    works = list_works(ORCID, StubOrcidOpenAlex([
        _typed("W1", "10.5281/zenodo.1", "Test ROCrate", 2020, "dataset"),
        _typed("W2", "10.5281/zenodo.2", "Test ROCrate", 2021, "dataset"),
        _typed("W3", "10.5281/zenodo.3", "Test ROCrate", 2021, "dataset", authors=("Someone Else",)),
    ]))["works"]
    assert all(w["duplicate_of"] is None for w in works)


def test_an_editors_editorials_stay_apart_even_in_one_year():
    """The reason the floor exists: every "Editorial" is not one paper.

    One author, one year, the same one-word title -- identical on every
    field a summary has. Only the type says these are separate pieces.
    """
    works = list_works(ORCID, StubOrcidOpenAlex([
        _typed("W1", "10.1/e1", "Editorial", 2021, "editorial", authors=("Jane Smith",)),
        _typed("W2", "10.1/e2", "Editorial", 2021, "editorial", authors=("Jane Smith",)),
    ]))["works"]
    assert all(w["duplicate_of"] is None for w in works)


def test_a_tool_and_the_preprint_about_it_stay_apart():
    """Same short title, year and authors, both mergeable types -- but a
    software release and the preprint describing it are two works."""
    works = list_works(ORCID, StubOrcidOpenAlex([
        _typed("W1", "10.5281/zenodo.9", "scTools", 2023, "software"),
        _typed("W2", "10.1101/2023.01.01.1", "scTools", 2023, "preprint"),
    ]))["works"]
    assert all(w["duplicate_of"] is None for w in works)


def test_chapters_sharing_a_truncated_book_title_stay_apart():
    """Two chapters of one book, both titled with the book's name.

    Measured on a real record: pages 3-27 and 254-270 of "Coexisting with
    Large Carnivores", same year, same two authors in either order --
    nothing in a summary separates them, so chapters never merge on a
    short title.
    """
    works = list_works(ORCID, StubOrcidOpenAlex([
        _typed("W1", "10.2307/jj.41003712.4", "Coexisting with Large Carnivores:", 2013,
               "book-chapter", authors=("Tim W. Clark", "Murray B. Rutherford")),
        _typed("W2", "10.2307/jj.41003712.11", "Coexisting with Large Carnivores:", 2013,
               "book-chapter", authors=("Murray B. Rutherford", "Tim W. Clark")),
    ]))["works"]
    assert all(w["duplicate_of"] is None for w in works)


def test_list_works_leaves_the_order_alone_when_nothing_is_duplicated():
    works = list_works(ORCID, StubOrcidOpenAlex(WORKS))["works"]
    assert [w["openalex_id"] for w in works] == ["W3", "W2", "W1"]
    assert all(w["duplicate_of"] is None for w in works)


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
