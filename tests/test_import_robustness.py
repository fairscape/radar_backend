"""Importing a researcher from ORCID / Prosopia is robust (FUNCTIONAL_SPEC O).

Offline, with the stubs of the two import suites: what starts ticked when
the ORCID record cannot be read, the institution hint, a title match that
must be by the person, Prosopia changing its shape, a second profile of
one person, what an import left out, two records of one paper, the
profile's fields following the source, and a failure part-way through
the writes.
"""

from __future__ import annotations

import httpx
import pytest

from rag_lib import orcid_registry as reg
from rag_lib.api.services import orcid as orcid_service
from rag_lib.api.services import prosopia as prosopia_service
from rag_lib.db import connect
from rag_lib.db.repos import profiles as profiles_repo
from rag_lib.prosopia_client import ApiProfile, PagedApiStorage, ProfileNotPublic, ProsopiaFormatChanged
from rag_lib.prosopia_seeds import Person, resolve_seed
from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex, _work as orcid_work
from tests.test_prosopia_import import (
    RECORD_DOI_ONLY,
    RECORD_FULL,
    RECORD_TITLE_ONLY,
    StubOpenAlex,
    StubProfile,
    StubProsopia,
    StubSummaries,
    _record,
    _stub_openalex,
    _stub_profile,
    _work,
)
from tests.test_researchers import _app, _count, _import, _request, env  # noqa: F401 -- fixture


def _q(env, sql, *params):
    conn = connect(env)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _with_institution(work, name, ror=None):
    work = dict(work)
    work["authorships"] = [dict(a) for a in work["authorships"]]
    work["authorships"][0] = {**work["authorships"][0],
                              "institutions": [{"display_name": name, "ror": ror and f"https://ror.org/{ror}"}]}
    return work


# --- O-1, O-2: what starts ticked -------------------------------------------------------


def test_o1_the_listing_says_when_the_orcid_record_could_not_be_read(monkeypatch):
    listing = orcid_service.list_works(ORCID, StubOrcidOpenAlex(WORKS))
    assert listing["registry_read"] is False
    assert {w["claimed"] for w in listing["works"]} == {None}


def test_o2_a_readable_record_marks_claimed_works(monkeypatch):
    claimed = reg.ClaimedWorks(dois={"10.1/one", "10.1/three"}, n_works=2)
    monkeypatch.setattr(orcid_service, "fetch_record",
                        lambda *a, **k: reg.RegistryRecord(found=True, claimed=claimed))
    listing = orcid_service.list_works(ORCID, StubOrcidOpenAlex(WORKS))
    assert listing["registry_read"] is True
    assert {w["openalex_id"]: w["claimed"] for w in listing["works"]} == {"W1": True, "W2": False, "W3": True}


# --- O-3: the institution hint -----------------------------------------------------------


def test_o3_a_work_at_none_of_the_orcid_affiliations_is_marked(monkeypatch):
    works = [
        _with_institution(WORKS[0], "University of Virginia", "0153tk833"),
        _with_institution(WORKS[1], "Tsinghua University", "03cve4549"),
        # Named without a ROR id, the way ORCID names Duke: still the same.
        _with_institution(WORKS[2], "The Duke University"),
    ]
    record = reg.RegistryRecord(found=True, affiliations=[
        reg.Affiliation("University of Virginia", "0153tk833"), reg.Affiliation("Duke University")])
    monkeypatch.setattr(orcid_service, "fetch_record", lambda *a, **k: record)
    listing = orcid_service.list_works(ORCID, StubOrcidOpenAlex(works))
    marks = {w["openalex_id"]: (w["other_institution"], w["institutions"]) for w in listing["works"]}
    assert marks == {"W1": (False, ["University of Virginia"]), "W2": (True, ["Tsinghua University"]),
                     "W3": (False, ["The Duke University"])}
    assert listing["institution_basis"] == "orcid"
    # Only a hint: every claimed-unknown work is still offered as it was.
    assert {w["claimed"] for w in listing["works"]} == {None}


def test_o3_without_orcid_affiliations_the_works_own_institutions_are_the_basis():
    uva = [_with_institution(orcid_work(f"W{i}", f"10.1/u{i}", f"UVA paper {i}", 2020), "University of Virginia",
                             "0153tk833") for i in range(10, 15)]
    other = _with_institution(orcid_work("W20", "10.1/x", "A namesake's paper", 2021), "Tsinghua University", "03cve4549")
    bare = orcid_work("W21", "10.1/y", "No institution recorded", 2021)   # not marked
    listing = orcid_service.list_works(ORCID, StubOrcidOpenAlex(uva + [other, bare]))
    marks = {w["openalex_id"]: w["other_institution"] for w in listing["works"]}
    # The top few institutions are the basis, so the namesake's one is too
    # when it is among them (a weak basis; said in O.2 A3): with 3 places
    # and only 2 institutions, nothing is marked.
    assert listing["institution_basis"] == "works"
    assert marks["W21"] is False


def test_o3_the_works_basis_marks_a_rare_institution(monkeypatch):
    monkeypatch.setattr(orcid_service, "TOP_WORK_INSTITUTIONS", 1)
    uva = [_with_institution(orcid_work(f"W{i}", f"10.1/u{i}", f"UVA paper {i}", 2020), "University of Virginia",
                             "0153tk833") for i in range(10, 15)]
    other = _with_institution(orcid_work("W20", "10.1/x", "A namesake's paper", 2021), "Tsinghua University", "03cve4549")
    listing = orcid_service.list_works(ORCID, StubOrcidOpenAlex(uva + [other]))
    marks = {w["openalex_id"]: w["other_institution"] for w in listing["works"]}
    assert marks["W20"] is True and not any(marks[f"W{i}"] for i in range(10, 15))


# --- O-4, O-5: a title match must be the person's ----------------------------------------


def _title_client(hit):
    return StubOpenAlex(title_hits={RECORD_TITLE_ONLY.name: [hit]})


SHEFFIELD = Person.of("Nathan C. Sheffield", "0000-0001-5643-4068")


def test_o4_a_title_hit_by_someone_else_is_not_taken():
    hit = _work("10.9/other", "W9", RECORD_TITLE_ONLY.name, 2025, authors=[("Jane Roe", None)])
    res = resolve_seed(RECORD_TITLE_ONLY, _title_client(hit), slug="s", person=SHEFFIELD)
    assert res.rung == "none" and res.openalex_id.startswith("prosopia:")


def test_o4_a_title_hit_two_years_off_is_not_taken():
    hit = _work("10.9/other", "W9", RECORD_TITLE_ONLY.name, 2022, authors=[("Nathan Sheffield", None)])
    res = resolve_seed(RECORD_TITLE_ONLY, _title_client(hit), slug="s", person=SHEFFIELD)
    assert res.rung == "none"


@pytest.mark.parametrize("author", [("N. C. Sheffield", None), ("Somebody", "0000-0001-5643-4068")])
def test_o5_a_title_hit_by_the_person_is_taken(author):
    hit = _work("10.9/ok", "W9", RECORD_TITLE_ONLY.name, 2026, authors=[author])
    res = resolve_seed(RECORD_TITLE_ONLY, _title_client(hit), slug="s", person=SHEFFIELD)
    assert res.rung == "title" and res.openalex_id == "https://openalex.org/W9"


# --- O-6: Prosopia changed its shape -------------------------------------------------------


def _storage(papers_json, detail_json):
    def handler(request):
        if request.url.path.endswith("/papers"):
            return httpx.Response(200, json=papers_json)
        return httpx.Response(200, json=detail_json)

    client = httpx.Client(base_url="http://p.test", transport=httpx.MockTransport(handler))
    return PagedApiStorage(slug="jane", base_url="http://p.test", client=client)


@pytest.mark.parametrize("papers, detail", [
    ({"items": [{"paper_id": "a", "title": "A"}]}, {"fields": {"name": "Jane"}}),     # no 'results'
    ({"results": [{"paper_id": "a", "title": "A"}]}, {"profile": {"name": "Jane"}}),   # no fields / metadata
    ({"results": [{"paper_id": "a", "title": 3}, {"paper_id": "b", "title": ["x"]}]}, {"fields": {"name": "Jane"}}),   # none readable
])
def test_o6_a_shape_change_is_an_error_not_an_empty_profile(papers, detail):
    with pytest.raises(ProsopiaFormatChanged, match="interface has changed"):
        ApiProfile(_storage(papers, detail))


def test_o6_the_route_says_so_and_creates_nothing(env):
    app = _app(StubProsopia(error=ProsopiaFormatChanged("the papers list is not a list")), _stub_openalex())
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "jane"})
    assert r.status_code == 502 and "interface has changed" in r.json()["detail"]
    r = _import(app, source="prosopia", ref="jane")
    assert r.status_code == 502 and "interface has changed" in r.json()["detail"]
    assert _q(env, "SELECT COUNT(*) FROM profiles") == [(0,)]
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(0,)]


def test_o15_a_private_prosopia_profile_is_said_to_be_not_public(env):
    app = _app(StubProsopia(error=ProfileNotPublic("the Prosopia profile 'jane' is not public, so it cannot be read")))
    r = _request(app, "GET", "/api/import/prosopia/works", params={"ref": "jane"})
    assert r.status_code == 403 and "not public" in r.json()["detail"]


def test_o15_a_401_from_prosopia_is_not_public(monkeypatch):
    import rag_lib.prosopia_client as pc

    client = httpx.Client(base_url="http://p.test",
                          transport=httpx.MockTransport(lambda request: httpx.Response(401, json={"detail": "auth"})))
    real_storage = pc.PagedApiStorage
    monkeypatch.setattr(pc, "PagedApiStorage", lambda **kw: real_storage(client=client, **kw))
    with pytest.raises(ProfileNotPublic, match="not public"):
        pc.ProsopiaClient("http://p.test").fetch_profile("jane")


# --- O-8: one person, one profile ---------------------------------------------------------


def test_o8_an_orcid_import_of_a_person_stored_from_prosopia_is_refused(env):
    conn = connect(env)
    uid = conn.execute("SELECT id FROM users").fetchone()[0]
    conn.execute("INSERT INTO researchers (user_id, source, key, name, orcid) VALUES (?, 'prosopia', 'jn', 'J N', ?)",
                 (uid, ORCID))
    conn.commit()
    conn.close()
    r = _import(_app(None, StubOrcidOpenAlex(WORKS)), source="orcid", ref=ORCID)
    assert r.status_code == 409, r.text
    assert r.json()["researcher"]["name"] == "J N"
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(1,)]


def test_o8_a_refresh_of_someone_elses_profile_is_404(env):
    r = _import(_app(None, StubOrcidOpenAlex(WORKS)), source="orcid", ref=ORCID, researcher_id=777)
    assert r.status_code == 404


def test_o8_a_refresh_names_one_profile_and_cannot_reach_another_person(env):
    """``researcher_id`` lets only that one profile be refreshed: an import
    whose person is stored as a different profile is still refused."""
    conn = connect(env)
    uid = conn.execute("SELECT id FROM users").fetchone()[0]
    other = conn.execute("INSERT INTO researchers (user_id, source, key, name) VALUES (?, 'orcid', '0000-0002-1694-233X', 'Other')",
                         (uid,)).lastrowid
    conn.execute("INSERT INTO researchers (user_id, source, key, name, orcid) VALUES (?, 'prosopia', 'jn', 'J N', ?)",
                 (uid, ORCID))
    conn.commit()
    conn.close()
    r = _import(_app(None, StubOrcidOpenAlex(WORKS)), source="orcid", ref=ORCID, researcher_id=other)
    assert r.status_code == 409, r.text
    assert r.json()["researcher"]["name"] == "J N"
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(2,)]


def test_o10_a_prosopia_wizard_import_paused_at_the_start_leaves_nothing(env):
    """O.2 F for Prosopia (revised 2026-10-31: a failed import keeps no
    draft): paused before the import starts, nothing is created."""
    from rag_lib import openalex_status

    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    openalex_status.pause("quota")
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "name": "Paused P"})
    assert r.status_code == 503, r.text
    assert "draft_slug" not in r.json()
    assert _q(env, "SELECT COUNT(*) FROM profiles") == [(0,)]
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(0,)]
    assert _q(env, "SELECT COUNT(*) FROM gather_runs") == [(0,)]


# --- O-13: the profile's fields follow the source ------------------------------------------


def test_o13_a_field_prosopia_no_longer_has_is_cleared(env):
    profile = _stub_profile()
    profile.affiliation, profile.expertise = "UVA", "Genomics"
    app = _app(StubProsopia(profile), _stub_openalex())
    rid = _import(app, source="prosopia", ref="sheffield-nathan").json()["researcher_id"]
    assert _q(env, "SELECT affiliation, expertise FROM researchers") == [("UVA", "Genomics")]
    profile.affiliation, profile.expertise = None, ""
    r = _import(app, source="prosopia", ref="sheffield-nathan", researcher_id=rid,
                paper_ids=[RECORD_FULL.paper_id])
    assert r.status_code == 200, r.text
    assert _q(env, "SELECT affiliation, expertise FROM researchers") == [(None, None)]


# --- O-14: two records of one paper ----------------------------------------------------------


def test_o14_two_records_of_one_paper_are_one_seed_and_never_new(env):
    twin = _record(name="Atacformer (preprint record)", paper_id="leroy2025twin",
                   doi="10.1101/2025.11.03.685753", datePublished="2025")
    profile = StubProfile("Nathan C. Sheffield", [RECORD_FULL, twin, RECORD_DOI_ONLY], StubSummaries({}))
    app = _app(StubProsopia(profile), _stub_openalex())
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan"})
    assert r.status_code == 200, r.text
    result = _request(app, "GET", f"/api/import/prosopia/{r.json()['run_id']}").json()["result"]
    assert result["drafted"] == 2
    assert _q(env, "SELECT COUNT(*) FROM profile_seeds") == [(2,)]
    # No researcher profile saved (M.2), so nothing to mark the twin on.
    assert _q(env, "SELECT COUNT(*) FROM researcher_work_marks") == [(0,)]
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "save_profile": True})
    assert r.status_code == 200, r.text
    assert _q(env, "SELECT COUNT(*) FROM profile_seeds") == [(4,)]
    assert _q(env, "SELECT work_id, mark FROM researcher_work_marks") == [("leroy2025twin", "duplicate")]
    rid = _q(env, "SELECT id FROM researchers")[0][0]
    works = _request(app, "GET", f"/api/researchers/{rid}/works").json()
    assert "leroy2025twin" in works["saved_ids"] and works["skipped_ids"] == []


# --- O-15: the ORCID iD itself ----------------------------------------------------------------


def test_o15_an_orcid_with_no_record_and_no_works_is_not_found(env, monkeypatch):
    monkeypatch.setattr(orcid_service, "fetch_record", lambda *a, **k: reg.RegistryRecord(found=False))
    other = "0000-0001-5643-4068"
    r = _request(_app(None, StubOrcidOpenAlex(WORKS)), "GET", f"/api/import/orcid/{other}/works")
    assert r.status_code == 404 and "no ORCID record" in r.json()["detail"]


def test_o15_a_wrong_check_digit_is_400(env):
    r = _request(_app(None, StubOrcidOpenAlex(WORKS)), "GET", "/api/import/orcid/0000-0002-1103-3883/works")
    assert r.status_code == 400 and "check digit" in r.json()["detail"]


def test_o15_exactly_the_cap_is_not_truncated():
    works = [orcid_work(f"W{i}", f"10.1/{i}", f"Paper number {i}", 2020) for i in range(4)]
    assert orcid_service.list_works(ORCID, StubOrcidOpenAlex(works[:3]), limit=3)["truncated"] is False
    cut = orcid_service.list_works(ORCID, StubOrcidOpenAlex(works), limit=3)
    assert cut["truncated"] is True and len(cut["works"]) == 3


# --- the registry read ----------------------------------------------------------------------------


def test_the_registry_record_reads_works_and_affiliations(monkeypatch, live_orcid_registry):
    body = {
        "works": {"group": [{"external-ids": {"external-id": [{"external-id-type": "doi", "external-id-value": "10.1/ONE"}]},
                             "work-summary": [{"title": {"title": {"value": "Vital signs of low-birth-weight infants"}}}]}]},
        "employments": {"affiliation-group": [{"summaries": [{"employment-summary": {"organization": {
            "name": "University of Virginia",
            "disambiguated-organization": {"disambiguation-source": "ROR",
                                           "disambiguated-organization-identifier": "https://ror.org/0153tk833"}}}}]}]},
        "educations": {"affiliation-group": [{"summaries": [{"education-summary": {"organization": {
            "name": "Duke University",
            "disambiguated-organization": {"disambiguation-source": "RINGGOLD",
                                           "disambiguated-organization-identifier": "3065"}}}}]}]},
    }

    class R:
        def __init__(self, code, data):
            self.status_code, self._data = code, data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

        def json(self):
            return self._data

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: R(200, body))
    rec = reg.fetch_record(ORCID)
    assert rec.found is True and rec.claimed.dois == {"10.1/one"}
    assert [(a.name, a.ror) for a in rec.affiliations] == [("University of Virginia", "0153tk833"), ("Duke University", None)]
    monkeypatch.setattr(requests, "get", lambda *a, **k: R(404, {}))
    assert reg.fetch_record(ORCID).found is False
    monkeypatch.setattr(requests, "get", lambda *a, **k: R(500, {}))
    assert reg.fetch_record(ORCID).found is None


# --- Prosopia by ORCID: a rid stored as a URL, a paged list ----------------------------------------


def test_resolve_orcid_is_one_read_by_rid(monkeypatch):
    """A URL-form ORCID is reduced to the bare id and read in one GET; no listing."""
    calls = []

    class R:
        def __init__(self, status, data):
            self.status_code, self._data = status, data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    def get(url, params=None, timeout=None):
        calls.append(url)
        if url.endswith("/0000-0001-5643-4068"):
            return R(200, {"slug": "sheffield-nathan", "rid": "0000-0001-5643-4068"})
        return R(404, {"detail": "not found"})

    import rag_lib.prosopia_client as pc
    monkeypatch.setattr(pc.requests, "get", get)
    client = pc.ProsopiaClient("http://p.test")
    assert client.resolve_orcid("https://orcid.org/0000-0001-5643-4068") == "sheffield-nathan"
    assert calls == ["http://p.test/api/v1/profiles/0000-0001-5643-4068"]
    with pytest.raises(pc.ProfileNotFound):
        client.resolve_orcid("0000-0002-1825-0097")


def test_resolve_orcid_tells_a_changed_record_from_a_missing_one(monkeypatch):
    """A 200 with no slug is the interface having changed, not "nobody has
    this ORCID"; a server error is neither."""

    class R:
        def __init__(self, status, data):
            self.status_code, self._data = status, data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._data

    import rag_lib.prosopia_client as pc
    client = pc.ProsopiaClient("http://p.test")
    monkeypatch.setattr(pc.requests, "get", lambda *a, **k: R(200, {"rid": "0000-0001-5643-4068"}))
    with pytest.raises(pc.ProsopiaFormatChanged):
        client.resolve_orcid("0000-0001-5643-4068")
    monkeypatch.setattr(pc.requests, "get", lambda *a, **k: R(500, {}))
    with pytest.raises(pc.ProsopiaError) as err:
        client.resolve_orcid("0000-0001-5643-4068")
    assert not isinstance(err.value, (pc.ProfileNotFound, pc.ProsopiaFormatChanged))


# --- a failure part-way through the writes takes them back ----------------------------------------


def test_a_failure_while_saving_adds_nothing_to_an_existing_draft(env, monkeypatch):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    d = _request(app, "POST", "/api/profiles/draft", json={"name": "Mine"}).json()["slug"]
    before = {t: _q(env, f"SELECT COUNT(*) FROM {t}")[0][0]
              for t in ("profile_seeds", "papers", "paper_embeddings", "researcher_papers")}
    real = profiles_repo.attach_seed
    calls = {"n": 0}

    def flaky(conn, *a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("disk full")
        return real(conn, *a, **k)

    monkeypatch.setattr(profiles_repo, "attach_seed", flaky)
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "into": d})
    assert r.status_code == 500
    after = {t: _q(env, f"SELECT COUNT(*) FROM {t}")[0][0] for t in before}
    assert after == before
    # The draft added to is as it was; the researcher the request made is gone.
    assert _q(env, "SELECT name, n_seed FROM profiles") == [("Mine", 0)]
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(0,)]


# --- after the review (2026-10-31) -----------------------------------------------------------


def test_o13_check_for_new_papers_alone_brings_the_fields_up_to_date(env):
    """F4: the fields follow the source even when no paper is imported --
    "check for new papers" with nothing new used to leave them stale."""
    profile = _stub_profile()
    profile.affiliation, profile.expertise = "UVA", "Genomics"
    app = _app(StubProsopia(profile), _stub_openalex())
    rid = _import(app, source="prosopia", ref="sheffield-nathan").json()["researcher_id"]
    profile.affiliation, profile.expertise = None, ""
    r = _request(app, "GET", f"/api/researchers/{rid}/works")
    assert r.status_code == 200, r.text
    assert _q(env, "SELECT affiliation, expertise FROM researchers") == [(None, None)]


def test_an_empty_profile_on_refresh_says_nothing_changed(env):
    profile = _stub_profile()
    app = _app(StubProsopia(profile), _stub_openalex())
    rid = _import(app, source="prosopia", ref="sheffield-nathan").json()["researcher_id"]
    profile.papers = []
    r = _request(app, "GET", f"/api/researchers/{rid}/works")
    assert r.status_code == 422, r.text
    assert "Nothing was changed" in r.json()["detail"] and "created" not in r.json()["detail"]


def test_an_import_that_cannot_start_leaves_no_draft(env):
    """The scheduler is off, so the request fails as it starts: the draft it
    opened goes with it."""
    from rag_lib.api.routers.prosopia import get_import_openalex_client

    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    app.dependency_overrides.pop(get_import_openalex_client)
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "name": "No start"})
    assert r.status_code == 503 and "scheduler" in r.text, r.text
    assert _q(env, "SELECT COUNT(*) FROM profiles") == [(0,)]
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(0,)]


def test_o8_a_profile_only_a_draft_brought_in_survives_the_draft_after_a_refusal(env):
    """The 409 points at that profile; discarding the draft must not then
    delete it."""
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    slug = _request(app, "POST", "/api/import/prosopia",
                    json={"ref": "sheffield-nathan", "save_profile": True}).json()["draft_slug"]
    r = _import(app, source="prosopia", ref="sheffield-nathan")
    assert r.status_code == 409, r.text
    rid = r.json()["researcher"]["id"]
    assert _request(app, "DELETE", f"/api/profiles/draft/{slug}").status_code == 200
    assert _q(env, "SELECT id FROM researchers") == [(rid,)]


def test_o9_a_refresh_marks_only_what_it_offered_and_left(env):
    """A refresh lists the saved works too; they are not "left out"."""
    app = _app(None, StubOrcidOpenAlex(WORKS))
    rid = _import(app, source="orcid", ref=ORCID, paper_ids=["W1"]).json()["researcher_id"]
    assert _import(app, source="orcid", ref=ORCID, paper_ids=["W2"], researcher_id=rid).status_code == 200
    assert _q(env, "SELECT work_id, mark FROM researcher_work_marks ORDER BY work_id") == [("W3", "skipped")]


def test_o15_an_orcid_with_no_record_on_check_for_new_papers_is_not_found(env, monkeypatch):
    conn = connect(env)
    uid = conn.execute("SELECT id FROM users").fetchone()[0]
    rid = conn.execute("INSERT INTO researchers (user_id, source, key, name, orcid) VALUES (?, 'orcid', ?, 'J', ?)",
                       (uid, ORCID, ORCID)).lastrowid
    conn.commit()
    conn.close()
    monkeypatch.setattr(orcid_service, "fetch_record", lambda *a, **k: reg.RegistryRecord(found=False))
    r = _request(_app(None, StubOrcidOpenAlex([])), "GET", f"/api/researchers/{rid}/works")
    assert r.status_code == 404, r.text


def test_b_a_person_with_no_name_and_no_orcid_does_not_refuse_title_hits():
    assert Person.of("", None).wrote({"authorships": [{"author": {"display_name": "Someone Else"}}]}) is True


def test_o15_a_403_from_prosopia_is_not_public(monkeypatch):
    import rag_lib.prosopia_client as pc

    client = httpx.Client(base_url="http://p.test",
                          transport=httpx.MockTransport(lambda request: httpx.Response(403, json={"detail": "no"})))
    real_storage = pc.PagedApiStorage
    monkeypatch.setattr(pc, "PagedApiStorage", lambda **kw: real_storage(client=client, **kw))
    with pytest.raises(ProfileNotPublic, match="not public"):
        pc.ProsopiaClient("http://p.test").fetch_profile("jane")
