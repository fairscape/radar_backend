"""A wizard or Edit interest import of a person the user already has a
researcher profile for (FUNCTIONAL_SPEC M.2a, M-37 to M-45).

Not refused any more. Unticked, the papers go to the draft only and the
profile is not touched. Ticked, the papers the profile lacks are added to
it -- not a second profile -- and stay whatever becomes of the draft; the
profile's own fields are left as they are.
"""

from __future__ import annotations

from rag_lib.db import connect
from rag_lib.db.repos import gather_runs as gather_runs_repo
from tests.test_draft_sandbox_import import Unanswering, _snapshot
from tests.test_m2_m3_import_topics import _profile
from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex
from tests.test_prosopia_import import StubProsopia, _stub_openalex
from tests.test_researchers import _app, _import, _request, env  # noqa: F401 -- fixture


def _q(env, sql, *params):
    conn = connect(env)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _stored(app, ids=("W1",)):
    """The person's profile, imported on the Profiles page with ``ids``."""
    r = _import(app, source="orcid", ref=ORCID, paper_ids=list(ids))
    assert r.status_code == 200, r.text
    return r.json()["researcher_id"]


def _wizard(app, ids, **body):
    r = _request(app, "POST", "/api/import/orcid", json={"orcid": ORCID, "openalex_ids": ids, **body})
    assert r.status_code == 200, r.text
    result = _request(app, "GET", f"/api/import/prosopia/{r.json()['run_id']}").json()["result"]
    assert result is not None
    return result


def _saved_interest(env, app, rid, name="X"):
    """An interest built from the profile, saved (as the commit leaves it;
    the commit's first scan is not what these tests are about)."""
    slug = _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": name}).json()["draft"]["slug"]
    conn = connect(env)
    try:
        conn.execute("UPDATE profiles SET is_draft = 0 WHERE slug = ?", (slug,))
        conn.commit()
    finally:
        conn.close()
    return slug


def _profile_papers(env, rid):
    return sorted(r[0].rsplit("/", 1)[-1] for r in _q(
        env, "SELECT openalex_id FROM researcher_papers WHERE researcher_id = ?", rid))


def _draft_seeds(env, slug):
    return sorted(r[0].rsplit("/", 1)[-1] for r in _q(
        env, "SELECT s.openalex_id FROM profile_seeds s JOIN profiles p ON p.id = s.profile_id WHERE p.slug = ?", slug))


PROFILE_TABLES = ("researchers", "researcher_papers")


def test_m37_unticked_adds_to_the_draft_and_leaves_the_profile(env):
    app = _app(None, StubOrcidOpenAlex(WORKS))
    rid = _stored(app)
    before = _snapshot(env, PROFILE_TABLES)
    result = _wizard(app, ["W1", "W2"], name="New")
    assert _draft_seeds(env, result["draft_slug"]) == ["W1", "W2"]
    assert result.get("added_to_profile") is None
    assert _snapshot(env, PROFILE_TABLES) == before
    # Into an existing draft as well.
    d = _request(app, "POST", "/api/profiles/draft", json={"name": "Mine"}).json()["slug"]
    _wizard(app, ["W3"], into=d)
    assert _draft_seeds(env, d) == ["W3"]
    assert _snapshot(env, PROFILE_TABLES) == before
    assert _profile_papers(env, rid) == ["W1"]


def test_m38_ticked_adds_only_the_missing_papers_to_the_profile(env):
    app = _app(None, StubOrcidOpenAlex(WORKS))
    rid = _stored(app)
    result = _wizard(app, ["W1", "W2"], name="New", save_profile=True)
    assert _profile_papers(env, rid) == ["W1", "W2"]
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(1,)]
    assert _q(env, "SELECT n_papers FROM researchers WHERE id = ?", rid) == [(2,)]
    assert (result["added_to_profile"], result["profile_name"]) == (1, "Justin C. Niestroy")
    assert result["researcher_id"] == rid
    assert _draft_seeds(env, result["draft_slug"]) == ["W1", "W2"]


def test_m39_what_was_added_to_the_profile_stays_when_the_draft_goes(env):
    app = _app(None, StubOrcidOpenAlex(WORKS))
    rid = _stored(app)
    slug = _wizard(app, ["W1", "W2"], name="New", save_profile=True)["draft_slug"]
    assert _request(app, "DELETE", f"/api/profiles/draft/{slug}").status_code in (200, 204)
    assert _profile_papers(env, rid) == ["W1", "W2"]
    assert sorted(r[0].rsplit("/", 1)[-1] for r in _q(env, "SELECT openalex_id FROM papers")) == ["W1", "W2"]
    assert _q(env, "SELECT COUNT(*) FROM paper_embeddings") == [(2,)]


def test_m39_and_when_an_edit_is_discarded(env):
    app = _app(None, StubOrcidOpenAlex(WORKS))
    rid = _stored(app)
    slug = _saved_interest(env, app, rid)
    copy = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    _wizard(app, ["W3"], into=copy, save_profile=True)
    assert _profile_papers(env, rid) == ["W1", "W3"]
    assert _request(app, "DELETE", f"/api/profiles/draft/{copy}").status_code in (200, 204)
    assert _profile_papers(env, rid) == ["W1", "W3"]
    assert _draft_seeds(env, slug) == ["W1"]


def test_m40_the_listing_names_the_saved_profile(env):
    app = _app(None, StubOrcidOpenAlex(WORKS))
    assert _request(app, "GET", f"/api/import/orcid/{ORCID}/works").json()["saved_profile"] is None
    rid = _stored(app)
    assert _request(app, "GET", f"/api/import/orcid/{ORCID}/works").json()["saved_profile"] == {
        "id": rid, "name": "Justin C. Niestroy"}


def test_m40_the_prosopia_listing_too(env):
    app = _app(StubProsopia(_profile()), _stub_openalex())
    works = lambda: _request(app, "GET", "/api/import/prosopia/works", params={"ref": "sheffield-nathan"}).json()  # noqa: E731
    assert works()["saved_profile"] is None
    rid = _import(app, source="prosopia", ref="sheffield-nathan").json()["researcher_id"]
    assert works()["saved_profile"] == {"id": rid, "name": "Nathan C. Sheffield"}


def test_m41_an_interest_built_from_the_profile_keeps_its_seeds(env):
    app = _app(None, StubOrcidOpenAlex(WORKS))
    rid = _stored(app)
    x = _saved_interest(env, app, rid)
    _wizard(app, ["W2", "W3"], name="New", save_profile=True)
    assert _profile_papers(env, rid) == ["W1", "W2", "W3"]
    assert _draft_seeds(env, x) == ["W1"]


def test_m42_the_same_person_under_the_other_source_is_added_to(env):
    """Stored from Prosopia, imported by ORCID: one person, one profile."""
    from rag_lib.db.repos import users as users_repo

    conn = connect(env)
    try:
        uid = int(users_repo.get_by_email(conn, "demo@example.com")["id"])
        rid = conn.execute("INSERT INTO researchers (user_id, source, key, name, orcid) VALUES (?, 'prosopia', 'jn', 'J N', ?)",
                           (uid, ORCID.lower())).lastrowid
        conn.commit()
    finally:
        conn.close()
    app = _app(None, StubOrcidOpenAlex(WORKS))
    assert _request(app, "GET", f"/api/import/orcid/{ORCID}/works").json()["saved_profile"] == {"id": rid, "name": "J N"}
    result = _wizard(app, ["W1"], name="New", save_profile=True)
    assert _q(env, "SELECT id, source, key, name FROM researchers") == [(rid, "prosopia", "jn", "J N")]
    assert _profile_papers(env, rid) == ["W1"]
    assert result["added_to_profile"] == 1


def test_m43_openalex_unavailable_changes_neither_draft_nor_profile(env):
    app = _app(None, StubOrcidOpenAlex(WORKS))
    _stored(app)
    before = _snapshot(env, bookkeeping=True)
    app = _app(None, Unanswering(WORKS))
    r = _request(app, "POST", "/api/import/orcid", json={"orcid": ORCID, "openalex_ids": ["W2"], "save_profile": True})
    if r.status_code == 200:
        status = _request(app, "GET", f"/api/import/prosopia/{r.json()['run_id']}").json()
        assert status["result"] is None and status["run"]["error"]
    else:
        assert r.status_code == 503, r.text
    assert _snapshot(env, bookkeeping=True) == before


def test_m44_refused_while_the_profiles_page_checks_the_person(env):
    app = _app(None, StubOrcidOpenAlex(WORKS))
    rid = _stored(app)
    conn = connect(env)
    try:
        uid = int(conn.execute("SELECT user_id FROM researchers WHERE id = ?", (rid,)).fetchone()[0])
        gather_runs_repo.start(conn, profile_id=None, user_id=uid, researcher_id=rid, tier_used="researcher_import")
        conn.commit()
    finally:
        conn.close()
    before = _snapshot(env)
    for ticked in (True, False):
        r = _request(app, "POST", "/api/import/orcid",
                     json={"orcid": ORCID, "openalex_ids": ["W2"], "save_profile": ticked})
        assert r.status_code == 409, r.text
    assert _snapshot(env) == before


def test_m45_the_profiles_own_fields_are_left_as_they_are(env):
    prosopia = StubProsopia(_profile())
    app = _app(prosopia, _stub_openalex())
    rid = _import(app, source="prosopia", ref="sheffield-nathan", paper_ids=["leroy2025atacformer"]).json()["researcher_id"]
    cols = ("name", "affiliation", "url", "expertise", "soul", "grants_json", "document_json", "imported_at", "mark")
    # Not last_run_id: the run to watch on the Profiles page is this one.
    fields = lambda: dict(zip(cols, _q(env, f"SELECT {', '.join(cols)} FROM researchers WHERE id = ?", rid)[0]))  # noqa: E731
    before = fields()
    # What Prosopia says of the person has moved on since.
    prosopia.profile.name = "N. C. Sheffield"
    prosopia.profile.metadata = {"research_interests": []}
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "save_profile": True})
    assert r.status_code == 200, r.text
    result = _request(app, "GET", f"/api/import/prosopia/{r.json()['run_id']}").json()["result"]
    assert (result["added_to_profile"], result["profile_name"]) == (1, "Nathan C. Sheffield")
    assert fields() == before
    assert _q(env, "SELECT COUNT(*) FROM researcher_papers WHERE researcher_id = ?", rid) == [(2,)]
    # The unticked papers of the listing were not marked as turned down.
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(1,)]


def test_m45_the_listings_other_papers_are_not_marked_turned_down(env):
    """Picking seeds for an interest is not going through the profile's
    papers: what was not picked keeps the mark it had (none here)."""
    app = _app(None, StubOrcidOpenAlex(WORKS))
    rid = _stored(app)
    conn = connect(env)
    try:
        conn.execute("DELETE FROM researcher_work_marks WHERE researcher_id = ?", (rid,))
        conn.commit()
    finally:
        conn.close()
    _wizard(app, ["W2"], name="New", save_profile=True)
    assert _q(env, "SELECT work_id, mark FROM researcher_work_marks WHERE researcher_id = ?", rid) == []
    assert _profile_papers(env, rid) == ["W1", "W2"]


def test_m42_a_prosopia_profile_of_someone_stored_by_orcid(env):
    """The Prosopia profile carries the person's ORCID (rid): the listing
    names the ORCID profile, and the ticked import adds to it."""
    prosopia_profile = _profile()
    prosopia_profile.rid = ORCID
    app = _app(StubProsopia(prosopia_profile), _stub_openalex())
    rid = _stored(_app(None, StubOrcidOpenAlex(WORKS)))
    listing = _request(app, "GET", "/api/import/prosopia/works", params={"ref": "sheffield-nathan"}).json()
    assert listing["saved_profile"] == {"id": rid, "name": "Justin C. Niestroy"}
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "save_profile": True})
    assert r.status_code == 200, r.text
    result = _request(app, "GET", f"/api/import/prosopia/{r.json()['run_id']}").json()["result"]
    assert result["researcher_id"] == rid and result["added_to_profile"] == 2
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(1,)]
    assert len(_profile_papers(env, rid)) == 3
