"""Imports after the 2026-11-02 review (FUNCTIONAL_SPEC CR.7, CR.8, L10)."""

from __future__ import annotations

import json

from rag_lib.api.services import wizard
from rag_lib.db import connect
from rag_lib.db.repos import embeddings as embeddings_repo, profiles as profiles_repo
from rag_lib.embedders import placeholder_embed
from rag_lib.openalex_status import OpenAlexUnavailable
from rag_lib.researcher_topics import code as _c
from tests.test_prosopia_import import RECORD_DOI_ONLY, RECORD_FULL, StubOpenAlex, StubProfile, StubProsopia, StubSummaries
from tests.test_researcher_topics import _topic
from tests.test_researchers import _app, _request, env  # noqa: F401 -- fixture
from tests.test_seed_removal_topics import _paper

OWN = "https://openalex.org/W9001"
DOC_B = {"interests": ["Quantum Widgets"],
         "research_interests": [_topic("T12345", "Quantum Widgets", weight=0.5, method="declared")]}


class Unanswering(StubOpenAlex):
    """OpenAlex out of quota for every lookup the import makes."""

    def paginate_filter(self, *a, **k):
        raise OpenAlexUnavailable("quota", detail="429")

    def get_work(self, *a, **k):
        raise OpenAlexUnavailable("quota", detail="429")


def _q(env, sql, *p):
    c = connect(env)
    try:
        return [tuple(r) for r in c.execute(sql, p).fetchall()]
    finally:
        c.close()


def _live(env):
    conn = connect(env)
    pid = profiles_repo.upsert(conn, user_id=1, name="Live", embedding_model="placeholder-v1",
                               n_seed=1, topic_filters={"topics": []}, threshold=0.5)
    conn.execute("UPDATE profiles SET is_draft = 0 WHERE id = ?", (pid,))
    _paper(conn, OWN, ("https://openalex.org/T10009", "Nine"))
    profiles_repo.attach_seed(conn, pid, OWN)
    embeddings_repo.upsert(conn, OWN, "placeholder-v1", placeholder_embed(OWN))
    slug = profiles_repo.get(conn, pid)["slug"]
    wizard.aggregate_draft_topics(conn, user_id=1, slug=slug)
    conn.commit()
    conn.close()
    return slug


def _someone_else():
    prof = StubProfile("Someone Else", [RECORD_FULL, RECORD_DOI_ONLY], StubSummaries({}))
    prof.metadata = DOC_B
    return prof


def _import(app, into):
    try:
        return _request(app, "POST", "/api/import/prosopia", json={"ref": "someone-else", "into": into})
    except OpenAlexUnavailable:      # the inline test path re-raises the job's error
        return None


def test_cr7_a_failed_import_into_an_edit_copy_leaves_no_prosopia_interest(env):
    """Review EI-2: OpenAlex refuses the import; the copy's Prosopia
    interests, its step-3 topics and, once saved, the interest's are as
    they were."""
    slug = _live(env)
    app = _app(StubProsopia(_someone_else()), Unanswering())
    copy = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    before = _q(env, "SELECT prosopia_docs_json FROM profiles WHERE slug = ?", copy)
    r = _import(app, copy)
    assert r is None or r.status_code == 503
    assert _q(env, "SELECT prosopia_docs_json FROM profiles WHERE slug = ?", copy) == before
    topics = _request(app, "GET", f"/api/profiles/{copy}/topics/preview").json()
    assert "T12345" not in {_c(t["id"]) for t in topics}
    assert _request(app, "POST", f"/api/profiles/draft/{copy}/save-edit").status_code == 200
    saved = json.loads(_q(env, "SELECT topic_filters_json FROM profiles WHERE slug = ?", slug)[0][0])
    assert "T12345" not in {_c(t["id"]) for t in saved.get("topics", [])}
    assert "someone-else" not in (_q(env, "SELECT prosopia_docs_json FROM profiles WHERE slug = ?", slug)[0][0] or "")


def test_cr7_a_failed_import_into_a_wizard_draft_leaves_no_prosopia_interest(env):
    """The other way in: the new-interest wizard adding a person to a draft
    it already has."""
    app = _app(StubProsopia(_someone_else()), Unanswering())
    d = _request(app, "POST", "/api/profiles/draft", json={"name": "Mine"}).json()["slug"]
    before = _q(env, "SELECT prosopia_docs_json FROM profiles WHERE slug = ?", d)
    r = _import(app, d)
    assert r is None or r.status_code == 503
    assert _q(env, "SELECT prosopia_docs_json FROM profiles WHERE slug = ?", d) == before


def test_cr7_a_successful_import_keeps_the_prosopia_interest(env):
    slug = _live(env)
    app = _app(StubProsopia(_someone_else()), StubOpenAlex())
    copy = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    r = _import(app, copy)
    assert r is not None and r.status_code == 200, r and r.text
    docs = json.loads(_q(env, "SELECT prosopia_docs_json FROM profiles WHERE slug = ?", copy)[0][0])
    assert [d["key"] for d in docs] == ["someone-else"] and d_has_interest(docs[0])


def d_has_interest(doc):
    return doc.get("document", {}).get("interests") == ["Quantum Widgets"]


# --- CR.8: one form of the ORCID from where a Prosopia profile is read in ---

def _people(env):
    return _q(env, "SELECT source, key, orcid FROM researchers ORDER BY id")


def _same_person_twice(env, rid_form):
    from tests.test_m2a_saved_researcher import _profile, _stored
    from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex
    from tests.test_prosopia_import import _stub_openalex
    from tests.test_researchers import _import

    _stored(_app(None, StubOrcidOpenAlex(WORKS)))         # Profiles page: by ORCID
    prof = _profile()
    prof.rid = rid_form
    app = _app(StubProsopia(prof), _stub_openalex())
    listing = _request(app, "GET", "/api/import/prosopia/works", params={"ref": "sheffield-nathan"}).json()
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "save_profile": True})
    again = _import(app, source="prosopia", ref="sheffield-nathan")
    return ORCID, listing, r, again


import pytest  # noqa: E402


@pytest.mark.parametrize("form", ["{o}", "https://orcid.org/{o}", "http://www.ORCID.org/{o}/", " {lo} "])
def test_cr8_a_prosopia_rid_in_any_form_is_the_person_stored_by_orcid(env, form):
    """Review EI-3: the rid as a bare iD, an orcid.org URL, or with a
    lower-case x and spaces is one person, one researcher profile."""
    from tests.test_orcid_import import ORCID
    rid = form.format(o=ORCID, lo=ORCID.lower())
    orcid, listing, r, again = _same_person_twice(env, rid)
    assert listing.get("saved_profile") and listing["saved_profile"]["id"] == 1
    assert r.status_code == 200, r.text
    assert again.status_code == 409, again.text
    assert [o for (_, _, o) in _people(env)] == [orcid]


def test_cr8_normalize_rid():
    from rag_lib.prosopia_client import normalize_rid
    assert normalize_rid("https://orcid.org/0000-0002-1103-388x") == "0000-0002-1103-388X"
    assert normalize_rid("orcid.org/0000-0002-1103-3882") == "0000-0002-1103-3882"
    assert normalize_rid("local:abc") == "local:abc"
    assert normalize_rid(None) == ""


def test_cr8_l10_the_pick_list_ids_are_the_seed_form_whether_or_not_openalex_answers(env):
    """Review EI-4: the same row reads the same with OpenAlex up and down."""
    from tests.test_prosopia_import import _work

    def listing(oa):
        prof = StubProfile("N S", [RECORD_FULL, RECORD_DOI_ONLY], StubSummaries({}))
        r = _request(_app(StubProsopia(prof), oa), "GET", "/api/import/prosopia/works",
                     params={"ref": "sheffield-nathan"})
        assert r.status_code == 200, r.text
        return {w["id"]: w["openalex_id"] for w in r.json()["works"]}

    up = listing(StubOpenAlex(by_id={"W4415881950": _work("10.1101/2025.11.03.685753", "W4415881950", "Atacformer", 2025)}))
    down = listing(Unanswering())
    assert up == down
    assert all(v is None or v.startswith("https://openalex.org/W") for v in up.values())
    assert any(v for v in up.values())


@pytest.mark.parametrize("rid", ["local:sheffield-nathan", "", "not-an-orcid"])
def test_cr8_a_rid_that_is_no_orcid_stores_no_orcid(env, rid):
    from tests.test_m2a_saved_researcher import _profile
    from tests.test_prosopia_import import _stub_openalex
    from tests.test_researchers import _import

    prof = _profile()
    prof.rid = rid
    r = _import(_app(StubProsopia(prof), _stub_openalex()), source="prosopia", ref="sheffield-nathan")
    assert r.status_code == 200, r.text
    assert _people(env) == [("prosopia", "sheffield-nathan", None)]


def test_b3_2_the_seed_form_of_ids():
    from rag_lib.api.services.prosopia import _seed_form
    assert _seed_form("W123") == _seed_form("w123") == _seed_form("https://openalex.org/W123") == "https://openalex.org/W123"
    for bad in (None, "", "  ", "https://doi.org/10.1234/abc.def", "10.1234/xyz", "openalex:W5"):
        assert _seed_form(bad) is None, bad


# --- B3-1 (the user's choice B): an existing draft is marked as built from
# the person only once the import into it succeeds ---

def _mark(env, slug):
    return _q(env, "SELECT researcher_id FROM profiles WHERE slug = ?", slug)[0][0]


def _empty_draft(app, name="Mine"):
    return _request(app, "POST", "/api/profiles/draft", json={"name": name}).json()["slug"]


def test_b3_1_a_failed_import_does_not_mark_the_draft(env):
    from tests.test_m2a_saved_researcher import _profile, _stored
    from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex

    _stored(_app(None, StubOrcidOpenAlex(WORKS)))
    prof = _profile()
    prof.rid = ORCID
    app = _app(StubProsopia(prof), Unanswering())
    d = _empty_draft(app)
    try:
        r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "into": d, "save_profile": True})
        assert r.status_code == 503, r.text
    except OpenAlexUnavailable:
        pass
    assert _mark(env, d) is None
    drafts = {x["slug"]: x for x in _request(app, "GET", "/api/profiles/drafts").json()}
    assert not drafts[d].get("researcher_name")


def test_b3_1_a_successful_import_marks_the_empty_draft(env):
    from tests.test_m2a_saved_researcher import _profile, _stored
    from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex
    from tests.test_prosopia_import import _stub_openalex

    rid = _stored(_app(None, StubOrcidOpenAlex(WORKS)))
    prof = _profile()
    prof.rid = ORCID
    app = _app(StubProsopia(prof), _stub_openalex())
    d = _empty_draft(app)
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "into": d, "save_profile": True})
    assert r.status_code == 200, r.text
    assert _mark(env, d) == rid


def test_b3_1_the_orcid_wizard_import_marks_the_empty_draft_too(env):
    from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex

    app = _app(None, StubOrcidOpenAlex(WORKS))
    d = _empty_draft(app)
    r = _request(app, "POST", "/api/import/orcid", json={"orcid": ORCID, "openalex_ids": ["W1"], "into": d, "save_profile": True})
    assert r.status_code == 200, r.text
    rid = _q(env, "SELECT id FROM researchers WHERE orcid = ?", ORCID)[0][0]
    assert _mark(env, d) == rid


def test_b3_1_a_draft_with_seeds_of_its_own_is_not_marked(env):
    from tests.test_m2a_saved_researcher import _profile, _stored
    from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex
    from tests.test_prosopia_import import _stub_openalex

    _stored(_app(None, StubOrcidOpenAlex(WORKS)))
    prof = _profile()
    prof.rid = ORCID
    app = _app(StubProsopia(prof), _stub_openalex())
    d = _empty_draft(app)
    conn = connect(env)
    pid = _q(env, "SELECT id FROM profiles WHERE slug = ?", d)[0][0]
    _paper(conn, OWN, ("https://openalex.org/T10009", "Nine"))
    profiles_repo.attach_seed(conn, pid, OWN)
    conn.commit()
    conn.close()
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "into": d, "save_profile": True})
    assert r.status_code == 200, r.text
    assert _mark(env, d) is None
