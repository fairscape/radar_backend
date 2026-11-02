"""A wizard import saves no researcher profile unless asked (FUNCTIONAL_SPEC
M.2), and the interests of a Prosopia profile imported join the topic list,
marked with where they come from and off (M.3).

Offline, with the stubs of the import suites. The seeds the stubs resolve
carry OpenAlex topic T10001 ("Test Topic").
"""

from __future__ import annotations

import json

import pytest

from rag_lib.db import connect
from rag_lib.db.repos import profiles as profiles_repo
from rag_lib.api.services import edit_copy
from rag_lib.researcher_topics import code as _c
from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex
from tests.test_prosopia_import import (
    RECORD_DOI_ONLY,
    RECORD_FULL,
    StubProfile,
    StubProsopia,
    StubSummaries,
    _stub_openalex,
)
from tests.test_researcher_topics import _topic
from tests.test_researchers import _app, _import, _request, env  # noqa: F401 -- fixture

SEED_TOPIC = "T10001"
DOC_A = {"research_interests": [
    _topic("T10515", "Cancer-related molecular mechanisms", weight=0.5, method="declared"),
    _topic("T10001", "Test Topic", weight=-0.5, method="declared"),
    _topic("T11986", "Scientific Computing", weight=-0.3, method="declared"),
]}
DOC_B = {"research_interests": [_topic("T10001", "Test Topic", weight=0.4, method="declared")]}



def _q(env, sql, *params):
    conn = connect(env)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _profile(name="Nathan C. Sheffield", doc=DOC_A, records=(RECORD_FULL, RECORD_DOI_ONLY)):
    p = StubProfile(name, list(records), StubSummaries({}))
    p.metadata = doc
    return p


def _wizard(app, **body):
    return _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", **body})


def _topics(app, slug):
    return {_c(t["id"]): t for t in _request(app, "GET", f"/api/profiles/draft/{slug}/topics").json()}


# --- M.2: no researcher profile unless asked ------------------------------------------------


def test_m7_m8_an_unticked_prosopia_import_saves_no_researcher(env):
    app = _app(StubProsopia(_profile()), _stub_openalex())
    r = _wizard(app)
    assert r.status_code == 200, r.text
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(0,)]
    assert _q(env, "SELECT researcher_id FROM profiles") == [(None,)]
    assert set(_q(env, "SELECT source, source_researcher_id FROM profile_seeds")) == {("prosopia", None)}
    assert _request(app, "GET", "/api/researchers").json() == []


def test_m7_m8_an_unticked_orcid_import_saves_no_researcher(env):
    app = _app(None, StubOrcidOpenAlex(WORKS))
    r = _request(app, "POST", "/api/import/orcid", json={"orcid": ORCID, "openalex_ids": ["W1", "W2"]})
    assert r.status_code == 200, r.text
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(0,)]
    assert set(_q(env, "SELECT source, source_researcher_id FROM profile_seeds")) == {("orcid", None)}


def test_m9_discarding_an_unticked_import_takes_its_papers(env):
    app = _app(StubProsopia(_profile()), _stub_openalex())
    slug = _wizard(app).json()["draft_slug"]
    assert _q(env, "SELECT COUNT(*) FROM papers")[0][0] > 0
    assert _request(app, "DELETE", f"/api/profiles/draft/{slug}").status_code in (200, 204)
    assert _q(env, "SELECT COUNT(*) FROM papers") == [(0,)]
    assert _q(env, "SELECT COUNT(*) FROM paper_embeddings") == [(0,)]


def test_m11_m12_a_ticked_import_saves_the_researcher_at_once_and_keeps_it(env):
    app = _app(StubProsopia(_profile()), _stub_openalex())
    slug = _wizard(app, save_profile=True).json()["draft_slug"]
    rid = _q(env, "SELECT id FROM researchers")[0][0]
    # Saved, not the draft's: not recorded as something the draft brought in.
    assert _q(env, "SELECT COUNT(*) FROM draft_owned WHERE kind = 'researcher'") == [(0,)]
    assert [r["id"] for r in _request(app, "GET", "/api/researchers").json()] == [rid]
    assert set(_q(env, "SELECT source, source_researcher_id FROM profile_seeds")) == {("researcher", rid)}
    n_papers = _q(env, "SELECT COUNT(*) FROM researcher_papers WHERE researcher_id = ?", rid)[0][0]
    assert _request(app, "DELETE", f"/api/profiles/draft/{slug}").status_code in (200, 204)
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(1,)]
    assert _q(env, "SELECT COUNT(*) FROM researcher_papers WHERE researcher_id = ?", rid) == [(n_papers,)]
    assert _q(env, "SELECT COUNT(*) FROM papers")[0][0] == n_papers


def test_m11_a_ticked_orcid_import_saves_the_researcher(env):
    app = _app(None, StubOrcidOpenAlex(WORKS))
    r = _request(app, "POST", "/api/import/orcid",
                 json={"orcid": ORCID, "openalex_ids": ["W1", "W2"], "save_profile": True})
    assert r.status_code == 200, r.text
    listing = _request(app, "GET", "/api/researchers").json()
    assert [(x["source"], x["n_papers"]) for x in listing] == [("orcid", 2)]
    assert set(_q(env, "SELECT source FROM profile_seeds")) == {("researcher",)}


# M-13 (a person with a saved profile refused, 409) gave way to M.2a: see
# tests/test_m2a_saved_researcher.py.


def test_m14_two_drafts_of_one_person_are_independent(env):
    app = _app(StubProsopia(_profile()), _stub_openalex())
    a = _wizard(app, name="A").json()["draft_slug"]
    b = _wizard(app, name="B").json()["draft_slug"]
    assert a != b
    seeds_b = _q(env, "SELECT s.openalex_id FROM profile_seeds s JOIN profiles p ON p.id = s.profile_id "
                      "WHERE p.slug = ? ORDER BY 1", b)
    assert seeds_b
    assert _request(app, "DELETE", f"/api/profiles/draft/{a}").status_code in (200, 204)
    assert _q(env, "SELECT s.openalex_id FROM profile_seeds s JOIN profiles p ON p.id = s.profile_id "
                   "WHERE p.slug = ? ORDER BY 1", b) == seeds_b
    assert _q(env, "SELECT COUNT(*) FROM papers") == [(len(seeds_b),)]


def test_m16_a_ticked_import_that_fails_leaves_no_researcher(env, monkeypatch):
    app = _app(StubProsopia(_profile()), _stub_openalex())

    def broken(conn, *a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(profiles_repo, "attach_seed", broken)
    r = _wizard(app, save_profile=True)
    assert r.status_code == 500                         # run inline under the test client
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(0,)]
    assert _q(env, "SELECT COUNT(*) FROM profiles") == [(0,)]
    assert _q(env, "SELECT COUNT(*) FROM papers") == [(0,)]


def test_m17_an_interest_from_a_saved_profile_takes_its_prosopia_interests(env):
    app = _app(StubProsopia(_profile()), _stub_openalex())
    rid = _import(app, source="prosopia", ref="sheffield-nathan").json()["researcher_id"]
    r = _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "From R"})
    assert r.status_code == 200, r.text
    t = _topics(app, r.json()["draft"]["slug"])["T10515"]
    assert t["on"] is False and t["prosopia"] == [{"profile": "Nathan C. Sheffield"}]


# --- M.3: the profile's interests join the topic list, marked and off ---------------------


def test_m18_an_interest_no_seed_brings_is_added_off_and_marked(env):
    app = _app(StubProsopia(_profile()), _stub_openalex())
    slug = _wizard(app).json()["draft_slug"]
    topics = _topics(app, slug)
    # DOC_A states one interest (T10515); its not interests (the seed topic
    # among them) do nothing: the seed topic is on as the seeds have it.
    assert set(topics) == {SEED_TOPIC, "T10515"}
    added = topics["T10515"]
    assert added["on"] is False and added["source"] == "prosopia"
    assert added["prosopia"] == [{"profile": "Nathan C. Sheffield"}] and added["papers"] == 0
    assert topics[SEED_TOPIC]["on"] is True and topics[SEED_TOPIC].get("prosopia") is None


def test_m18_a_seed_topic_a_profile_lists_is_only_marked(env):
    app = _app(StubProsopia(_profile()), _stub_openalex())
    slug = _wizard(app).json()["draft_slug"]
    # Switched off by the user first: a profile listing it later moves no switch.
    _request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": ["T10515"]})
    b = _app(StubProsopia(_profile(name="Someone Else", doc=DOC_B)), _stub_openalex())
    assert _request(b, "POST", "/api/import/prosopia", json={"ref": "someone-else", "into": slug}).status_code == 200
    t = _topics(app, slug)[SEED_TOPIC]
    assert t["on"] is False and t["source"] != "prosopia"
    assert t["prosopia"] == [{"profile": "Someone Else"}]


def test_m20_a_prosopia_interest_switches_as_any_topic_and_stays_so(env):
    app = _app(StubProsopia(_profile()), _stub_openalex())
    slug = _wizard(app).json()["draft_slug"]
    r = _request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": [SEED_TOPIC, "T10515"]})
    assert r.status_code == 200, r.text
    assert _topics(app, slug)["T10515"]["on"] is True
    # More seeds since (a rebuild of the list): kept as switched.
    more = _app(StubProsopia(_profile(name="Someone Else", doc=DOC_B)), _stub_openalex())
    assert _request(more, "POST", "/api/import/prosopia", json={"ref": "someone-else", "into": slug}).status_code == 200
    assert _topics(app, slug)["T10515"]["on"] is True
    _request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": [SEED_TOPIC]})
    t = _topics(app, slug)["T10515"]
    assert t["on"] is False and t["prosopia"] == [{"profile": "Nathan C. Sheffield"}]


def test_m21_a_topic_two_profiles_list_is_marked_with_both(env):
    both = {"research_interests": [_topic("T10515", "Cancer", weight=0.5, method="declared")]}
    app = _app(StubProsopia(_profile()), _stub_openalex())
    slug = _wizard(app).json()["draft_slug"]
    b = _app(StubProsopia(_profile(name="Someone Else", doc=both)), _stub_openalex())
    assert _request(b, "POST", "/api/import/prosopia", json={"ref": "someone-else", "into": slug}).status_code == 200
    t = _topics(app, slug)["T10515"]
    assert t["on"] is False
    assert t["prosopia"] == [{"profile": "Nathan C. Sheffield"}, {"profile": "Someone Else"}]


def test_m3_the_pick_routes_are_gone(env):
    app = _app(StubProsopia(_profile()), _stub_openalex())
    slug = _wizard(app).json()["draft_slug"]
    assert _request(app, "GET", f"/api/profiles/{slug}/prosopia-topics").status_code in (404, 405)
    assert _request(app, "PUT", f"/api/profiles/{slug}/prosopia-topics",
                    json={"side": "interest", "topic_id": "T10515", "profile": "sheffield-nathan",
                          "picked": True}).status_code in (404, 405)


def test_m23_a_ticked_import_lists_the_same(env):
    app = _app(StubProsopia(_profile()), _stub_openalex())
    slug = _wizard(app, save_profile=True).json()["draft_slug"]
    t = _topics(app, slug)["T10515"]
    assert t["on"] is False and t["prosopia"] == [{"profile": "Nathan C. Sheffield"}]


def test_m22_an_edit_copy_carries_the_profiles_and_saves_them_back():
    assert "prosopia_docs_json" in edit_copy._COPIED and "prosopia_docs_json" in edit_copy._SAVED


def test_m24_the_copy_goes_with_the_draft(env):
    app = _app(StubProsopia(_profile()), _stub_openalex())
    slug = _wizard(app).json()["draft_slug"]
    assert json.loads(_q(env, "SELECT prosopia_docs_json FROM profiles")[0][0])[0]["key"] == "sheffield-nathan"
    assert _request(app, "DELETE", f"/api/profiles/draft/{slug}").status_code in (200, 204)
    assert _q(env, "SELECT COUNT(*) FROM profiles") == [(0,)]


# --- a second request for the person being imported joins that run -------------------------
# With no researcher profile saved (M.2) the run itself says whom it imports.


def _plan_and_run(conn, plan):
    from rag_lib.db.repos import gather_runs as gather_runs_repo
    # As the router opens it (launch_import).
    return gather_runs_repo.start(conn, profile_id=plan.profile_id, researcher_id=plan.researcher_id,
                                  user_id=1, tier_used="prosopia_import", import_orcid=plan.import_orcid,
                                  import_key=plan.import_key, opened_draft=plan.new_draft)


def test_an_unsaved_import_running_is_found_by_orcid_from_either_source(tmp_path):
    import pytest as _pytest
    from rag_lib.api import settings as settings_module
    from rag_lib.api.services import orcid as orcid_service, prosopia as prosopia_service
    from rag_lib.db import apply_migrations
    from tests.test_review_round4 import _Client

    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    s = settings_module.get_settings()
    plan = prosopia_service.prepare_import(conn, s, user_id=1, ref="doe-jane", prosopia_client=_Client())
    assert plan.researcher_id is None
    run = _plan_and_run(conn, plan)
    # The same person by ORCID, from the ORCID import: that run.
    with _pytest.raises(orcid_service.ImportAlreadyRunning) as e:
        orcid_service.refuse_if_importing(conn, 1, "0000-0001-2345-6789", None, key="orcid:0000-0001-2345-6789")
    assert (e.value.slug, e.value.run_id) == (plan.draft_slug, run)
    # Into some other draft: not that one (slug None, a 409).
    with _pytest.raises(orcid_service.ImportAlreadyRunning) as e:
        orcid_service.refuse_if_importing(conn, 1, "0000-0001-2345-6789", "other-draft")
    assert e.value.slug is None
    # Someone else: nothing running.
    orcid_service.refuse_if_importing(conn, 1, "0000-0009-9999-9999", None, key="orcid:0000-0009-9999-9999")


def test_an_added_to_draft_run_is_not_a_new_drafts_answer(tmp_path):
    import pytest as _pytest
    from rag_lib.api import settings as settings_module
    from rag_lib.api.services import orcid as orcid_service, prosopia as prosopia_service, wizard as wizard_service
    from rag_lib.db import apply_migrations
    from rag_lib.db.repos import users as users_repo
    from tests.test_review_round4 import _Client

    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    users_repo.upsert(conn, "demo@example.com")
    s = settings_module.get_settings()
    other = wizard_service.create_draft(conn, user_id=1, name="Other", embedding_model="placeholder-v1",
                                        selector=s.RADAR_DEFAULT_SELECTOR)["slug"]
    plan = prosopia_service.prepare_import(conn, s, user_id=1, ref="doe-jane", prosopia_client=_Client(),
                                           into=other)
    assert plan.new_draft is False
    _plan_and_run(conn, plan)
    # A new-draft request for the same person: the running one fills
    # "Other", which is not the draft it asked for.
    with _pytest.raises(orcid_service.ImportAlreadyRunning) as e:
        prosopia_service.prepare_import(conn, s, user_id=1, ref="doe-jane", prosopia_client=_Client())
    assert e.value.slug is None


def test_the_route_records_whom_each_import_imports(env):
    app = _app(StubProsopia(_profile()), StubOrcidOpenAlex(WORKS))
    slug = _wizard(app).json()["draft_slug"]
    assert _request(app, "POST", "/api/import/orcid",
                    json={"orcid": ORCID, "openalex_ids": ["W1"], "into": slug}).status_code == 200
    assert _q(env, "SELECT tier_used, import_key, import_orcid, opened_draft FROM gather_runs ORDER BY id") == [
        ("prosopia_import", "prosopia:sheffield-nathan", None, 1),
        ("orcid_import", f"orcid:{ORCID}", ORCID, 0),
    ]


def test_an_added_topic_is_written_as_the_list_writes_its_ids():
    """OpenAlex's own lists use URLs (the stubs here use bare codes): a
    topic a Prosopia interest adds takes the list's form, so one topic is one id."""
    from rag_lib import prosopia_interests

    listed = [{"id": "https://openalex.org/T10515", "name": "Cancer", "profile": "p", "profile_name": "P"}]
    for have, want in (("https://openalex.org/T10222", "https://openalex.org/T10515"), ("T10222", "T10515")):
        out = prosopia_interests.apply({"topics": [{"id": have, "on": True}]}, {}, listed)
        assert [t["id"] for t in out["topics"]] == [have, want]
        assert out["topics"][1]["on"] is False
    # Switched on by the user since (the stored state under the list's id): stays on.
    out = prosopia_interests.apply({"topics": [{"id": "https://openalex.org/T10222"}]},
                                   {"https://openalex.org/T10515": True}, listed)
    assert out["topics"][1]["on"] is True


def test_m3_the_interests_are_what_the_profile_states_not_what_prosopia_infers(monkeypatch):
    """A real profile (sheffield-nathan) states 5 interests; its
    research_interests carry 15 more topics Prosopia inferred from the
    papers. Only the stated ones are brought in; its not interests none."""
    import types
    from rag_lib import prosopia_interests, researcher_topics
    from tests.test_researcher_topics import _text

    doc = {
        "interests": ["Cancer-related molecular mechanisms research", "Scientific Computing and Data Management",
                      "A stated interest with no topic entry"],
        "not_interests": ["Genomics and Phylogenetic Studies"],
        "research_interests": [
            _text("phylogenomics", weight=-0.5, method="inferred"),
            _topic("T10515", "Cancer-related molecular mechanisms research"),
            _topic("T10015", "Genomics and Phylogenetic Studies"),
            _topic("T10887", "Bioinformatics and Genomic Networks"),
            _topic("T11986", "Scientific Computing and Data Management", weight=0.5, method="declared"),
        ],
    }
    view = prosopia_interests.stated(doc)
    assert [t["id"] for t in view["wanted"]] == ["https://openalex.org/T10515", "https://openalex.org/T11986"]
    assert view["wanted_labels"] == ["A stated interest with no topic entry"]
    assert view["unwanted"] == [] and view["unwanted_labels"] == []
    raw = json.dumps([{"key": "p", "name": "P", "document": doc}])
    assert [t["name"] for t in prosopia_interests.interests(raw, settings=None)] == [
        "Cancer-related molecular mechanisms research", "Scientific Computing and Data Management"]
    # Labels matched (UMLS on): the label's nearest topic joins, even one
    # the not interests name outright.
    monkeypatch.setattr(researcher_topics, "_nearest_topic", lambda label, settings: {
        "id": "https://openalex.org/T10015", "display_name": "Genomics and Phylogenetic Studies", "similarity": 0.9})
    got = prosopia_interests.interests(raw, settings=types.SimpleNamespace(RADAR_UMLS_ENABLED=True))
    assert [(_c(t["id"]), t["label"]) for t in got][-1] == ("T10015", "A stated interest with no topic entry")
    # Without the field: the declared, wanted entries, not the inferred.
    bare = {"research_interests": doc["research_interests"]}
    assert [t["id"] for t in prosopia_interests.stated(bare)["wanted"]] == ["https://openalex.org/T11986"]


def test_m3_step_three_stores_a_drafts_ticks_without_a_rerank(env, monkeypatch):
    """The topic step stores the ticks as they change (a seed added since
    re-reads them from the draft); a draft has no pool, so no rerank."""
    from rag_lib.scheduler import jobs

    asked: list[int] = []
    monkeypatch.setattr(jobs, "request_rerank_refresh", lambda pid: asked.append(pid))
    app = _app(StubProsopia(_profile()), _stub_openalex())
    slug = _wizard(app).json()["draft_slug"]
    r = _request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": [SEED_TOPIC, "T10515"]})
    assert r.status_code == 200, r.text
    assert asked == [] and _topics(app, slug)["T10515"]["on"] is True
