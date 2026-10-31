"""The wizard's imports and the draft sandbox (FUNCTIONAL_SPEC D, S5-S8):
a draft never changes a stored researcher, a researcher and papers it
imported go when it is discarded, and an import OpenAlex cannot answer
leaves nothing behind."""

from __future__ import annotations

from rag_lib.db import connect
from rag_lib.openalex_status import OpenAlexUnavailable
from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex, _app, _request, env  # noqa: F401

TABLES = ("profiles", "papers", "paper_embeddings", "profile_seeds", "researchers",
          "researcher_papers", "draft_owned")


def _q(env, sql, *a):
    conn = connect(env)
    try:
        return [tuple(r) for r in conn.execute(sql, a).fetchall()]
    finally:
        conn.close()


# Bookkeeping a failed import may leave: when the draft it was added to was
# last touched, and the researcher's last run (the failed one, which the
# Profiles page then reports).
_BOOKKEEPING = {"profiles": ("updated_at",), "researchers": ("last_run_id", "updated_at")}


def _snapshot(env, tables=TABLES, *, bookkeeping=False):
    out = {}
    for t in tables:
        cols = [r[1] for r in _q(env, f"PRAGMA table_info({t})")]
        if bookkeeping:
            cols = [c for c in cols if c not in _BOOKKEEPING.get(t, ())]
        out[t] = sorted(map(repr, _q(env, f"SELECT {', '.join(cols)} FROM {t}")))
    return out


def _wizard_import(app, ids, orcid=ORCID, **body):
    return _request(app, "POST", "/api/import/orcid", json={"orcid": orcid, "openalex_ids": ids, **body})


class Unanswering(StubOrcidOpenAlex):
    """Lists the author's works, then cannot answer for them (quota)."""

    def paginate_filter(self, filter_str, **kw):
        if filter_str.startswith("doi:"):
            raise OpenAlexUnavailable("quota", detail="429")
        return super().paginate_filter(filter_str, **kw)

    def get_work(self, openalex_id):
        raise OpenAlexUnavailable("quota", detail="429")


def test_s5_discarding_a_wizard_import_removes_the_new_researcher_and_its_papers(env):
    before = _snapshot(env)
    app = _app(StubOrcidOpenAlex(WORKS))
    r = _wizard_import(app, ["W1", "W3"], name="S5")
    assert r.status_code == 200, r.text
    slug, run_id = r.json()["draft_slug"], r.json()["run_id"]
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(1,)]
    assert _q(env, "SELECT COUNT(*) FROM paper_embeddings") == [(2,)]
    assert len(_q(env, "SELECT * FROM draft_owned")) == 3        # the researcher, W1, W3
    assert _request(app, "GET", "/api/researchers").json() != []

    r = _request(app, "DELETE", f"/api/profiles/draft/{slug}")
    assert r.status_code == 200, r.text
    assert _snapshot(env) == before
    assert _q(env, "SELECT 1 FROM gather_runs WHERE id = ?", run_id) == []
    assert _request(app, "GET", "/api/researchers").json() == []


def _stored_researcher(app):
    """R, imported from the Profiles page: the user's own, not a draft's."""
    r = _request(app, "POST", "/api/researchers/import", json={"source": "orcid", "ref": ORCID})
    assert r.status_code == 200, r.text
    return r.json()["researcher_id"]


def test_s6_a_draft_built_from_a_stored_researcher_leaves_it_as_it_was(env):
    app = _app(StubOrcidOpenAlex(WORKS))
    rid = _stored_researcher(app)
    before = _snapshot(env, ("papers", "paper_embeddings", "researchers", "researcher_papers"))
    r = _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "From R"})
    assert r.status_code == 200, r.text
    assert _q(env, "SELECT * FROM draft_owned") == []
    r = _request(app, "DELETE", f"/api/profiles/draft/{r.json()['draft']['slug']}")
    assert r.status_code == 200, r.text
    assert _snapshot(env, ("papers", "paper_embeddings", "researchers", "researcher_papers")) == before


def test_s7_the_wizard_does_not_import_a_stored_researcher_again(env):
    app = _app(StubOrcidOpenAlex(WORKS))
    rid = _stored_researcher(app)
    before = _snapshot(env)
    r = _wizard_import(app, ["W1"], name="Again")
    assert r.status_code == 409, r.text
    assert r.json()["researcher"]["id"] == rid and "Nothing was changed" in r.json()["detail"]
    assert _snapshot(env) == before
    # Nor into an existing draft.
    d = _request(app, "POST", "/api/profiles/draft", json={"name": "Mine"}).json()["slug"]
    before = _snapshot(env)
    r = _wizard_import(app, ["W1"], into=d)
    assert r.status_code == 409 and _snapshot(env) == before


def test_the_same_person_under_prosopia_is_a_stored_researcher_too(env):
    from rag_lib.db.repos import users as users_repo

    conn = connect(env)
    try:
        uid = int(users_repo.get_by_email(conn, "demo@example.com")["id"])
        conn.execute("INSERT INTO researchers (user_id, source, key, name, orcid) VALUES (?, 'prosopia', 'jn', 'J N', ?)",
                     (uid, ORCID.lower()))
        conn.commit()
    finally:
        conn.close()
    app = _app(StubOrcidOpenAlex(WORKS))
    r = _wizard_import(app, ["W1"])
    assert r.status_code == 409, r.text


def test_more_papers_of_the_drafts_own_researcher_can_be_added(env):
    app = _app(StubOrcidOpenAlex(WORKS))
    slug = _wizard_import(app, ["W1"], name="Mine").json()["draft_slug"]
    r = _wizard_import(app, ["W3"], into=slug)
    assert r.status_code == 200, r.text
    assert len(_q(env, "SELECT * FROM researcher_papers")) == 2
    _request(app, "DELETE", f"/api/profiles/draft/{slug}")
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(0,)]
    assert _q(env, "SELECT COUNT(*) FROM papers") == [(0,)]


def test_s8_a_wizard_import_openalex_cannot_answer_leaves_nothing(env):
    before = _snapshot(env)
    app = _app(Unanswering(WORKS))
    r = _wizard_import(app, ["W1", "W3"], name="S8")
    assert r.status_code == 503, r.text
    assert "Nothing was changed" in r.json()["detail"]
    assert _snapshot(env) == before
    # The run stays, detached, so a poll still says why.
    (run,) = _q(env, "SELECT profile_id, researcher_id, error FROM gather_runs")
    assert run[:2] == (None, None) and "daily request limit" in run[2]


def test_an_import_into_a_draft_openalex_cannot_answer_keeps_the_draft_as_it_was(env):
    app = _app(StubOrcidOpenAlex(WORKS))
    d = _request(app, "POST", "/api/profiles/draft", json={"name": "Kept"}).json()["slug"]
    before = _snapshot(env, bookkeeping=True)
    r = _wizard_import(_app(Unanswering(WORKS)), ["W1"], into=d)
    assert r.status_code == 503, r.text
    assert _snapshot(env, bookkeeping=True) == before


def test_a_profiles_page_import_openalex_cannot_answer_leaves_no_new_researcher(env):
    before = _snapshot(env)
    r = _request(_app(Unanswering(WORKS)), "POST", "/api/researchers/import", json={"source": "orcid", "ref": ORCID})
    assert r.status_code == 503, r.text
    assert _snapshot(env) == before


def test_a_profiles_page_refresh_openalex_cannot_answer_keeps_the_researcher(env):
    app = _app(StubOrcidOpenAlex(WORKS))
    _stored_researcher(app)
    before = _snapshot(env, bookkeeping=True)
    r = _request(_app(Unanswering(WORKS)), "POST", "/api/researchers/import", json={"source": "orcid", "ref": ORCID})
    assert r.status_code == 503, r.text
    assert _snapshot(env, bookkeeping=True) == before


def test_a_researcher_a_draft_imported_is_the_users_once_imported_from_the_profiles_page(env):
    app = _app(StubOrcidOpenAlex(WORKS))
    slug = _wizard_import(app, ["W1"], name="Mine").json()["draft_slug"]
    _stored_researcher(app)
    _request(app, "DELETE", f"/api/profiles/draft/{slug}")
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(1,)]
    assert _q(env, "SELECT COUNT(*) FROM researcher_papers") == [(len(WORKS),)]


def test_an_import_that_cannot_start_leaves_nothing(env, monkeypatch):
    """No scheduler: the run cannot start, and the draft and researcher the
    request made go again."""
    from rag_lib.api.app import create_app
    from rag_lib.api.services import orcid as orcid_service

    monkeypatch.setattr(orcid_service, "OpenAlexClient", lambda **_: StubOrcidOpenAlex(WORKS))
    before = _snapshot(env)
    r = _request(create_app(), "POST", "/api/import/orcid", json={"orcid": ORCID, "openalex_ids": ["W1"]})
    assert r.status_code == 503, r.text
    assert _snapshot(env) == before


def test_the_background_import_openalex_cannot_answer_leaves_nothing(env, monkeypatch):
    """The scheduler's job, not the inline test path: it closes the run with
    the reason and removes the draft and researcher the request made."""
    from rag_lib.api import settings as settings_module
    from rag_lib.api.services import orcid as orcid_service
    from rag_lib.api.services import prosopia as prosopia_service
    from rag_lib.db.repos import gather_runs as gather_runs_repo
    from rag_lib.db.repos import users as users_repo
    from rag_lib.scheduler.jobs import import_prosopia_profile

    before = _snapshot(env)
    settings = settings_module.get_settings()
    conn = connect(env)
    try:
        uid = int(users_repo.get_by_email(conn, "demo@example.com")["id"])
        plan = orcid_service.prepare_import(conn, settings, user_id=uid, orcid=ORCID, openalex_ids=["W1"],
                                            openalex_client=StubOrcidOpenAlex(WORKS))
        run_id = gather_runs_repo.start(conn, profile_id=plan.profile_id, researcher_id=plan.researcher_id,
                                        user_id=uid, tier_used="orcid_import")
    finally:
        conn.close()
    monkeypatch.setattr(prosopia_service, "OpenAlexClient", lambda **_: Unanswering(WORKS))
    import_prosopia_profile(plan, settings=settings, run_id=run_id)
    assert _snapshot(env) == before
    (run,) = _q(env, "SELECT profile_id, researcher_id, finished_at IS NOT NULL, error FROM gather_runs")
    assert run[:3] == (None, None, 1) and run[3].startswith("OpenAlex's daily request limit")


def test_each_seed_says_where_it_came_from(env):
    """FUNCTIONAL_SPEC D.2-3: an upload, a researcher's papers (whose), the
    vault, the feed; the first source stays when a paper is added again."""
    app = _app(StubOrcidOpenAlex(WORKS))
    rid = _stored_researcher(app)
    r = _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "Mixed"})
    slug = r.json()["draft"]["slug"]
    detail = _request(app, "GET", f"/api/profiles/{slug}/detail").json()
    assert {(s["source"], s["sourceResearcherId"], s["sourceResearcherName"]) for s in detail["seeds"]} == \
        {("researcher", rid, "Justin C. Niestroy")}
    w1 = detail["seeds"][0]["id"]
    _request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": w1})
    # A new interest's draft has no feed: "feed" is refused (E.11), "vault" is not.
    r = _request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [w1], "source": "feed"})
    assert r.status_code == 200 and r.json()["rejected"] == [w1], r.text
    r = _request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [w1], "source": "vault"})
    assert r.status_code == 200 and r.json()["attached"] == 1, r.text
    r = _request(app, "POST", f"/api/profiles/{slug}/seeds",
                 json={"openalex_ids": [w1], "source": "researcher", "researcher_id": rid})
    seeds = {s["id"]: s for s in _request(app, "GET", f"/api/profiles/{slug}/detail").json()["seeds"]}
    assert (seeds[w1]["source"], seeds[w1]["sourceResearcherId"]) == ("vault", None)
    # A researcher that is not the user's is refused.
    r = _request(app, "POST", f"/api/profiles/{slug}/seeds",
                 json={"openalex_ids": [w1], "source": "researcher", "researcher_id": 999})
    assert r.status_code == 404
    r = _request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [w1], "source": "nowhere"})
    assert r.status_code == 422
