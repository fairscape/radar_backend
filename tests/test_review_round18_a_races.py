"""Review 18 (a): races between requests, forced deterministically.

Each test makes the "other tab" act at a precise point inside the first
request by wrapping one internal step; the other tab uses its own
connection, as a second request would.
"""

from __future__ import annotations

from rag_lib.db import connect
from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    _go_live, _import_researcher, _ok, _one, _q, _request, _upload, app, env,
)


def _pid(env, slug):
    return _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]


def _draft_ready(app, name):
    rid = _import_researcher(app)
    papers = _ok(_request(app, "GET", f"/api/researchers/{rid}"))["papers"]
    slug = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests",
                        json={"name": name, "openalex_ids": [p["id"] for p in papers]}))["draft"]["slug"]
    topics = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    return slug, rid, [t["id"] for t in topics if t["on"]]


def test_commit_while_draft_discarded_in_other_tab(app, env, monkeypatch):
    """Discard in tab 2 lands while tab 1's commit is fitting."""
    from rag_lib.api.services import wizard as wizard_service

    slug, _rid, on = _draft_ready(app, "Commit Race")
    pid = _pid(env, slug)
    real = wizard_service._load_profile

    def load_then_discard(conn, row, **kw):
        out = real(conn, row, **kw)
        other = connect(env)
        try:
            ok = wizard_service.delete_draft(other, user_id=1, slug=slug)
            assert ok
        finally:
            other.close()
        return out

    monkeypatch.setattr(wizard_service, "_load_profile", load_then_discard)
    resp = _request(app, "POST", "/api/profiles", json={
        "slug": slug, "threshold": 0.1, "selected_topic_ids": on, "years": None})
    print("COMMIT", resp.status_code, resp.text[:300])
    assert _one(env, "SELECT 1 FROM profiles WHERE id = ?", pid) is None
    # The draft is gone: a 404, not an unhandled IntegrityError.
    assert resp.status_code == 404, resp.text


def test_two_tabs_remove_the_last_two_seeds(app, env, monkeypatch):
    """A live interest with two seeds; each tab removes one at once."""
    from rag_lib.api.services import wizard as wizard_service

    slug, _rid = _go_live(app, env, "Seed Race", seeds_from_researcher=False)
    pid = _pid(env, slug)
    seeds = [r[0] for r in _q(env, "SELECT openalex_id FROM profile_seeds WHERE profile_id=?", pid)]
    assert len(seeds) == 2
    real = wizard_service.embed_missing
    fired = []

    def embed_then_other_tab(conn, model, ids):
        out = real(conn, model, ids)
        if not fired:
            fired.append(1)
            other = connect(env)
            try:
                # Tab 2: removes the other seed; sees two seeds, so allowed.
                assert wizard_service.remove_seed(other, user_id=1, slug=slug, openalex_id=seeds[1])
            finally:
                other.close()
        return out

    monkeypatch.setattr(wizard_service, "embed_missing", embed_then_other_tab)
    resp = _request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": seeds[0]})
    n_left = _one(env, "SELECT COUNT(*) FROM profile_seeds WHERE profile_id=?", pid)[0]
    print("REMOVE", resp.status_code, resp.text[:300], "left", n_left)
    # The rule "an interest needs at least one seed" must hold.
    assert n_left >= 1, (resp.status_code, resp.text)


def test_new_interest_while_researcher_forgotten(app, env, monkeypatch):
    """Tab 2 forgets the researcher while tab 1 builds an interest from them."""
    from rag_lib.api.services import researchers as researchers_service
    from rag_lib.api.services import wizard as wizard_service

    rid = _import_researcher(app)
    papers = _ok(_request(app, "GET", f"/api/researchers/{rid}"))["papers"]
    real = wizard_service.embed_missing

    def embed_then_forget(conn, model, ids):
        out = real(conn, model, ids)
        other = connect(env)
        try:
            researchers_service.delete_researcher(other, 1, rid)
        finally:
            other.close()
        return out

    monkeypatch.setattr(wizard_service, "embed_missing", embed_then_forget)
    resp = _request(app, "POST", f"/api/researchers/{rid}/interests",
                    json={"name": "Gone", "openalex_ids": [p["id"] for p in papers]})
    drafts = _q(env, "SELECT slug, n_seed FROM profiles WHERE is_draft = 1")
    print("CREATE", resp.status_code, resp.text[:300], [tuple(d) for d in drafts])
    assert resp.status_code in (200, 404), resp.text
    if resp.status_code == 404:
        assert drafts == [], "a failed create left an empty draft behind"


def test_reimport_while_researcher_forgotten(app, env, monkeypatch):
    """Tab 2 forgets the researcher while tab 1's "check for new papers"
    import is being prepared (the person row written, the run not yet open)."""
    from rag_lib.api.services import researchers as researchers_service

    rid = _import_researcher(app)
    real = researchers_service.record_prosopia_researcher

    def record_then_forget(conn, **kw):
        out = real(conn, **kw)
        other = connect(env)
        try:
            researchers_service.delete_researcher(other, 1, out)
        finally:
            other.close()
        return out

    monkeypatch.setattr(researchers_service, "record_prosopia_researcher", record_then_forget)
    resp = _request(app, "POST", "/api/researchers/import",
                    json={"source": "prosopia", "ref": "sheffield-nathan"})
    print("REIMPORT", rid, resp.status_code, resp.text[:300])
    assert resp.status_code < 500, resp.text


def test_draft_trial_while_draft_discarded(app, env, monkeypatch):
    """Tab 2 discards the draft while tab 1's "Run trial" saves its topics."""
    from rag_lib.api.services import wizard as wizard_service

    slug, _rid, on = _draft_ready(app, "Trial Race")
    real = wizard_service.save_topics

    def save_then_discard(conn, **kw):
        out = real(conn, **kw)
        other = connect(env)
        try:
            assert wizard_service.delete_draft(other, user_id=1, slug=slug)
        finally:
            other.close()
        return out

    monkeypatch.setattr(wizard_service, "save_topics", save_then_discard)
    resp = _request(app, "POST", f"/api/profiles/draft/{slug}/dry-run",
                    json={"days": 30, "selected_topic_ids": on})
    print("TRIAL", resp.status_code, resp.text[:300])
    assert resp.status_code == 404, resp.text


def test_add_papers_into_draft_discarded_meanwhile(app, env, monkeypatch):
    """Wizard "add papers from a Prosopia profile" into a draft that tab 2
    discards while the profile is being read."""
    from rag_lib.api.services import prosopia as prosopia_service
    from rag_lib.api.services import wizard as wizard_service

    slug = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Into Race"}))["slug"]
    real = prosopia_service._resolve_profile

    def resolve_then_discard(*a, **kw):
        out = real(*a, **kw)
        other = connect(env)
        try:
            assert wizard_service.delete_draft(other, user_id=1, slug=slug)
        finally:
            other.close()
        return out

    monkeypatch.setattr(prosopia_service, "_resolve_profile", resolve_then_discard)
    resp = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "into": slug})
    print("INTO", resp.status_code, resp.text[:300])
    assert resp.status_code == 404, resp.text
