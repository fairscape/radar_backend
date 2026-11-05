"""Review round 3 (reviewer b): multi-user isolation.

User A owns a researcher, a draft built from it, a plain draft, a live
interest with candidates, and an uploaded PDF. User B then hits every
read and mutating endpoint with A's ids/slugs. Expected: 404 (or 400/409
for list-shaped bodies), never a 2xx that touches A's data and never a 5xx.
Afterwards A's state is re-read and must be byte-for-byte what it was.

Every API route is either hit by B here or on ``NO_USER_DATA`` with its
reason; a route added later without either fails the sweep
(FUNCTIONAL_SPEC 2.4).
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from rag_lib.db import connect
from rag_lib.db.repos import (
    candidates as candidates_repo,
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)
from tests.edit_helpers import edit_saved
from tests.test_orcid_import import ORCID, WORKS
from tests.test_prosopia_import import StubProsopia, _stub_profile
from tests.test_researcher_marks import _both_stubs
from tests.test_researchers import _app, _request, env  # noqa: F401 -- fixture

A = {"X-User-Email": "alice@example.com"}
B = {"X-User-Email": "bob@example.com"}


# Routes that read nothing of a user's: what B gets is what A gets.
NO_USER_DATA = {
    ("GET", "/api/health"): "service health",
    ("GET", "/api/openalex/status"): "OpenAlex reachability",
    ("GET", "/api/profiles/wizard/options"): "the embedders and selectors installed",
    ("GET", "/api/users/me"): "the caller's own row (test_api_auth.py)",
    ("PATCH", "/api/users/me"): "the caller's own row (test_api_auth.py)",
    # Sign-in (records/21). None takes a user id or returns anyone's data;
    # who the cookie names is tested in test_signin_identity.py and
    # test_api_oidc.py.
    ("GET", "/auth/config"): "which sign-in buttons are on, and the provider of the caller's own cookie",
    ("GET", "/auth/login/{provider}"): "starts a sign-in: nobody is signed in yet",
    ("GET", "/auth/login/callback"): "finishes a sign-in into the address the provider verified",
    ("GET", "/auth/logout"): "clears the caller's own cookie",
    # Needs a sign-in session (self_rid), which this header-model world
    # has none of; B-cannot-see-A's-import is test_api_my_prosopia.py::
    # test_another_users_import_is_not_mine.
    ("POST", "/api/users/me/lease"): "the caller's own lease (test_write_lease.py::test_accounts_are_separate)",
    ("POST", "/api/users/me/lease/release"): "the caller's own lease (test_write_lease.py::test_accounts_are_separate)",
    ("GET", "/api/users/me/prosopia"): "the caller's own Prosopia profile (tested for isolation in test_api_my_prosopia.py)",
}

# Routes left out on purpose, with the reason.
NOT_TESTED = {
    # Vault chat is not tested and is being closed before launch; without a
    # vector store it 500s here for anyone (to be settled when it is closed).
    ("POST", "/api/chat"): "Vault chat (FUNCTIONAL_SPEC: not tested, closed before launch)",
}


def _api_routes(app):
    """Full API path of every route object, by ``id()``.

    The routers are included with a prefix; the route that answers a
    request (``scope["route"]``) carries only its path inside its router.
    Checked against the OpenAPI listing so a FastAPI that lays its routes
    out differently fails here, not silently.
    """
    from fastapi.routing import APIRoute

    full = {}
    for entry in app.routes:
        if isinstance(entry, APIRoute):
            full[id(entry)] = (entry, entry.path)
        elif hasattr(entry, "original_router"):
            prefix = entry.include_context.prefix
            for r in entry.original_router.routes:
                if isinstance(r, APIRoute):
                    full[id(r)] = (r, prefix + r.path)
    ops = {(m, path) for r, path in full.values() for m in r.methods if path.startswith(("/api", "/auth"))}
    import re

    listed = {(m.upper(), path) for path, item in app.openapi()["paths"].items() for m in item}
    assert {(m, re.sub(r":\w+}", "}", p)) for m, p in ops} == listed  # OpenAPI drops ":path"
    return ops, {k: path for k, (_, path) in full.items()}


def _request_served(app, method, path, **kwargs):
    """``_request``, also returning the route that answered."""
    import asyncio

    import httpx

    seen = {}

    async def recorder(scope, receive, send):
        await app(scope, receive, send)
        if scope["type"] == "http":
            seen["route"] = scope.get("route")

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=recorder),
                                         base_url="http://test") as client:
                return await client.request(method, path, **kwargs)

    r = asyncio.run(run())
    route = seen.get("route")
    return r, route


class _FakeScheduler:
    def __init__(self):
        self.jobs = []

    def add_job(self, *a, **k):
        self.jobs.append((a, k))

    def remove_job(self, *a, **k):
        pass


@pytest.fixture()
def world(env, monkeypatch):
    from rag_lib.api.deps import get_scheduler
    from rag_lib.api.services import vault as vault_service
    from rag_lib.vault import PdfIngestRecord

    def _fake_ingest_pdf(path):
        from pathlib import Path
        p = Path(path)
        return PdfIngestRecord(path=str(p), title=f"Title for {p.stem[:8]}",
                               body_text="hello world from a fake pdf", doi=None, n_pages=3)

    class _NoNetClient:
        def __init__(self, **_): pass
        def lookup_by_doi(self, doi): return None
        def lookup_by_title(self, *a, **kw): return None
        def search_by_title(self, *a, **kw): return []
        def paper_from_work(self, *a, **kw): return None

    monkeypatch.setattr(vault_service, "ingest_pdf", _fake_ingest_pdf)
    monkeypatch.setattr(vault_service, "OpenAlexClient", _NoNetClient)

    app = _app(StubProsopia(_stub_profile()), _both_stubs())
    fake = _FakeScheduler()
    app.dependency_overrides[get_scheduler] = lambda: fake

    # --- A's world -------------------------------------------------------
    r = _request(app, "POST", "/api/researchers/import",
                 json={"source": "prosopia", "ref": "sheffield-nathan"}, headers=A)
    assert r.status_code == 200, r.text
    rid, run_id = r.json()["researcher_id"], r.json()["run_id"]

    r = _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "A from nathan"}, headers=A)
    assert r.status_code == 200, r.text
    draft_slug = r.json()["draft"]["slug"]
    assert r.json()["n_seeds"] == 5

    r = _request(app, "POST", "/api/profiles/draft", json={"name": "A plain"}, headers=A)
    assert r.status_code == 200, r.text
    plain_slug = r.json()["slug"]

    # A live interest with seeds and candidates, written straight to the DB.
    conn = connect(env)
    try:
        a_uid = int(users_repo.get_by_email(conn, "alice@example.com")["id"])
        rng = np.random.default_rng(7)
        live_id = profiles_repo.upsert(
            conn, user_id=a_uid, name="A live", embedding_model="placeholder-v1", n_seed=3,
            topic_filters={"topics": [{"id": "T1", "display_name": "T one", "count": 3, "on": True}]},
            threshold=0.5, coherence_median=0.7, coherence_iqr=0.05, coherence_bimodal=False,
        )
        for i in range(3):
            oa = f"https://openalex.org/W{9000000 + i}"
            papers_repo.upsert(conn, {"openalex_id": oa, "title": f"A seed {i}", "year": 2024, "source": "user_csv"})
            v = rng.standard_normal(64).astype("float32"); v /= np.linalg.norm(v)
            embeddings_repo.upsert(conn, oa, "placeholder-v1", v)
            profiles_repo.attach_seed(conn, live_id, oa)
        g = gather_runs_repo.start(conn, profile_id=live_id, user_id=a_uid, tier_used="manual")
        cards = []
        for i in range(5):
            oa = f"https://openalex.org/W{9100000 + i}"
            papers_repo.upsert(conn, {"openalex_id": oa, "title": f"A cand {i}", "year": 2026,
                                      "source": "openalex_gatherer", "abstract": "x"})
            candidates_repo.insert_dedup(conn, profile_id=live_id, openalex_id=oa, score=0.9 - i * 0.01,
                                         score_raw=0.9 - i * 0.01, gather_run_id=g)
            cards.append(oa)
        gather_runs_repo.finish(conn, g, n_fetched=5, n_new=5, n_redup=0)
        live_slug = profiles_repo.get(conn, live_id)["slug"]
    finally:
        conn.close()

    # A saved interest takes uploads only through Edit interest (spec E):
    # upload into the copy, then save it back onto 'A live'.
    files = {"file": ("alpha.pdf", io.BytesIO(b"%PDF-1.4\nalice-bytes"), "application/pdf")}
    with edit_saved(lambda m, p, **kw: _request(app, m, p, headers=A, **kw), live_slug) as copy:
        r = _request(app, "POST", "/api/vault/upload", files=files, data={"profile_slug": copy}, headers=A)
        assert r.status_code == 200, r.text
        upload_id = r.json()["id"]

    seed_of_draft = _request(app, "GET", f"/api/researchers/{rid}", headers=A).json()["papers"][0]["id"]

    return dict(app=app, env=env, rid=rid, run_id=run_id, draft=draft_slug, plain=plain_slug,
                live=live_slug, live_id=live_id, cards=cards, upload=upload_id, seed=seed_of_draft,
                fake=fake)


def _snapshot(env):
    conn = connect(env)
    try:
        q = lambda sql: [tuple(r) for r in conn.execute(sql).fetchall()]  # noqa: E731
        return {
            "researchers": q("SELECT id, user_id, mark, position, name FROM researchers ORDER BY id"),
            "profiles": q("SELECT id, user_id, slug, name, is_draft, threshold, year_from, year_to, "
                          "topic_filters_json, researcher_id, interest_terms_json FROM profiles ORDER BY id"),
            "seeds": q("SELECT profile_id, openalex_id FROM profile_seeds ORDER BY 1, 2"),
            "cands": q("SELECT profile_id, openalex_id, saved_at, dismissed_at, vaulted_at, unsaved_at "
                       "FROM profile_candidates ORDER BY 1, 2"),
            "sched": q("SELECT profile_id, cron, tz, enabled FROM profile_schedules ORDER BY 1"),
            "rpapers": q("SELECT researcher_id, openalex_id FROM researcher_papers ORDER BY 1, 2"),
            "uploads": q("SELECT user_id, openalex_id FROM paper_uploads ORDER BY 1, 2"),
            "feedback": q("SELECT user_id, profile_id, openalex_id, action FROM feedback_events ORDER BY id"),
            # CR.1: what keeps a paper in a Vault, and the papers themselves.
            "kept": q("SELECT user_id, openalex_id FROM vault_kept ORDER BY 1, 2"),
            "deleted": q("SELECT profile_id, openalex_id FROM deleted_seeds ORDER BY 1, 2"),
            "owned": q("SELECT profile_id, kind, item FROM draft_owned ORDER BY 1, 2, 3"),
            "versions": q("SELECT version_id, user_id, canonical_openalex_id, file_hash FROM paper_versions ORDER BY 1"),
            "papers": q("SELECT openalex_id FROM papers ORDER BY 1"),
            "vectors": q("SELECT openalex_id, embedding_model FROM paper_embeddings ORDER BY 1, 2"),
        }
    finally:
        conn.close()


def test_user_b_cannot_read_or_touch_user_a(world):
    app, env = world["app"], world["env"]
    rid, run_id = world["rid"], world["run_id"]
    # A is mid-Edit interest on 'A live', and has an ORCID researcher saved.
    r = _request(app, "POST", f"/api/profiles/{world['live']}/edit", headers=A)
    assert r.status_code == 200, r.text
    copy = r.json()["slug"]
    r = _request(app, "POST", "/api/researchers/import", headers=A,
                 json={"source": "orcid", "ref": ORCID, "paper_ids": [WORKS[0]["id"]]})
    assert r.status_code == 200, r.text
    orcid_rid = r.json()["researcher_id"]
    # What A is told it already has; B must be told nothing.
    r = _request(app, "GET", f"/api/import/orcid/{ORCID}/works", headers=A)
    assert r.status_code == 200 and r.json()["saved_profile"], r.text
    r = _request(app, "GET", "/api/import/prosopia/works?ref=sheffield-nathan", headers=A)
    assert r.status_code == 200 and r.json()["saved_profile"], r.text
    before = _snapshot(env)
    a_profiles_before = _request(app, "GET", "/api/profiles", headers=A).json()
    a_researchers_before = _request(app, "GET", "/api/researchers", headers=A).json()

    bad: list[tuple[str, str, int, str]] = []
    served: set[tuple[str, str]] = set()
    ops, full_path = _api_routes(app)

    def hit(method, path, expect=(404,), **kw):
        r, route = _request_served(app, method, path, headers=B, **kw)
        served.add((method, full_path.get(id(route))))
        if r.status_code not in expect:
            bad.append((method, path, r.status_code, r.text[:200]))
        return r

    # ---- researchers --------------------------------------------------------
    hit("GET", f"/api/researchers/{rid}")
    hit("GET", f"/api/researchers/{rid}/works")
    hit("GET", f"/api/researchers/{rid}/suggestions")
    hit("PATCH", f"/api/researchers/{rid}", json={"mark": "self"})
    hit("PUT", "/api/researchers/order", expect=(400,), json={"ids": [rid]})
    hit("POST", f"/api/researchers/{rid}/interests", json={"name": "stolen"})
    hit("DELETE", f"/api/researchers/{rid}")
    hit("GET", f"/api/researchers/{orcid_rid}")
    r = hit("GET", "/api/researchers", expect=(200,)); assert r.json() == []

    # ---- import runs / into A's draft ----------------------------------------
    hit("GET", f"/api/import/prosopia/{run_id}")
    hit("POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "into": world["draft"]})
    hit("POST", "/api/import/orcid",
        json={"orcid": ORCID, "openalex_ids": [WORKS[0]["id"]], "into": world["draft"]})
    r = hit("GET", f"/api/import/orcid/{ORCID}/works", expect=(200,))
    assert r.json()["saved_profile"] is None, r.text
    r = hit("GET", "/api/import/prosopia/works?ref=sheffield-nathan", expect=(200,))
    assert r.json()["saved_profile"] is None, r.text

    # ---- every /{key} route, for a draft and a live interest of A's ---------
    r = hit("GET", "/api/profiles", expect=(200,)); assert r.json() == []
    r = hit("GET", "/api/profiles/drafts", expect=(200,)); assert r.json() == []
    # B's "discard my left-behind edits" leaves A's open copy alone.
    r = hit("POST", "/api/profiles/drafts/discard-edits", expect=(200,), json={"tab": ""})
    assert r.json() == {"discarded": 0}
    for s in (world["draft"], world["live"], world["plain"], copy):
        # L11: 404, as for a key that does not exist -- not A's row.
        r = hit("GET", f"/api/profiles/{s}", expect=(404,))
        assert r.json() == {"detail": "No such interest."}, (s, r.text)
        r = hit("GET", f"/api/profiles/{s}/detail", expect=(404,))
        assert r.json() == {"detail": "No such interest."}, (s, r.text)
        hit("PATCH", f"/api/profiles/{s}", json={"threshold": 0.25})
        hit("PATCH", f"/api/profiles/{s}", json={"years": {"year_from": 2000, "year_to": 2010}})
        hit("POST", f"/api/profiles/{s}/recompute-coherence")
        hit("GET", f"/api/profiles/{s}/umls")
        hit("POST", f"/api/profiles/{s}/umls")
        hit("GET", f"/api/profiles/{s}/terms")
        hit("POST", f"/api/profiles/{s}/terms/generate", json={})
        hit("PUT", f"/api/profiles/{s}/terms", json={"terms": [{"label": "x"}], "removed": [], "not_interests": []})
        hit("GET", f"/api/profiles/{s}/topics/preview")
        hit("PUT", f"/api/profiles/{s}/topics", json={"selected_topic_ids": ["T1"]})
        hit("POST", f"/api/profiles/{s}/recompute-topics")
        hit("GET", f"/api/profiles/{s}/researcher-topics")
        hit("PUT", f"/api/profiles/{s}/researcher-topics", json={"enabled": True})
        hit("PUT", f"/api/profiles/{s}/researcher-topics", json={"enabled": False})
        hit("PUT", f"/api/profiles/{s}/researcher-topics/not-interest", json={"topic_id": "T1", "choice": "add"})
        hit("POST", f"/api/profiles/{s}/llm-topics")
        hit("GET", f"/api/profiles/{s}/scan-topics")
        hit("POST", f"/api/profiles/{s}/edit")
        hit("POST", f"/api/profiles/{s}/refit")
        hit("POST", f"/api/profiles/{s}/dry-run")
        hit("POST", f"/api/profiles/{s}/gather-now?days=1&limit=10")
        hit("POST", f"/api/profiles/{s}/backfill", json={"year_from": 2020, "year_to": 2021})
        hit("GET", f"/api/profiles/{s}/saved")
        hit("GET", f"/api/profiles/{s}/feedback")
        hit("GET", f"/api/profiles/{s}/runs")
        hit("GET", f"/api/profiles/{s}/reranker-comparison")
        hit("GET", f"/api/profiles/{s}/topic-yield")
        hit("PATCH", f"/api/profiles/{s}/schedule", json={"enabled": True})
        hit("POST", f"/api/profiles/{s}/seeds", json={"openalex_ids": [world["seed"], world["upload"]]})
        hit("DELETE", f"/api/profiles/{s}/seeds?openalex_id={world['seed']}")
        # draft-only routes
        hit("POST", f"/api/profiles/draft/{s}/coherence")
        hit("GET", f"/api/profiles/draft/{s}/topics")
        hit("POST", f"/api/profiles/draft/{s}/dry-run", json={"days": 7})
        hit("GET", f"/api/profiles/draft/{s}/dry-run/{run_id}")
        hit("DELETE", f"/api/profiles/draft/{s}")
        hit("DELETE", f"/api/profiles/draft/{s}/seeds/{world['seed']}")
        hit("POST", f"/api/profiles/draft/{s}/seeds", json={"openalex_ids": [world["seed"]]})
        hit("POST", "/api/profiles", json={"slug": s, "threshold": 0.7, "selected_topic_ids": []})
        # One tab at a time on a draft (W.L): B neither takes nor frees A's.
        hit("POST", f"/api/profiles/draft/{s}/hold")
        hit("POST", f"/api/profiles/draft/{s}/release")
        # Edit interest's copy routes
        hit("POST", f"/api/profiles/draft/{s}/edit-alive", json={"tab": "t"})
        hit("GET", f"/api/profiles/draft/{s}/edit-state")
        hit("GET", f"/api/profiles/draft/{s}/feed-papers")
        hit("POST", f"/api/profiles/draft/{s}/save-edit")
        hit("POST", f"/api/profiles/draft/{s}/discard-edit", json={"tab": "t"})
        hit("DELETE", f"/api/profiles/{s}")

    # ---- radar cards ----------------------------------------------------------
    for card in world["cards"][:2]:
        hit("POST", "/api/radar/cards/save", json={"card_id": card, "profile": world["live"], "on": True})
        hit("POST", "/api/radar/cards/save", json={"card_id": card})
        hit("POST", "/api/radar/cards/dismiss", json={"card_id": card, "profile": world["live"], "on": True})
        hit("POST", "/api/radar/cards/dismiss", json={"card_id": card, "on": True})
    r = hit("GET", f"/api/radar/daily?profile={world['live']}", expect=(200,))
    assert r.json()["cards"] == [] and r.json()["states"] == {}
    r = hit("GET", "/api/radar/daily", expect=(200,))
    assert r.json()["cards"] == []

    # ---- vault ------------------------------------------------------------
    r = hit("GET", "/api/vault/docs", expect=(200,)); assert r.json() == []
    r = hit("GET", f"/api/vault/docs?tag={world['live']}", expect=(200,)); assert r.json() == []
    r = hit("GET", "/api/vault/tags", expect=(200,)); assert r.json() == {"all": 0}
    r = hit("GET", "/api/vault/stats", expect=(200,)); assert r.json()["docs"] == 0
    r = hit("GET", "/api/vault/meta", expect=(200,)); assert world["upload"] not in r.text
    hit("DELETE", f"/api/vault/docs/{world['upload']}")
    # Upload "into" A's interest: refused (404), nothing stored for B either.
    files = {"file": ("bob.pdf", io.BytesIO(b"%PDF-1.4\nbob-bytes"), "application/pdf")}
    r = hit("POST", "/api/vault/upload", expect=(404,), files=files, data={"profile_slug": world["live"]})
    assert "no interest" in r.json()["detail"]
    r = hit("GET", "/api/vault/stats", expect=(200,)); assert r.json()["docs"] == 0
    # B's own draft cannot take A's papers as seeds.
    r = hit("POST", "/api/profiles/draft", expect=(200,), json={"name": "B draft"})
    b_slug = r.json()["slug"]
    r = hit("POST", f"/api/profiles/draft/{b_slug}/seeds", expect=(200,),
            json={"openalex_ids": [world["seed"], world["upload"], world["cards"][0]]})
    assert r.json()["attached"] == 0
    assert set(r.json()["rejected"]) == {world["seed"], world["upload"], world["cards"][0]}

    # ---- chat history is per user ---------------------------------------
    r = hit("GET", "/api/chat/history", expect=(200,)); assert r.json() == []
    hit("DELETE", "/api/chat/history", expect=(200, 204))
    hit("GET", "/api/chat/providers", expect=(200,))

    # ---- last, as it gives B papers of B's own ------------------------------
    # B importing the same person makes B's own profile (and is then told of that one), never refreshes A's.
    r = hit("POST", "/api/researchers/import", expect=(200,),
            json={"source": "prosopia", "ref": "sheffield-nathan"})
    r_b = r.json()["researcher_id"]
    assert r_b not in (rid, orcid_rid)
    r = hit("GET", "/api/import/prosopia/works?ref=sheffield-nathan", expect=(200,))
    assert r.json()["saved_profile"]["id"] == r_b, r.text

    # ---- every route was hit or is listed as reading no user data ------------
    assert set(NO_USER_DATA) | set(NOT_TESTED) <= ops, "a listed route does not exist"
    missing = ops - served - set(NO_USER_DATA) - set(NOT_TESTED)
    assert not missing, "routes B never tried: " + ", ".join(sorted(f"{m} {p}" for m, p in missing))

    assert not bad, "\n".join(f"{m} {p} -> {c}: {t}" for m, p, c, t in bad)
    assert not world["fake"].jobs, "B scheduled a job on A's interest"

    # ---- A's state is untouched (B's own additions are excluded) ----------
    after = _snapshot(env)
    conn = connect(env)
    try:
        b_uid = int(users_repo.get_by_email(conn, "bob@example.com")["id"])
        b_pids = {int(r[0]) for r in conn.execute("SELECT id FROM profiles WHERE user_id = ?", (b_uid,))}
        b_rids = {int(r[0]) for r in conn.execute("SELECT id FROM researchers WHERE user_id = ?", (b_uid,))}
    finally:
        conn.close()
    after["profiles"] = [r for r in after["profiles"] if r[1] != b_uid]
    after["researchers"] = [r for r in after["researchers"] if r[1] != b_uid]
    after["rpapers"] = [r for r in after["rpapers"] if r[0] not in b_rids]
    after["uploads"] = [r for r in after["uploads"] if r[0] != b_uid]
    after["seeds"] = [r for r in after["seeds"] if r[0] not in b_pids]
    after["sched"] = [r for r in after["sched"] if r[0] not in b_pids]
    after["kept"] = [r for r in after["kept"] if r[0] != b_uid]
    after["deleted"] = [r for r in after["deleted"] if r[0] not in b_pids]
    after["owned"] = [r for r in after["owned"] if r[0] not in b_pids]
    after["versions"] = [r for r in after["versions"] if r[1] != b_uid]
    # Papers and their vectors are shared: B may add public ones; none of
    # A's may go (CR.1).
    for shared in ("papers", "vectors"):
        assert set(before[shared]) <= set(after.pop(shared)), shared
        before.pop(shared)
    assert after == before
    assert _request(app, "GET", "/api/profiles", headers=A).json() == a_profiles_before
    assert _request(app, "GET", "/api/researchers", headers=A).json() == a_researchers_before


def test_user_b_sees_only_own_import_runs_and_drafts(world):
    """Reads that list by user, with A's rows present."""
    app = world["app"]
    assert _request(app, "GET", "/api/profiles/drafts", headers=B).json() == []
    assert _request(app, "GET", "/api/profiles", headers=B).json() == []
    assert _request(app, "GET", "/api/researchers", headers=B).json() == []
    # And A still sees everything.
    assert {d["slug"] for d in _request(app, "GET", "/api/profiles/drafts", headers=A).json()} == {world["draft"], world["plain"]}
    assert len(_request(app, "GET", "/api/researchers", headers=A).json()) == 1
