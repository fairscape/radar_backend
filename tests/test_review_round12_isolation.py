"""Review round 3 (reviewer b): multi-user isolation.

User A owns a researcher, a draft built from it, a plain draft, a live
interest with candidates, and an uploaded PDF. User B then hits every
read and mutating endpoint with A's ids/slugs. Expected: 404 (or 400/409
for list-shaped bodies), never a 2xx that touches A's data and never a 5xx.
Afterwards A's state is re-read and must be byte-for-byte what it was.
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
            "cands": q("SELECT profile_id, openalex_id, saved_at, dismissed_at FROM profile_candidates ORDER BY 1, 2"),
            "sched": q("SELECT profile_id, cron, tz, enabled FROM profile_schedules ORDER BY 1"),
            "rpapers": q("SELECT researcher_id, openalex_id FROM researcher_papers ORDER BY 1, 2"),
            "uploads": q("SELECT user_id, openalex_id FROM paper_uploads ORDER BY 1, 2"),
            "feedback": q("SELECT user_id, profile_id, openalex_id, action FROM feedback_events ORDER BY id"),
        }
    finally:
        conn.close()


def test_user_b_cannot_read_or_touch_user_a(world):
    app, env = world["app"], world["env"]
    rid, run_id = world["rid"], world["run_id"]
    before = _snapshot(env)
    a_profiles_before = _request(app, "GET", "/api/profiles", headers=A).json()
    a_researchers_before = _request(app, "GET", "/api/researchers", headers=A).json()

    bad: list[tuple[str, str, int, str]] = []

    def hit(method, path, expect=(404,), **kw):
        r = _request(app, method, path, headers=B, **kw)
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

    # ---- import runs / into A's draft ----------------------------------------
    hit("GET", f"/api/import/prosopia/{run_id}")
    hit("POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "into": world["draft"]})
    hit("POST", "/api/import/orcid",
        json={"orcid": ORCID, "openalex_ids": [WORKS[0]["id"]], "into": world["draft"]})

    # ---- every /{key} route, for a draft and a live interest of A's ---------
    for s in (world["draft"], world["live"], world["plain"]):
        r = hit("GET", f"/api/profiles/{s}", expect=(200,))
        assert r.json() is None, (s, r.text)                       # by design: null, not A's row
        r = hit("GET", f"/api/profiles/{s}/detail", expect=(200,))
        assert r.json() is None, (s, r.text)
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
        hit("POST", f"/api/profiles/{s}/refit")
        hit("POST", f"/api/profiles/{s}/dry-run")
        hit("POST", f"/api/profiles/{s}/gather-now?days=1&limit=10")
        hit("POST", f"/api/profiles/{s}/backfill", json={"year_from": 2020, "year_to": 2021})
        hit("GET", f"/api/profiles/{s}/feedback-threshold")
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

    assert not bad, "\n".join(f"{m} {p} -> {c}: {t}" for m, p, c, t in bad)
    assert not world["fake"].jobs, "B scheduled a job on A's interest"

    # ---- A's state is untouched (B's own additions are excluded) ----------
    after = _snapshot(env)
    conn = connect(env)
    try:
        b_uid = int(users_repo.get_by_email(conn, "bob@example.com")["id"])
        b_pids = {int(r[0]) for r in conn.execute("SELECT id FROM profiles WHERE user_id = ?", (b_uid,))}
    finally:
        conn.close()
    after["profiles"] = [r for r in after["profiles"] if r[1] != b_uid]
    after["uploads"] = [r for r in after["uploads"] if r[0] != b_uid]
    after["seeds"] = [r for r in after["seeds"] if r[0] not in b_pids]
    after["sched"] = [r for r in after["sched"] if r[0] not in b_pids]
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
