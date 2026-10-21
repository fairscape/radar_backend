"""Review round 3 (reviewer b): multi-step data flows over the newest features.

Looks for 500s, orphans and wrong counts when papers move between the
feed, the seeds and the vault; when a researcher is forgotten under a
live interest and imported again; and when two users share one person.
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
from tests.test_prosopia_import import StubProsopia, _stub_profile
from tests.test_researcher_marks import _both_stubs
from tests.test_researchers import _app, _request, env  # noqa: F401 -- fixture

A = {"X-User-Email": "alice@example.com"}
B = {"X-User-Email": "bob@example.com"}


@pytest.fixture()
def app(env, monkeypatch):
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
    return _app(StubProsopia(_stub_profile()), _both_stubs())


def _live(env, email, name="L") -> tuple[str, list[str]]:
    conn = connect(env)
    try:
        uid = int(users_repo.upsert(conn, email)["id"])
        rng = np.random.default_rng(3)
        base = rng.standard_normal(64).astype("float32"); base /= np.linalg.norm(base)

        def near(scale):
            v = base + scale * rng.standard_normal(64).astype("float32")
            return v / np.linalg.norm(v)

        pid = profiles_repo.upsert(
            conn, user_id=uid, name=name, embedding_model="placeholder-v1", n_seed=3,
            topic_filters={"topics": [{"id": "T1", "display_name": "T one", "count": 3, "on": True}]},
            threshold=0.5, coherence_median=0.7, coherence_iqr=0.05, coherence_bimodal=False,
        )
        for i in range(3):
            oa = f"https://openalex.org/W{9000000 + i}"
            papers_repo.upsert(conn, {"openalex_id": oa, "title": f"seed {i}", "year": 2024, "source": "user_csv"})
            embeddings_repo.upsert(conn, oa, "placeholder-v1", near(0.03))
            profiles_repo.attach_seed(conn, pid, oa)
        g = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
        cards = []
        for i in range(6):
            oa = f"https://openalex.org/W{9100000 + i}"
            papers_repo.upsert(conn, {"openalex_id": oa, "title": f"cand {i}", "year": 2026,
                                      "source": "openalex_gatherer", "abstract": "x"})
            # the last two sit below the 0.5 threshold (and stay there after a refit)
            embeddings_repo.upsert(conn, oa, "placeholder-v1", near(0.03) if i < 4 else -near(0.03))
            s = 0.9 - i * 0.1
            candidates_repo.insert_dedup(conn, profile_id=pid, openalex_id=oa, score=s, score_raw=s, gather_run_id=g)
            cards.append(oa)
        gather_runs_repo.finish(conn, g, n_fetched=6, n_new=6, n_redup=0)
        return profiles_repo.get(conn, pid)["slug"], cards
    finally:
        conn.close()


def _ok(r, *codes):
    assert r.status_code in (codes or (200,)), (r.request.method, r.request.url, r.status_code, r.text[:300])
    return r


def test_feed_paper_becomes_seed_and_back(app, env):
    slug, cards = _live(env, "alice@example.com")
    c1 = cards[0]
    _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": c1, "profile": slug, "on": True}, headers=A))
    feed = _ok(_request(app, "GET", f"/api/radar/daily?profile={slug}", headers=A)).json()
    assert feed["states"][c1] == "saved"
    # The feed paper can be a seed (owned through its candidate row).
    r = _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [c1]}, headers=A)).json()
    assert r == {"attached": 1, "rejected": [], "rescored": r["rescored"]}
    feed = _ok(_request(app, "GET", f"/api/radar/daily?profile={slug}", headers=A)).json()
    assert c1 not in {c["id"] for c in feed["cards"]}          # a seed is not shown to itself
    detail = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail", headers=A)).json()
    assert len(detail["seeds"]) == 4 and detail["profile"]["seeds"] == 4
    vault = _ok(_request(app, "GET", f"/api/vault/docs?tag={slug}", headers=A)).json()
    assert [d["id"] for d in vault].count(c1) == 1 and [d for d in vault if d["id"] == c1][0]["tags"] == [slug]
    tags = _ok(_request(app, "GET", "/api/vault/tags", headers=A)).json()
    assert tags[slug] == 4 and tags["all"] == 4
    stats = _ok(_request(app, "GET", "/api/vault/stats", headers=A)).json()
    assert stats["docs"] == 4
    dry = _ok(_request(app, "POST", f"/api/profiles/{slug}/dry-run", headers=A)).json()
    assert c1 not in {p["id"] for p in dry["papers"]} and dry["n"] == 5
    # And back out again: the candidate row (still saved) returns to the feed.
    _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds?openalex_id={c1}", headers=A))
    feed = _ok(_request(app, "GET", f"/api/radar/daily?profile={slug}", headers=A)).json()
    assert feed["states"][c1] == "saved"
    detail = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail", headers=A)).json()
    assert len(detail["seeds"]) == 3 and detail["profile"]["seeds"] == 3
    # Removing down to the last seed is refused; the one before it is fine.
    for s in [d["id"] for d in detail["seeds"]][:2]:
        _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds?openalex_id={s}", headers=A))
    last = _request(app, "GET", f"/api/profiles/{slug}/detail", headers=A).json()["seeds"][0]["id"]
    _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds?openalex_id={last}", headers=A), 409)
    # A below-threshold card can still be saved, and then shows in the vault.
    low = cards[-1]
    _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": low, "profile": slug, "on": True}, headers=A))
    assert low in {d["id"] for d in _request(app, "GET", "/api/vault/docs", headers=A).json()}
    fb = _ok(_request(app, "GET", f"/api/profiles/{slug}/feedback", headers=A)).json()
    assert [e["openalex_id"] for e in fb] == [low, c1]


def test_patch_threshold_and_years_validation(app, env):
    slug, _ = _live(env, "alice@example.com")
    p = _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"threshold": 0.6, "years": {"year_from": 2020, "year_to": None}}, headers=A)).json()
    assert p["threshold"] == 0.6 and p["yearFrom"] == 2020 and p["yearTo"] is None
    p = _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"years": {"year_from": None, "year_to": None}}, headers=A)).json()
    assert p["yearFrom"] is None and p["yearTo"] is None and p["threshold"] == 0.6
    _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"years": {"year_from": 2030, "year_to": None}}, headers=A), 400)
    _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"years": {"year_from": 2020, "year_to": 2010}}, headers=A), 400)
    _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"threshold": 1.5}, headers=A), 422)
    _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"threshold": "nan"}, headers=A), 422)
    _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={}, headers=A), 400)
    _ok(_request(app, "POST", f"/api/profiles/{slug}/backfill", json={"year_from": 1800}, headers=A), 400, 503)
    _ok(_request(app, "POST", f"/api/profiles/{slug}/backfill", json={"year_to": 2020}, headers=A), 400, 503)


def test_forget_under_a_live_interest_then_import_again(app, env):
    r = _ok(_request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"}, headers=A)).json()
    rid = r["researcher_id"]
    d = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "Nathan"}, headers=A)).json()
    slug = d["draft"]["slug"]
    topics = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics", headers=A)).json()
    _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/coherence", headers=A))
    live = _ok(_request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.5, "selected_topic_ids": [t["id"] for t in topics]}, headers=A)).json()
    assert live["isDraft"] is False and live["researcherId"] == rid and live["seeds"] == 5
    assert _request(app, "GET", f"/api/researchers/{rid}", headers=A).json()["researcher"]["n_interests"] == 1

    _ok(_request(app, "DELETE", f"/api/researchers/{rid}", headers=A))
    assert _request(app, "GET", "/api/researchers", headers=A).json() == []
    detail = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail", headers=A)).json()
    assert len(detail["seeds"]) == 5 and detail["profile"]["researcherId"] is None
    rt = _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics", headers=A)).json()
    assert rt["researcher_id"] is None and rt["applied"] is False
    _ok(_request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}, headers=A), 400)
    _ok(_request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}, headers=A))
    assert _ok(_request(app, "GET", "/api/vault/stats", headers=A)).json()["docs"] == 5    # seeds stay in the vault
    assert len(_ok(_request(app, "GET", f"/api/vault/docs?tag={slug}", headers=A)).json()) == 5
    for path in ("refit", "recompute-coherence", "recompute-topics", "dry-run"):
        _ok(_request(app, "POST", f"/api/profiles/{slug}/{path}", headers=A))
    _ok(_request(app, "GET", f"/api/profiles/{slug}/topics/preview", headers=A))
    _ok(_request(app, "GET", f"/api/profiles/{slug}/runs", headers=A))

    # Import the same person again: a new row, the old interest is not theirs.
    r2 = _ok(_request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"}, headers=A)).json()
    assert r2["researcher_id"] != rid and r2["already_running"] is False
    det = _ok(_request(app, "GET", f"/api/researchers/{r2['researcher_id']}", headers=A)).json()
    assert det["researcher"]["n_papers"] == 5 and det["researcher"]["n_interests"] == 0 and det["interests"] == []
    assert _ok(_request(app, "GET", "/api/vault/stats", headers=A)).json()["docs"] == 5
    # The old interest's seeds can be re-attached from the new researcher's papers.
    ids = [p["id"] for p in det["papers"]]
    r = _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": ids}, headers=A)).json()
    assert r["attached"] == 0 and r["rejected"] == []
    # Marks and order with the new row.
    _ok(_request(app, "PATCH", f"/api/researchers/{r2['researcher_id']}", json={"mark": "self"}, headers=A))
    _ok(_request(app, "PUT", "/api/researchers/order", json={"ids": [r2["researcher_id"]]}, headers=A))
    _ok(_request(app, "PUT", "/api/researchers/order", json={"ids": [rid]}, headers=A), 400)


def test_two_users_share_a_person(app, env):
    ra = _ok(_request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"}, headers=A)).json()
    rb = _ok(_request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"}, headers=B)).json()
    assert ra["researcher_id"] != rb["researcher_id"]
    assert rb["already_running"] is False
    da = _ok(_request(app, "POST", f"/api/researchers/{ra['researcher_id']}/interests", json={"name": "Nathan"}, headers=A)).json()
    db_ = _ok(_request(app, "POST", f"/api/researchers/{rb['researcher_id']}/interests", json={"name": "Nathan"}, headers=B)).json()
    assert da["draft"]["slug"] == db_["draft"]["slug"] == "nathan"       # slugs are per user
    _ok(_request(app, "PATCH", f"/api/researchers/{rb['researcher_id']}", json={"mark": "self"}, headers=B))
    _ok(_request(app, "DELETE", f"/api/researchers/{rb['researcher_id']}", headers=B))
    det = _ok(_request(app, "GET", f"/api/researchers/{ra['researcher_id']}", headers=A)).json()
    assert det["researcher"]["n_papers"] == 5 and det["researcher"]["n_interests"] == 1 and det["researcher"]["mark"] is None
    assert _ok(_request(app, "GET", "/api/vault/stats", headers=A)).json()["docs"] == 5
    assert _ok(_request(app, "GET", "/api/vault/stats", headers=B)).json()["docs"] == 5    # B's draft seeds stay
    assert len(_ok(_request(app, "GET", "/api/profiles/drafts", headers=B)).json()) == 1
    # A's draft seeds, B's draft seeds: the same paper rows, separate links.
    conn = connect(env)
    try:
        assert conn.execute("SELECT COUNT(*) FROM profile_seeds").fetchone()[0] == 10
        assert conn.execute("SELECT COUNT(*) FROM researcher_papers").fetchone()[0] == 5
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
    # Same bytes uploaded by both: two upload rows, one paper.
    for h in (A, B):
        files = {"file": ("p.pdf", io.BytesIO(b"%PDF-1.4\nshared"), "application/pdf")}
        _ok(_request(app, "POST", "/api/vault/upload", files=files, headers=h))
    conn = connect(env)
    try:
        assert conn.execute("SELECT COUNT(*) FROM paper_uploads").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM papers WHERE openalex_id LIKE 'local:%'").fetchone()[0] == 1
    finally:
        conn.close()
    assert _ok(_request(app, "GET", "/api/vault/stats", headers=B)).json()["docs"] == 6
