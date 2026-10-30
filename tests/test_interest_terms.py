"""Research topics generated the Prosopia way (rag_lib/interest_terms.py).

No network and no real key: a fake embedding backend stands in for MiniLM
and a fake LLMClient for Anthropic. The key tests use a unique string and
look for it everywhere it must not be.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os

import httpx
import numpy as np
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import papers as papers_repo, profiles as profiles_repo, users as users_repo
from tests.edit_helpers import edit_saved

SLUG = "chromatin"
KEY = "sk-ant-TEST-" + "q7Zp" * 6          # unique enough to grep for


class FakeBackend:
    name = "fake:16"
    dim = 16

    def embed(self, texts):
        out = []
        for t in texts:
            h = hashlib.sha256((t or "").encode()).digest()
            v = np.frombuffer(h[: self.dim], dtype=np.uint8).astype(np.float32) - 127.0
            out.append((v / (np.linalg.norm(v) or 1.0)).tolist())
        return out


@pytest.fixture()
def app(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    monkeypatch.setenv("RADAR_DB_PATH", str(db))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "false")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings_module.get_settings.cache_clear()

    conn = connect(db)
    apply_migrations(conn)
    uid = int(users_repo.upsert(conn, "demo@example.com")["id"])
    pid = profiles_repo.upsert(conn, user_id=uid, name="Chromatin", embedding_model="placeholder-v1",
                               n_seed=6, topic_filters={"topics": []}, threshold=0.8)
    for i in range(6):
        oa = f"https://openalex.org/W90{i}"
        papers_repo.upsert(conn, {"openalex_id": oa, "title": f"Seed {i}: chromatin accessibility study {i}",
                                  "abstract": f"We profile chromatin accessibility with ATAC-seq in cohort {i}. " * 3,
                                  "year": 2024, "source": "user_csv"})
        profiles_repo.attach_seed(conn, pid, oa)
    conn.close()

    # MiniLM stands in as a fake: the SDK builds its index with whatever
    # backend get_backend hands back.
    import researcher_profiles.embeddings.backends as be
    monkeypatch.setattr(be, "get_backend", lambda spec=None: FakeBackend())

    from rag_lib.api.app import create_app
    yield create_app()
    settings_module.get_settings.cache_clear()


def _request(app, method, path, **kwargs) -> httpx.Response:
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)
    return asyncio.run(_run())


def _edit(app, **kw):
    """Edit interest on the saved interest: its research topics change on
    this copy and are saved with it (FUNCTIONAL_SPEC E)."""
    return edit_saved(lambda method, path, **k: _request(app, method, path, **k), SLUG, **kw)


def _stored(tmp_path, slug=SLUG):
    conn = connect(tmp_path / "radar.db")
    try:
        raw = conn.execute("SELECT interest_terms_json FROM profiles WHERE slug = ?", (slug,)).fetchone()[0]
    finally:
        conn.close()
    return json.loads(raw) if raw else None


def _assert_key_nowhere(tmp_path, *texts):
    """The key is in no byte of the database, the SDK's profile files,
    captured output, the response, or the process environment."""
    needle = KEY.encode()
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert needle not in p.read_bytes(), f"key found in {p}"
    for t in texts:
        assert KEY not in t
    assert all(KEY not in v for v in os.environ.values())


class FakeLLM:
    seen_keys: list = []

    def __init__(self, api_key=None, **kw):
        FakeLLM.seen_keys.append(api_key)

    def complete(self, system, messages, **kw):
        class R:
            text = json.dumps([{"label": "chromatin accessibility profiling", "weight": 0.7},
                               {"label": "ATAC-seq cohort analysis", "weight": 0.3}])
            model = "claude-test"
            usage = {"input_tokens": 120, "output_tokens": 40}
        return R()


def test_without_a_key_the_sdk_cluster_method_runs_and_topics_name_their_papers(app, tmp_path):
    with _edit(app) as copy:
        resp = _request(app, "POST", f"/api/profiles/{copy}/terms/generate", json={})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["source"] == "rp-sdk-cluster" and body["terms"]
        # The SDK collects paper ids only from summary chunks; a lite profile's
        # chunks are abstracts, so without the chunk mapping every list was empty.
        assert all(t["paper_ids"] for t in body["terms"])
        assert all(p.startswith("https://openalex.org/W90") for t in body["terms"] for p in t["paper_ids"])
        # A preview: nothing is saved until the user keeps it.
        assert body["saved"] is False and _stored(tmp_path, copy) is None
        kept = _request(app, "PUT", f"/api/profiles/{copy}/terms", json={**body, "edited": False})
        assert kept.status_code == 200 and kept.json()["saved"] is True
        assert _stored(tmp_path, copy)["source"] == "rp-sdk-cluster"
        # Kept on the copy; the interest gets them when the copy is saved.
        assert _stored(tmp_path) is None
    assert _stored(tmp_path)["source"] == "rp-sdk-cluster" and _stored(tmp_path)["edited"] is False
    root = tmp_path / "rp_profiles"
    assert any(root.rglob("profile.jsonld")) and any(root.rglob("papers.jsonld"))


def test_the_llm_method_uses_the_key_typed_in_and_stores_it_nowhere(app, tmp_path, monkeypatch, caplog, capsys):
    import researcher_profiles.generative.llm as llm
    FakeLLM.seen_keys = []
    monkeypatch.setattr(llm, "LLMClient", FakeLLM)
    caplog.set_level(logging.DEBUG)

    resp = _request(app, "POST", f"/api/profiles/{SLUG}/terms/generate", json={"api_key": KEY})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "rp-sdk-llm" and body["model"] == "claude-test"
    assert [t["label"] for t in body["terms"]] == ["chromatin accessibility profiling", "ATAC-seq cohort analysis"]
    assert all(t["paper_ids"] for t in body["terms"])
    assert FakeLLM.seen_keys == [KEY]                     # passed explicitly, not via the environment
    out = capsys.readouterr()
    _assert_key_nowhere(tmp_path, resp.text, caplog.text, out.out, out.err)


def test_a_rejected_key_is_401_stores_nothing_and_is_not_echoed(app, tmp_path, monkeypatch, caplog, capsys):
    import anthropic
    import researcher_profiles.generative.llm as llm

    class Rejecting(FakeLLM):
        def complete(self, *a, **kw):
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages", headers={"x-api-key": KEY})
            raise anthropic.AuthenticationError("invalid x-api-key", response=httpx.Response(401, request=req), body=None)

    monkeypatch.setattr(llm, "LLMClient", Rejecting)
    caplog.set_level(logging.DEBUG)
    resp = _request(app, "POST", f"/api/profiles/{SLUG}/terms/generate", json={"api_key": KEY})
    assert resp.status_code == 401
    assert _stored(tmp_path) is None
    out = capsys.readouterr()
    _assert_key_nowhere(tmp_path, resp.text, caplog.text, out.out, out.err)


def test_a_malformed_request_does_not_echo_the_key(app):
    resp = _request(app, "POST", f"/api/profiles/{SLUG}/terms/generate", json={"api_key": [KEY]})
    assert resp.status_code == 422
    assert KEY not in resp.text


def test_a_prosopia_backed_interest_uses_the_profiles_own_fields(app, tmp_path, monkeypatch):
    from rag_lib import interest_terms

    conn = connect(tmp_path / "radar.db")
    conn.execute("INSERT INTO researchers (user_id, source, key, base_url, name) VALUES (1, 'prosopia', 'sheffield-nathan', 'https://prosopia.example', 'N')")
    rid = conn.execute("SELECT id FROM researchers").fetchone()[0]
    conn.execute("UPDATE profiles SET researcher_id = ? WHERE slug = ?", (rid, SLUG))
    conn.commit()
    conn.close()

    class Resp:
        status_code = 200

        def json(self):
            return {"fields": {"research_interests": {"top": [{"label": "Genomics and Chromatin Dynamics"}]},
                               "expertise": ["atac-seq-pipelines"], "subfields": ["regulatory genomics"],
                               "not_interests": {"top": ["phylogenomics"]}}}
    seen = []
    monkeypatch.setattr(interest_terms.requests, "get", lambda url, timeout=0: seen.append(url) or Resp())

    body = _request(app, "POST", f"/api/profiles/{SLUG}/terms/generate", json={}).json()
    assert seen == ["https://prosopia.example/api/v1/profiles/sheffield-nathan"]
    assert body["source"] == "prosopia"
    assert [(t["label"], t["kind"]) for t in body["terms"]] == [
        ("Genomics and Chromatin Dynamics", "research_interest"), ("atac seq pipelines", "expertise"),
        ("regulatory genomics", "subfield")]
    assert body["not_interests"] == ["phylogenomics"]


def test_edits_are_kept_and_a_seed_change_marks_the_topics_stale(app, tmp_path):
    with _edit(app) as copy:
        preview = _request(app, "POST", f"/api/profiles/{copy}/terms/generate", json={}).json()
        resp = _request(app, "PUT", f"/api/profiles/{copy}/terms",
                        json={**preview, "terms": [{"label": "my own topic"}], "not_interests": ["wet lab"], "edited": True})
        assert resp.status_code == 200
        assert resp.json()["edited"] is True and resp.json()["stale"] is False
    assert [t["label"] for t in _stored(tmp_path)["terms"]] == ["my own topic"]
    assert _request(app, "GET", f"/api/profiles/{SLUG}/terms").json()["stale"] is False

    conn = connect(tmp_path / "radar.db")
    conn.execute("DELETE FROM profile_seeds WHERE openalex_id = 'https://openalex.org/W905'")
    conn.commit()
    conn.close()
    assert _request(app, "GET", f"/api/profiles/{SLUG}/terms").json()["stale"] is True


def test_terms_mode_reranks_with_the_topics_and_falls_back_to_topic_names():
    from rag_lib.profile import Profile
    from rag_lib.rerankers.medcpt import MedCPTReranker

    r = MedCPTReranker(query_mode="terms")
    with_terms = Profile(name="x", interest_terms=["chromatin accessibility profiling"],
                         topic_filters={"topics": [{"id": "T1", "display_name": "Genomics", "on": True}]})
    assert r._load_queries(with_terms) == ["chromatin accessibility profiling"]
    without = Profile(name="x", topic_filters={"topics": [{"id": "T1", "display_name": "Genomics", "on": True}]})
    assert r._load_queries(without) == ["Genomics"]


def test_struck_out_topics_are_kept_apart_and_never_used_as_queries(app, tmp_path):
    """Removing a topic used to drop it; it is now kept, struck out, so the
    user can put it back. Only the kept topics reach the reranker."""
    from rag_lib import interest_terms
    from rag_lib.db.repos import profiles as profiles_repo

    # On the Edit interest copy, saved each time (FUNCTIONAL_SPEC E).
    with _edit(app) as copy:
        preview = _request(app, "POST", f"/api/profiles/{copy}/terms/generate", json={}).json()
        first, *rest = preview["terms"]
        body = _request(app, "PUT", f"/api/profiles/{copy}/terms",
                        json={**preview, "terms": rest, "removed": [first]}).json()
        assert [t["label"] for t in body["removed"]] == [first["label"]]
        assert first["label"] not in [t["label"] for t in body["terms"]]

    conn = connect(tmp_path / "radar.db")
    row = profiles_repo.get_by_slug(conn, 1, SLUG)
    assert [t["label"] for t in interest_terms.load(row)["removed"]] == [first["label"]]
    assert first["label"] not in interest_terms.query_labels(interest_terms.load(row))
    conn.close()

    # Put back: it leaves the struck-out list.
    with _edit(app) as copy:
        body = _request(app, "PUT", f"/api/profiles/{copy}/terms",
                        json={"terms": [first, *rest], "removed": [first]}).json()
        assert body["removed"] == [] and first["label"] in [t["label"] for t in body["terms"]]
    assert _stored(tmp_path)["removed"] == [] and first["label"] in [t["label"] for t in _stored(tmp_path)["terms"]]
