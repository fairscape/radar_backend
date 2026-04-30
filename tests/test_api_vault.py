"""Vault API — Phase 6.

Stand up the FastAPI app against a tmp DB + tmp vault dir, patch
``ingest_pdf`` and ``OpenAlexClient`` so tests never hit pdfplumber or
the network, then exercise:
  - upload (happy path)
  - upload idempotency (same bytes from same user)
  - upload tagged to a profile (shows up in profile-filtered list_docs)
  - list_docs returns user-uploaded docs only (gather candidates excluded)
  - list_docs ?tag= filters
  - stats / meta / tags
"""

from __future__ import annotations

import asyncio
import io
import sqlite3

import httpx
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.api.services import vault as vault_service
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import (
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)
from rag_lib.vault import PdfIngestRecord


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _seed_db(db_path) -> dict:
    """Empty DB + a profile so upload?profile_slug=... has somewhere to attach."""
    conn = connect(db_path)
    apply_migrations(conn)
    user = users_repo.upsert(conn, "demo@example.com")
    user_id = int(user["id"])
    pid = profiles_repo.upsert(
        conn,
        user_id=user_id,
        name="neonatal_vitals",
        embedding_model="placeholder-v1",
        n_seed=0,
        topic_filters={},
    )
    # An unrelated paper coming in via the gather path — must NOT show up
    # in the vault listing for this user (uploaded_by_user_id is NULL).
    papers_repo.upsert(conn, {
        "openalex_id": "W_GATHER_1",
        "title": "Gathered candidate",
        "source": "openalex_gatherer",
    })
    conn.close()
    return {"user_id": user_id, "profile_id": pid}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    monkeypatch.setenv("RADAR_DB_PATH", str(db))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "false")
    settings_module.get_settings.cache_clear()

    _seed_db(db)

    # Replace ingest_pdf with a deterministic record so tests don't need
    # pdfplumber. The hash-derived stem distinguishes uploads.
    def _fake_ingest_pdf(path):
        from pathlib import Path
        p = Path(path)
        return PdfIngestRecord(
            path=str(p),
            title=f"Title for {p.stem[:8]}",
            body_text="hello world from a fake pdf",
            doi=None,
            n_pages=3,
        )

    monkeypatch.setattr(vault_service, "ingest_pdf", _fake_ingest_pdf)

    # And replace OpenAlexClient so the enrichment path is offline.
    class _NoNetClient:
        def __init__(self, **_): pass
        def lookup_by_doi(self, doi): return None
        def lookup_by_title(self, *a, **kw): return None
        def paper_from_work(self, *a, **kw): return None
    monkeypatch.setattr(vault_service, "OpenAlexClient", _NoNetClient)

    from rag_lib.api.app import create_app
    app = create_app()
    yield app

    settings_module.get_settings.cache_clear()


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                return await c.request(method, path, **kwargs)
    return asyncio.run(_run())


def _upload(app, content: bytes, filename: str = "paper.pdf",
            profile_slug: str | None = None) -> httpx.Response:
    files = {"file": (filename, io.BytesIO(content), "application/pdf")}
    data = {"profile_slug": profile_slug} if profile_slug else {}
    return _request(app, "POST", "/api/vault/upload", files=files, data=data)


# -----------------------------------------------------------------------------
# Upload
# -----------------------------------------------------------------------------


def test_upload_creates_vault_doc(app):
    resp = _upload(app, b"hello-bytes-A", filename="alpha.pdf")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["title"]
    assert body["pages"] == 3
    assert body["chunks"] == 0
    assert body["tags"] == []  # no profile_slug supplied


def test_upload_is_idempotent_for_same_bytes(app):
    a = _upload(app, b"identical-bytes", filename="x.pdf").json()
    b = _upload(app, b"identical-bytes", filename="x-renamed.pdf").json()
    assert a["id"] == b["id"]

    docs = _request(app, "GET", "/api/vault/docs").json()
    assert sum(1 for d in docs if d["id"] == a["id"]) == 1


def test_upload_tagged_to_profile_appears_under_tag(app):
    up = _upload(app, b"unique-bytes", profile_slug="neonatal-vitals").json()
    assert "neonatal-vitals" in up["tags"]

    docs = _request(app, "GET", "/api/vault/docs",
                    params={"tag": "neonatal-vitals"}).json()
    assert any(d["id"] == up["id"] for d in docs)


def test_upload_rejects_empty_file(app):
    resp = _upload(app, b"", filename="empty.pdf")
    assert resp.status_code == 400


# -----------------------------------------------------------------------------
# List / stats / meta / tags
# -----------------------------------------------------------------------------


def test_list_docs_excludes_gather_sourced_papers(app):
    """The gathered paper in the seed fixture must not leak into the vault."""
    _upload(app, b"vault-bytes-1")
    docs = _request(app, "GET", "/api/vault/docs").json()
    assert len(docs) == 1
    assert all(d["id"] != "W_GATHER_1" for d in docs)


def test_list_docs_tag_all_returns_everything(app):
    _upload(app, b"a")
    _upload(app, b"b")
    docs = _request(app, "GET", "/api/vault/docs", params={"tag": "all"}).json()
    assert len(docs) == 2


def test_list_docs_unknown_tag_returns_empty(app):
    _upload(app, b"some-bytes")
    docs = _request(app, "GET", "/api/vault/docs",
                    params={"tag": "no-such-profile"}).json()
    assert docs == []


def test_stats_reflects_uploads(app):
    _upload(app, b"stats-1")
    _upload(app, b"stats-2")
    stats = _request(app, "GET", "/api/vault/stats").json()
    assert stats["docs"] == 2
    assert stats["pages"] == 6  # n_pages=3 each
    assert stats["chunks"] == 0
    assert stats["lastIngest"]


def test_meta_contains_paths_and_chunk_policy(app):
    meta = _request(app, "GET", "/api/vault/meta").json()
    assert "rootPath" in meta and meta["rootPath"]
    assert "indexPath" in meta and meta["indexPath"]
    assert meta["chunkSize"] == "500 tok + 50 overlap"


def test_tags_endpoint_includes_all_total(app):
    _upload(app, b"u1", profile_slug="neonatal-vitals")
    _upload(app, b"u2")  # untagged
    counts = _request(app, "GET", "/api/vault/tags").json()
    # multi-tagged docs would only count once under "all"
    assert counts["all"] == 2
    assert counts.get("neonatal-vitals") == 1
