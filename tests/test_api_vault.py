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
        def search_by_title(self, *a, **kw): return []
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
    # Uploads must carry a PDF header now; the fake ingest ignores the rest.
    if content and not content.startswith(b"%PDF-"):
        content = b"%PDF-1.4\n" + content
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


# -----------------------------------------------------------------------------
# Rejected uploads, per-user uploads, title matching (review round 3)
# -----------------------------------------------------------------------------


def _raw_upload(app, content: bytes, filename: str = "paper.pdf", headers=None):
    files = {"file": (filename, io.BytesIO(content), "application/pdf")}
    return _request(app, "POST", "/api/vault/upload", files=files, headers=headers or {})


def test_non_pdf_upload_is_rejected_and_not_kept(app, tmp_path):
    resp = _raw_upload(app, b"just some notes\n", filename="notes.txt")
    assert resp.status_code == 415, resp.text
    assert "not a PDF" in resp.json()["detail"]
    assert not any((tmp_path / "vault").rglob("*.pdf"))
    assert _request(app, "GET", "/api/vault/docs").json() == []


def test_unreadable_pdf_is_rejected_and_not_kept(app, tmp_path, monkeypatch):
    def _broken(path):
        raise ValueError("No /Root object! - Is this really a PDF?")
    monkeypatch.setattr(vault_service, "ingest_pdf", _broken)
    resp = _raw_upload(app, b"%PDF-1.4\n\x00garbage")
    assert resp.status_code == 422, resp.text
    assert "could not be read" in resp.json()["detail"]
    assert not [p for p in (tmp_path / "vault").rglob("*") if p.is_file()]


def test_upload_size_limit(app, monkeypatch):
    monkeypatch.setenv("RADAR_UPLOAD_MAX_MB", "1")
    settings_module.get_settings.cache_clear()
    resp = _raw_upload(app, b"%PDF-1.4\n" + b"x" * (1024 * 1024 + 10))
    assert resp.status_code == 413, resp.text


def test_reupload_says_duplicate(app):
    a = _upload(app, b"dup-bytes").json()
    b = _upload(app, b"dup-bytes").json()
    assert not a.get("duplicate")
    assert b["duplicate"] is True


def test_two_users_uploading_one_paper_both_keep_it(app, monkeypatch):
    # Both PDFs resolve to the same OpenAlex work.
    work = {"id": "https://openalex.org/W42", "title": "Shared paper"}

    class _Client:
        def __init__(self, **_): pass
        def lookup_by_doi(self, doi): return work
        def search_by_title(self, *a, **kw): return []
        def paper_from_work(self, w, source=None):
            from rag_lib.paper import Paper
            return Paper(doi="10.1/x", openalex_id="https://openalex.org/W42", title="Shared paper")
    monkeypatch.setattr(vault_service, "OpenAlexClient", _Client)

    texts = iter(["first user's text for Shared paper", "second user's text for Shared paper"])

    def _ingest(path):
        return PdfIngestRecord(path=str(path), title="Shared paper",
                               body_text=next(texts), doi="10.1/x", n_pages=2)
    monkeypatch.setattr(vault_service, "ingest_pdf", _ingest)

    a = _upload(app, b"copy-one")
    assert a.status_code == 200, a.text
    b_hdr = {"X-User-Email": "other@example.com"}
    b = _raw_upload(app, b"%PDF-1.4\ncopy-two", headers=b_hdr)
    assert b.status_code == 200, b.text

    mine = [d["id"] for d in _request(app, "GET", "/api/vault/docs").json()]
    theirs = [d["id"] for d in _request(app, "GET", "/api/vault/docs", headers=b_hdr).json()]
    assert "https://openalex.org/W42" in mine
    assert "https://openalex.org/W42" in theirs
    # The shared text is the first upload's; the second did not rewrite it.
    import os
    conn = connect(os.environ["RADAR_DB_PATH"])
    body = conn.execute("SELECT body_text FROM papers WHERE openalex_id='https://openalex.org/W42'").fetchone()[0]
    n = conn.execute("SELECT count(*) FROM paper_uploads WHERE openalex_id='https://openalex.org/W42'").fetchone()[0]
    conn.close()
    assert body.startswith("first user's")
    assert n == 2


def _rec(title, body, doi=None):
    return PdfIngestRecord(path="x.pdf", title=title, body_text=body, doi=doi, n_pages=1)


def _work(title, *authors, doi=None):
    return {"title": title, "doi": doi,
            "authorships": [{"author": {"display_name": a}} for a in authors]}


_GOODFELLOW = _work("Deep Learning", "Ian Goodfellow", "Yoshua Bengio", "Aaron Courville")
_BOOK_TEXT = "Deep Learning\nIan Goodfellow, Yoshua Bengio and Aaron Courville\nMIT Press"


def test_the_lab_note_is_not_the_book():
    notes = _rec("Deep learning", "Lab notes, week 3. Tried a bigger batch size.")
    assert vault_service.title_rule(notes, _GOODFELLOW)          # same title...
    assert not vault_service.author_rule(notes, _GOODFELLOW)     # ...nobody's name
    assert vault_service._verified_title_match(notes, [_GOODFELLOW]) is None
    book = _rec("Deep Learning", _BOOK_TEXT)
    assert vault_service._verified_title_match(book, [_GOODFELLOW]) is _GOODFELLOW


def test_title_rule_is_exact_after_normalizing():
    w = _work("Chromatin accessibility: a review", "A Smith")
    assert vault_service.title_rule(_rec("CHROMATIN ACCESSIBILITY — A REVIEW", ""), w)
    assert vault_service.title_rule(_rec("Chromatin accessibility", ""), w)          # no subtitle
    assert not vault_service.title_rule(_rec("Chromatin accessibility in mice", ""), w)
    assert vault_service.title_rule(_rec("Résumé of García's work", ""), _work("Resume of Garcia's work"))
    # A mangled extracted title, but the real one is printed on page one.
    assert vault_service.title_rule(_rec("bioRxiv preprint", "Chromatin accessibility: a review\nA. Smith"), w)


def test_author_rule_needs_the_first_author_and_one_more():
    w = _work("T", "Jane Doe", "Wei Zhang", "Ana Pérez")
    assert vault_service.author_rule(_rec("T", "Jane Doe1, Ana Perez2"), w)
    assert not vault_service.author_rule(_rec("T", "Jane Doe only"), w)            # one of three
    assert not vault_service.author_rule(_rec("T", "Wei Zhang and Ana Perez"), w)  # first author missing
    assert vault_service.author_rule(_rec("T", "Jane Doe only"), w, first_only=True)
    assert not vault_service.author_rule(_rec("T", "John Doerr wrote this"), _work("T", "Jane Doe"))
    assert not vault_service.author_rule(_rec("T", ""), w)                         # nothing to check


def test_author_rule_reads_bylines_as_pdfs_print_them():
    # Spaces lost, affiliation marks glued on (seen in the live vault).
    assert vault_service.author_rule(
        _rec("T", "AlexMallen1,∗MadelineBrumley2,JuliaKharchenko2"),
        _work("T", "Alex Mallen", "Brumley, Madeline"))
    assert vault_service.author_rule(
        _rec("T", "KathleenC.Frasera,JedA.Meltzerb andFrankRudzicza,c"),
        _work("T", "Kathleen Fraser", "Jed A. Meltzer"))
    # "Last, First" from OpenAlex, with a particle.
    assert vault_service.author_rule(
        _rec("T", "ChuhanZhang⋆ GuillaumeLeMoing⋆"), _work("T", "Zhang, Chuhan", "Moing, Guillaume Le"))
    # Initials only.
    assert vault_service.author_rule(_rec("T", "K. C. Fraser, J. A. Meltzer"),
                                     _work("T", "Kathleen Fraser", "Jed Meltzer"))
    # A group author, its members listed at the end.
    assert vault_service.author_rule(_rec("T", "DeepSeek-AI\nresearch@deepseek.com"),
                                     _work("T", "DeepSeek-AI", "Xu, Anyi"))


def test_a_preprint_doi_does_not_rule_out_the_published_work():
    w = _work("AllTracker: Efficient Dense Point Tracking", "Adam W. Harley", "Yang You",
              doi="https://doi.org/10.1109/iccv.2025.1")
    rec = _rec("AllTracker: Efficient Dense Point Tracking", "AdamW.Harley1 YangYou1",
               doi="10.48550/arXiv.2506.07310")
    assert vault_service._verified_title_match(rec, [w]) is w


def test_a_doi_from_the_references_is_not_the_paper():
    cited = _work("Some cited method", "Bob Ray", "Eve Long")
    body = "My own paper\nJane Doe\n" + "Body text. " * 800 + "References: Ray B, Long E. Some cited method."
    mine = _rec("My own paper", body)
    assert not vault_service._verified_doi_match(mine, cited)
    assert vault_service._verified_doi_match(_rec("Some cited method", "x"), cited)
