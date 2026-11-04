"""Uploads after the 2026-11-02 review (FUNCTIONAL_SPEC CR.2, CR.3, CR.11
L2 L3 L5 L6). Each test compares with the world before it, which already
holds the fixture's own upload and papers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from rag_lib.db import connect
from rag_lib.db.repos import gather_runs as gather_runs_repo
from tests.test_draft_sandbox import _discard, _draft, _q, _snapshot
from tests.test_private_upload_versions import W, _post, pdfs  # noqa: F401 -- fixture
from tests.test_researchers import env  # noqa: F401 -- fixture
from tests.test_review_round12_isolation import A, _request, world  # noqa: F401 -- fixture


def test_cr2_a_draft_discarded_during_its_upload_keeps_nothing_of_it(pdfs, monkeypatch):
    """Review U1: the draft is discarded (another tab) while its PDF is
    read. The upload says so and keeps nothing: no row, no file."""
    from rag_lib.api.services import vault as vault_service

    d = _draft(pdfs, "Race")
    before = _snapshot(pdfs)
    orig = vault_service._enrich_via_openalex

    def enrich(record, **kw):
        out = orig(record, **kw)
        r = _request(pdfs["app"], "DELETE", f"/api/profiles/draft/{d}", headers=A)
        assert r.status_code == 200, r.text
        return out

    monkeypatch.setattr(vault_service, "_enrich_via_openalex", enrich)
    r = _post(pdfs, b"manuscript race", A, d)
    assert r.status_code == 404, r.text
    assert r.json()["detail"] == "The draft was discarded; the PDF was not kept."
    after = _snapshot(pdfs)
    # The draft's own rows went with the discard; nothing of the upload came in.
    assert after == before
    assert not [p for p in (Path(pdfs["env"]).parent / "vault").rglob("*.part")]


def test_cr3_an_upload_that_fails_midway_leaves_nothing_and_a_retry_is_the_drafts(pdfs, monkeypatch):
    """Review U3: the database refuses partway (locked); the upload fails
    and leaves nothing; the retry is the draft's, and the discard takes it."""
    from rag_lib.db.repos import papers as papers_repo

    d = _draft(pdfs, "Fail")
    before = _snapshot(pdfs)
    orig = papers_repo.record_upload
    calls = {"n": 0}

    def flaky(conn, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return orig(conn, **kw)

    monkeypatch.setattr(papers_repo, "record_upload", flaky)
    try:
        r = _post(pdfs, b"manuscript F", A, d)
        assert r.status_code >= 500
    except sqlite3.OperationalError:
        pass
    assert _snapshot(pdfs) == before

    r = _post(pdfs, b"manuscript F", A, d)
    assert r.status_code == 200, r.text
    assert _q(pdfs, "SELECT item FROM draft_owned WHERE kind = 'paper'") == [(r.json()["id"],)]
    _discard(pdfs, d)
    after = _snapshot(pdfs)
    for t in ("papers", "paper_uploads", "paper_versions", "paper_embeddings", "files"):
        assert after[t] == before[t], t


def test_l2_a_doi_read_from_the_pdf_stays_on_the_users_version(pdfs, monkeypatch):
    """OpenAlex matched the work but has no DOI; the DOI on the PDF is the
    user's, not the public record's (M.1 rule 1)."""
    from rag_lib.api.services import vault as vault_service

    def enrich(record, *, mailto, filename, file_hash):
        return W, {"openalex_id": W, "title": "Public Title", "year": 2020, "source": "user_pdf",
                   "doi": "10.9/from-the-pdf", "openalex_doi": None}

    monkeypatch.setattr(vault_service, "_enrich_via_openalex", enrich)
    d = _draft(pdfs, "Doi")
    r = _post(pdfs, b"matched doi", A, d)
    assert r.status_code == 200, r.text
    assert _q(pdfs, "SELECT doi FROM papers WHERE openalex_id = ?", W) == [(None,)]
    assert _q(pdfs, "SELECT doi FROM papers WHERE openalex_id = ?", r.json()["id"]) == [("10.9/from-the-pdf",)]


def test_l3_no_upload_into_an_interest_whose_first_scan_is_running(pdfs):
    d = _draft(pdfs, "Saving")
    pid, uid = _q(pdfs, "SELECT id, user_id FROM profiles WHERE slug = ?", d)[0]
    conn = connect(pdfs["env"])
    gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    conn.commit()
    conn.close()
    before = _snapshot(pdfs)
    r = _post(pdfs, b"manuscript late", A, d)
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == f"'{d}' is being saved; add seeds to it once that finishes."
    assert _snapshot(pdfs) == before


def test_l5_the_vault_delete_checks_under_the_write_lock(pdfs):
    """The refusals are read inside BEGIN IMMEDIATE: nothing can start a
    scan or remove a seed between the check and the delete."""
    from rag_lib.api.services import vault as vault_service

    d = _draft(pdfs, "Lock")
    doc = _post(pdfs, b"manuscript lock", A, d).json()["id"]
    uid = _q(pdfs, "SELECT user_id FROM profiles WHERE slug = ?", d)[0][0]
    conn = connect(pdfs["env"])
    seen = []
    conn.set_trace_callback(seen.append)
    vault_service.delete_upload(conn, None, user_id=uid, doc_id=doc)
    conn.close()
    begin = next(i for i, s in enumerate(seen) if s.strip().upper().startswith("BEGIN IMMEDIATE"))
    first_check = next(i for i, s in enumerate(seen) if "FROM paper_uploads WHERE user_id" in s)
    assert begin < first_check


def test_l6_the_vault_delete_drops_chat_chunks_when_chat_has_a_collection(pdfs, monkeypatch):
    import rag_lib.rag.indexer as rag_indexer
    from rag_lib.api.services import vault as vault_service

    dropped = []

    class Coll:
        def __init__(self, name):
            self.name = name

        def delete(self, where):
            dropped.append((self.name, where["openalex_id"]))

    monkeypatch.setattr(rag_indexer, "index_user_collection", lambda s, u: Coll("retrieval"))
    monkeypatch.setattr(rag_indexer, "index_user_chat_collection", lambda s, u: Coll("chat"))
    uid = None
    for model, want in (("", {"retrieval"}), ("chat-model", {"retrieval", "chat"})):
        d = _draft(pdfs, f"Chunks {model or 'none'}")
        doc = _post(pdfs, f"manuscript {model}".encode(), A, d).json()["id"]
        uid = _q(pdfs, "SELECT user_id FROM profiles WHERE slug = ?", d)[0][0]
        dropped.clear()
        conn = connect(pdfs["env"])
        vault_service.delete_upload(conn, SimpleNamespace(RADAR_CHAT_EMBEDDING_MODEL=model), user_id=uid, doc_id=doc)
        conn.close()
        assert {n for n, i in dropped if i == doc} == want, model


def test_the_same_untitled_file_in_two_interests_is_two_numbered_versions(pdfs, monkeypatch):
    """A PDF without a title (its name stands in) that OpenAlex does not
    know, uploaded into two interests: one paper, versions 1 and 2 in the
    Vault (CR.10, CR.4) -- the same file, not only the same title."""
    from rag_lib.api.services import vault as vault_service
    from rag_lib.vault import PdfIngestRecord

    def ingest(path):
        return PdfIngestRecord(path=str(path), title=None, body_text="scanned", doi=None, n_pages=1)

    monkeypatch.setattr(vault_service, "ingest_pdf", ingest)
    ids = [_post(pdfs, b"scan bytes", A, _draft(pdfs, name)).json()["id"] for name in ("One", "Two")]
    other = _post(pdfs, b"other scan", A, _draft(pdfs, "Three")).json()["id"]
    docs = {d["id"]: d for d in _request(pdfs["app"], "GET", "/api/vault/docs", headers=A).json()}
    assert [docs[i]["version"] for i in ids] == [1, 2]
    assert docs[other]["version"] is None


def test_r1_each_upload_has_its_own_file(pdfs, monkeypatch):
    """Review R1: one file in two interests is two files on disk. Deleting
    the first from the Vault while the second is still being written
    (another tab) leaves the second its file."""
    import os

    from rag_lib.api.services import vault as vault_service

    x = _draft(pdfs, "X")
    v1 = _post(pdfs, b"manuscript same", A, x).json()
    uid = _q(pdfs, "SELECT user_id FROM profiles WHERE slug = ?", x)[0][0]
    p1 = _q(pdfs, "SELECT local_path FROM paper_uploads WHERE openalex_id = ?", v1["id"])[0][0]
    real, fired = os.replace, []

    def replace(src, dst):
        real(src, dst)
        if str(src).endswith(".part") and not fired:
            fired.append(1)
            c = connect(pdfs["env"])
            vault_service.delete_upload(c, None, user_id=uid, doc_id=v1["id"])
            c.close()

    monkeypatch.setattr(os, "replace", replace)
    r = _post(pdfs, b"manuscript same", A, _draft(pdfs, "Y"))
    assert r.status_code == 200 and fired, r.text
    p2 = _q(pdfs, "SELECT local_path FROM paper_uploads WHERE openalex_id = ?", r.json()["id"])[0][0]
    assert p2 != p1 and Path(p2).is_file() and not Path(p1).exists()


def test_r2_an_upload_that_cannot_take_the_write_lock_leaves_no_file(pdfs, monkeypatch):
    """Review R2: another writer holds the database past the busy timeout;
    the upload fails and the file it moved into place goes again."""
    import os

    real, holder = os.replace, {}

    def replace(src, dst):
        real(src, dst)
        if str(src).endswith(".part"):
            c = sqlite3.connect(str(pdfs["env"]), isolation_level=None, check_same_thread=False)
            c.execute("BEGIN IMMEDIATE")
            holder["c"], holder["dst"] = c, dst

    monkeypatch.setattr(os, "replace", replace)
    before = _snapshot(pdfs)
    try:
        r = _post(pdfs, b"manuscript locked", A, _draft(pdfs, "Locked"))
        assert r.status_code >= 500
    except sqlite3.OperationalError:
        pass
    finally:
        if "c" in holder:
            holder["c"].execute("ROLLBACK")
            holder["c"].close()
    assert "dst" in holder and not Path(holder["dst"]).exists()
    after = _snapshot(pdfs)
    assert after["files"] == before["files"]
    assert not [p for p in (Path(pdfs["env"]).parent / "vault").rglob("*.part")]


def test_an_upload_into_a_draft_is_embedded_under_the_drafts_model_too(pdfs, monkeypatch):
    """Moved from test_review_round9: vectors under the default model and
    the target interest's, computed before anything is written (CR.3)."""
    from rag_lib.api.services import vault as vault_service
    from rag_lib.embedders import placeholder_embed

    monkeypatch.setattr(vault_service, "get_embedder", lambda model: placeholder_embed)
    d = _draft(pdfs, "Other model")
    conn = connect(pdfs["env"])
    conn.execute("UPDATE profiles SET embedding_model = 'specter2' WHERE slug = ?", (d,))
    conn.commit()
    conn.close()
    doc = _post(pdfs, b"manuscript models", A, d).json()["id"]
    models = {m for (m,) in _q(pdfs, "SELECT embedding_model FROM paper_embeddings WHERE openalex_id = ?", doc)}
    assert models == {"placeholder-v1", "specter2"}
