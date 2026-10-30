"""Each user's upload of a PDF is their own version of the paper.

Before migration 0029 an upload was stored on the paper's public row: a
second user uploading the same paper was scored with the first user's
text and vectors, and a revised file replaced the old one. Now the public
row is bibliographic only and every upload is ``local:upload:<user>:<hash>``
(``rag_lib.db.repos.versions``).
"""

from __future__ import annotations

import io
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from rag_lib.db import connect
from rag_lib.db.migrate import MIGRATIONS_DIR, apply_migrations
from rag_lib.interest_terms import seed_hash
from rag_lib.db.repos import (
    candidates as candidates_repo,
    embeddings as embeddings_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
    versions as versions_repo,
)
from tests.edit_helpers import edit_saved
from tests.test_review_round12_isolation import A, B, _request, world  # noqa: F401 -- fixture
from tests.test_researchers import env  # noqa: F401 -- fixture
from tests.test_db_migrate import EXPECTED_VERSIONS

W = "https://openalex.org/W77"
TOPIC = {"id": "https://openalex.org/T11", "display_name": "Public topic", "score": 0.9}


@pytest.fixture()
def pdfs(world, monkeypatch):
    """Uploads read their text from the bytes; one whose bytes say
    ``matched`` is OpenAlex's W77, any other matches nothing."""
    from rag_lib.api.services import vault as vault_service
    from rag_lib.vault import PdfIngestRecord

    def ingest(path):
        text = Path(path).read_bytes().decode("latin-1")
        return PdfIngestRecord(path=str(path), title="Title on the page", body_text=f"body of {text}",
                               doi=None, n_pages=2)

    def enrich(record, *, mailto, filename, file_hash):
        d = {"title": record.title, "doi": None, "source": "user_pdf"}
        if "matched" in record.body_text:
            d.update(openalex_id=W, title="Public Title", abstract="public abstract", year=2020,
                     doi="10.1/w77", primary_topic=TOPIC, topics=[TOPIC])
            return W, d
        d["openalex_id"] = f"local:{file_hash}"
        return d["openalex_id"], d

    content_calls = []

    def content(title, abstract, body, *, settings):
        content_calls.append(body)
        return json.dumps({"source": "pdf_content", "topics": [{"id": "T-content", "display_name": "From text"}]})

    monkeypatch.setattr(vault_service, "ingest_pdf", ingest)
    monkeypatch.setattr(vault_service, "_enrich_via_openalex", enrich)
    monkeypatch.setattr(vault_service.content_topics, "try_topics_json", content)
    world["content_calls"] = content_calls
    return world


def _inbox(world, headers) -> str:
    """The user's draft to upload into when a test names no interest: every
    upload is into one (FUNCTIONAL_SPEC 1.1)."""
    boxes = world.setdefault("inboxes", {})
    who = headers["X-User-Email"]
    if who not in boxes:
        r = _request(world["app"], "POST", "/api/profiles/draft", json={"name": "Inbox"}, headers=headers)
        assert r.status_code == 200, r.text
        boxes[who] = r.json()["slug"]
    return boxes[who]


def _post(world, data: bytes, headers, slug=None):
    files = {"file": ("p.pdf", io.BytesIO(b"%PDF-1.4\n" + data), "application/pdf")}
    return _request(world["app"], "POST", "/api/vault/upload", files=files,
                    data={"profile_slug": slug or _inbox(world, headers)}, headers=headers)


def _upload(world, data: bytes, headers, slug=None):
    r = _post(world, data, headers, slug)
    assert r.status_code == 200, r.text
    return r.json()


def _upload_saved(world, data: bytes, headers, slug):
    """Upload into saved interest ``slug``: only through Edit interest
    (spec E), so into its copy, saved back on the way out."""
    with edit_saved(lambda m, p, **kw: _request(world["app"], m, p, headers=headers, **kw), slug) as copy:
        return _upload(world, data, headers, slug=copy)


def _row(env, oid):
    conn = connect(env)
    try:
        r = conn.execute("SELECT * FROM papers WHERE openalex_id = ?", (oid,)).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def _vector(env, oid):
    conn = connect(env)
    try:
        v = embeddings_repo.get(conn, oid, "placeholder-v1")
        return None if v is None else np.asarray(v)
    finally:
        conn.close()


def test_two_users_uploading_the_same_paper_get_their_own_versions(pdfs):
    env = pdfs["env"]
    a = _upload(pdfs, b"matched alice", A)
    b = _upload(pdfs, b"matched bob", B)
    assert a["id"] != b["id"] and a["id"].startswith("local:upload:") and b["id"].startswith("local:upload:")
    assert a["canonicalId"] == b["canonicalId"] == W
    assert "alice" in _row(env, a["id"])["body_text"] and "bob" in _row(env, b["id"])["body_text"]
    assert not np.allclose(_vector(env, a["id"]), _vector(env, b["id"]))
    public = _row(env, W)
    assert public["title"] == "Public Title"
    assert [public[k] for k in ("body_text", "local_path", "file_hash", "uploaded_by_user_id")] == [None] * 4
    assert _vector(env, W) is None
    # Identical bytes from two users are still two versions.
    c = _upload(pdfs, b"matched alice", B)
    assert c["id"] not in (a["id"], b["id"]) and not c.get("duplicate")


def test_a_matched_upload_keeps_openalex_title_and_topics(pdfs):
    a = _upload(pdfs, b"matched alice", A)
    row = _row(pdfs["env"], a["id"])
    assert row["title"] == "Public Title"
    topics = json.loads(row["topics_json"])
    assert topics.get("source") != "pdf_content" and topics["primary_topic"]["id"] == TOPIC["id"]
    assert pdfs["content_calls"] == []
    u = _upload(pdfs, b"nothing known", A)
    assert u["canonicalId"] is None
    assert json.loads(_row(pdfs["env"], u["id"])["topics_json"])["source"] == "pdf_content"
    assert len(pdfs["content_calls"]) == 1


def test_a_revised_file_is_a_new_version_and_the_old_one_stays_a_seed(pdfs):
    env, live = pdfs["env"], pdfs["live"]
    v1 = _upload_saved(pdfs, b"matched draft one", A, live)
    v2 = _upload(pdfs, b"matched draft two", A)
    assert v2["id"] != v1["id"] and v2["canonicalId"] == v1["canonicalId"] == W
    assert "draft one" in _row(env, v1["id"])["body_text"]
    assert _vector(env, v1["id"]) is not None
    conn = connect(env)
    try:
        seeds = profiles_repo.list_seed_openalex_ids(conn, pdfs["live_id"])
    finally:
        conn.close()
    assert v1["id"] in seeds and v2["id"] not in seeds
    docs = {d["id"]: d for d in _request(pdfs["app"], "GET", "/api/vault/docs", headers=A).json()}
    assert {v1["id"], v2["id"]} <= set(docs)
    # Two files of one paper: numbered in upload order (FUNCTIONAL_SPEC 1.7).
    assert (docs[v1["id"]]["version"], docs[v2["id"]]["version"]) == (1, 2)


def test_the_same_bytes_again_are_a_duplicate_unless_the_file_is_gone(pdfs):
    env = pdfs["env"]
    first = _upload(pdfs, b"matched alice", A)
    assert _upload(pdfs, b"matched alice", A)["duplicate"] is True
    path = Path(_row(env, first["id"])["local_path"])
    path.unlink()
    again = _upload(pdfs, b"matched alice", A)
    assert again["id"] == first["id"] and not again.get("duplicate")
    assert path.is_file()
    assert _upload(pdfs, b"matched alice", A)["duplicate"] is True


def test_a_seeded_version_keeps_its_public_work_out_of_the_feed(pdfs):
    env, live_id = pdfs["env"], pdfs["live_id"]
    v = _upload_saved(pdfs, b"matched alice", A, pdfs["live"])
    conn = connect(env)
    try:
        ids, _ = profiles_repo.seed_keys(conn, live_id)
        assert W in ids and v["id"] in ids
        conn.execute("UPDATE papers SET doi = NULL WHERE openalex_id IN (?, ?)", (W, v["id"]))
        conn.commit()
        candidates_repo.insert_dedup(conn, profile_id=live_id, openalex_id=W, score=0.99, score_raw=0.99)
        shown = [r["openalex_id"] for r in candidates_repo.top_for_profile(conn, live_id, limit=50)]
    finally:
        conn.close()
    assert W not in shown and pdfs["cards"][0] in shown


def test_an_upload_is_a_seed_beside_its_works_public_record(pdfs):
    """FUNCTIONAL_SPEC 1.7: both stay seeds, the user removes one if they
    want; the public record named again (a search, an import) adds nothing."""
    env, live_id = pdfs["env"], pdfs["live_id"]
    conn = connect(env)
    try:
        papers_repo.upsert(conn, {"openalex_id": W, "title": "Public Title", "source": "openalex"})
        assert profiles_repo.attach_seed(conn, live_id, W) == 1
    finally:
        conn.close()
    v = _upload_saved(pdfs, b"matched alice", A, pdfs["live"])
    conn = connect(env)
    try:
        seeds = profiles_repo.list_seed_openalex_ids(conn, live_id)
        assert v["id"] in seeds and W in seeds
        assert profiles_repo.attach_seed(conn, live_id, W) == 0
        assert profiles_repo.list_seed_openalex_ids(conn, live_id) == seeds
    finally:
        conn.close()


def test_another_users_version_cannot_be_used(pdfs):
    env = pdfs["env"]
    a = _upload(pdfs, b"matched alice", A)
    conn = connect(env)
    try:
        users_repo.upsert(conn, "bob@example.com")
        b_uid = int(users_repo.get_by_email(conn, "bob@example.com")["id"])
        pid = profiles_repo.upsert(conn, user_id=b_uid, name="B's", embedding_model="placeholder-v1", n_seed=0,
                                   topic_filters={"topics": []}, threshold=0.5)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO profile_seeds (profile_id, openalex_id) VALUES (?, ?)", (pid, a["id"]))
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            papers_repo.record_upload(conn, user_id=b_uid, openalex_id=a["id"], file_hash="x",
                                      local_path=None, n_pages=None)
        conn.rollback()
        slug = profiles_repo.get(conn, pid)["slug"]
        assert papers_repo.get_by_doi(conn, "10.1/w77")["openalex_id"] == W
        # Only the upload has the DOI: it still resolves to no paper.
        conn.execute("UPDATE papers SET doi = NULL WHERE openalex_id = ?", (W,))
        conn.commit()
        assert papers_repo.get_by_doi(conn, "10.1/w77") is None
    finally:
        conn.close()
    # B's interest is saved: seeds go through Edit interest (spec E). It
    # has no seeds of its own, so the copy cannot be saved; discard it.
    with edit_saved(lambda m, p, **kw: _request(pdfs["app"], m, p, headers=B, **kw), slug, save=False) as copy:
        r = _request(pdfs["app"], "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [a["id"]]}, headers=B)
        assert r.status_code in (200, 400) and a["id"] in r.json().get("rejected", [a["id"]])
    conn = connect(env)
    try:
        assert a["id"] not in profiles_repo.list_seed_openalex_ids(conn, pid)
    finally:
        conn.close()


def test_the_vault_lists_a_work_once(pdfs):
    v = _upload(pdfs, b"matched alice", A)
    conn = connect(pdfs["env"])
    try:
        rid = int(conn.execute("SELECT id FROM researchers LIMIT 1").fetchone()[0])
        papers_repo.upsert(conn, {"openalex_id": W, "title": "Public Title", "source": "openalex"})
        conn.execute("INSERT INTO researcher_papers (researcher_id, openalex_id) VALUES (?, ?)", (rid, W))
        conn.commit()
    finally:
        conn.close()
    ids = [d["id"] for d in _request(pdfs["app"], "GET", "/api/vault/docs", headers=A).json()]
    assert v["id"] in ids and W not in ids


# --- migration 0029 ---------------------------------------------------------

def _db_before_0029(tmp_path) -> Path:
    mdir = tmp_path / "migrations"
    mdir.mkdir()
    for p in MIGRATIONS_DIR.iterdir():
        if p.suffix in (".sql", ".py") and p.stem[:4].isdigit() and p.stem < "0029":
            shutil.copy(p, mdir / p.name)
    db = tmp_path / "legacy.db"
    conn = connect(db)
    apply_migrations(conn, mdir)
    conn.close()
    return db


def _legacy_world(tmp_path):
    """A uploaded W1 (its only upload, the row's text); C seeds W1 from
    ORCID; A also uploaded an unmatched PDF ``local:hA2``. W4 has two
    uploads: A's file is what the row records, B's is another file, and
    either could be the row's text."""
    db = _db_before_0029(tmp_path)
    conn = connect(db)
    uid = {n: users_repo.upsert(conn, f"{n}@example.com") for n in "abc"}
    uid = {n: int(users_repo.get_by_email(conn, f"{n}@example.com")["id"]) for n in "abc"}
    w1, loc = "https://openalex.org/W1", "local:hA2"
    papers_repo.upsert(conn, {"openalex_id": w1, "title": "W one", "abstract": "abs", "source": "user_pdf",
                              "doi": "10.1/one", "body_text": "alice's text", "local_path": "vault/a/hA.pdf",
                              "file_hash": "hA", "uploaded_by_user_id": uid["a"], "n_pages": 4,
                              "primary_topic": TOPIC, "topics": [TOPIC]})
    papers_repo.upsert(conn, {"openalex_id": loc, "title": "Local one", "source": "user_pdf",
                              "body_text": "alice's other text", "local_path": "vault/a/hA2.pdf",
                              "file_hash": "hA2", "uploaded_by_user_id": uid["a"]})
    papers_repo.upsert(conn, {"openalex_id": "https://openalex.org/W2", "title": "W two", "source": "openalex"})
    papers_repo.upsert(conn, {"openalex_id": W4, "title": "W four", "source": "user_pdf",
                              "body_text": "whose text?", "local_path": "vault/a/hA4.pdf",
                              "file_hash": "hA4", "uploaded_by_user_id": uid["a"]})
    conn.execute("UPDATE papers SET umls_concepts_json = '[\"C1\"]', umls_mapped_topics_json = '[]' WHERE openalex_id = ?", (w1,))
    rng = np.random.default_rng(3)
    for oid in (w1, loc, "https://openalex.org/W2"):
        embeddings_repo.upsert(conn, oid, "placeholder-v1", rng.standard_normal(8).astype("float32"))
    for who, oid, h in (("a", w1, "hA"), ("a", loc, "hA2"), ("a", W4, "hA4"), ("b", W4, "hB")):
        papers_repo.record_upload(conn, user_id=uid[who], openalex_id=oid, file_hash=h,
                                  local_path=f"vault/{who}/{h}.pdf", n_pages=4)
    pids = {}
    for who, seeds in (("a", [w1, loc, "https://openalex.org/W2"]), ("b", [W4]), ("c", [w1])):
        topics = {"topics": [{"id": "T1", "display_name": "one", "count": 2, "on": False, "source": "umls",
                              "seeds": list(seeds)},
                             {"id": "T2", "display_name": "two", "count": 1, "on": True}]}
        pid = profiles_repo.upsert(conn, user_id=uid[who], name=f"{who} interest", embedding_model="placeholder-v1",
                                   n_seed=len(seeds), topic_filters=topics, threshold=0.4,
                                   selector_config={"type": "centroid", "fit_seed_ids": sorted(seeds),
                                                    "centroid": [0.1] * 8, "seed_matrix": [[0.0]] * len(seeds)})
        for s in seeds:     # attach_seed reads paper_versions, which 0029 makes
            conn.execute("INSERT INTO profile_seeds (profile_id, openalex_id) VALUES (?, ?)", (pid, s))
        conn.execute("UPDATE profiles SET interest_terms_json = ? WHERE id = ?", (json.dumps(
            {"seed_hash": seed_hash(seeds), "terms": [{"label": "t", "paper_ids": list(seeds)}]}), pid))
        pids[who] = pid
    # An older fit, from before fit seed ids were recorded.
    pids["a_old"] = profiles_repo.upsert(
        conn, user_id=uid["a"], name="a older", embedding_model="placeholder-v1", n_seed=1, threshold=0.4,
        selector_config={"type": "centroid", "centroid": [0.1] * 8, "seed_matrix": [[0.0]]})
    conn.execute("INSERT INTO profile_seeds (profile_id, openalex_id) VALUES (?, ?)", (pids["a_old"], w1))
    conn.commit()
    conn.close()
    return db, uid, pids


W4 = "https://openalex.org/W4"


def _terms(conn, pid):
    return json.loads(conn.execute("SELECT interest_terms_json FROM profiles WHERE id = ?", (pid,)).fetchone()[0])


def _profile(conn, pid):
    r = conn.execute("SELECT topic_filters_json, selector_config_json, threshold FROM profiles WHERE id = ?",
                     (pid,)).fetchone()
    return json.loads(r[0]), json.loads(r[1]), r[2], sorted(profiles_repo.list_seed_openalex_ids(conn, pid))


def test_0029_moves_each_upload_to_its_own_version(tmp_path):
    db, uid, pids = _legacy_world(tmp_path)
    w1 = "https://openalex.org/W1"
    conn = connect(db)
    before = {who: _profile(conn, pid) for who, pid in pids.items()}
    vec_a = embeddings_repo.get(conn, w1, "placeholder-v1")
    conn.close()

    conn = connect(db)
    assert apply_migrations(conn) == EXPECTED_VERSIONS[EXPECTED_VERSIONS.index("0029_private_upload_versions"):]
    va, va2, vb = (versions_repo.version_id(uid["a"], "hA"), versions_repo.version_id(uid["a"], "hA2"),
                   versions_repo.version_id(uid["b"], "hB"))
    try:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        # A's upload is the row's text: everything moves with it.
        row = conn.execute("SELECT * FROM papers WHERE openalex_id = ?", (va,)).fetchone()
        assert (row["body_text"], row["title"], row["umls_concepts_json"]) == ("alice's text", "W one", '["C1"]')
        assert json.loads(row["topics_json"])["primary_topic"]["id"] == TOPIC["id"]
        assert np.allclose(embeddings_repo.get(conn, va, "placeholder-v1"), vec_a)
        state = "SELECT parsed, canonical_openalex_id FROM paper_versions WHERE version_id = ?"
        assert tuple(conn.execute(state, (va,)).fetchone()) == (1, w1)
        assert tuple(conn.execute(state, (va2,)).fetchone()) == (1, None)
        # A's interest is unchanged but for the ids: choices, fit, seeds.
        ren = {w1: va, "local:hA2": va2}
        topics, cfg, thr, seeds = _profile(conn, pids["a"])
        assert (topics, cfg, thr, seeds) == (
            versions_repo.rename_ids(before["a"][0], ren), versions_repo.rename_ids(before["a"][1], ren),
            before["a"][2], sorted(versions_repo.rename_ids(before["a"][3], ren)))
        assert sorted(cfg["fit_seed_ids"]) == seeds          # not stale: no refit, no rebuild
        assert _profile(conn, pids["a_old"])[1] == before["a_old"][1]   # its matrix too
        terms = _terms(conn, pids["a"])
        assert terms["seed_hash"] == seed_hash(seeds) and sorted(terms["terms"][0]["paper_ids"]) == seeds
        # W4's text may be A's or B's file: neither keeps it, both are read again.
        va4 = versions_repo.version_id(uid["a"], "hA4")
        for v in (va4, vb):
            row = conn.execute("SELECT * FROM papers WHERE openalex_id = ?", (v,)).fetchone()
            assert row["body_text"] is None and row["umls_concepts_json"] is None and row["title"] == "W four"
            assert embeddings_repo.get(conn, v, "placeholder-v1") is None
            assert conn.execute("SELECT parsed FROM paper_versions WHERE version_id = ?", (v,)).fetchone()[0] == 0
        topics, cfg, _, seeds = _profile(conn, pids["b"])
        assert seeds == [vb] and topics == versions_repo.rename_ids(before["b"][0], {W4: vb})
        assert cfg.get("fit_seed_ids") != seeds              # stale: the next scan refits it
        assert _terms(conn, pids["b"])["seed_hash"] != seed_hash(seeds)    # and its terms
        # C seeds the public work, which keeps only its metadata: C's
        # interest is the same, but was fitted from A's text, so is stale.
        topics, cfg, thr, seeds = _profile(conn, pids["c"])
        assert (topics, thr, seeds) == (before["c"][0], before["c"][2], before["c"][3])
        assert cfg.get("fit_seed_ids") != seeds
        pub = conn.execute("SELECT * FROM papers WHERE openalex_id = ?", (w1,)).fetchone()
        assert [pub[k] for k in ("body_text", "local_path", "file_hash", "uploaded_by_user_id",
                                 "umls_concepts_json")] == [None] * 5
        assert pub["title"] == "W one" and pub["doi"] == "10.1/one"
        assert embeddings_repo.get(conn, w1, "placeholder-v1") is None
        assert conn.execute("SELECT 1 FROM papers WHERE openalex_id = 'local:hA2'").fetchone() is None
        assert sorted(r[0] for r in conn.execute("SELECT openalex_id FROM paper_uploads")) == sorted([va, va2, va4, vb])
    finally:
        conn.close()


def test_a_waiting_version_reads_its_own_file_or_is_scored_on_metadata(tmp_path, monkeypatch):
    db, uid, pids = _legacy_world(tmp_path)
    conn = connect(db)
    apply_migrations(conn)
    vb = versions_repo.version_id(uid["b"], "hB")
    good = tmp_path / "b.pdf"
    good.write_bytes(b"bob's file")
    import rag_lib.vault as vault_mod
    from rag_lib.vault import PdfIngestRecord
    monkeypatch.setattr(vault_mod, "compute_file_hash", lambda p: "hB" if Path(p) == good else "other")
    monkeypatch.setattr(vault_mod, "ingest_pdf", lambda p: PdfIngestRecord(
        path=str(p), title="hB", body_text="bob's own text", doi=None, n_pages=5))
    try:
        # Missing file: marked unreadable, nothing raised, still a seed.
        assert versions_repo.parse_pending(conn) == 0
        assert conn.execute("SELECT parsed FROM paper_versions WHERE version_id = ?", (vb,)).fetchone()[0] == -1
        assert vb in profiles_repo.list_seed_openalex_ids(conn, pids["b"])
        # A file there that is other bytes (rewritten in place): the same.
        changed = tmp_path / "changed.pdf"
        changed.write_bytes(b"something else")
        conn.execute("UPDATE paper_versions SET parsed = 0 WHERE version_id = ?", (vb,))
        conn.execute("UPDATE paper_uploads SET local_path = ? WHERE openalex_id = ?", (str(changed), vb))
        conn.commit()
        assert versions_repo.parse_pending(conn) == 0
        assert conn.execute("SELECT body_text FROM papers WHERE openalex_id = ?", (vb,)).fetchone()[0] is None
        # The file is there and is these bytes: its text, the title kept.
        conn.execute("UPDATE paper_versions SET parsed = 0 WHERE version_id = ?", (vb,))
        conn.execute("UPDATE paper_uploads SET local_path = ? WHERE openalex_id = ?", (str(good), vb))
        conn.execute("UPDATE papers SET topics_json = ? WHERE openalex_id = ?",
                     (json.dumps({"source": "pdf_content", "topics": [TOPIC]}), vb))
        conn.commit()
        assert versions_repo.parse_pending(conn) == 1
        # Topics read from the old text go too; the next fit reads the new.
        assert conn.execute("SELECT topics_json FROM papers WHERE openalex_id = ?", (vb,)).fetchone()[0] is None
        row = conn.execute("SELECT title, body_text FROM papers WHERE openalex_id = ?", (vb,)).fetchone()
        assert tuple(row) == ("W four", "bob's own text")
        assert conn.execute("SELECT parsed FROM paper_versions WHERE version_id = ?", (vb,)).fetchone()[0] == 1
    finally:
        conn.close()


def test_a_cli_ingested_pdf_is_the_users_version(env):
    from rag_lib.paper import Paper
    from rag_lib.persistence import store_profile_from_object
    from rag_lib.profile import Profile

    conn = connect(env)
    try:
        uid = int(users_repo.get_by_email(conn, "demo@example.com")["id"])
        paper = Paper(doi="10.1/cli", openalex_id="https://openalex.org/W5", title="CLI paper",
                      abstract="a", body_text="text read from my pdf",
                      embeddings={"placeholder-v1": [0.5] * 8})
        profile = Profile(name="cli", papers=[paper], embedding_model="placeholder-v1",
                          selector_config={"type": "centroid", "fit_seed_ids": ["https://openalex.org/W5"]})
        pid = store_profile_from_object(conn, user_id=uid, profile=profile)
        seeds = profiles_repo.list_seed_openalex_ids(conn, pid)
        assert len(seeds) == 1 and versions_repo.is_version(seeds[0])
        assert versions_repo.canonical_id(conn, seeds[0]) == "https://openalex.org/W5"
        assert conn.execute("SELECT body_text FROM papers WHERE openalex_id = ?", (seeds[0],)).fetchone()[0] \
            == "text read from my pdf"
        assert conn.execute("SELECT body_text FROM papers WHERE openalex_id = 'https://openalex.org/W5'").fetchone()[0] is None
        assert embeddings_repo.get(conn, seeds[0], "placeholder-v1") is not None
        cfg = json.loads(conn.execute("SELECT selector_config_json FROM profiles WHERE id = ?", (pid,)).fetchone()[0])
        assert cfg["fit_seed_ids"] == seeds
    finally:
        conn.close()


def test_0029_cli_pdfs_and_one_file_under_two_papers(tmp_path):
    """A CLI-ingested PDF (its file on the row, no upload row) is the upload
    of the user the row names; with none, nobody's, also where one user
    alone seeds it (that does not say whose file it was). One
    file a user uploaded as two papers is one version, of the first; the
    second paper stays a seed, on its public record."""
    db = _db_before_0029(tmp_path)
    conn = connect(db)
    for n in "cd":
        users_repo.upsert(conn, f"{n}@example.com")
    uid = {n: int(users_repo.get_by_email(conn, f"{n}@example.com")["id"]) for n in "cd"}
    w5, w6, w7, w8, w9 = (f"https://openalex.org/W{i}" for i in (5, 6, 7, 8, 9))
    for oid, extra in ((w6, {"body_text": "d's cli text", "local_path": "vault/w6.pdf",
                             "uploaded_by_user_id": uid["d"]}),
                       (w5, {"body_text": "whose cli text?", "local_path": "vault/w5.pdf"}),
                       (w7, {"body_text": "whose cli text?", "local_path": "vault/w7.pdf"}),
                       (w8, {"body_text": "c's file", "local_path": "vault/c/hS.pdf",
                             "file_hash": "hS", "uploaded_by_user_id": uid["c"]}),
                       (w9, {})):
        papers_repo.upsert(conn, {"openalex_id": oid, "title": oid[-2:], "source": "user_pdf", **extra})
    for oid in (w8, w9):
        papers_repo.record_upload(conn, user_id=uid["c"], openalex_id=oid, file_hash="hS",
                                  local_path="vault/c/hS.pdf", n_pages=1)
    pids = {}
    for who, seeds in (("c", [w7, w8, w9]), ("d", [w5, w6, w7])):
        pid = profiles_repo.upsert(conn, user_id=uid[who], name=f"{who}", embedding_model="placeholder-v1",
                                   n_seed=len(seeds), threshold=0.4,
                                   topic_filters={"topics": [{"id": "T1", "seeds": list(seeds)}]},
                                   selector_config={"type": "centroid", "fit_seed_ids": sorted(seeds)})
        for x in seeds:
            conn.execute("INSERT INTO profile_seeds (profile_id, openalex_id) VALUES (?, ?)", (pid, x))
        pids[who] = pid
    conn.commit()
    conn.close()

    conn = connect(db)
    try:
        apply_migrations(conn)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        import hashlib
        v6 = versions_repo.version_id(uid["d"], hashlib.sha256(b"d's cli text").hexdigest())
        vs = versions_repo.version_id(uid["c"], "hS")
        state = "SELECT parsed, canonical_openalex_id FROM paper_versions WHERE version_id = ?"
        assert tuple(conn.execute(state, (v6,)).fetchone()) == (1, w6)
        assert conn.execute("SELECT body_text FROM papers WHERE openalex_id = ?", (v6,)).fetchone()[0] == "d's cli text"
        assert tuple(conn.execute(state, (vs,)).fetchone()) == (1, w8)
        # W5 (only D seeds it) and W7 (two users do): nobody's file; public.
        for w in (w5, w7):
            assert conn.execute("SELECT 1 FROM paper_versions WHERE canonical_openalex_id = ?", (w,)).fetchone() is None
            assert conn.execute("SELECT body_text, local_path FROM papers WHERE openalex_id = ?",
                                (w,)).fetchone()[:] == (None, None)
        _, cfg, _, seeds = _profile(conn, pids["d"])
        assert seeds == sorted([w5, v6, w7]) and not {w5, w7} & set(cfg["fit_seed_ids"])   # fitted from their text: stale
        # C's file is one version, of W8; W9 is still C's seed, not swallowed.
        topics, cfg, _, seeds = _profile(conn, pids["c"])
        assert seeds == sorted([vs, w7, w9])
        assert sorted(topics["topics"][0]["seeds"]) == seeds
        assert [r[0] for r in conn.execute("SELECT openalex_id FROM paper_uploads WHERE user_id = ?",
                                           (uid["c"],))] == [vs]
        assert w7 not in cfg["fit_seed_ids"]
    finally:
        conn.close()


def test_each_uploaded_file_is_a_seed_of_its_own(env):
    conn = connect(env)
    try:
        uid = int(users_repo.get_by_email(conn, "demo@example.com")["id"])
        w = "https://openalex.org/W10"
        papers_repo.upsert(conn, {"openalex_id": w, "title": "ten", "source": "openalex"})
        v1, v2 = versions_repo.version_id(uid, "h1"), versions_repo.version_id(uid, "h2")
        for v, h in ((v1, "h1"), (v2, "h2")):
            papers_repo.upsert(conn, {"openalex_id": v, "title": "ten", "source": "user_pdf_version"})
            versions_repo.register(conn, user_id=uid, paper_id=v, canonical_id=w, file_hash=h)
        pid = profiles_repo.upsert(conn, user_id=uid, name="rev", embedding_model="placeholder-v1", n_seed=0,
                                   threshold=0.4, topic_filters={"topics": [{"id": "T1", "seeds": [w]}]})
        conn.commit()
        seeds = lambda: profiles_repo.list_seed_openalex_ids(conn, pid)  # noqa: E731
        topic_seeds = lambda: json.loads(conn.execute(  # noqa: E731
            "SELECT topic_filters_json FROM profiles WHERE id = ?", (pid,)).fetchone()[0])["topics"][0]["seeds"]
        assert profiles_repo.attach_seed(conn, pid, w) == 1 and seeds() == [w]
        # A file of the work, then a revised file: each added beside the
        # others, nothing renamed (FUNCTIONAL_SPEC 1.7).
        assert profiles_repo.attach_seed(conn, pid, v1) == 1 and seeds() == [w, v1] and topic_seeds() == [w]
        assert profiles_repo.attach_seed(conn, pid, v2) == 1 and seeds() == [w, v1, v2] and topic_seeds() == [w]
        assert profiles_repo.attach_seed(conn, pid, v2) == 0
        assert profiles_repo.attach_seed(conn, pid, w) == 0 and seeds() == [w, v1, v2]
        assert conn.execute("SELECT n_seed FROM profiles WHERE id = ?", (pid,)).fetchone()[0] == 3
        # The public record is held back while a file of it is a seed (a
        # search or import naming the paper again), and only then.
        conn.execute("DELETE FROM profile_seeds WHERE profile_id = ?", (pid,))
        conn.commit()
        assert profiles_repo.attach_seed(conn, pid, v2) == 1
        assert profiles_repo.attach_seed(conn, pid, w) == 0 and seeds() == [v2]
        conn.execute("DELETE FROM profile_seeds WHERE profile_id = ?", (pid,))
        conn.commit()
        assert profiles_repo.attach_seed(conn, pid, w) == 1 and seeds() == [w]
        assert versions_repo.rename_ids(["a", "b", "c", 1], {"a": "v", "b": "v"}) == ["v", "c", 1]
    finally:
        conn.close()


def test_vault_stats_count_a_work_once(env):
    from rag_lib.db.repos import vault as vault_repo

    conn = connect(env)
    try:
        uid = int(users_repo.get_by_email(conn, "demo@example.com")["id"])
        w = "https://openalex.org/W11"
        v = versions_repo.version_id(uid, "h11")
        papers_repo.upsert(conn, {"openalex_id": w, "title": "eleven", "source": "openalex"})
        papers_repo.upsert(conn, {"openalex_id": v, "title": "eleven", "source": "user_pdf_version"})
        versions_repo.register(conn, user_id=uid, paper_id=v, canonical_id=w, file_hash="h11")
        papers_repo.record_upload(conn, user_id=uid, openalex_id=v, file_hash="h11", local_path="x.pdf", n_pages=3)
        pid = profiles_repo.upsert(conn, user_id=uid, name="s", embedding_model="placeholder-v1", n_seed=0,
                                   threshold=0.4)
        # Saved from the feed before it was uploaded: the same work.
        conn.execute("INSERT INTO profile_candidates (profile_id, openalex_id, score, saved_at) "
                     "VALUES (?, ?, 0.9, datetime('now'))", (pid, w))
        conn.commit()
        assert vault_repo.stats_for_user(conn, uid)["docs"] == 1
    finally:
        conn.close()


def test_a_file_added_beside_its_works_seed_leaves_the_fit_stale(env, monkeypatch):
    """The fit was made without the new file: the end-of-scan check must
    refit, also when the attach's own refit was skipped."""
    from rag_lib.api.services import wizard as wizard_service
    from rag_lib.scheduler import jobs

    conn = connect(env)
    try:
        uid = int(users_repo.get_by_email(conn, "demo@example.com")["id"])
        w = "https://openalex.org/W10"
        papers_repo.upsert(conn, {"openalex_id": w, "title": "ten", "source": "openalex"})
        v1, v2 = versions_repo.version_id(uid, "h1"), versions_repo.version_id(uid, "h2")
        for v, h in ((v1, "h1"), (v2, "h2")):
            papers_repo.upsert(conn, {"openalex_id": v, "title": "ten", "source": "user_pdf_version"})
            versions_repo.register(conn, user_id=uid, paper_id=v, canonical_id=w, file_hash=h)
        calls = []
        monkeypatch.setattr(wizard_service, "refit_profile", lambda *a, **k: calls.append(k) or {})
        for n, (old, new, cfg) in enumerate(((w, v1, {"type": "centroid", "fit_seed_ids": [w]}),
                              (v1, v2, {"type": "centroid", "fit_seed_ids": [v1]}),
                                              (w, v1, {"type": "centroid", "seed_matrix": [[0.0]]}))):  # an older fit
            pid = profiles_repo.upsert(conn, user_id=uid, name=f"fit {n}",
                                       embedding_model="placeholder-v1", n_seed=0, threshold=0.4,
                                       selector_config=cfg)
            profiles_repo.attach_seed(conn, pid, old)
            conn.execute("UPDATE profiles SET selector_config_json = ? WHERE id = ?", (json.dumps(cfg), pid))
            conn.commit()
            jobs._refit_if_fit_stale(conn, uid, pid, None, when="end")
            assert calls == [], cfg                       # current before
            profiles_repo.attach_seed(conn, pid, new)
            jobs._refit_if_fit_stale(conn, uid, pid, None, when="end")
            assert len(calls) == 1, cfg
            calls.clear()
    finally:
        conn.close()


def test_uploading_the_file_again_reads_a_version_without_its_text(pdfs, monkeypatch):
    a = _upload(pdfs, b"matched alice", A)
    vid = a["id"]
    conn = connect(pdfs["env"])
    try:
        # As a failed read leaves it: no text, marked unreadable, file intact.
        conn.execute("UPDATE papers SET body_text = NULL WHERE openalex_id = ?", (vid,))
        conn.execute("UPDATE paper_versions SET parsed = -1 WHERE version_id = ?", (vid,))
        conn.commit()
    finally:
        conn.close()
    again = _upload(pdfs, b"matched alice", A)
    conn = connect(pdfs["env"])
    try:
        assert again["id"] == vid and not again.get("duplicate")
        assert conn.execute("SELECT parsed FROM paper_versions WHERE version_id = ?", (vid,)).fetchone()[0] == 1
        assert conn.execute("SELECT body_text FROM papers WHERE openalex_id = ?", (vid,)).fetchone()[0]
    finally:
        conn.close()
    # With its text, the same bytes again are a duplicate.
    assert _upload(pdfs, b"matched alice", A).get("duplicate") is True


def test_the_vault_shows_a_work_once_with_all_its_interests(pdfs):
    from rag_lib.api.services import vault as vault_service
    from rag_lib.db.repos import vault as vault_repo

    conn = connect(pdfs["env"])
    try:
        papers_repo.upsert(conn, {"openalex_id": W, "title": "Public Title", "source": "openalex"})
        profiles_repo.attach_seed(conn, pdfs["live_id"], W)
        slug = profiles_repo.get(conn, pdfs["live_id"])["slug"]
    finally:
        conn.close()
    v = _upload(pdfs, b"matched alice", A)["id"]      # into another interest
    docs = _request(pdfs["app"], "GET", "/api/vault/docs", headers=A).json()
    mine = [d for d in docs if d["id"] in (W, v)]
    assert [d["id"] for d in mine] == [v] and slug in mine[0]["tags"]
    # The interest's own list shows the same entry, and counts it once.
    docs = _request(pdfs["app"], "GET", f"/api/vault/docs?tag={slug}", headers=A).json()
    assert [d["id"] for d in docs if d["id"] in (W, v)] == [v]
    conn = connect(pdfs["env"])
    try:
        from rag_lib.db.repos import vault as vault_repo2
        uid0 = int(users_repo.get_by_email(conn, A["X-User-Email"])["id"])
        assert vault_repo2.tag_counts_for_user(conn, uid0)[slug] == len(docs)
    finally:
        conn.close()
    conn = connect(pdfs["env"])
    try:
        uid = int(users_repo.get_by_email(conn, A["X-User-Email"])["id"])
        # Saved from the feed too, then seeded as the upload: one entry.
        conn.execute("INSERT INTO profile_candidates (profile_id, openalex_id, score, saved_at) "
                     "VALUES (?, ?, 0.9, datetime('now'))", (pdfs["live_id"], W))
        conn.commit()
        profiles_repo.attach_seed(conn, pdfs["live_id"], v)
        listed = vault_service.list_docs(conn, uid, tag=slug)
        assert vault_repo.tag_counts_for_user(conn, uid)[slug] == len(listed)
        assert [d["id"] for d in listed if d["id"] in (W, v)] == [v]
    finally:
        conn.close()


def test_the_least_alike_pair_names_an_uploads_work(env):
    from rag_lib.api.services import wizard as wizard_service

    conn = connect(env)
    try:
        uid = int(users_repo.get_by_email(conn, "demo@example.com")["id"])
        w, other = "https://openalex.org/W12", "https://openalex.org/W13"
        v = versions_repo.version_id(uid, "h12")
        for oid in (w, other, v):
            papers_repo.upsert(conn, {"openalex_id": oid, "title": oid[-6:], "source": "openalex"})
        versions_repo.register(conn, user_id=uid, paper_id=v, canonical_id=w, file_hash="h12")
        embeddings_repo.upsert(conn, v, "placeholder-v1", np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype="float32"))
        embeddings_repo.upsert(conn, other, "placeholder-v1", np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype="float32"))
        pid = profiles_repo.upsert(conn, user_id=uid, name="pair", embedding_model="placeholder-v1", n_seed=0,
                                   threshold=0.4)
        for oid in (v, other):
            profiles_repo.attach_seed(conn, pid, oid)
        slug = profiles_repo.get(conn, pid)["slug"]
        pair = wizard_service.compute_draft_coherence(conn, user_id=uid, slug=slug).least_similar
        named = {pair.a_id: pair.a_canonical_id, pair.b_id: pair.b_canonical_id}
        assert named == {v: w, other: None}
    finally:
        conn.close()


def test_attach_reads_the_works_seeds_under_the_write_lock(env, monkeypatch):
    """Two requests attaching two files of one work at once must not both
    see no other seed: the read is in the write transaction."""
    seen = []
    real = versions_repo.same_work_seeds
    monkeypatch.setattr(versions_repo, "same_work_seeds",
                        lambda conn, *a: seen.append(conn.in_transaction) or real(conn, *a))
    conn = connect(env)
    try:
        uid = int(users_repo.get_by_email(conn, "demo@example.com")["id"])
        papers_repo.upsert(conn, {"openalex_id": "https://openalex.org/W14", "title": "t", "source": "openalex"})
        pid = profiles_repo.upsert(conn, user_id=uid, name="lock", embedding_model="placeholder-v1", n_seed=0,
                                   threshold=0.4)
        conn.commit()
        profiles_repo.attach_seed(conn, pid, "https://openalex.org/W14")
        assert seen == [True] and not conn.in_transaction
        other = connect(env)
        try:    # and the lock is the write lock
            other.execute("PRAGMA busy_timeout = 0")
            monkeypatch.setattr(versions_repo, "same_work_seeds", lambda conn, *a: (
                pytest.raises(sqlite3.OperationalError, other.execute, "BEGIN IMMEDIATE"), [])[1])
            profiles_repo.attach_seed(conn, pid, "https://openalex.org/W14")
        finally:
            other.close()
    finally:
        conn.close()


def test_public_papers_that_lost_their_vector_are_embedded_again(tmp_path, monkeypatch):
    db, uid, pids = _legacy_world(tmp_path)
    w1 = "https://openalex.org/W1"
    conn = connect(db)
    try:
        apply_migrations(conn)
        # C seeds W1: its vector, made from A's text, is gone and queued.
        assert embeddings_repo.get(conn, w1, "placeholder-v1") is None
        assert [r[0] for r in conn.execute("SELECT openalex_id FROM papers_to_embed")] == [w1]
        texts = []
        import rag_lib.embedders as embedders_mod
        monkeypatch.setattr(embedders_mod, "get_embedder", lambda m: lambda t: texts.append(t) or [0.5] * 8)
        assert versions_repo.embed_queued(conn, default_model="placeholder-v1") == 1
        assert embeddings_repo.get(conn, w1, "placeholder-v1") is not None
        assert texts and "alice's text" not in texts[0]          # from the metadata
        assert conn.execute("SELECT COUNT(*) FROM papers_to_embed").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_version_follows_its_public_records_bibliography(env):
    conn = connect(env)
    try:
        uid = int(users_repo.get_by_email(conn, "demo@example.com")["id"])
        w, v = "https://openalex.org/W15", versions_repo.version_id(uid, "h15")
        papers_repo.upsert(conn, {"openalex_id": w, "title": "Public", "source": "openalex"})
        papers_repo.upsert(conn, {"openalex_id": v, "title": "Mine", "source": "user_pdf_version"})
        versions_repo.register(conn, user_id=uid, paper_id=v, canonical_id=w, file_hash="h15")
        conn.commit()
        papers_repo.upsert(conn, {"openalex_id": w, "title": "Public renamed", "source": "openalex",
                                  "authors": ["Ada"], "year": 2020, "publication_date": "2020-01-02"})
        row = conn.execute("SELECT title, authors_json, year, publication_date FROM papers WHERE openalex_id = ?",
                           (v,)).fetchone()
        assert (row[0], json.loads(row[1]), row[2], row[3]) == ("Mine", ["Ada"], 2020, "2020-01-02")
    finally:
        conn.close()


def test_a_repaired_upload_reads_its_topics_from_its_text(pdfs):
    v = _upload(pdfs, b"nothing on openalex", A)["id"]
    conn = connect(pdfs["env"])
    try:
        # As a read that failed left it: no text, topics from the title alone.
        title_only = json.dumps({"source": "pdf_content", "topics": [{"id": "T-title", "display_name": "x"}]})
        conn.execute("UPDATE papers SET body_text = NULL, topics_json = ? WHERE openalex_id = ?", (title_only, v))
        conn.execute("UPDATE paper_versions SET parsed = -1 WHERE version_id = ?", (v,))
        conn.commit()
    finally:
        conn.close()
    _upload(pdfs, b"nothing on openalex", A)
    raw = _row(pdfs["env"], v)["topics_json"]
    assert [t["id"] for t in json.loads(raw)["topics"]] == ["T-content"]


def test_a_profile_file_naming_another_users_upload_is_refused_before_any_write(tmp_path):
    """Codex review, MEDIUM 1: the refusal came from the seed trigger, after
    the paper rows (another user's text) had been overwritten and committed."""
    from rag_lib.paper import Paper
    from rag_lib.persistence import store_profile_from_object
    from rag_lib.profile import Profile

    conn = connect(tmp_path / "foreign.db")
    try:
        apply_migrations(conn)
        a, b = (int(users_repo.upsert(conn, f"{n}@example.com")["id"]) for n in "ab")
        mine = Paper(doi=None, openalex_id="https://openalex.org/W9", title="A title", body_text="A private text",
                     embeddings={"placeholder-v1": [1.0] * 8})
        store_profile_from_object(conn, user_id=a, profile=Profile(name="A interest", papers=[mine]))
        vid = mine.openalex_id
        snapshot = lambda: (tuple(conn.execute("SELECT * FROM papers WHERE openalex_id = ?", (vid,)).fetchone()),  # noqa: E731
                            embeddings_repo.get(conn, vid, "placeholder-v1").tolist())
        before = snapshot()
        n_profiles = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
        for who, foreign_id in ((b, vid), (b, versions_repo.version_id(a, "not-yet-stored")),
                                (a, versions_repo.version_id(a, "not-yet-stored"))):    # nobody's, not even A's
            foreign = Paper(doi=None, openalex_id=foreign_id, title="B title", body_text="B replacement",
                            embeddings={"placeholder-v1": [0.0] * 8})
            with pytest.raises(ValueError, match="not one of your uploads"):
                store_profile_from_object(conn, user_id=who, profile=Profile(name="B interest", papers=[foreign]))
        # A's own registered version, imported again by A, is fine.
        again = Paper(doi=None, openalex_id=vid, title="A title", body_text="A private text",
                      embeddings={"placeholder-v1": [1.0] * 8})
        store_profile_from_object(conn, user_id=a, profile=Profile(name="A again", papers=[again]))
        n_profiles += 1
        assert snapshot() == before
        assert conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == n_profiles
        assert conn.execute("SELECT 1 FROM papers WHERE openalex_id = ?",
                            (versions_repo.version_id(a, "not-yet-stored"),)).fetchone() is None
    finally:
        conn.close()


def test_a_public_paper_embedded_again_rescores_the_candidates_holding_it(tmp_path, monkeypatch):
    """Codex review, MEDIUM 2: the new vector was made but the candidate kept
    the score made from the old one (an upload's text)."""
    db, uid, pids = _legacy_world(tmp_path)
    w1 = "https://openalex.org/W1"
    conn = connect(db)
    try:
        pid = pids["b"]      # another user's interest, holding W1 as a candidate
        cfg = json.loads(conn.execute("SELECT selector_config_json FROM profiles WHERE id = ?", (pid,)).fetchone()[0])
        cfg.update(centroid=[1.0] + [0.0] * 7, seed_matrix=[[1.0] + [0.0] * 7])
        conn.execute("UPDATE profiles SET selector_config_json = ? WHERE id = ?", (json.dumps(cfg), pid))
        candidates_repo.insert_dedup(conn, profile_id=pid, openalex_id=w1, score=0.77, score_raw=0.77)
        apply_migrations(conn)
        import rag_lib.embedders as embedders_mod
        monkeypatch.setattr(embedders_mod, "get_embedder", lambda m: lambda t: [0.0, 1.0] + [0.0] * 6)
        assert versions_repo.embed_queued(conn, default_model="placeholder-v1") == 1
        score = lambda: conn.execute("SELECT score_raw FROM profile_candidates WHERE profile_id = ? AND openalex_id = ?",  # noqa: E731
                                     (pid, w1)).fetchone()[0]
        assert score() == pytest.approx(0.0)          # from the new vector
        assert conn.execute("SELECT COUNT(*) FROM papers_to_embed").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_rescore_that_fails_keeps_the_paper_queued(tmp_path, monkeypatch):
    db, uid, pids = _legacy_world(tmp_path)
    w1 = "https://openalex.org/W1"
    conn = connect(db)
    try:
        cfg = json.loads(conn.execute("SELECT selector_config_json FROM profiles WHERE id = ?", (pids["b"],)).fetchone()[0])
        cfg.update(centroid=[1.0] + [0.0] * 7, seed_matrix=[[1.0] + [0.0] * 7])
        conn.execute("UPDATE profiles SET selector_config_json = ? WHERE id = ?", (json.dumps(cfg), pids["b"]))
        candidates_repo.insert_dedup(conn, profile_id=pids["b"], openalex_id=w1, score=0.77, score_raw=0.77)
        apply_migrations(conn)
        import rag_lib.embedders as embedders_mod
        import rag_lib.scoring.rescore as rescore_mod
        monkeypatch.setattr(embedders_mod, "get_embedder", lambda m: lambda t: [0.5] * 8)

        def boom(conn, profile_id):
            raise RuntimeError("scorer down")
        monkeypatch.setattr(rescore_mod, "rescore_candidates", boom)
        versions_repo.embed_queued(conn, default_model="placeholder-v1")
        assert [r[0] for r in conn.execute("SELECT openalex_id FROM papers_to_embed")] == [w1]
        monkeypatch.undo()
        monkeypatch.setattr(embedders_mod, "get_embedder", lambda m: lambda t: [0.5] * 8)
        versions_repo.embed_queued(conn, default_model="placeholder-v1")      # next sweep
        assert conn.execute("SELECT COUNT(*) FROM papers_to_embed").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_repair_openalex_no_longer_matches_keeps_the_confirmed_match(pdfs, monkeypatch):
    """Codex review, LOW 3: the title was overwritten with the page's while
    the vector made from the confirmed title stayed. (OpenAlex answering
    "no match" now; not answering is refused, see the test below.)"""
    from rag_lib.api.services import vault as vault_service

    first = _upload(pdfs, b"matched original", A)
    conn = connect(pdfs["env"])
    try:
        row = dict(conn.execute("SELECT * FROM papers WHERE openalex_id = ?", (first["id"],)).fetchone())
        vec = lambda: conn.execute("SELECT vector FROM paper_embeddings WHERE openalex_id = ?",  # noqa: E731
                                   (first["id"],)).fetchone()
        old_vector = bytes(vec()[0])
    finally:
        conn.close()
    Path(row["local_path"]).unlink()
    monkeypatch.setattr(vault_service, "_enrich_via_openalex", lambda record, *, mailto, filename, file_hash: (
        f"local:{file_hash}", {"openalex_id": f"local:{file_hash}", "title": record.title, "source": "user_pdf"}))
    repaired = _upload(pdfs, b"matched original", A)
    after = _row(pdfs["env"], first["id"])
    assert repaired["canonicalId"] == W and not repaired.get("duplicate")
    assert [after[k] for k in ("title", "abstract", "doi", "year", "topics_json")] == \
        [row[k] for k in ("title", "abstract", "doi", "year", "topics_json")]
    conn = connect(pdfs["env"])
    try:
        assert bytes(vec()[0]) == old_vector               # nothing it was made from changed
    finally:
        conn.close()


def test_a_reupload_that_changes_what_the_vector_is_made_of_makes_a_new_one(pdfs, monkeypatch):
    from rag_lib.api.services import vault as vault_service

    first = _upload(pdfs, b"matched original", A)
    conn = connect(pdfs["env"])
    try:
        vec = lambda: conn.execute("SELECT vector FROM paper_embeddings WHERE openalex_id = ?",  # noqa: E731
                                   (first["id"],)).fetchone()
        old_vector = bytes(vec()[0])
        conn.execute("UPDATE papers SET abstract = 'an abstract from before' WHERE openalex_id = ?", (first["id"],))
        conn.commit()
    finally:
        conn.close()
    Path(_row(pdfs["env"], first["id"])["local_path"]).unlink()
    texts = []
    import rag_lib.api.services.vault as vs
    real = vs.get_embedder
    monkeypatch.setattr(vs, "get_embedder", lambda m: (lambda e: (lambda t: texts.append(t) or e(t)))(real(m)))
    _upload(pdfs, b"matched original", A)      # OpenAlex's abstract again replaces the edited one
    conn = connect(pdfs["env"])
    try:
        assert texts and "public abstract" in texts[0]
        assert vec() is not None
    finally:
        conn.close()


def test_a_version_embedded_when_openalex_no_longer_matches_uses_its_confirmed_metadata(pdfs, monkeypatch):
    from rag_lib.api.services import vault as vault_service

    first = _upload(pdfs, b"matched original", A)
    conn = connect(pdfs["env"])
    try:
        # A version whose read failed: no text, no vector, its file intact.
        conn.execute("UPDATE papers SET body_text = NULL WHERE openalex_id = ?", (first["id"],))
        conn.execute("DELETE FROM paper_embeddings WHERE openalex_id = ?", (first["id"],))
        conn.execute("UPDATE paper_versions SET parsed = -1 WHERE version_id = ?", (first["id"],))
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(vault_service, "_enrich_via_openalex", lambda record, *, mailto, filename, file_hash: (
        f"local:{file_hash}", {"openalex_id": f"local:{file_hash}", "title": record.title, "source": "user_pdf"}))
    texts = []
    real = vault_service.get_embedder
    monkeypatch.setattr(vault_service, "get_embedder", lambda m: (lambda e: (lambda t: texts.append(t) or e(t)))(real(m)))
    again = _upload(pdfs, b"matched original", A)
    assert again["id"] == first["id"] and again["canonicalId"] == W
    assert len(texts) == 1 and "Public Title" in texts[0] and "public abstract" in texts[0]
    assert "body of" in texts[0]                       # and its text, read again


def test_a_paper_whose_rescore_keeps_failing_does_not_hold_up_the_queue(tmp_path, monkeypatch):
    db, uid, pids = _legacy_world(tmp_path)
    conn = connect(db)
    try:
        apply_migrations(conn)
        w1, w2 = "https://openalex.org/W1", "https://openalex.org/W2"
        conn.execute("INSERT OR IGNORE INTO papers_to_embed (openalex_id) VALUES (?)", (w2,))
        candidates_repo.insert_dedup(conn, profile_id=pids["b"], openalex_id=w1, score=0.5, score_raw=0.5)
        conn.commit()
        import rag_lib.embedders as embedders_mod
        import rag_lib.scoring.rescore as rescore_mod
        monkeypatch.setattr(embedders_mod, "get_embedder", lambda m: lambda t: [0.5] * 8)

        def boom(conn, profile_id):
            raise RuntimeError("this interest's fit cannot score")
        monkeypatch.setattr(rescore_mod, "rescore_candidates", boom)
        queued = lambda: sorted(r[0] for r in conn.execute("SELECT openalex_id FROM papers_to_embed"))  # noqa: E731
        assert queued() == [w1, w2]
        versions_repo.embed_queued(conn, default_model="placeholder-v1", limit=1)    # W1 first, fails
        assert queued() == [w1, w2]
        versions_repo.embed_queued(conn, default_model="placeholder-v1", limit=1)    # W2 gets its turn
        assert queued() == [w1]
    finally:
        conn.close()


# --- OpenAlex not answering (FUNCTIONAL_SPEC D2, U11) -------------------------

def _state(env):
    conn = connect(env)
    try:
        return {t: conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall()
                for t in ("papers", "paper_versions", "paper_uploads", "profile_seeds", "paper_embeddings")}
    finally:
        conn.close()


def _files(world):
    from rag_lib.api import settings as settings_module
    root = Path(settings_module.get_settings().RADAR_VAULT_DIR)
    return sorted(str(p) for p in root.rglob("*") if p.is_file())


@pytest.mark.parametrize("failure", ["quota", "down"])
def test_an_upload_openalex_cannot_answer_stores_nothing(pdfs, monkeypatch, failure):
    from rag_lib import openalex_status
    from rag_lib.api.services import vault as vault_service

    slug = _inbox(pdfs, A)
    before, files = _state(pdfs["env"]), _files(pdfs)

    def enrich(record, *, mailto, filename, file_hash):
        if failure == "quota":
            raise openalex_status.OpenAlexUnavailable("quota", until=openalex_status.pause("quota", reset_seconds=600))
        raise openalex_status.OpenAlexUnavailable("down", detail="ReadTimeout")
    monkeypatch.setattr(vault_service, "_enrich_via_openalex", enrich)
    r = _post(pdfs, b"matched never stored", A, slug)
    assert r.status_code == 503, r.text
    assert "Nothing was changed" in r.json()["detail"]
    assert _state(pdfs["env"]) == before and _files(pdfs) == files
    status = _request(pdfs["app"], "GET", "/api/openalex/status", headers=A).json()
    assert status["available"] is (failure != "quota")
    if failure == "quota":
        # Paused: the next upload is refused before anything is read.
        monkeypatch.setattr(vault_service, "ingest_pdf", lambda p: pytest.fail("read while paused"))
        r = _post(pdfs, b"matched later", A, slug)
        assert r.status_code == 503 and r.json()["openalex"]["available"] is False
        assert _state(pdfs["env"]) == before and _files(pdfs) == files


def test_a_repair_openalex_cannot_answer_changes_nothing(pdfs, monkeypatch):
    from rag_lib import openalex_status
    from rag_lib.api.services import vault as vault_service

    first = _upload(pdfs, b"matched original", A)
    Path(_row(pdfs["env"], first["id"])["local_path"]).unlink()
    before, files = _state(pdfs["env"]), _files(pdfs)
    monkeypatch.setattr(vault_service, "_enrich_via_openalex", lambda *a, **k: (_ for _ in ()).throw(
        openalex_status.OpenAlexUnavailable("down")))
    assert _post(pdfs, b"matched original", A).status_code == 503
    assert _state(pdfs["env"]) == before and _files(pdfs) == files


def test_the_client_tells_not_answering_from_no_match(monkeypatch):
    """Only an answer from OpenAlex is "not found": a 429, 401, 5xx, timeout
    or non-JSON 200 raise OpenAlexUnavailable; a 404 is None; 429 and 401
    pause every later call without sending it."""
    import requests
    from rag_lib import openalex_status
    from rag_lib.openalex_client import OpenAlexClient

    class Resp:
        def __init__(self, code, headers=None, body=None):
            self.status_code, self.headers, self._body, self.text = code, headers or {}, body, "x"

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(response=self)

        def json(self):
            if self._body is None:
                raise ValueError("not json")
            return self._body

    sent = []

    class Session:
        headers: dict = {}

        def __init__(self, answer):
            self.answer = answer

        def get(self, url, params=None, timeout=None):
            sent.append(url)
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer

    def client(answer):
        return OpenAlexClient(mailto="t@example.com", session=Session(answer), rate_limit_sleep=0)

    assert client(Resp(404)).lookup_by_doi("10.1/x") is None
    assert client(Resp(200, body={"results": []})).search_by_title("x") == []
    for answer, kind in ((Resp(503), "down"), (requests.ReadTimeout("slow"), "down"),
                         (requests.ConnectionError("no route"), "down"), (Resp(200), "down")):
        with pytest.raises(openalex_status.OpenAlexUnavailable) as e:
            client(answer).lookup_by_doi("10.1/x")
        assert e.value.kind == kind and openalex_status.current() is None
    with pytest.raises(openalex_status.OpenAlexUnavailable) as e:
        client(Resp(429, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "3600"})).lookup_by_doi("10.1/x")
    assert e.value.kind == "quota" and 3500 < openalex_status.current()["until"] - __import__("time").time() <= 3600
    n = len(sent)
    with pytest.raises(openalex_status.OpenAlexUnavailable):
        client(Resp(200, body={})).lookup_by_doi("10.1/y")
    assert len(sent) == n                               # paused: not sent
    openalex_status.reset()
    with pytest.raises(openalex_status.OpenAlexUnavailable) as e:
        client(Resp(401)).get_work("W1")
    assert e.value.kind == "auth" and openalex_status.current()["kind"] == "auth"
    openalex_status.reset()
    # A burst 429 (quota left) pauses only briefly.
    with pytest.raises(openalex_status.OpenAlexUnavailable):
        client(Resp(429, {"X-RateLimit-Remaining": "5"})).lookup_by_doi("10.1/x")
    assert openalex_status.current()["until"] - __import__("time").time() <= openalex_status.BURST_PAUSE_SECONDS


def test_an_openalex_refusal_of_the_query_is_no_match(monkeypatch):
    """A 400 for a title OpenAlex cannot parse is an answer: not found."""
    import requests
    from rag_lib.api.services import vault as vault_service
    from rag_lib.vault import PdfIngestRecord

    class Refusing:
        def __init__(self, **_): pass
        def lookup_by_doi(self, doi): return None
        def search_by_title(self, *a, **k):
            raise requests.HTTPError(response=type("R", (), {"status_code": 400})())
    monkeypatch.setattr(vault_service, "OpenAlexClient", Refusing)
    record = PdfIngestRecord(path="x", title="Odd ((title", body_text="body", doi=None, n_pages=1)
    oid, d = vault_service._enrich_via_openalex(record, mailto="t@example.com", filename="x.pdf", file_hash="h")
    assert oid == "local:h" and d["title"] == "Odd ((title"


def test_files_of_one_paper_are_numbered_in_upload_order(env):
    """FUNCTIONAL_SPEC 1.3/1.7: the same paper is the same OpenAlex work, or
    for a file OpenAlex has no record of, the same title (case, accents and
    punctuation aside) -- not a title that is only the file's name, nor
    "Untitled PDF". Numbers follow this user's upload times; other users'
    files and public records take none."""
    conn = connect(env)
    try:
        uid = int(users_repo.get_by_email(conn, "demo@example.com")["id"])
        other = users_repo.upsert(conn, "other@example.com")
        other = int(users_repo.get_by_email(conn, "other@example.com")["id"])
        w = "https://openalex.org/W55"
        papers_repo.upsert(conn, {"openalex_id": w, "title": "Fifty Five", "source": "openalex"})

        def add(user, h, title, canonical=None, *, at, from_file=False):
            v = versions_repo.version_id(user, h)
            papers_repo.upsert(conn, {"openalex_id": v, "title": title, "source": "user_pdf_version"})
            versions_repo.register(conn, user_id=user, paper_id=v, canonical_id=canonical, file_hash=h,
                                   title_from_file=from_file)
            papers_repo.record_upload(conn, user_id=user, openalex_id=v, file_hash=h, local_path=None, n_pages=1)
            conn.execute("UPDATE paper_uploads SET uploaded_at = ? WHERE openalex_id = ?", (at, v))
            return v

        late = add(uid, "h1", "Fifty Five", w, at="2026-10-02 00:00:00")
        early = add(uid, "h2", "Fifty Five", w, at="2026-10-01 00:00:00")
        page = add(uid, "h3", "fifty-five!", None, at="2026-10-03 00:00:00")     # unmatched, same title
        m1 = add(uid, "h4", "Lab Notebook: Draft", None, at="2026-10-04 00:00:00")
        m2 = add(uid, "h5", "lab notebook draft", None, at="2026-10-05 00:00:00")
        f1 = add(uid, "h6", "scan", None, at="2026-10-06 00:00:00", from_file=True)
        f2 = add(uid, "h7", "scan", None, at="2026-10-07 00:00:00", from_file=True)
        u1 = add(uid, "h8", "Untitled PDF", None, at="2026-10-08 00:00:00")
        u2 = add(uid, "h9", "Untitled PDF", None, at="2026-10-09 00:00:00")
        solo = add(uid, "h10", "Only One", None, at="2026-10-10 00:00:00")
        # Within one second: the order they came in, not their hashes'.
        s1 = add(uid, "zz", "Same Second", None, at="2026-10-11 00:00:00")
        s2 = add(uid, "aa", "Same Second", None, at="2026-10-11 00:00:00")
        theirs = add(other, "h1", "Fifty Five", w, at="2026-09-01 00:00:00")
        conn.commit()
        n = versions_repo.version_numbers(conn, uid)
        assert {k: v[0] for k, v in n.items()} == {early: 1, late: 2, page: 3, m1: 1, m2: 2, s1: 1, s2: 2}
        assert n[early][1].startswith("2026-10-01")
        for v in (f1, f2, u1, u2, solo, theirs):
            assert v not in n
        assert versions_repo.version_numbers(conn, other) == {}
    finally:
        conn.close()


def test_the_seeds_list_and_the_vault_show_version_numbers(pdfs):
    # A fresh draft: the world's live interest has hand-made 64-d seed
    # vectors, which do not mix with the uploads' model vectors.
    r = _request(pdfs["app"], "POST", "/api/profiles/draft", json={"name": "Versions"}, headers=A)
    live = r.json()["slug"]
    v1 = _upload(pdfs, b"matched draft one", A, slug=live)
    v2 = _upload(pdfs, b"matched draft two", A, slug=live)
    assert (v1["version"], v2["version"]) == (None, 2)
    detail = _request(pdfs["app"], "GET", f"/api/profiles/{live}/detail", headers=A).json()
    seeds = {s["id"]: s for s in detail["seeds"]}
    assert (seeds[v1["id"]]["version"], seeds[v2["id"]]["version"]) == (1, 2)
    assert seeds[v2["id"]]["uploaded"]
    docs = {d["id"]: d for d in _request(pdfs["app"], "GET", f"/api/vault/docs?tag={live}", headers=A).json()}
    assert (docs[v1["id"]]["version"], docs[v2["id"]]["version"]) == (1, 2)


def test_an_upload_must_name_an_interest(pdfs):
    """FUNCTIONAL_SPEC U13: no upload outside an interest; nothing stored."""
    before, files = _state(pdfs["env"]), _files(pdfs)
    files_ = {"file": ("p.pdf", io.BytesIO(b"%PDF-1.4\nmatched loose"), "application/pdf")}
    r = _request(pdfs["app"], "POST", "/api/vault/upload", files=files_, headers=A)
    assert r.status_code == 400, r.text
    assert _state(pdfs["env"]) == before and _files(pdfs) == files


def test_forgetting_a_versions_fit_makes_every_fit_with_it_stale(env):
    """A re-upload whose vector input changed (``forget_fit``): fits that
    recorded the version drop it, an older fit with no record loses its
    matrix (so the scan refits), and a fit without the version is kept."""
    conn = connect(env)
    try:
        uid = int(users_repo.get_by_email(conn, "demo@example.com")["id"])
        v = versions_repo.version_id(uid, "hf")
        papers_repo.upsert(conn, {"openalex_id": v, "title": "f", "source": "user_pdf_version"})
        versions_repo.register(conn, user_id=uid, paper_id=v, canonical_id=None, file_hash="hf")
        cfgs = {"recorded": {"type": "centroid", "fit_seed_ids": [v, "W1"]},
                "legacy": {"type": "centroid", "seed_matrix": [[0.0]]},
                "other": {"type": "centroid", "fit_seed_ids": ["W1"]}}
        pids = {}
        for name, cfg in cfgs.items():
            pids[name] = profiles_repo.upsert(conn, user_id=uid, name=f"ff {name}", embedding_model="placeholder-v1",
                                              n_seed=0, threshold=0.4, selector_config=cfg)
            if name != "other":
                profiles_repo.attach_seed(conn, pids[name], v)
            conn.execute("UPDATE profiles SET selector_config_json = ? WHERE id = ?", (json.dumps(cfg), pids[name]))
        conn.commit()
        versions_repo.forget_fit(conn, v)
        conn.commit()
        got = {n: json.loads(conn.execute("SELECT selector_config_json FROM profiles WHERE id = ?", (p,)).fetchone()[0])
               for n, p in pids.items()}
        assert got["recorded"]["fit_seed_ids"] == ["W1"]
        assert "seed_matrix" not in got["legacy"]
        assert got["other"] == cfgs["other"]
    finally:
        conn.close()


def test_a_pause_ends_at_its_time():
    from rag_lib import openalex_status

    now = 1_000_000.0
    until = openalex_status.pause("quota", reset_seconds=30, now=now)
    assert until == now + 30
    assert openalex_status.current(now=now + 29)["kind"] == "quota"
    assert openalex_status.current(now=now + 30) is None          # over: OpenAlex is asked again
    assert openalex_status.current(now=now + 1) is None           # and stays forgotten
    # Without a time from OpenAlex: until the next UTC midnight.
    day = 86400 * 20000
    assert openalex_status.pause("auth", now=day + 3600) == day + 86400
