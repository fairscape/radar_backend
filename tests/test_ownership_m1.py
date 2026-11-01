"""Each user's papers are theirs (FUNCTIONAL_SPEC M.1).

A public work's row holds what OpenAlex says of it and is shared; what a
user's own source adds -- a Prosopia abstract OpenAlex has none for, the
summary of a record OpenAlex has nothing for -- is stored as theirs, so
one user's import, refresh or delete never changes another's papers.
"""

from __future__ import annotations

from rag_lib.db import connect
from rag_lib.db.repos import users as users_repo
from tests.test_prosopia_import import (
    RECORD_DOI_ONLY,
    RECORD_NOTHING,
    StubOpenAlex,
    StubProfile,
    StubProsopia,
    StubSummaries,
    _record,
    _work,
)
from tests.test_private_upload_versions import pdfs  # noqa: F401 -- fixture
from tests.test_researchers import _app, _import, _request, env  # noqa: F401 -- fixture
from tests.test_review_round12_isolation import world  # noqa: F401 -- fixture

B = {"X-User-Email": "someone-else@example.com"}


def _import_as(app, headers, **body):
    return _request(app, "POST", "/api/researchers/import", json=body, headers=headers)
W = "https://openalex.org/W4386953349"
PROSOPIA_ABSTRACT = "Prosopia's own abstract of the renin paper."


def _q(env, sql, *params):
    conn = connect(env)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _uid(env, email):
    conn = connect(env)
    try:
        return int(users_repo.get_by_email(conn, email)["id"])
    finally:
        conn.close()


def _openalex_without_abstract():
    work = _work(RECORD_DOI_ONLY.doi, "W4386953349", RECORD_DOI_ONLY.name, 2023)
    work.pop("abstract_inverted_index", None)
    return StubOpenAlex(by_doi={RECORD_DOI_ONLY.doi: work})


def _with_abstract():
    return _record(**{**RECORD_DOI_ONLY.model_dump(exclude_none=True), "abstract": PROSOPIA_ABSTRACT})


def _vector(env, oa_id):
    rows = _q(env, "SELECT vector FROM paper_embeddings WHERE openalex_id = ?", oa_id)
    return rows[0][0] if rows else None


# --- M-1: a Prosopia abstract goes on the importer's own copy ----------------------------


def test_m1_a_prosopia_abstract_is_the_importers_copy_and_leaves_the_public_row(env):
    oa = _openalex_without_abstract()
    plain = StubProfile("Nathan C. Sheffield", [RECORD_DOI_ONLY], StubSummaries({}))
    a = _import(_app(StubProsopia(plain), oa), source="prosopia", ref="sheffield-nathan")
    assert a.status_code == 200, a.text
    assert _q(env, "SELECT openalex_id FROM researcher_papers") == [(W,)]
    before = (_q(env, "SELECT abstract, title FROM papers WHERE openalex_id = ?", W), _vector(env, W))
    assert before[1] is not None

    rich = StubProfile("Nathan C. Sheffield", [_with_abstract()], StubSummaries({}))
    b = _import_as(_app(StubProsopia(rich), oa), B, source="prosopia", ref="sheffield-nathan")
    assert b.status_code == 200, b.text
    uid = _uid(env, B["X-User-Email"])
    copy = f"local:prosopia:{uid}:W4386953349"

    # A's paper: the public row and its vector, as they were.
    assert (_q(env, "SELECT abstract, title FROM papers WHERE openalex_id = ?", W), _vector(env, W)) == before
    # B's: their own copy, with Prosopia's abstract, embedded from it.
    assert _q(env, "SELECT abstract FROM papers WHERE openalex_id = ?", copy) == [(PROSOPIA_ABSTRACT,)]
    assert _q(env, "SELECT user_id, canonical_openalex_id FROM paper_versions WHERE version_id = ?",
              copy) == [(uid, W)]
    assert _vector(env, copy) is not None and _vector(env, copy) != before[1]
    links = _q(env, "SELECT r.user_id, p.openalex_id FROM researcher_papers p "
                    "JOIN researchers r ON r.id = p.researcher_id ORDER BY r.user_id")
    assert links == [(_uid(env, "demo@example.com"), W), (uid, copy)]


def test_m1_an_abstract_openalex_has_needs_no_copy(env):
    work = _work(RECORD_DOI_ONLY.doi, "W4386953349", RECORD_DOI_ONLY.name, 2023)
    oa = StubOpenAlex(by_doi={RECORD_DOI_ONLY.doi: work})
    rich = StubProfile("Nathan C. Sheffield", [_with_abstract()], StubSummaries({}))
    assert _import(_app(StubProsopia(rich), oa), source="prosopia", ref="sheffield-nathan").status_code == 200
    assert _q(env, "SELECT openalex_id FROM researcher_papers") == [(W,)]
    assert _q(env, "SELECT abstract FROM papers WHERE openalex_id = ?", W) == [("an abstract",)]
    assert _q(env, "SELECT COUNT(*) FROM paper_versions") == [(0,)]


def test_m1_the_copy_is_a_seed_and_the_vault_lists_the_work_once(env):
    oa = _openalex_without_abstract()
    rich = StubProfile("Nathan C. Sheffield", [_with_abstract()], StubSummaries({}))
    r = _request(_app(StubProsopia(rich), oa), "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan"})
    assert r.status_code == 200, r.text
    uid = _uid(env, "demo@example.com")
    copy = f"local:prosopia:{uid}:W4386953349"
    assert _q(env, "SELECT openalex_id FROM profile_seeds") == [(copy,)]
    docs = _request(_app(), "GET", "/api/vault/docs").json()
    docs = docs.get("docs", docs) if isinstance(docs, dict) else docs
    assert [(d["id"], d["canonicalId"]) for d in docs] == [(copy, W)]


def test_m1_discarding_the_draft_takes_the_copy_and_its_new_public_row(env):
    oa = _openalex_without_abstract()
    rich = StubProfile("Nathan C. Sheffield", [_with_abstract()], StubSummaries({}))
    app = _app(StubProsopia(rich), oa)
    slug = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan"}).json()["draft_slug"]
    assert _q(env, "SELECT COUNT(*) FROM papers") == [(2,)]
    assert _request(app, "DELETE", f"/api/profiles/draft/{slug}").status_code in (200, 204)
    assert _q(env, "SELECT COUNT(*) FROM papers") == [(0,)]
    assert _q(env, "SELECT COUNT(*) FROM paper_versions") == [(0,)]


# --- M-2, M-3: a record OpenAlex has nothing for is each user's own ----------------------


def test_m2_m3_an_unmatched_record_is_a_row_per_user_written_from_their_own_import(env):
    """Before M.1 the two users shared ``prosopia:<slug>:<id>``: the second
    importer got the first one's row -- the summary as it was then -- and
    their own import of the profile could not change it."""
    oa = StubOpenAlex()
    old = StubProfile("Nathan C. Sheffield", [RECORD_NOTHING], StubSummaries({"ghost2020nowhere": "First summary."}))
    assert _import(_app(StubProsopia(old), oa), source="prosopia", ref="sheffield-nathan").status_code == 200
    new = StubProfile("Nathan C. Sheffield", [RECORD_NOTHING], StubSummaries({"ghost2020nowhere": "Revised summary."}))
    b = _import_as(_app(StubProsopia(new), oa), B, source="prosopia", ref="sheffield-nathan")
    assert b.status_code == 200, b.text
    ua, ub = _uid(env, "demo@example.com"), _uid(env, B["X-User-Email"])
    mine = f"prosopia:{ua}:sheffield-nathan:ghost2020nowhere"
    theirs = f"prosopia:{ub}:sheffield-nathan:ghost2020nowhere"
    body = {oa_id: text for oa_id, text in _q(env, "SELECT openalex_id, body_text FROM papers")}
    assert set(body) == {mine, theirs}
    assert "First summary." in body[mine] and "Revised" not in body[mine]
    assert "Revised summary." in body[theirs] and "First" not in body[theirs]
    links = _q(env, "SELECT r.user_id, p.openalex_id FROM researcher_papers p "
                    "JOIN researchers r ON r.id = p.researcher_id ORDER BY r.user_id")
    assert links == [(ua, mine), (ub, theirs)]


# --- M-4: one user's delete leaves the other's papers ------------------------------------


def test_m4_deleting_a_researcher_leaves_the_other_users_paper_and_vector(env):
    oa = _openalex_without_abstract()
    plain = StubProfile("Nathan C. Sheffield", [RECORD_DOI_ONLY], StubSummaries({}))
    app = _app(StubProsopia(plain), oa)
    a = _import(app, source="prosopia", ref="sheffield-nathan").json()["researcher_id"]
    assert _import_as(app, B, source="prosopia", ref="sheffield-nathan").status_code == 200
    vector = _vector(env, W)
    assert _request(app, "DELETE", f"/api/researchers/{a}").status_code in (200, 204)
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(1,)]
    assert _q(env, "SELECT COUNT(*) FROM papers WHERE openalex_id = ?", W) == [(1,)]
    assert _vector(env, W) == vector


def test_m4_discarding_a_draft_leaves_the_paper_another_user_imported(env):
    oa = _openalex_without_abstract()
    plain = StubProfile("Nathan C. Sheffield", [RECORD_DOI_ONLY], StubSummaries({}))
    app = _app(StubProsopia(plain), oa)
    slug = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan"}).json()["draft_slug"]
    other = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan"}, headers=B)
    assert other.status_code == 200, other.text
    vector = _vector(env, W)
    assert _request(app, "DELETE", f"/api/profiles/draft/{slug}").status_code in (200, 204)
    assert _q(env, "SELECT COUNT(*) FROM profile_seeds WHERE openalex_id = ?", W) == [(1,)]
    assert _q(env, "SELECT COUNT(*) FROM papers WHERE openalex_id = ?", W) == [(1,)]
    assert _vector(env, W) == vector


# --- M-5: uploads of the same paper by two users -----------------------------------------


def test_m5_discarding_one_users_upload_leaves_the_other_users(pdfs):
    from tests.test_private_upload_versions import A as UA, B as UB, W as UW, _inbox, _row, _upload
    from tests.test_private_upload_versions import _vector as upload_vector

    env = pdfs["env"]
    a = _upload(pdfs, b"matched alice", UA)
    b = _upload(pdfs, b"matched bob", UB)
    their = (_row(env, b["id"])["body_text"], upload_vector(env, b["id"]))
    assert _request(pdfs["app"], "DELETE", f"/api/profiles/draft/{_inbox(pdfs, UA)}",
                    headers=UA).status_code in (200, 204)
    assert _row(env, a["id"]) is None
    assert _row(env, b["id"])["body_text"] == their[0] and (upload_vector(env, b["id"]) == their[1]).all()
    assert _row(env, UW) is not None      # still B's upload's work


def test_m1_a_failure_after_the_copy_is_stored_takes_back_the_copy_and_its_public_row(env, monkeypatch):
    from rag_lib.db.repos import profiles as profiles_repo

    oa = _openalex_without_abstract()
    rich = StubProfile("Nathan C. Sheffield", [_with_abstract()], StubSummaries({}))
    app = _app(StubProsopia(rich), oa)
    d = _request(app, "POST", "/api/profiles/draft", json={"name": "Mine"}).json()["slug"]

    def broken(conn, *a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(profiles_repo, "attach_seed", broken)
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "into": d})
    assert r.status_code == 500
    assert _q(env, "SELECT COUNT(*) FROM papers") == [(0,)]
    assert _q(env, "SELECT COUNT(*) FROM paper_versions") == [(0,)]
    assert _q(env, "SELECT COUNT(*) FROM paper_embeddings") == [(0,)]


def test_m1_a_prosopia_copy_is_not_numbered_as_an_upload_version(env):
    """Version numbers ("version 2, uploaded ...") are for two files of one
    paper; a Prosopia copy is not a file."""
    from rag_lib.db.repos import versions as versions_repo

    oa = _openalex_without_abstract()
    rich = StubProfile("Nathan C. Sheffield", [_with_abstract()], StubSummaries({}))
    assert _import(_app(StubProsopia(rich), oa), source="prosopia", ref="sheffield-nathan").status_code == 200
    uid = _uid(env, "demo@example.com")
    conn = connect(env)
    try:
        # One upload of the same work beside the copy.
        conn.execute("INSERT INTO papers (openalex_id, title, source) VALUES ('local:upload:1:abc', 't', 'user_pdf')")
        versions_repo.register(conn, user_id=uid, paper_id="local:upload:1:abc", canonical_id=W, file_hash="abc")
        conn.commit()
        assert versions_repo.version_numbers(conn, uid) == {}
    finally:
        conn.close()
