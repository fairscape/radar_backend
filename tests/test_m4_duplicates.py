"""Two seeds of one interest that may be one paper (FUNCTIONAL_SPEC M.4).

OpenAlex works with one DOI are one seed; seeds with one title are marked
"title duplicate" (an upload or an unmatched record among them) or
"possible duplicate" (OpenAlex works), listed together and both kept.
"""

from __future__ import annotations

import pytest

from rag_lib import seed_duplicates
from rag_lib.db import connect
from rag_lib.db.repos import profiles as profiles_repo
from rag_lib.db.repos import users as users_repo
from rag_lib.db.repos import versions as versions_repo
from tests.test_researchers import _request, env  # noqa: F401 -- fixture

TITLE = "Chromatin accessibility across human cell types at single-cell resolution"


def _conn(env):
    return connect(env)


def _user(conn, email="demo@example.com") -> int:
    return int(users_repo.upsert(conn, email)["id"])


def _profile(conn, uid, slug="int-a") -> int:
    cur = conn.execute(
        "INSERT INTO profiles (user_id, name, slug, embedding_model, topic_filters_json, n_seed)"
        " VALUES (?, ?, ?, 'placeholder-v1', '{}', 0)", (uid, slug, slug))
    conn.commit()
    return int(cur.lastrowid)


def _paper(conn, pid, title=TITLE, doi=None, source="openalex"):
    conn.execute("INSERT INTO papers (openalex_id, title, doi, year, source) VALUES (?, ?, ?, 2024, ?)",
                 (pid, title, doi, source))
    conn.commit()


def _upload(conn, uid, title=TITLE, digest="abc", canonical=None) -> str:
    vid = versions_repo.version_id(uid, digest)
    _paper(conn, vid, title=title, source="upload")
    versions_repo.register(conn, user_id=uid, paper_id=vid, canonical_id=canonical, file_hash=digest)
    conn.execute("INSERT INTO paper_uploads (user_id, openalex_id, file_hash) VALUES (?, ?, ?)", (uid, vid, digest))
    conn.commit()
    return vid


def _seeds(conn, profile_id):
    return profiles_repo.list_seed_openalex_ids(conn, profile_id)


# --- DOI: one seed --------------------------------------------------------------------------


@pytest.mark.parametrize("second", ["10.1/abc", "10.1/ABC", "https://doi.org/10.1/abc", "doi:10.1/Abc"])
def test_m27_an_openalex_work_with_a_seeds_doi_is_not_a_second_seed(env, second):
    conn = _conn(env)
    uid = _user(conn)
    pid = _profile(conn, uid)
    _paper(conn, "https://openalex.org/W1", doi="https://doi.org/10.1/abc")
    _paper(conn, "https://openalex.org/W2", title="Another title for the same record", doi=second)
    assert profiles_repo.attach_seed(conn, pid, "https://openalex.org/W1") == 1
    assert profiles_repo.attach_seed(conn, pid, "https://openalex.org/W2") == 0
    assert _seeds(conn, pid) == ["https://openalex.org/W1"]
    assert conn.execute("SELECT n_seed FROM profiles WHERE id = ?", (pid,)).fetchone()[0] == 1


def test_m27_different_dois_and_other_interests_are_left_alone(env):
    conn = _conn(env)
    uid = _user(conn)
    a, b = _profile(conn, uid, "int-a"), _profile(conn, uid, "int-b")
    _paper(conn, "W1", doi="10.1/abc")
    _paper(conn, "W2", title="x" * 40, doi="10.1/abcd")
    _paper(conn, "W3", title="y" * 40, doi="10.1/abc")
    assert profiles_repo.attach_seed(conn, a, "W1") == 1
    assert profiles_repo.attach_seed(conn, a, "W2") == 1      # another DOI
    assert profiles_repo.attach_seed(conn, b, "W3") == 1      # another interest
    _paper(conn, "W4", title="z" * 40, doi=None)
    assert profiles_repo.attach_seed(conn, a, "W4") == 1      # no DOI


def test_m27_an_upload_or_an_unmatched_record_is_never_merged_by_doi(env):
    conn = _conn(env)
    uid = _user(conn)
    pid = _profile(conn, uid)
    _paper(conn, "W1", doi="10.1/abc")
    vid = _upload(conn, uid)
    conn.execute("UPDATE papers SET doi = '10.1/abc' WHERE openalex_id = ?", (vid,))
    _paper(conn, "prosopia:sheffield-nathan:p1", doi="10.1/abc", source="prosopia")
    assert profiles_repo.attach_seed(conn, pid, "W1") == 1
    assert profiles_repo.attach_seed(conn, pid, vid) == 1
    assert profiles_repo.attach_seed(conn, pid, "prosopia:sheffield-nathan:p1") == 1


def test_m27_a_prosopia_copy_carries_its_works_doi(env):
    conn = _conn(env)
    uid = _user(conn)
    pid = _profile(conn, uid)
    _paper(conn, "W1", doi="10.1/abc")
    copy = versions_repo.prosopia_copy_id(uid, "W1")
    _paper(conn, copy, doi=None, source="prosopia")
    versions_repo.register(conn, user_id=uid, paper_id=copy, canonical_id="W1", file_hash="prosopia:W1")
    _paper(conn, "W9", title="q" * 40, doi="10.1/ABC")
    assert profiles_repo.attach_seed(conn, pid, copy) == 1
    assert profiles_repo.attach_seed(conn, pid, "W9") == 0


def test_m27_a_prosopia_copy_with_the_doi_of_a_seed_is_not_a_second_seed(env):
    conn = _conn(env)
    uid = _user(conn)
    pid = _profile(conn, uid)
    _paper(conn, "W1", doi="10.1/abc")
    copy = versions_repo.prosopia_copy_id(uid, "W1")
    _paper(conn, copy, doi=None, source="prosopia")
    versions_repo.register(conn, user_id=uid, paper_id=copy, canonical_id="W1", file_hash="prosopia:W1")
    _paper(conn, "W9", title="q" * 40, doi="10.1/ABC")
    assert profiles_repo.attach_seed(conn, pid, "W9") == 1
    assert profiles_repo.attach_seed(conn, pid, copy) == 0


def test_m27_an_upload_seed_does_not_keep_out_a_work_with_its_doi(env):
    conn = _conn(env)
    uid = _user(conn)
    pid = _profile(conn, uid)
    vid = _upload(conn, uid)
    conn.execute("UPDATE papers SET doi = '10.1/abc' WHERE openalex_id = ?", (vid,))
    _paper(conn, "W1", title="r" * 40, doi="10.1/abc")
    assert profiles_repo.attach_seed(conn, pid, vid) == 1
    assert profiles_repo.attach_seed(conn, pid, "W1") == 1


# --- same title: marked, listed together, both kept -----------------------------------------


def test_m25_an_upload_with_an_openalex_seeds_title_is_a_title_duplicate(env):
    conn = _conn(env)
    uid = _user(conn)
    pid = _profile(conn, uid)
    _paper(conn, "W1")
    _paper(conn, "W2", title="An unrelated paper about something else entirely")
    vid = _upload(conn, uid, title="  CHROMATIN accessibility across human cell-types, at single cell resolution. ")
    for p in ("W1", "W2", vid):
        assert profiles_repo.attach_seed(conn, pid, p) == 1
    m = seed_duplicates.marks(conn, pid)
    assert m == {"W1": {"kind": "title", "group": "W1"}, vid: {"kind": "title", "group": "W1"}}
    assert len(_seeds(conn, pid)) == 3     # both kept, counted as two


def test_m26_an_unmatched_prosopia_record_with_a_seeds_title_is_a_title_duplicate(env):
    conn = _conn(env)
    uid = _user(conn)
    pid = _profile(conn, uid)
    _paper(conn, "W1")
    _paper(conn, "prosopia:sheffield-nathan:p1", source="prosopia")
    for p in ("W1", "prosopia:sheffield-nathan:p1"):
        profiles_repo.attach_seed(conn, pid, p)
    assert {k: v["kind"] for k, v in seed_duplicates.marks(conn, pid).items()} == {
        "W1": "title", "prosopia:sheffield-nathan:p1": "title"}


def test_m28_two_openalex_works_with_one_title_are_a_possible_duplicate(env):
    conn = _conn(env)
    uid = _user(conn)
    pid = _profile(conn, uid)
    _paper(conn, "W1", doi="10.1101/2024.01.01.000001")      # the preprint
    _paper(conn, "W2", doi="10.1038/s41586-024-00001-1")     # the published version
    for p in ("W1", "W2"):
        assert profiles_repo.attach_seed(conn, pid, p) == 1
    assert {k: v["kind"] for k, v in seed_duplicates.marks(conn, pid).items()} == {"W1": "possible", "W2": "possible"}


def test_m30_titles_only_alike_or_too_short_are_not_marked(env):
    conn = _conn(env)
    uid = _user(conn)
    pid = _profile(conn, uid)
    _paper(conn, "W1", title=TITLE + ": Part I")
    _paper(conn, "W2", title=TITLE + ": Part II")
    _paper(conn, "W3", title="Editorial")
    _paper(conn, "W4", title="Editorial")
    for p in ("W1", "W2", "W3", "W4"):
        profiles_repo.attach_seed(conn, pid, p)
    assert seed_duplicates.marks(conn, pid) == {}


def test_m32_two_uploads_of_one_paper_stay_seeds_and_are_marked(env):
    conn = _conn(env)
    uid = _user(conn)
    pid = _profile(conn, uid)
    v1, v2 = _upload(conn, uid, digest="d1"), _upload(conn, uid, digest="d2")
    assert profiles_repo.attach_seed(conn, pid, v1) == 1
    assert profiles_repo.attach_seed(conn, pid, v2) == 1
    assert {k: v["kind"] for k, v in seed_duplicates.marks(conn, pid).items()} == {v1: "title", v2: "title"}


def test_m33_only_within_one_interest_of_one_user(env):
    conn = _conn(env)
    a = _user(conn)
    b = _user(conn, "b@example.com")
    pa, pa2, pb = _profile(conn, a, "int-a"), _profile(conn, a, "int-a2"), _profile(conn, b, "int-b")
    _paper(conn, "W1")
    _paper(conn, "W2")
    profiles_repo.attach_seed(conn, pa, "W1")
    profiles_repo.attach_seed(conn, pa2, "W2")
    profiles_repo.attach_seed(conn, pb, "W2")
    assert seed_duplicates.marks(conn, pa) == {}
    assert seed_duplicates.marks(conn, pb) == {}


def test_grouped_lists_the_copies_together_in_their_order():
    marks = {"c": {"kind": "possible", "group": "a"}, "a": {"kind": "possible", "group": "a"},
             "e": {"kind": "title", "group": "b"}, "b": {"kind": "title", "group": "b"}}
    assert seed_duplicates.grouped(list("abcde"), marks, lambda x: x) == list("acbed")


def test_doi_key():
    assert {seed_duplicates.doi_key(d) for d in
            ("10.1/X", " https://doi.org/10.1/x ", "http://dx.doi.org/10.1/x", "doi:10.1/x")} == {"10.1/x"}
    assert seed_duplicates.doi_key(None) == ""


# --- what the pages read ----------------------------------------------------------------


def _app():
    from rag_lib.api.app import create_app
    return create_app()


def _three_seeds(env):
    """demo's interest "int-a": W1, an unrelated W2, then W3 with W1's title."""
    conn = _conn(env)
    uid = _user(conn)
    pid = _profile(conn, uid)
    _paper(conn, "https://openalex.org/W1")
    _paper(conn, "https://openalex.org/W2", title="An unrelated paper about something else entirely")
    _paper(conn, "https://openalex.org/W3", title=TITLE.upper())
    for p in ("W1", "W2", "W3"):
        profiles_repo.attach_seed(conn, pid, f"https://openalex.org/{p}")
    conn.close()
    return pid


def test_m28_the_detail_page_marks_them_and_lists_them_together(env):
    _three_seeds(env)
    r = _request(_app(), "GET", "/api/profiles/int-a/detail")
    assert r.status_code == 200, r.text
    seeds = [(s["id"].rsplit("/", 1)[-1], s["idx"], s["sameTitle"]) for s in r.json()["seeds"]]
    assert seeds == [("W1", 1, "possible"), ("W3", 2, "possible"), ("W2", 3, None)]
    groups = {s["sameTitleGroup"] for s in r.json()["seeds"] if s["sameTitle"]}
    assert len(groups) == 1


def test_m28_the_wizard_list_marks_them_and_lists_them_together(env):
    _three_seeds(env)
    docs = _request(_app(), "GET", "/api/vault/docs", params={"tag": "int-a"}).json()
    got = [(d["id"].rsplit("/", 1)[-1], d.get("sameTitle")) for d in docs]
    i1 = got.index(("W1", "possible"))
    i3 = got.index(("W3", "possible"))
    assert abs(i1 - i3) == 1
    assert ("W2", None) in got
    # The whole vault is no interest's list: nothing marked.
    assert all(d.get("sameTitle") is None for d in _request(_app(), "GET", "/api/vault/docs").json())


def test_m29_removing_one_takes_the_mark_away(env):
    pid = _three_seeds(env)
    conn = _conn(env)
    assert profiles_repo.detach_seed(conn, pid, "https://openalex.org/W3") == 1
    conn.close()
    seeds = _request(_app(), "GET", "/api/profiles/int-a/detail").json()["seeds"]
    assert [s["sameTitle"] for s in seeds] == [None, None]


# --- an import adding to an interest reports what it added -------------------------------


def test_m31_an_import_into_a_draft_counts_only_the_seeds_it_added(env):
    from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex
    from tests.test_researchers import _app as _import_app

    app = _import_app(None, StubOrcidOpenAlex(WORKS))
    first = _request(app, "POST", "/api/import/orcid", json={"orcid": ORCID, "openalex_ids": ["W1"]})
    assert first.status_code == 200, first.text
    result = _request(app, "GET", f"/api/import/prosopia/{first.json()['run_id']}").json()["result"]
    assert (result["drafted"], result["seeded"]) == (1, 1)
    slug = result["draft_slug"]
    again = _request(app, "POST", "/api/import/orcid",
                     json={"orcid": ORCID, "openalex_ids": ["W1", "W2"], "into": slug})
    assert again.status_code == 200, again.text
    result = _request(app, "GET", f"/api/import/prosopia/{again.json()['run_id']}").json()["result"]
    assert (result["drafted"], result["seeded"]) == (2, 1)


def test_m32_the_wizard_list_marks_nothing_whose_copy_it_does_not_list(env):
    """The public record of a work whose upload is a seed is not listed
    beside the upload (vault.list_docs): the upload alone is not marked,
    the detail page, listing both, marks both."""
    conn = _conn(env)
    uid = _user(conn)
    pid = _profile(conn, uid)
    _paper(conn, "https://openalex.org/W1")
    profiles_repo.attach_seed(conn, pid, "https://openalex.org/W1")
    vid = _upload(conn, uid, canonical="https://openalex.org/W1")
    assert profiles_repo.attach_seed(conn, pid, vid) == 1
    conn.close()
    docs = _request(_app(), "GET", "/api/vault/docs", params={"tag": "int-a"}).json()
    assert [(d["id"], d.get("sameTitle")) for d in docs] == [(vid, None)]
    seeds = _request(_app(), "GET", "/api/profiles/int-a/detail").json()["seeds"]
    assert {s["sameTitle"] for s in seeds} == {"title"}
