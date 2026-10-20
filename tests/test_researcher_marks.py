"""Marking a researcher as the user or as followed, and arranging the list.

Marks are presentation: a marked researcher moves to the top of their
group, and the user can then drag the list into any order, which is
stored as ``position`` and comes back the same way. Offline, through the
researchers API with the import stubs.
"""

from __future__ import annotations

from rag_lib.db import connect
from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex
from tests.test_prosopia_import import StubProsopia, _stub_openalex, _stub_profile
from tests.test_researchers import _app, _request, env  # noqa: F401 -- fixture


def _two(app) -> tuple[int, int]:
    """Nathan (Prosopia) and Justin (ORCID); Justin sorts first by name."""
    a = _request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"})
    b = _request(app, "POST", "/api/researchers/import", json={"source": "orcid", "ref": ORCID})
    assert a.status_code == 200 and b.status_code == 200, (a.text, b.text)
    return a.json()["researcher_id"], b.json()["researcher_id"]


def _order(app, headers=None) -> list[tuple[int, str | None, int | None]]:
    rows = _request(app, "GET", "/api/researchers", headers=headers).json()
    return [(r["id"], r["mark"], r["position"]) for r in rows]


def _both_stubs():
    """One OpenAlex stub that serves the Prosopia works and the ORCID works."""
    oa = StubOrcidOpenAlex(WORKS)
    prosopia = _stub_openalex()
    for attr in ("by_id", "by_doi", "by_pmcid", "title_hits"):
        getattr(oa, attr).update(getattr(prosopia, attr))
    return oa


def test_unmarked_researchers_list_by_name_with_no_position(env):
    app = _app(StubProsopia(_stub_profile()), _both_stubs())
    nathan, justin = _two(app)
    assert _order(app) == [(justin, None, None), (nathan, None, None)]


def test_marking_moves_the_person_to_the_top_of_their_group(env):
    app = _app(StubProsopia(_stub_profile()), _both_stubs())
    nathan, justin = _two(app)

    resp = _request(app, "PATCH", f"/api/researchers/{nathan}", json={"mark": "self"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["mark"] == "self" and resp.json()["position"] == 0
    assert _order(app) == [(nathan, "self", 0), (justin, None, 1)]

    # Followed goes behind the user themself, ahead of the unmarked.
    c = _request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "0000-0001-5643-4068"})
    assert c.status_code == 200, c.text
    assert c.json()["researcher_id"] == nathan          # the same person, refreshed
    _request(app, "PATCH", f"/api/researchers/{justin}", json={"mark": "followed"})
    assert _order(app) == [(nathan, "self", 0), (justin, "followed", 1)]
    _request(app, "PATCH", f"/api/researchers/{nathan}", json={"mark": None})
    assert _order(app) == [(justin, "followed", 0), (nathan, None, 1)]
    _request(app, "PATCH", f"/api/researchers/{nathan}", json={"mark": "self"})
    assert _order(app) == [(nathan, "self", 0), (justin, "followed", 1)]

    detail = _request(app, "GET", f"/api/researchers/{nathan}").json()["researcher"]
    assert detail["mark"] == "self" and detail["position"] == 0


def test_the_user_arranges_the_list_and_it_comes_back_that_way(env):
    app = _app(StubProsopia(_stub_profile()), _both_stubs())
    nathan, justin = _two(app)
    _request(app, "PATCH", f"/api/researchers/{nathan}", json={"mark": "self"})
    # Dragging the followed/unmarked one above "self" is allowed: the mark is a
    # badge and a default place, the order is the user's.
    resp = _request(app, "PUT", "/api/researchers/order", json={"ids": [justin, nathan]})
    assert resp.status_code == 200, resp.text
    assert [(r["id"], r["position"]) for r in resp.json()] == [(justin, 0), (nathan, 1)]
    assert _order(app) == [(justin, None, 0), (nathan, "self", 1)]

    # A later mark places that person at the top of their group again: a
    # followed researcher goes behind the user themself.
    _request(app, "PATCH", f"/api/researchers/{justin}", json={"mark": "followed"})
    assert _order(app) == [(nathan, "self", 0), (justin, "followed", 1)]


def test_a_partial_or_padded_order_is_refused(env):
    app = _app(StubProsopia(_stub_profile()), _both_stubs())
    nathan, justin = _two(app)
    for ids in ([nathan], [nathan, justin, 999], [nathan, nathan, justin], [999]):
        resp = _request(app, "PUT", "/api/researchers/order", json={"ids": ids})
        assert resp.status_code == 400, (ids, resp.text)
    assert _request(app, "PUT", "/api/researchers/order", json={"ids": []}).status_code == 422
    assert _order(app) == [(justin, None, None), (nathan, None, None)]


def test_marks_are_validated_and_per_user(env):
    app = _app(StubProsopia(_stub_profile()), _both_stubs())
    nathan, justin = _two(app)
    assert _request(app, "PATCH", f"/api/researchers/{nathan}", json={"mark": "hero"}).status_code == 422
    assert _request(app, "PATCH", "/api/researchers/999", json={"mark": "self"}).status_code == 404
    other = {"X-User-Email": "someone-else@example.com"}
    assert _request(app, "PATCH", f"/api/researchers/{nathan}", json={"mark": "self"}, headers=other).status_code == 404
    assert _request(app, "PUT", "/api/researchers/order", json={"ids": [nathan, justin]}, headers=other).status_code == 400
    assert _order(app) == [(justin, None, None), (nathan, None, None)]
    # The repo refuses a bad mark too, in case a caller bypasses the schema.
    import pytest
    from rag_lib.db.repos import researchers as repo
    conn = connect(env)
    try:
        with pytest.raises(ValueError):
            repo.set_mark(conn, 1, nathan, "hero")
        with pytest.raises(LookupError):
            repo.set_mark(conn, 2, nathan, "self")
    finally:
        conn.close()


def test_rows_from_before_the_marks_sort_behind_the_placed_ones(env):
    app = _app(StubProsopia(_stub_profile()), _both_stubs())
    nathan, justin = _two(app)
    _request(app, "PUT", "/api/researchers/order", json={"ids": [nathan, justin]})
    conn = connect(env)
    try:
        # An old row: no position, but a mark (as if written by hand).
        conn.execute("UPDATE researchers SET position = NULL, mark = 'self' WHERE id = ?", (justin,))
        conn.commit()
    finally:
        conn.close()
    assert _order(app) == [(nathan, None, 0), (justin, "self", None)]
    # Marking anyone places every row.
    _request(app, "PATCH", f"/api/researchers/{justin}", json={"mark": "self"})
    assert _order(app) == [(justin, "self", 0), (nathan, None, 1)]


def test_a_patch_without_the_field_is_not_a_clear(env):
    app = _app(StubProsopia(_stub_profile()), _both_stubs())
    nathan, _ = _two(app)
    _request(app, "PATCH", f"/api/researchers/{nathan}", json={"mark": "self"})
    assert _request(app, "PATCH", f"/api/researchers/{nathan}", json={}).status_code == 422
    assert _request(app, "GET", f"/api/researchers/{nathan}").json()["researcher"]["mark"] == "self"
    assert _request(app, "PATCH", f"/api/researchers/{nathan}", json={"mark": None}).status_code == 200
    assert _request(app, "GET", f"/api/researchers/{nathan}").json()["researcher"]["mark"] is None


def test_marking_is_one_transaction(env, monkeypatch):
    """The list is read and every position written under one write lock, so
    a second writer waits and reads the first's result: the read happens
    inside the transaction, not before it."""
    from rag_lib.db.repos import researchers as repo
    app = _app(StubProsopia(_stub_profile()), _both_stubs())
    nathan, justin = _two(app)
    seen = []
    real = repo.list_for_user
    monkeypatch.setattr(repo, "list_for_user", lambda conn, uid: seen.append(conn.in_transaction) or real(conn, uid))
    conn = connect(env)
    try:
        repo.set_mark(conn, 1, justin, "followed")
        repo.reorder(conn, 1, [nathan, justin])
        assert seen == [True, True]                # both reads were under the write lock
        assert not conn.in_transaction             # and it was committed
        assert [r["id"] for r in real(conn, 1)] == [nathan, justin]
        # A writer that cannot get the lock fails cleanly, leaving nothing open.
        other = connect(env)
        other.execute("PRAGMA busy_timeout = 100")
        conn.execute("BEGIN IMMEDIATE")
        import sqlite3
        import pytest
        with pytest.raises(sqlite3.OperationalError):
            repo.set_mark(other, 1, nathan, "self")
        assert not other.in_transaction
        conn.rollback()
        other.close()
    finally:
        conn.close()


def test_marking_refuses_a_connection_with_a_transaction_open(env):
    """The helper commits; a caller's open work is not its to commit."""
    import pytest
    from rag_lib.db.repos import researchers as repo
    app = _app(StubProsopia(_stub_profile()), _both_stubs())
    nathan, justin = _two(app)
    conn = connect(env)
    try:
        conn.execute("INSERT INTO users (email) VALUES ('pending@example.com')")
        with pytest.raises(RuntimeError):
            repo.set_mark(conn, 1, nathan, "self")
        with pytest.raises(RuntimeError):
            repo.reorder(conn, 1, [nathan, justin])
        conn.rollback()
        assert repo.get(conn, nathan)["mark"] is None
    finally:
        conn.close()
