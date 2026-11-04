"""A papers row goes only when nothing names it, for any user
(FUNCTIONAL_SPEC CR.1). ``paper_rows.delete_if_unused`` is the one place
that deletes one; what counts as a use is read from the schema."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from rag_lib.db import connect
from rag_lib.db.migrate import apply_migrations
from rag_lib.db.repos import draft_owned, paper_rows, papers, profiles, researchers, users, vault

ROOT = Path(__file__).resolve().parent.parent

# Any spelling of a delete from the papers table (not papers_to_embed).
_DELETE_PAPERS = re.compile(r"""delete\s+from\s+["'`\[]?papers\b(?!_)["'`\]]?""", re.IGNORECASE)

# Tables whose rows are part of a paper: they go with it (ON DELETE CASCADE).
_PART_OF_A_PAPER = {"paper_embeddings", "paper_uploads", "papers_to_embed", "paper_versions"}


@pytest.fixture()
def conn(tmp_path):
    c = connect(tmp_path / "radar.db")
    apply_migrations(c)
    users.upsert(c, "demo@example.com")
    yield c
    c.close()


def _paper(conn, oa="https://openalex.org/W111"):
    papers.upsert(conn, {"openalex_id": oa, "title": "X", "source": "openalex"})
    conn.commit()
    return oa


def _n(conn, oa):
    return conn.execute("SELECT COUNT(*) FROM papers WHERE openalex_id = ?", (oa,)).fetchone()[0]


# --- the guard ------------------------------------------------------------

def test_the_guard_sees_every_spelling():
    for s in ("DELETE FROM papers WHERE", "delete  from\n  papers", 'DELETE FROM "papers" WHERE',
              "Delete From [papers]", "DELETE FROM papers;"):
        assert _DELETE_PAPERS.search(s), s
    for s in ("DELETE FROM papers_to_embed WHERE", "DELETE FROM paper_uploads", "DELETE FROM researcher_papers"):
        assert not _DELETE_PAPERS.search(s), s


def test_only_paper_rows_deletes_a_paper():
    found = set()
    for top in ("rag_lib", "cli"):
        for path in (ROOT / top).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if "/migrations/" in rel:
                continue
            if _DELETE_PAPERS.search(path.read_text()):
                found.add(rel)
    assert found == {"rag_lib/db/repos/paper_rows.py"}


def test_every_foreign_key_to_papers_is_a_use_or_part_of_the_paper(conn):
    """A cascading key takes its rows with the paper: only a paper's own
    parts may. A new one fails here until it is looked at (vault_kept
    cascaded, and a deleted paper took it out of other users' Vaults)."""
    cascading, uses = set(), set()
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
        for fk in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            if fk[2] != "papers":
                continue
            (cascading if str(fk[6]).upper() == "CASCADE" else uses).add(table)
    assert cascading == _PART_OF_A_PAPER
    assert "vault_kept" in uses and "profile_seeds" in uses
    sqls = paper_rows._uses(conn)
    for t in uses:
        assert any(f'FROM "{t}"' in s for s in sqls), t


# --- the check ------------------------------------------------------------

def test_an_unused_paper_goes_with_its_parts(conn):
    oa = _paper(conn)
    assert paper_rows.delete_if_unused(conn, oa) is True
    assert _n(conn, oa) == 0


def test_a_paper_another_user_keeps_in_the_vault_stays(conn):
    oa = _paper(conn)
    b = int(users.upsert(conn, "b@x.org")["id"])
    conn.execute("INSERT INTO vault_kept (user_id, openalex_id) VALUES (?, ?)", (b, oa))
    conn.commit()
    assert paper_rows.in_use(conn, oa)
    assert paper_rows.delete_if_unused(conn, oa) is False
    assert _n(conn, oa) == 1


def test_a_paper_another_draft_brought_in_stays(conn):
    oa = _paper(conn)
    b = int(users.upsert(conn, "b@x.org")["id"])
    pid = profiles.create_draft(conn, user_id=b, name="d", slug="d")
    draft_owned.own(conn, pid, "paper", oa)
    conn.commit()
    assert paper_rows.delete_if_unused(conn, oa) is False


def test_a_table_added_later_is_a_use_from_the_start(conn):
    oa = _paper(conn)
    conn.execute("CREATE TABLE later_notes (openalex_id TEXT REFERENCES papers(openalex_id))")
    conn.execute("INSERT INTO later_notes VALUES (?)", (oa,))
    conn.commit()
    assert paper_rows.delete_if_unused(conn, oa) is False
    assert _n(conn, oa) == 1


def test_the_foreign_key_is_the_last_word(conn, monkeypatch):
    """Past a check that missed a use, the database refuses: the row stays
    and the caller's transaction goes on."""
    oa = _paper(conn)
    b = int(users.upsert(conn, "b@x.org")["id"])
    conn.execute("INSERT INTO vault_kept (user_id, openalex_id) VALUES (?, ?)", (b, oa))
    conn.commit()
    monkeypatch.setattr(paper_rows, "in_use", lambda c, i: False)
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE papers SET title = 'still here' WHERE openalex_id = ?", (oa,))
    assert paper_rows.delete_if_unused(conn, oa) is False
    conn.commit()
    assert conn.execute("SELECT title FROM papers WHERE openalex_id = ?", (oa,)).fetchone()[0] == "still here"
    assert conn.execute("SELECT COUNT(*) FROM vault_kept").fetchone()[0] == 1


def test_vault_kept_does_not_cascade(conn):
    oa = _paper(conn)
    b = int(users.upsert(conn, "b@x.org")["id"])
    conn.execute("INSERT INTO vault_kept (user_id, openalex_id) VALUES (?, ?)", (b, oa))
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM papers WHERE openalex_id = ?", (oa,))
    conn.rollback()


# --- the paths (review A1: another user's discard) -------------------------

def test_a_discard_leaves_a_paper_another_user_kept_in_their_vault(conn):
    """B's draft brought X in; A's researcher had X too; A forgot the
    researcher (X kept in A's Vault); B discards the draft: X stays in A's
    Vault (F1, CR.1)."""
    a = int(users.upsert(conn, "a@x.org")["id"])
    b = int(users.upsert(conn, "b@x.org")["id"])
    pid_b = profiles.create_draft(conn, user_id=b, name="bd", slug="bd")
    x = _paper(conn)
    draft_owned.own(conn, pid_b, "paper", x)
    conn.commit()
    rid = researchers.upsert(conn, user_id=a, source="orcid", key="0000-0001-0000-0001",
                             name="R", orcid="0000-0001-0000-0001", url="u")
    researchers.attach_paper(conn, rid, x, paper_id=None, resolved_by="doi", summary=None)
    conn.commit()
    vault.keep_researcher(conn, rid)
    researchers.delete(conn, rid)
    conn.commit()
    before = [r["openalex_id"] for r in vault.list_for_user(conn, a)]
    assert before == [x]

    draft_owned.discard(conn, pid_b)
    conn.commit()
    assert [r["openalex_id"] for r in vault.list_for_user(conn, a)] == before
    assert _n(conn, x) == 1


# --- every way a draft or copy goes (CR.1 point 4) --------------------------

def _b_keeps_x_and_a_copy_owns_it(conn):
    """B forgot a researcher whose paper X stays in B's Vault; A's Edit
    copy of a saved interest brought X in (owns it)."""
    a = int(users.upsert(conn, "a@x.org")["id"])
    b = int(users.upsert(conn, "b@x.org")["id"])
    x = _paper(conn)
    rid = researchers.upsert(conn, user_id=b, source="orcid", key="0000-0002-0000-0002",
                             name="R", orcid="0000-0002-0000-0002", url="u")
    researchers.attach_paper(conn, rid, x, paper_id=None, resolved_by="doi", summary=None)
    conn.commit()
    vault.keep_researcher(conn, rid)
    researchers.delete(conn, rid)
    saved = profiles.create_draft(conn, user_id=a, name="saved", slug="saved")
    conn.execute("UPDATE profiles SET is_draft = 0 WHERE id = ?", (saved,))
    copy = profiles.create_draft(conn, user_id=a, name="saved (editing)", slug="saved-edit-1")
    conn.execute("UPDATE profiles SET edit_of = ?, edit_seen_at = NULL WHERE id = ?", (saved, copy))
    draft_owned.own(conn, copy, "paper", x)
    conn.commit()
    return a, b, x, saved, copy


def _leave_edit(conn, a, saved, copy):
    from rag_lib.api.services import edit_copy
    edit_copy.discard_copy(conn, user_id=a, slug="saved-edit-1")


def _stale_copy(conn, a, saved, copy):
    from rag_lib.api.services import edit_copy
    edit_copy.open_copy_of(conn, saved)


def _failed_import(conn, a, saved, copy):
    from types import SimpleNamespace
    from rag_lib.api.services import prosopia
    plan = SimpleNamespace(profile_id=copy, researcher_id=None, embedding_model="m")
    prosopia._take_back(conn, plan, [("paper", "https://openalex.org/W111")])


def _discard(conn, a, saved, copy):
    draft_owned.discard(conn, copy)


@pytest.mark.parametrize("leave", [_discard, _leave_edit, _stale_copy, _failed_import],
                         ids=["discard draft", "leave edit", "stale copy", "failed import"])
def test_no_way_out_of_a_draft_or_copy_shrinks_another_users_vault(conn, leave):
    a, b, x, saved, copy = _b_keeps_x_and_a_copy_owns_it(conn)
    before = {u: [r["openalex_id"] for r in vault.list_for_user(conn, u)] for u in (a, b)}
    assert x in before[b]
    leave(conn, a, saved, copy)
    conn.commit()
    after = {u: [r["openalex_id"] for r in vault.list_for_user(conn, u)] for u in (a, b)}
    assert after == before
    assert _n(conn, x) == 1
