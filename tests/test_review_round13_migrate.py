"""Review round 13 (c): migrations 0021-0023 on a populated 0020 database.

Builds a file-backed WAL database (foreign keys ON, as ``connect`` opens it)
at migration 0020 with rows in every table that references ``profiles`` or
is touched by 0021-0023, migrates to head, and checks that every row
survived, the new constraints hold and a re-run is a no-op.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from rag_lib.db import apply_migrations, connect
from rag_lib.db.migrate import MIGRATIONS_DIR
from tests.test_db_migrate import EXPECTED_VERSIONS

TABLES = [
    "users", "profiles", "papers", "paper_embeddings", "profile_seeds",
    "gather_runs", "profile_candidates", "profile_schedules", "feedback_events",
    "chat_turns", "researchers", "researcher_papers", "paper_uploads",
]


def _migrations_upto(tmp_path: Path, last: str) -> Path:
    d = tmp_path / "mig"
    d.mkdir()
    for p in sorted(MIGRATIONS_DIR.iterdir(), key=lambda p: p.stem):
        if p.suffix in (".sql", ".py") and p.stem[:4].isdigit() and p.stem <= last:
            shutil.copy(p, d / p.name)
    return d


def _populate_0020(conn: sqlite3.Connection) -> None:
    c = conn
    c.execute("INSERT INTO users (id, email, mailto) VALUES (2, 'a@x.org', 'a@x.org')")
    c.execute("INSERT INTO users (id, email, mailto) VALUES (3, 'b@x.org', 'b@x.org')")
    c.execute("INSERT INTO researchers (id, user_id, source, key, name, orcid) VALUES (1, 2, 'orcid', '0000-0001-0000-0001', 'R One', '0000-0001-0000-0001')")
    c.execute("INSERT INTO researchers (id, user_id, source, key, name) VALUES (2, 3, 'prosopia', 'slug-b', 'R Two')")
    for i in range(1, 5):
        c.execute("INSERT INTO papers (openalex_id, title, source) VALUES (?, ?, 'openalex')", (f"W{i}", f"Paper {i}"))
        c.execute("INSERT INTO paper_embeddings (openalex_id, embedding_model, vector) VALUES (?, 'specter2', x'00')", (f"W{i}",))
    # Two users, distinct names (0020 still has the global UNIQUE), one draft.
    rows = [
        (10, "Alpha", 2, "alpha", 0, 1),
        (11, "Beta", 2, "beta", 1, None),
        (12, "Gamma", 3, "gamma", 0, 2),
    ]
    for pid, name, uid, slug, draft, rid in rows:
        c.execute(
            "INSERT INTO profiles (id, name, embedding_model, topic_filters_json, n_seed, user_id, slug, is_draft, researcher_id, threshold, year_from) "
            "VALUES (?, ?, 'specter2', '{}', 1, ?, ?, ?, ?, 0.8, 2020)",
            (pid, name, uid, slug, draft, rid),
        )
    # A deleted interest raises the AUTOINCREMENT high-water mark above max(id).
    c.execute("INSERT INTO profiles (id, name, embedding_model, topic_filters_json, n_seed, user_id, slug) VALUES (13, 'Gone', 'specter2', '{}', 1, 3, 'gone')")
    c.execute("DELETE FROM profiles WHERE id = 13")
    for pid in (10, 11, 12):
        c.execute("INSERT INTO profile_seeds (profile_id, openalex_id) VALUES (?, 'W1')", (pid,))
        c.execute("INSERT INTO profile_schedules (profile_id, cron, tz, enabled) VALUES (?, '0 4 * * *', 'UTC', 0)", (pid,))
        c.execute("INSERT INTO gather_runs (profile_id, user_id, tier_used) VALUES (?, 2, 'manual')", (pid,))
    c.execute("INSERT INTO gather_runs (researcher_id, user_id, tier_used) VALUES (1, 2, 'researcher_import')")
    run_id = c.execute("SELECT id FROM gather_runs WHERE profile_id = 10").fetchone()[0]
    # One pre-0003 style row (score_raw NULL) for 0022's backfill, one normal.
    c.execute("INSERT INTO profile_candidates (profile_id, openalex_id, score, gather_run_id, score_raw) VALUES (10, 'W2', 0.9, ?, NULL)", (run_id,))
    c.execute("INSERT INTO profile_candidates (profile_id, openalex_id, score, gather_run_id, score_raw) VALUES (10, 'W3', 0.75, ?, 0.5)", (run_id,))
    c.execute("INSERT INTO profile_candidates (profile_id, openalex_id, score, gather_run_id, score_raw) VALUES (12, 'W4', 0.6, NULL, 0.2)")
    c.execute("INSERT INTO feedback_events (user_id, profile_id, openalex_id, action) VALUES (2, 10, 'W2', 'saved')")
    c.execute("INSERT INTO feedback_events (user_id, profile_id, openalex_id, action) VALUES (3, 12, 'W4', 'dismissed')")
    c.execute("INSERT INTO chat_turns (user_id, role, body) VALUES (2, 'user', 'hi')")
    c.execute("INSERT INTO researcher_papers (researcher_id, openalex_id) VALUES (1, 'W1')")
    c.execute("INSERT INTO researcher_papers (researcher_id, openalex_id) VALUES (2, 'W2')")
    c.execute("INSERT INTO paper_uploads (user_id, openalex_id, file_hash) VALUES (2, 'W1', 'h1')")
    c.commit()


@pytest.fixture()
def db_at_0020(tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_SQLITE_JOURNAL_MODE", "wal")
    path = tmp_path / "radar.db"
    conn = connect(path)
    try:
        applied = apply_migrations(conn, _migrations_upto(tmp_path, "0020_paper_uploads"))
        assert applied[-1] == "0020_paper_uploads"
        _populate_0020(conn)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()
    return path


def _counts(conn):
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


def test_0021_to_head_keeps_every_row_and_adds_the_new_rules(db_at_0020):
    conn = connect(db_at_0020)
    try:
        before = _counts(conn)
        seq_before = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'profiles'").fetchone()[0]
        assert seq_before == 13
        cols_before = [r[1] for r in conn.execute("PRAGMA table_info(profiles)")]
        profiles_before = [tuple(r) for r in conn.execute("SELECT * FROM profiles ORDER BY id")]

        applied = apply_migrations(conn)
        assert applied == EXPECTED_VERSIONS[EXPECTED_VERSIONS.index("0021_profile_names_per_user"):]

        # 0029: user 2's upload of W1 became their own paper. W1's row had
        # no record of whose file its text was, so its vector (made from
        # that text) is dropped and the copy waits for user 2's file.
        before.update(papers=before["papers"] + 1, paper_embeddings=before["paper_embeddings"] - 1)
        assert _counts(conn) == before
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        # Same rows, same ids, same column values (0021 copies every column);
        # 0033 and 0034 add columns, empty on these rows (none has a researcher's
        # topics applied, which is what fills ``researcher_doc_json``).
        cols_after = [r[1] for r in conn.execute("PRAGMA table_info(profiles)")]
        assert cols_after == cols_before + ["edit_of", "researcher_doc_json", "edit_tab", "edit_seen_at"]
        old = ", ".join(cols_before)
        assert [tuple(r) for r in conn.execute(f"SELECT {old} FROM profiles ORDER BY id")] == profiles_before
        assert conn.execute("SELECT COUNT(*) FROM profiles WHERE edit_of IS NOT NULL OR researcher_doc_json IS NOT NULL").fetchone()[0] == 0
        # Children still point at the surviving parents.
        assert conn.execute("SELECT COUNT(*) FROM profile_candidates pc JOIN profiles p ON p.id = pc.profile_id").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM gather_runs g JOIN profiles p ON p.id = g.profile_id").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM profile_schedules s JOIN profiles p ON p.id = s.profile_id").fetchone()[0] == 3
        # High-water mark kept: a new interest does not reuse id 13.
        assert conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'profiles'").fetchone()[0] == 13
        conn.execute("INSERT INTO profiles (name, embedding_model, topic_filters_json, n_seed, user_id, slug) VALUES ('New', 'specter2', '{}', 1, 3, 'new')")
        assert conn.execute("SELECT MAX(id) FROM profiles").fetchone()[0] == 14
        # Per-user uniqueness: another user may use an existing name/slug...
        conn.execute("INSERT INTO profiles (name, embedding_model, topic_filters_json, n_seed, user_id, slug) VALUES ('Alpha', 'specter2', '{}', 1, 3, 'alpha')")
        # ...the same user may not.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO profiles (name, embedding_model, topic_filters_json, n_seed, user_id, slug) VALUES ('Alpha', 'specter2', '{}', 1, 2, 'alpha-x')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO profiles (name, embedding_model, topic_filters_json, n_seed, user_id, slug) VALUES ('Alpha X', 'specter2', '{}', 1, 2, 'alpha')")
        conn.rollback()
        idx = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'profiles'")}
        assert {"idx_profiles_user", "idx_profiles_user_name", "idx_profiles_user_slug"} <= idx
        assert "idx_profiles_slug" not in idx
        # Cascades still work through the rebuilt parent.
        conn.execute("DELETE FROM profiles WHERE id = 12")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM profile_candidates WHERE profile_id = 12").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM profile_schedules WHERE profile_id = 12").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM feedback_events WHERE profile_id = 12").fetchone()[0] == 0
        # 0022: the pre-0003 row got its raw cosine back; the others are untouched.
        raw = dict(conn.execute("SELECT openalex_id, score_raw FROM profile_candidates WHERE profile_id = 10"))
        assert raw["W2"] == pytest.approx(2 * 0.9 - 1)
        assert raw["W3"] == 0.5
        # 0023: new columns default to NULL and the CHECK holds.
        assert tuple(conn.execute("SELECT mark, position FROM researchers WHERE id = 1").fetchone()) == (None, None)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE researchers SET mark = 'friend' WHERE id = 1")
        conn.rollback()
        # FK enforcement is back on after the python migration.
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        # Idempotent.
        assert apply_migrations(conn) == []
    finally:
        conn.close()


def test_migrating_with_a_second_connection_open_in_wal(db_at_0020):
    """A CLI or a lingering reader holding the file open must not break the rebuild."""
    reader = connect(db_at_0020)
    try:
        reader.execute("SELECT COUNT(*) FROM profiles").fetchone()
        conn = connect(db_at_0020)
        try:
            applied = apply_migrations(conn)
            assert applied[0] == "0021_profile_names_per_user"
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        finally:
            conn.close()
        assert reader.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 3
    finally:
        reader.close()
