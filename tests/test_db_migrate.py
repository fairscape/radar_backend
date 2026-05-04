"""Migration runner: schema is built correctly and re-running is a no-op."""

from __future__ import annotations

import sqlite3

import pytest

from rag_lib.db import apply_migrations, applied_versions


EXPECTED_TABLES = {
    "_migrations_applied",
    "profiles",
    "papers",
    "paper_embeddings",
    "profile_seeds",
    "gather_runs",
    "profile_candidates",
    "users",
    "profile_schedules",
    "feedback_events",
    "chat_turns",
}


EXPECTED_VERSIONS = [
    "0001_initial",
    "0002_users",
    "0003_scores",
    "0004_schedules",
    "0005_feedback",
    "0006_chat",
    "0007_vault",
    "0008_drafts",
    "0009_oa",
    "0010_gather_run_progress",
    "0011_gather_run_result",
]


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r[0] for r in rows}


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    yield c
    c.close()


def test_first_run_creates_all_expected_tables(conn):
    applied = apply_migrations(conn)
    assert applied == EXPECTED_VERSIONS
    assert EXPECTED_TABLES.issubset(_table_names(conn))


def test_second_run_is_noop(conn):
    apply_migrations(conn)
    applied_again = apply_migrations(conn)
    assert applied_again == []


def test_applied_versions_returns_in_order(conn):
    apply_migrations(conn)
    versions = [v for v, _ in applied_versions(conn)]
    assert versions == EXPECTED_VERSIONS


def test_default_demo_user_exists(conn):
    apply_migrations(conn)
    row = conn.execute("SELECT id, email FROM users WHERE id = 1").fetchone()
    assert row is not None
    assert row["email"] == "demo@example.com"


def test_profiles_has_user_id_and_slug(conn):
    apply_migrations(conn)
    cols = _column_names(conn, "profiles")
    assert "user_id" in cols
    assert "slug" in cols


def test_gather_runs_has_user_id(conn):
    apply_migrations(conn)
    assert "user_id" in _column_names(conn, "gather_runs")


def test_unique_slug_index(conn):
    apply_migrations(conn)
    conn.execute(
        """
        INSERT INTO profiles
          (user_id, name, slug, embedding_model, topic_filters_json, n_seed)
        VALUES (1, 'Alpha', 'alpha', 'placeholder-v1', '{}', 0)
        """
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO profiles
              (user_id, name, slug, embedding_model, topic_filters_json, n_seed)
            VALUES (1, 'Beta', 'alpha', 'placeholder-v1', '{}', 0)
            """
        )


def test_foreign_keys_are_enforced(conn):
    apply_migrations(conn)
    # profile_seeds references both profiles and papers; inserting orphans should fail
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO profile_seeds (profile_id, openalex_id) VALUES (?, ?)",
            (999, "https://openalex.org/W999"),
        )
