"""Repo round-trips against an in-memory SQLite."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from rag_lib.db import apply_migrations
from rag_lib.db.repos import (
    candidates,
    embeddings,
    gather_runs,
    papers,
    profiles,
    users,
)


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    apply_migrations(c)
    yield c
    c.close()


# ---------------------------------------------------------------------- users

def test_users_demo_seeded(conn):
    row = users.get(conn, 1)
    assert row["email"] == "demo@example.com"


def test_users_upsert_lowercases(conn):
    row = users.upsert(conn, "  Alice@Example.COM ")
    assert row["email"] == "alice@example.com"
    again = users.upsert(conn, "alice@example.com")
    assert again["id"] == row["id"]


def test_users_upsert_updates_mailto(conn):
    users.upsert(conn, "bob@example.com")
    row = users.upsert(conn, "bob@example.com", mailto="other@example.com")
    assert row["mailto"] == "other@example.com"


# ------------------------------------------------------------------- profiles

def test_profiles_upsert_creates_then_updates(conn):
    pid = profiles.upsert(
        conn,
        user_id=1, name="neonatal_vitals", embedding_model="specter2",
        n_seed=10, topic_filters={"topics": [1, 2]},
        coherence_median=0.92, coherence_iqr=0.05, coherence_bimodal=False,
    )
    assert pid >= 1
    row = profiles.get_by_slug(conn, 1, "neonatal-vitals")
    assert row["id"] == pid
    assert row["n_seed"] == 10
    assert row["coherence_bimodal"] == 0

    pid2 = profiles.upsert(
        conn,
        user_id=1, name="neonatal_vitals", embedding_model="specter2",
        n_seed=12, topic_filters={"topics": [1, 2, 3]},
        threshold=0.85,
    )
    assert pid2 == pid
    row = profiles.get(conn, pid)
    assert row["n_seed"] == 12
    assert row["threshold"] == 0.85
    assert row["updated_at"] is not None


def test_profiles_attach_seed_idempotent(conn):
    pid = profiles.upsert(
        conn, user_id=1, name="A", embedding_model="x",
        n_seed=0, topic_filters={},
    )
    papers.upsert(conn, {
        "openalex_id": "W1", "title": "T1", "source": "openalex_gatherer",
    })
    assert profiles.attach_seed(conn, pid, "W1") == 1
    assert profiles.attach_seed(conn, pid, "W1") == 0
    assert profiles.list_seed_openalex_ids(conn, pid) == ["W1"]


def test_profiles_slug_collision_raises(conn):
    profiles.upsert(
        conn, user_id=1, name="Alpha", embedding_model="x",
        n_seed=0, topic_filters={},
    )
    with pytest.raises(sqlite3.IntegrityError):
        profiles.upsert(
            conn, user_id=1, name="ALPHA", embedding_model="x",
            n_seed=0, topic_filters={},
        )


# --------------------------------------------------------------------- papers

def test_papers_upsert_round_trip(conn):
    papers.upsert(conn, {
        "openalex_id": "W1", "doi": "10.1/1", "title": "Hello",
        "abstract": "abs", "year": 2024, "venue": "J",
        "source": "openalex_gatherer",
    })
    row = papers.get_by_openalex_id(conn, "W1")
    assert row["doi"] == "10.1/1"
    assert row["title"] == "Hello"

    # Upsert with a new title; abstract preserved via COALESCE since not provided
    papers.upsert(conn, {
        "openalex_id": "W1", "title": "Hello v2",
        "source": "openalex_gatherer",
    })
    row = papers.get_by_openalex_id(conn, "W1")
    assert row["title"] == "Hello v2"
    assert row["abstract"] == "abs"


def test_papers_get_by_doi(conn):
    papers.upsert(conn, {
        "openalex_id": "W1", "doi": "10.1/1", "title": "T",
        "source": "openalex_gatherer",
    })
    assert papers.get_by_doi(conn, "10.1/1")["openalex_id"] == "W1"


# ----------------------------------------------------------------- embeddings

def test_embeddings_round_trip(conn):
    papers.upsert(conn, {"openalex_id": "W1", "title": "T", "source": "openalex_gatherer"})
    vec = np.random.RandomState(0).rand(8).astype(np.float32)
    embeddings.upsert(conn, "W1", "specter2", vec)
    out = embeddings.get(conn, "W1", "specter2")
    assert out is not None
    np.testing.assert_allclose(out, vec, rtol=0, atol=0)
    assert embeddings.has(conn, "W1", "specter2")
    assert not embeddings.has(conn, "W1", "placeholder-v1")
    assert embeddings.models_for(conn, "W1") == ["specter2"]


def test_embeddings_upsert_overwrites(conn):
    papers.upsert(conn, {"openalex_id": "W1", "title": "T", "source": "openalex_gatherer"})
    embeddings.upsert(conn, "W1", "m", [0.0, 1.0])
    embeddings.upsert(conn, "W1", "m", [2.0, 3.0])
    out = embeddings.get(conn, "W1", "m")
    np.testing.assert_allclose(out, [2.0, 3.0])


# ----------------------------------------------------------------- candidates

def _seed_profile_and_paper(conn, oa_id: str = "W1") -> int:
    pid = profiles.upsert(
        conn, user_id=1, name="P", embedding_model="x",
        n_seed=0, topic_filters={},
    )
    papers.upsert(conn, {"openalex_id": oa_id, "title": "T", "source": "openalex_gatherer"})
    return pid


def test_candidates_insert_dedup(conn):
    pid = _seed_profile_and_paper(conn)
    assert candidates.insert_dedup(conn, profile_id=pid, openalex_id="W1", score=0.9) == 1
    assert candidates.insert_dedup(conn, profile_id=pid, openalex_id="W1", score=0.95) == 0
    row = conn.execute(
        "SELECT score FROM profile_candidates WHERE profile_id=? AND openalex_id=?",
        (pid, "W1"),
    ).fetchone()
    # Resurface refreshes the score columns (migration 0013): a later
    # gather's reranker output has to land on rows that were first seen
    # before the reranker was enabled. The return value still reports
    # 0 = "not new" so gather_runs.n_new / n_redup stay meaningful.
    assert row["score"] == pytest.approx(0.95)


def test_candidates_insert_dedup_refreshes_reranker_columns(conn):
    """Resurfacing a candidate writes the new reranker breakdown."""
    pid = _seed_profile_and_paper(conn)
    candidates.insert_dedup(conn, profile_id=pid, openalex_id="W1", score=0.9)
    row = conn.execute(
        "SELECT score_blended FROM profile_candidates "
        "WHERE profile_id=? AND openalex_id=?",
        (pid, "W1"),
    ).fetchone()
    assert row["score_blended"] is None

    # Second gather, this time with a reranker active.
    assert candidates.insert_dedup(
        conn, profile_id=pid, openalex_id="W1", score=0.62,
        score_reranker_raw=1.4, score_reranker_norm=0.8, score_blended=0.62,
    ) == 0
    row = conn.execute(
        "SELECT score, score_reranker_raw, score_reranker_norm, score_blended "
        "FROM profile_candidates WHERE profile_id=? AND openalex_id=?",
        (pid, "W1"),
    ).fetchone()
    assert row["score_reranker_raw"] == pytest.approx(1.4)
    assert row["score_reranker_norm"] == pytest.approx(0.8)
    assert row["score_blended"] == pytest.approx(0.62)
    assert row["score"] == pytest.approx(0.62)


def test_candidates_save_dismiss_toggles(conn):
    pid = _seed_profile_and_paper(conn)
    candidates.insert_dedup(conn, profile_id=pid, openalex_id="W1", score=0.9)
    assert candidates.mark_saved(conn, pid, "W1") == "saved"
    assert candidates.mark_saved(conn, pid, "W1") is None
    assert candidates.mark_dismissed(conn, pid, "W1") == "dismissed"
    # Switching from dismissed → saved clears dismissed_at.
    assert candidates.mark_saved(conn, pid, "W1") == "saved"
    row = conn.execute(
        "SELECT saved_at, dismissed_at FROM profile_candidates WHERE profile_id=? AND openalex_id=?",
        (pid, "W1"),
    ).fetchone()
    assert row["saved_at"] is not None
    assert row["dismissed_at"] is None


def test_candidates_unshown_excludes_triaged(conn):
    pid = _seed_profile_and_paper(conn, "W1")
    papers.upsert(conn, {"openalex_id": "W2", "title": "T2", "source": "openalex_gatherer"})
    candidates.insert_dedup(conn, profile_id=pid, openalex_id="W1", score=0.9)
    candidates.insert_dedup(conn, profile_id=pid, openalex_id="W2", score=0.8)
    candidates.mark_dismissed(conn, pid, "W2")
    rows = candidates.unshown_for_profile(conn, pid)
    assert [r["openalex_id"] for r in rows] == ["W1"]


def test_candidates_mark_shown_bulk(conn):
    pid = _seed_profile_and_paper(conn, "W1")
    papers.upsert(conn, {"openalex_id": "W2", "title": "T2", "source": "openalex_gatherer"})
    candidates.insert_dedup(conn, profile_id=pid, openalex_id="W1", score=0.9)
    candidates.insert_dedup(conn, profile_id=pid, openalex_id="W2", score=0.8)
    n = candidates.mark_shown(conn, pid, ["W1", "W2"])
    assert n == 2
    # Calling again with the same ids: shown_at already set, so 0 rows updated.
    assert candidates.mark_shown(conn, pid, ["W1", "W2"]) == 0


# --------------------------------------------------------------- gather_runs

def test_gather_runs_start_finish(conn):
    pid = _seed_profile_and_paper(conn)
    rid = gather_runs.start(
        conn, profile_id=pid, user_id=1,
        since_date="2026-04-01", filter_string="topic:T1", tier_used="must-have-AND",
    )
    assert rid >= 1
    gather_runs.finish(
        conn, rid, n_fetched=10, n_new=7, n_redup=3, api_calls=5,
    )
    row = gather_runs.get(conn, rid)
    assert row["finished_at"] is not None
    assert row["n_fetched"] == 10
    assert row["n_new"] == 7
    assert row["n_redup"] == 3
    assert row["error"] is None


def test_gather_runs_finish_records_error(conn):
    pid = _seed_profile_and_paper(conn)
    rid = gather_runs.start(conn, profile_id=pid, user_id=1)
    gather_runs.finish(
        conn, rid, n_fetched=0, n_new=0, n_redup=0, error="HTTP 429",
    )
    row = gather_runs.get(conn, rid)
    assert row["error"] == "HTTP 429"
