"""Unit tests for the persistence bridge (rag_lib/persistence/db_store.py)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from rag_lib.db import apply_migrations
from rag_lib.db.repos import (
    candidates as candidates_repo,
    embeddings as embeddings_repo,
    papers as papers_repo,
    profiles as profiles_repo,
)
from rag_lib.paper import Paper, Topic, TopicNode
from rag_lib.persistence import (
    dedup_and_insert_candidates,
    resolve_user_id,
    store_papers,
    store_profile_from_object,
)
from rag_lib.profile import Profile


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    apply_migrations(c)
    yield c
    c.close()


def _topic(tid: str = "T1") -> Topic:
    return Topic(
        id=tid, display_name=f"Topic {tid}",
        subfield=TopicNode(id="SF1", display_name="Subfield"),
        field=TopicNode(id="F1", display_name="Field"),
        domain=TopicNode(id="D1", display_name="Domain"),
        score=0.9,
    )


def _paper(oa_id: str = "W1", *, with_embedding: bool = True) -> Paper:
    p = Paper(
        doi=f"10.1/{oa_id.lower()}",
        openalex_id=oa_id,
        title=f"Paper {oa_id}",
        abstract="abs",
        year=2024,
        venue="J",
        primary_topic=_topic(),
        topics=[_topic()],
        source="openalex_gatherer",
    )
    if with_embedding:
        p.embeddings["specter2"] = list(np.random.RandomState(int(oa_id[1:])).rand(8).astype(float))
    return p


# ------------------------------------------------------------------ resolve

def test_resolve_user_id_creates_then_reuses(conn):
    uid1 = resolve_user_id(conn, "alice@example.com")
    uid2 = resolve_user_id(conn, "ALICE@example.com")  # case-insensitive
    assert uid1 == uid2
    # Distinct user gets a distinct id.
    uid3 = resolve_user_id(conn, "bob@example.com")
    assert uid3 != uid1


# ------------------------------------------------------------------- papers

def test_store_papers_writes_paper_and_embedding(conn):
    n = store_papers(conn, [_paper("W1"), _paper("W2")])
    assert n == 2
    assert papers_repo.get_by_openalex_id(conn, "W1")["title"] == "Paper W1"
    assert embeddings_repo.has(conn, "W1", "specter2")
    assert embeddings_repo.has(conn, "W2", "specter2")


def test_store_papers_skips_missing_openalex_id(conn, capsys):
    p = _paper("W1")
    p.openalex_id = None
    n = store_papers(conn, [p, _paper("W2")])
    assert n == 1
    captured = capsys.readouterr()
    assert "skipping" in captured.err
    assert papers_repo.get_by_openalex_id(conn, "W2") is not None


def test_store_papers_does_not_re_embed_when_present(conn):
    p = _paper("W1")
    store_papers(conn, [p])
    # Mutate the in-memory vector after first write — second store_papers
    # call should NOT overwrite the persisted vector.
    original = embeddings_repo.get(conn, "W1", "specter2")
    p.embeddings["specter2"] = [9.0] * 8
    store_papers(conn, [p], skip_re_embed=True)
    after = embeddings_repo.get(conn, "W1", "specter2")
    np.testing.assert_allclose(after, original)


def test_store_papers_re_embeds_when_skip_false(conn):
    p = _paper("W1")
    store_papers(conn, [p])
    p.embeddings["specter2"] = [9.0] * 8
    store_papers(conn, [p], skip_re_embed=False)
    after = embeddings_repo.get(conn, "W1", "specter2")
    np.testing.assert_allclose(after, [9.0] * 8)


# ----------------------------------------------------------------- profiles

def test_store_profile_round_trip(conn):
    user_id = resolve_user_id(conn, "demo@example.com")
    profile = Profile(
        name="neonatal_vitals",
        papers=[_paper("W1"), _paper("W2")],
        topic_filters={"topics": [{"id": "T1", "display_name": "X", "count": 2}]},
        embedding_model="specter2",
        selector_config={
            "type": "centroid",
            "centroid": [0.1, 0.2, 0.3, 0.4],
            "diagnostics_snapshot": {
                "coherence_median": 0.92,
                "coherence_iqr": 0.05,
                "coherence_bimodal": False,
            },
        },
        gatherer_config={"type": "openalex", "mailto": "demo@example.com"},
        threshold=0.85,
    )
    pid = store_profile_from_object(conn, user_id=user_id, profile=profile)
    assert pid >= 1

    row = profiles_repo.get(conn, pid)
    assert row["name"] == "neonatal_vitals"
    assert row["slug"] == "neonatal-vitals"
    assert row["embedding_model"] == "specter2"
    assert row["n_seed"] == 2
    assert row["threshold"] == pytest.approx(0.85)
    assert row["coherence_median"] == pytest.approx(0.92)
    assert row["coherence_bimodal"] == 0
    assert row["centroid"] is not None
    decoded = np.frombuffer(row["centroid"], dtype=np.float32)
    np.testing.assert_allclose(decoded, [0.1, 0.2, 0.3, 0.4], rtol=0, atol=1e-6)

    seeds = profiles_repo.list_seed_openalex_ids(conn, pid)
    assert sorted(seeds) == ["W1", "W2"]


def test_store_profile_idempotent_on_name(conn):
    user_id = resolve_user_id(conn, "demo@example.com")
    profile = Profile(
        name="P", papers=[_paper("W1")], embedding_model="specter2",
        topic_filters={}, selector_config={"type": "x"},
    )
    pid1 = store_profile_from_object(conn, user_id=user_id, profile=profile)
    profile.papers.append(_paper("W2"))
    pid2 = store_profile_from_object(conn, user_id=user_id, profile=profile)
    assert pid1 == pid2
    seeds = profiles_repo.list_seed_openalex_ids(conn, pid1)
    assert sorted(seeds) == ["W1", "W2"]
    assert profiles_repo.get(conn, pid1)["n_seed"] == 2


# --------------------------------------------------------------- candidates

def test_dedup_and_insert_first_run_all_new(conn):
    user_id = resolve_user_id(conn, "demo@example.com")
    pid = store_profile_from_object(conn, user_id=user_id, profile=Profile(
        name="P", papers=[_paper("W0")], embedding_model="specter2",
        topic_filters={}, selector_config={"type": "x"},
    ))
    ranked = [(0.9, _paper("W10")), (0.8, _paper("W11"))]
    n_new, n_redup = dedup_and_insert_candidates(
        conn, profile_id=pid, gather_run_id=None,
        ranked=ranked, tier_used="must-have-AND",
    )
    assert (n_new, n_redup) == (2, 0)
    assert candidates_repo.count_for_profile(conn, pid) == 2
    # Candidate papers also got papers + embeddings rows.
    assert papers_repo.get_by_openalex_id(conn, "W10") is not None
    assert embeddings_repo.has(conn, "W10", "specter2")


def test_dedup_preserves_prior_triage(conn):
    user_id = resolve_user_id(conn, "demo@example.com")
    pid = store_profile_from_object(conn, user_id=user_id, profile=Profile(
        name="P", papers=[_paper("W0")], embedding_model="specter2",
        topic_filters={}, selector_config={"type": "x"},
    ))
    # First gather: two candidates
    dedup_and_insert_candidates(
        conn, profile_id=pid, gather_run_id=None,
        ranked=[(0.9, _paper("W10")), (0.8, _paper("W11"))],
    )
    # User saves one
    candidates_repo.mark_saved(conn, pid, "W10")
    # Second gather: same two candidates plus a new one
    n_new, n_redup = dedup_and_insert_candidates(
        conn, profile_id=pid, gather_run_id=None,
        ranked=[
            (0.95, _paper("W10")),  # higher score on resurface — refreshed
            (0.85, _paper("W11")),
            (0.7, _paper("W12")),
        ],
    )
    assert (n_new, n_redup) == (1, 2)
    # Triage survives the resurface, but the score columns are refreshed
    # to the latest gather's values so reranker output lands on rows that
    # predate it.
    row = conn.execute(
        "SELECT score, saved_at FROM profile_candidates WHERE profile_id=? AND openalex_id=?",
        (pid, "W10"),
    ).fetchone()
    assert row["saved_at"] is not None
    assert row["score"] == pytest.approx(0.95)


def test_dedup_skips_papers_without_openalex_id(conn):
    user_id = resolve_user_id(conn, "demo@example.com")
    pid = store_profile_from_object(conn, user_id=user_id, profile=Profile(
        name="P", papers=[_paper("W0")], embedding_model="specter2",
        topic_filters={}, selector_config={"type": "x"},
    ))
    bad = _paper("W10")
    bad.openalex_id = None
    n_new, n_redup = dedup_and_insert_candidates(
        conn, profile_id=pid, gather_run_id=None,
        ranked=[(0.9, bad), (0.8, _paper("W11"))],
    )
    assert (n_new, n_redup) == (1, 0)
