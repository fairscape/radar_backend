"""Phase 8 — feedback-driven threshold autotune (sketch).

Synthetic distribution: 60 events where saves cluster around 0.85 and
dismisses cluster around 0.55. The recommended threshold should land
between the two clusters (>= 0.6, <= 0.85). Below the min-events
floor, the recommendation must be ``None``.
"""

from __future__ import annotations

import sqlite3

import pytest

from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import (
    feedback as feedback_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)
from rag_lib.feedback.autotune import (
    apply_recommended,
    recommend_threshold,
)


def _seed_min(db_path) -> tuple[int, int]:
    """Bare-minimum seed: one user, one profile, one paper. Returns (user_id, profile_id)."""
    conn = connect(db_path)
    apply_migrations(conn)
    user = users_repo.upsert(conn, "demo@example.com")
    user_id = int(user["id"])
    profile_id = profiles_repo.upsert(
        conn,
        user_id=user_id,
        name="autotune-test",
        embedding_model="placeholder-v1",
        n_seed=0,
        topic_filters={"topics": []},
        threshold=0.50,
    )
    papers_repo.upsert(conn, {
        "openalex_id": "W3000000",
        "title": "anchor",
        "year": 2026,
        "source": "openalex_gatherer",
    })
    conn.close()
    return user_id, profile_id


def _conn(db_path) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def _insert_events(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    profile_id: int,
    saves: list[float],
    dismisses: list[float],
) -> None:
    for s in saves:
        feedback_repo.insert(
            conn,
            user_id=user_id, profile_id=profile_id,
            openalex_id="W3000000", action="saved", score=s,
        )
    for s in dismisses:
        feedback_repo.insert(
            conn,
            user_id=user_id, profile_id=profile_id,
            openalex_id="W3000000", action="dismissed", score=s,
        )


def test_recommend_threshold_returns_none_when_too_few_events(tmp_path):
    user_id, profile_id = _seed_min(tmp_path / "radar.db")
    conn = _conn(tmp_path / "radar.db")
    _insert_events(
        conn, user_id=user_id, profile_id=profile_id,
        saves=[0.9] * 5, dismisses=[0.5] * 5,
    )
    assert recommend_threshold(conn, profile_id, min_events=30) is None
    conn.close()


def test_recommend_threshold_finds_separation_band(tmp_path):
    """Saves at ~0.85, dismisses at ~0.55. Threshold should land in the gap."""
    user_id, profile_id = _seed_min(tmp_path / "radar.db")
    conn = _conn(tmp_path / "radar.db")
    saves = [round(0.80 + i * 0.005, 3) for i in range(30)]      # 0.800..0.945
    dismisses = [round(0.50 + i * 0.003, 3) for i in range(30)]  # 0.500..0.587
    _insert_events(
        conn, user_id=user_id, profile_id=profile_id,
        saves=saves, dismisses=dismisses,
    )
    th = recommend_threshold(conn, profile_id, min_events=30)
    assert th is not None
    # The dismiss cluster tops out near 0.587 and the save cluster starts
    # at 0.800; any threshold strictly between should achieve F1=1.0,
    # so the recommendation should land in that gap.
    assert 0.59 <= th <= 0.80, f"unexpected recommended threshold {th}"
    conn.close()


def test_recommend_threshold_with_overlapping_distributions(tmp_path):
    """Even when distributions overlap, the recommendation stays in [0.5, 0.95]."""
    user_id, profile_id = _seed_min(tmp_path / "radar.db")
    conn = _conn(tmp_path / "radar.db")
    saves = [round(0.65 + (i % 7) * 0.04, 3) for i in range(30)]
    dismisses = [round(0.55 + (i % 9) * 0.03, 3) for i in range(30)]
    _insert_events(
        conn, user_id=user_id, profile_id=profile_id,
        saves=saves, dismisses=dismisses,
    )
    th = recommend_threshold(conn, profile_id, min_events=30)
    assert th is not None
    assert 0.50 <= th <= 0.95
    conn.close()


def test_apply_recommended_writes_threshold(tmp_path):
    user_id, profile_id = _seed_min(tmp_path / "radar.db")
    conn = _conn(tmp_path / "radar.db")
    saves = [round(0.80 + i * 0.005, 3) for i in range(30)]
    dismisses = [round(0.50 + i * 0.003, 3) for i in range(30)]
    _insert_events(
        conn, user_id=user_id, profile_id=profile_id,
        saves=saves, dismisses=dismisses,
    )
    result = apply_recommended(conn, profile_id, min_events=30)
    assert result["applied"] is True
    assert result["old"] == pytest.approx(0.50)
    assert 0.59 <= result["new"] <= 0.80
    assert result["n_events"] == 60

    row = conn.execute(
        "SELECT threshold FROM profiles WHERE id = ?", (profile_id,)
    ).fetchone()
    assert row["threshold"] == pytest.approx(result["new"])
    conn.close()


def test_apply_recommended_skips_when_insufficient(tmp_path):
    user_id, profile_id = _seed_min(tmp_path / "radar.db")
    conn = _conn(tmp_path / "radar.db")
    _insert_events(
        conn, user_id=user_id, profile_id=profile_id,
        saves=[0.9] * 3, dismisses=[0.5] * 3,
    )
    result = apply_recommended(conn, profile_id, min_events=30)
    assert result["applied"] is False
    assert result["new"] is None
    row = conn.execute(
        "SELECT threshold FROM profiles WHERE id = ?", (profile_id,)
    ).fetchone()
    assert row["threshold"] == pytest.approx(0.50)
    conn.close()
