"""profiles repo.

Profile metadata + centroid + selector/gatherer config + coherence
diagnostics. Embeddings live in ``paper_embeddings``; seed membership
lives in ``profile_seeds``.

Profile JSON (the existing on-disk artifact from Phase 1B) maps to this
schema directly. Phase 2's ``db_store`` is the bridge.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any


def slugify(name: str) -> str:
    """Profile slug: lowercase, alnum + dashes, collapse runs.

    Used by the CLI when creating new profiles and by the wizard
    (Phase 11) when committing a draft. The 0002_users migration
    backfilled existing rows directly in SQL using the same shape.
    """
    s = name.strip().lower()
    # Underscores collapse into hyphens to match the 0002_users SQL backfill,
    # which rewrites both spaces and underscores as hyphens.
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "profile"


def get(conn: sqlite3.Connection, profile_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM profiles WHERE id = ?", (profile_id,)
    ).fetchone()


def get_by_slug(
    conn: sqlite3.Connection, user_id: int, slug: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM profiles WHERE user_id = ? AND slug = ?",
        (user_id, slug),
    ).fetchone()


def get_by_name(
    conn: sqlite3.Connection, user_id: int, name: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM profiles WHERE user_id = ? AND name = ?",
        (user_id, name),
    ).fetchone()


def list_for_user(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    include_drafts: bool = False,
) -> list[sqlite3.Row]:
    """Return the user's profiles, drafts hidden by default.

    The wizard (Phase 11) opens a draft as a real ``profiles`` row with
    ``is_draft=1`` so the seeds it accumulates can hang off
    ``profile_seeds`` without a parallel scratch table. Read paths must
    not surface drafts to the sidebar / Radar / detail UI; only the
    wizard endpoints opt in via ``include_drafts=True``.
    """
    if include_drafts:
        return conn.execute(
            "SELECT * FROM profiles WHERE user_id = ? ORDER BY id",
            (user_id,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM profiles WHERE user_id = ? AND is_draft = 0 ORDER BY id",
        (user_id,),
    ).fetchall()


def create_draft(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    name: str,
    slug: str,
    embedding_model: str = "placeholder-v1",
) -> int:
    """Insert a draft profiles row. Returns the new id.

    Topics, threshold, and centroid stay NULL until the wizard's commit
    step calls ``upsert`` to fill them in. ``n_seed`` is 0 at create
    time and updated at commit (the wizard counts seeds via
    ``profile_seeds`` so we don't need to keep this in sync mid-wizard).
    """
    cur = conn.execute(
        """
        INSERT INTO profiles (
          user_id, name, slug, embedding_model, topic_filters_json,
          n_seed, is_draft, threshold
        ) VALUES (?, ?, ?, ?, '{}', 0, 1, 0.7)
        """,
        (user_id, name, slug, embedding_model),
    )
    conn.commit()
    return int(cur.lastrowid)


def commit_draft(
    conn: sqlite3.Connection,
    profile_id: int,
    *,
    threshold: float | None,
    topic_filters: dict | None,
    centroid: bytes | None,
    selector_config: dict | None,
    coherence_median: float | None,
    coherence_iqr: float | None,
    coherence_bimodal: bool | None,
    n_seed: int,
) -> None:
    """Flip a draft to a real profile and persist its calibrated fields.

    Called by the wizard's commit endpoint after the user picks a
    threshold and selected topics. Centroid + selector_config are the
    output of ``CentroidSelector.fit`` serialized for fast restart.
    """
    tf_json = json.dumps(topic_filters or {})
    sel_json = json.dumps(selector_config) if selector_config is not None else None
    bimodal_int = (
        None if coherence_bimodal is None else (1 if coherence_bimodal else 0)
    )
    conn.execute(
        """
        UPDATE profiles SET
          is_draft           = 0,
          threshold          = ?,
          topic_filters_json = ?,
          centroid           = ?,
          selector_config_json = ?,
          coherence_median   = ?,
          coherence_iqr      = ?,
          coherence_bimodal  = ?,
          n_seed             = ?,
          updated_at         = datetime('now')
        WHERE id = ?
        """,
        (
            threshold, tf_json, centroid, sel_json,
            coherence_median, coherence_iqr, bimodal_int,
            n_seed, profile_id,
        ),
    )
    conn.commit()


def delete(conn: sqlite3.Connection, profile_id: int) -> None:
    """Hard-delete a profile. ``profile_seeds`` rows go via ON DELETE CASCADE."""
    conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
    conn.commit()


def upsert(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    name: str,
    embedding_model: str,
    n_seed: int,
    topic_filters: dict[str, Any] | None = None,
    centroid: bytes | None = None,
    threshold: float | None = None,
    selector_config: dict | None = None,
    gatherer_config: dict | None = None,
    coherence_median: float | None = None,
    coherence_iqr: float | None = None,
    coherence_bimodal: bool | None = None,
    slug: str | None = None,
) -> int:
    """Create or update a profile keyed by ``(user_id, name)``. Returns id.

    Slug defaults to ``slugify(name)`` when not given. JSON-typed columns
    are encoded here; callers pass plain dicts.
    """
    slug = slug or slugify(name)
    tf_json = json.dumps(topic_filters or {})
    sel_json = json.dumps(selector_config) if selector_config is not None else None
    gat_json = json.dumps(gatherer_config) if gatherer_config is not None else None
    bimodal_int = (
        None if coherence_bimodal is None else (1 if coherence_bimodal else 0)
    )

    existing = get_by_name(conn, user_id, name)
    if existing is None:
        cur = conn.execute(
            """
            INSERT INTO profiles (
              user_id, name, slug, embedding_model, centroid, threshold,
              topic_filters_json, selector_config_json, gatherer_config_json,
              coherence_median, coherence_iqr, coherence_bimodal, n_seed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, name, slug, embedding_model, centroid, threshold,
                tf_json, sel_json, gat_json,
                coherence_median, coherence_iqr, bimodal_int, n_seed,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    conn.execute(
        """
        UPDATE profiles SET
          embedding_model = ?, centroid = ?, threshold = ?,
          topic_filters_json = ?, selector_config_json = ?,
          gatherer_config_json = ?, coherence_median = ?, coherence_iqr = ?,
          coherence_bimodal = ?, n_seed = ?, slug = ?,
          updated_at = datetime('now')
        WHERE id = ?
        """,
        (
            embedding_model, centroid, threshold,
            tf_json, sel_json, gat_json,
            coherence_median, coherence_iqr, bimodal_int, n_seed, slug,
            existing["id"],
        ),
    )
    conn.commit()
    return int(existing["id"])


def attach_seed(
    conn: sqlite3.Connection, profile_id: int, openalex_id: str
) -> int:
    """Idempotent. Returns 1 if a new seed row was created, 0 if already present."""
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO profile_seeds (profile_id, openalex_id)
        VALUES (?, ?)
        """,
        (profile_id, openalex_id),
    )
    conn.commit()
    return cur.rowcount


def list_seed_openalex_ids(
    conn: sqlite3.Connection, profile_id: int
) -> list[str]:
    rows = conn.execute(
        "SELECT openalex_id FROM profile_seeds WHERE profile_id = ? ORDER BY added_at",
        (profile_id,),
    ).fetchall()
    return [r["openalex_id"] for r in rows]


def update_threshold(
    conn: sqlite3.Connection, profile_id: int, threshold: float
) -> None:
    conn.execute(
        "UPDATE profiles SET threshold = ?, updated_at = datetime('now') WHERE id = ?",
        (threshold, profile_id),
    )
    conn.commit()


def topic_filters(conn: sqlite3.Connection, profile_id: int) -> dict:
    row = conn.execute(
        "SELECT topic_filters_json FROM profiles WHERE id = ?", (profile_id,)
    ).fetchone()
    if row is None or row["topic_filters_json"] is None:
        return {}
    return json.loads(row["topic_filters_json"])


def coherence_bins(
    conn: sqlite3.Connection, profile_id: int, *, n_bins: int = 16
) -> list[int]:
    """16-bin pairwise-cosine histogram of the seed embeddings.

    Cosines are bucketed across ``[0.0, 1.0]`` (cosine on SPECTER2
    embeddings is already non-negative for any sane seed corpus).
    Returns an empty list when there are fewer than two seed vectors.

    The heavy lift here is loading the seed embeddings; for the API's
    ``getProfileDetail`` path this is fine because each profile is fit
    once and the bin counts could be cached on the row in a follow-up.
    """
    import numpy as np

    from ..codec import decode_vector

    profile_row = conn.execute(
        "SELECT embedding_model FROM profiles WHERE id = ?", (profile_id,)
    ).fetchone()
    if profile_row is None:
        return []
    model = profile_row["embedding_model"]

    rows = conn.execute(
        """
        SELECT pe.vector
        FROM profile_seeds ps
        JOIN paper_embeddings pe USING (openalex_id)
        WHERE ps.profile_id = ? AND pe.embedding_model = ?
        """,
        (profile_id, model),
    ).fetchall()
    if len(rows) < 2:
        return []

    vecs = np.vstack([decode_vector(r["vector"]) for r in rows]).astype(float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / (norms + 1e-12)
    sim = normed @ normed.T
    iu = np.triu_indices(vecs.shape[0], k=1)
    cos = sim[iu]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    hist, _ = np.histogram(np.clip(cos, 0.0, 1.0), bins=edges)
    return [int(c) for c in hist]
