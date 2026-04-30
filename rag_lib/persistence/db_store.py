"""Profile/Paper ↔ SQLite bridge.

Two write paths through this module:

  store_profile_from_object — write a Profile and its seed papers (with
    embeddings + topics) to the DB; idempotent on (user_id, name).
  dedup_and_insert_candidates — write a list of (score, Paper) results
    from a gather run, dedup against profile_candidates so the user's
    prior triage state survives a re-fetch.

Both call into ``rag_lib.db.repos.*`` which is the only direct user of
sqlite3. Higher layers (CLI, API services) never construct SQL.

Papers without an ``openalex_id`` are skipped — the schema is
content-addressed by that key. In practice this only matters for CSV
seeds that never resolved on OpenAlex (paywalled, not indexed); the
runtime warning surfaces it.
"""

from __future__ import annotations

import sqlite3
import sys

from ..db import encode_vector
from ..db.repos import (
    candidates as candidates_repo,
    embeddings as embeddings_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    schedules as schedules_repo,
    users as users_repo,
)
from ..paper import Paper
from ..profile import Profile


def resolve_user_id(conn: sqlite3.Connection, email: str) -> int:
    """Get-or-create a user by email; return its id."""
    return int(users_repo.upsert(conn, email)["id"])


def _paper_dict_for_repo(paper: Paper) -> dict:
    d = paper.to_dict()
    return {
        "openalex_id": d.get("openalex_id"),
        "doi": d.get("doi"),
        "title": d.get("title") or "",
        "abstract": d.get("abstract"),
        "year": d.get("year"),
        "venue": d.get("venue"),
        "publication_date": None,  # Paper has no publication_date field today
        "primary_topic": d.get("primary_topic"),
        "topics": d.get("topics") or [],
        "source": d.get("source") or "unknown",
        "local_path": d.get("local_path"),
        "body_text": d.get("body_text"),
        "pdf_url": d.get("pdf_url"),
        "oa_status": d.get("oa_status"),
    }


def store_papers(
    conn: sqlite3.Connection,
    papers: list[Paper],
    *,
    skip_re_embed: bool = True,
) -> int:
    """Upsert each paper and any embeddings it carries.

    With ``skip_re_embed=True`` (default), an embedding is written only
    if no row exists for ``(openalex_id, model)``. This is the path the
    gather CLI uses to avoid recomputing SPECTER2 vectors for papers we
    have already seen.

    Returns the count of papers written (excluding skipped ones).
    """
    written = 0
    for p in papers:
        if not p.openalex_id:
            print(f"warn: skipping paper without openalex_id: {p.title!r}",
                  file=sys.stderr)
            continue
        papers_repo.upsert(conn, _paper_dict_for_repo(p))
        for model, vec in (p.embeddings or {}).items():
            if skip_re_embed and embeddings_repo.has(conn, p.openalex_id, model):
                continue
            embeddings_repo.upsert(conn, p.openalex_id, model, vec)
        written += 1
    return written


def store_profile_from_object(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    profile: Profile,
) -> int:
    """Persist a Profile and its seed corpus. Returns ``profile_id``.

    The fitted centroid (when present in ``profile.selector_config``) is
    stored as a float32 BLOB on ``profiles.centroid``; the full
    selector_config still goes into ``selector_config_json`` so nothing
    is lost. Coherence diagnostics, when surfaced by the selector,
    populate the dedicated columns for fast read-side queries.
    """
    sel_cfg = profile.selector_config or {}

    centroid_blob: bytes | None = None
    centroid_list = sel_cfg.get("centroid") if isinstance(sel_cfg, dict) else None
    if centroid_list:
        centroid_blob = encode_vector(centroid_list)

    diag = sel_cfg.get("diagnostics_snapshot") if isinstance(sel_cfg, dict) else None
    diag = diag or {}
    coh_med = diag.get("coherence_median")
    coh_iqr = diag.get("coherence_iqr")
    coh_bm = diag.get("coherence_bimodal")

    profile_id = profiles_repo.upsert(
        conn,
        user_id=user_id,
        name=profile.name,
        embedding_model=profile.embedding_model,
        n_seed=len(profile.papers),
        topic_filters=profile.topic_filters,
        centroid=centroid_blob,
        threshold=profile.threshold,
        selector_config=sel_cfg or None,
        gatherer_config=profile.gatherer_config or None,
        coherence_median=coh_med,
        coherence_iqr=coh_iqr,
        coherence_bimodal=coh_bm,
    )

    store_papers(conn, profile.papers)
    for paper in profile.papers:
        if not paper.openalex_id:
            continue
        profiles_repo.attach_seed(conn, profile_id, paper.openalex_id)

    # Ensure every profile has a default daily schedule. Idempotent: the
    # 0004 migration backfilled existing rows; this covers profiles
    # created post-migration (CLI gather, Phase 11 wizard) so the
    # scheduler picks them up on the next boot without manual setup.
    if schedules_repo.get(conn, profile_id) is None:
        schedules_repo.upsert(conn, profile_id=profile_id)

    return profile_id


def dedup_and_insert_candidates(
    conn: sqlite3.Connection,
    *,
    profile_id: int,
    gather_run_id: int | None,
    ranked: list[tuple],
    tier_used: str | None = None,
) -> tuple[int, int]:
    """Persist ranked gather results and dedup against prior candidates.

    ``ranked`` accepts either the legacy 2-tuple shape ``(score, Paper)``
    or the Phase 3 3-tuple ``(score, Paper, breakdown)`` where
    ``breakdown`` may carry ``score_raw``, ``score_max_seed``, and
    ``score_pct`` floats. Missing breakdown fields write ``NULL`` to
    their respective columns.

    Each entry triggers (in order):
      1. ``papers.upsert`` (idempotent; later sources don't blank fields).
      2. ``embeddings.upsert`` for every embedding the paper carries
         (skipped when an entry for that model already exists).
      3. ``candidates.insert_dedup`` — INSERT OR IGNORE so prior
         ``shown_at``/``saved_at``/``dismissed_at`` survive resurface.

    Returns ``(n_new, n_redup)``.
    """
    store_papers(conn, [_paper_from_entry(e) for e in ranked])

    n_new = 0
    n_redup = 0
    for entry in ranked:
        score, paper, breakdown = _unpack_entry(entry)
        if not paper.openalex_id:
            continue
        added = candidates_repo.insert_dedup(
            conn,
            profile_id=profile_id,
            openalex_id=paper.openalex_id,
            score=float(score),
            tier_used=tier_used,
            gather_run_id=gather_run_id,
            score_raw=_breakdown_float(breakdown, "score_raw"),
            score_max_seed=_breakdown_float(breakdown, "score_max_seed"),
            score_pct=_breakdown_float(breakdown, "score_pct"),
        )
        if added:
            n_new += 1
        else:
            n_redup += 1
    return n_new, n_redup


def _unpack_entry(entry) -> tuple[float, Paper, dict]:
    """Accept either 2-tuple or 3-tuple ranked entries; return canonical 3-tuple."""
    if len(entry) == 3:
        score, paper, breakdown = entry
        return score, paper, dict(breakdown or {})
    score, paper = entry
    return score, paper, {}


def _paper_from_entry(entry) -> Paper:
    return entry[1]


def _breakdown_float(breakdown: dict, key: str) -> float | None:
    v = breakdown.get(key)
    return None if v is None else float(v)
