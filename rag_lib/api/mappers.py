"""Row → Pydantic adapters.

Thin functions that translate ``sqlite3.Row`` (from ``rag_lib.db.repos``)
into the API schemas. Routers call these so the SQL layer stays
ignorant of HTTP and the schemas stay ignorant of SQL.

Conventions:
- Every adapter takes ``row`` first and any computed values as kwargs.
- Empty / unknown fields default to safe values (empty string, empty
  list, ``"low"`` bucket) — never None where the frontend expects a
  primitive.
- ``hue`` is derived from a stable hash of the slug (no DB column).
- ``bucket`` is computed from ``score`` using the constants in
  ``schemas.py``.
- ``health`` is derived from coherence + 30d feedback counts.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Iterable

from .schemas import (
    BUCKET_HIGH,
    BUCKET_MEDIUM,
    Bucket,
    Card,
    HealthStatus,
    Profile,
    Seed,
    SweepRow,
    Topic,
)


def _hue_from_slug(slug: str) -> int:
    """Stable hash of slug into [0, 360). Avoids a DB column for hue."""
    h = hashlib.sha1(slug.encode("utf-8")).digest()
    return int.from_bytes(h[:2], "big") % 360


def _bucket_for(score: float) -> Bucket:
    if score >= BUCKET_HIGH:
        return "high"
    if score >= BUCKET_MEDIUM:
        return "medium"
    return "low"


def _health_for(
    coherence_median: float | None, saves30: int, dismisses30: int
) -> HealthStatus:
    """err < 0.5 < warn < 0.6 < ok-only-when-engaged.

    Engagement bonus: even with healthy coherence, fall back to ``warn``
    when the user hasn't interacted with the profile in the last 30
    days — matches the spec note in phase-05 doc.
    """
    if coherence_median is None:
        return "warn"
    if coherence_median < 0.5:
        return "err"
    if coherence_median < 0.6:
        return "warn"
    if (saves30 + dismisses30) == 0:
        return "warn"
    return "ok"


def profile_row_to_profile(
    row: sqlite3.Row,
    *,
    saves30: int,
    dismisses30: int,
) -> Profile:
    slug = row["slug"] if row["slug"] else row["name"]
    coh = row["coherence_median"]
    return Profile(
        key=slug,
        name=row["name"],
        hue=_hue_from_slug(slug),
        health=_health_for(coh, saves30, dismisses30),
        threshold=float(row["threshold"]) if row["threshold"] is not None else 0.0,
        coherence=float(coh) if coh is not None else 0.0,
        seeds=int(row["n_seed"] or 0),
        saves30=saves30,
        dismisses30=dismisses30,
        isDraft=bool(row["is_draft"]),
    )


def _decode_topics(topics_json: str | None) -> dict[str, Any]:
    if not topics_json:
        return {"primary_topic": None, "topics": []}
    try:
        return json.loads(topics_json)
    except json.JSONDecodeError:
        return {"primary_topic": None, "topics": []}


def _topic_terms(topics_payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    primary = topics_payload.get("primary_topic")
    if primary and primary.get("display_name"):
        out.append(primary["display_name"])
    for t in topics_payload.get("topics") or []:
        if t and t.get("display_name") and t["display_name"] not in out:
            out.append(t["display_name"])
    return out


def _matched_terms(
    topics_payload: dict[str, Any],
    active_topic_ids: Iterable[str],
) -> list[str]:
    active = {t for t in active_topic_ids if t}
    if not active:
        return []
    out: list[str] = []
    primary = topics_payload.get("primary_topic")
    if primary and primary.get("id") in active and primary.get("display_name"):
        out.append(primary["display_name"])
    for t in topics_payload.get("topics") or []:
        if t and t.get("id") in active and t.get("display_name") and t["display_name"] not in out:
            out.append(t["display_name"])
    return out


def _topic_match_for(
    topics_payload: dict[str, Any],
    active_topic_ids: Iterable[str],
) -> float:
    """Fraction of the candidate's topics that overlap the profile's active set."""
    active = {t for t in active_topic_ids if t}
    if not active:
        return 0.0
    candidate_ids: set[str] = set()
    primary = topics_payload.get("primary_topic")
    if primary and primary.get("id"):
        candidate_ids.add(primary["id"])
    for t in topics_payload.get("topics") or []:
        if t and t.get("id"):
            candidate_ids.add(t["id"])
    if not candidate_ids:
        return 0.0
    overlap = len(candidate_ids & active)
    return round(overlap / len(candidate_ids), 3)


def _read_minutes(abstract: str | None) -> int:
    """Heuristic reading time. ``max(2, round(words / 200))``."""
    if not abstract:
        return 2
    words = len(abstract.split())
    return max(2, round(words / 200))


def candidate_row_to_card(
    row: sqlite3.Row,
    *,
    profile_slug: str,
    active_topic_ids: Iterable[str],
) -> Card:
    """Build a ``Card`` from a ``profile_candidates JOIN papers`` row."""
    score = float(row["score"])
    topics_payload = _decode_topics(row["topics_json"])
    abstract = row["abstract"] or ""
    pub_date = row["publication_date"] or ""
    if not pub_date and row["year"] is not None:
        pub_date = f"{int(row['year'])}-01-01"

    terms = _topic_terms(topics_payload)
    matched = _matched_terms(topics_payload, active_topic_ids)

    return Card(
        id=row["openalex_id"],
        title=row["title"] or "",
        authors=[],  # papers table doesn't carry authors today; Phase 6+ enriches.
        venue=row["venue"] or "",
        date=pub_date,
        doi=row["doi"],
        openalex=row["openalex_id"],
        profile=profile_slug,
        score=round(score, 4),
        bucket=_bucket_for(score),
        abstract=abstract,
        mesh=[],
        terms=terms,
        matched=matched,
        topicMatch=_topic_match_for(topics_payload, active_topic_ids),
        centroidCos=round(score, 4),
        noveltyDelta=0.0,
        mins=_read_minutes(abstract),
    )


def seed_row_to_seed(
    row: sqlite3.Row,
    *,
    idx: int,
    coh_to_centroid: float | None = None,
) -> Seed:
    return Seed(
        id=row["openalex_id"],
        idx=idx,
        title=row["title"] or "",
        year=int(row["year"]) if row["year"] is not None else 0,
        coh=float(coh_to_centroid) if coh_to_centroid is not None else 0.0,
    )


def topic_filters_to_topics(topic_filters: dict | None) -> list[Topic]:
    """Read the ``profiles.topic_filters_json`` blob → ``Topic[]``.

    Uses the ``topics`` level (not subfields/fields/domains); those
    levels are filter knobs the gatherer uses, not user-facing tags.
    """
    if not topic_filters:
        return []
    items = topic_filters.get("topics") or []
    out: list[Topic] = []
    for it in items:
        tid = it.get("id") or ""
        name = it.get("display_name") or ""
        if not tid:
            continue
        out.append(Topic(
            id=tid,
            name=name,
            count=int(it.get("count") or 0),
            on=True,
        ))
    return out


def dry_run_result_to_sweep_rows(result: dict | None) -> list[SweepRow]:
    """Translate ``rag_lib.radar.dry_run`` output into ``SweepRow[]``."""
    if not result:
        return []
    out: list[SweepRow] = []
    for thr_str, payload in (result.get("results") or {}).items():
        try:
            th = float(thr_str)
        except (TypeError, ValueError):
            continue
        n = int(payload.get("count") or 0)
        top_titles = payload.get("top_titles") or []
        top = top_titles[0] if top_titles else ""
        out.append(SweepRow(th=th, n=n, top=top))
    out.sort(key=lambda r: r.th)
    return out
