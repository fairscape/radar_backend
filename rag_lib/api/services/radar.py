"""Radar service.

Reads ``profile_candidates`` joined with ``papers`` for the daily card
list, marks rows as shown on read, and toggles save/dismiss state via
the candidates repo.

Phase 8 added ``card_feedback_context`` so the router can build a
``FeedbackEvent`` without re-resolving the card (selector info, doi,
and ``score_pct`` come back in one query).
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone

from rag_lib.db.repos import (
    candidates as candidates_repo,
    profiles as profiles_repo,
)
from rag_lib.feedback.log import selector_config_hash

from ..mappers import candidate_row_to_card
from ..schemas import (
    Bucket,
    Card,
    CardActionResponse,
    CardState,
    DailyRadarResponse,
)


_DAILY_LIMIT = 200


def _today_label() -> str:
    """Mock-api formats date as 'THU 16 APR 2026'. Match that on the API."""
    return datetime.now(timezone.utc).strftime("%a %d %b %Y").upper()


def _active_topic_ids(topic_filters: dict | None) -> list[str]:
    if not topic_filters:
        return []
    items = topic_filters.get("topics") or []
    return [it["id"] for it in items if it.get("id")]


def _last_run_metrics(
    conn: sqlite3.Connection,
    profile_ids: list[int],
) -> tuple[str, int]:
    """``(fetchedAt, candidatesScored)`` aggregated over the latest run(s).

    For a single profile, takes the most-recent finished run.
    For ``profile=all`` (multiple ids), sums n_fetched and uses the most
    recent finished_at across runs.
    """
    if not profile_ids:
        return "", 0
    placeholders = ",".join("?" for _ in profile_ids)
    row = conn.execute(
        f"""
        SELECT
          MAX(finished_at) AS finished_at,
          SUM(COALESCE(n_fetched, 0)) AS scored
        FROM gather_runs
        WHERE profile_id IN ({placeholders}) AND finished_at IS NOT NULL
        """,
        profile_ids,
    ).fetchone()

    finished_at = row["finished_at"] or ""
    # The frontend renders "06:59:12"; pull the time half of an ISO ts.
    if "T" in finished_at:
        fetched_at = finished_at.split("T", 1)[1][:8]
    elif " " in finished_at:
        fetched_at = finished_at.split(" ", 1)[1][:8]
    else:
        fetched_at = finished_at[:8]

    return fetched_at, int(row["scored"] or 0)


def daily(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    profile_slug: str | None = None,
    bucket: Bucket | None = None,
) -> DailyRadarResponse:
    """Build the ``DailyRadarResponse`` for the user / profile / bucket.

    ``profile_slug=None`` or ``"all"`` returns cards across every profile
    the user owns. Bucketing happens once, in ``candidate_row_to_card``;
    the filter here reads the card's own value so the two cannot drift.
    Each returned card's ``shown_at`` is updated so the next gather run
    can dedup correctly.
    """
    t0 = time.perf_counter()

    if profile_slug and profile_slug != "all":
        row = profiles_repo.get_by_slug(conn, user_id, profile_slug)
        if row is None:
            return DailyRadarResponse(
                date=_today_label(),
                fetchedAt="",
                fetchMs=0,
                candidatesScored=0,
                cards=[],
                states={},
            )
        profile_rows = [row]
    else:
        profile_rows = profiles_repo.list_for_user(conn, user_id)

    cards: list[Card] = []
    states: dict[str, CardState | None] = {}
    profile_ids: list[int] = []

    per_profile_cap = max(1, _DAILY_LIMIT // max(1, len(profile_rows)))

    for prow in profile_rows:
        pid = int(prow["id"])
        profile_ids.append(pid)
        slug = prow["slug"] or prow["name"]
        topic_filters = profiles_repo.topic_filters(conn, pid)
        active = _active_topic_ids(topic_filters)
        rows = candidates_repo.top_for_profile(conn, pid, limit=per_profile_cap)
        for crow in rows:
            card = candidate_row_to_card(
                crow, profile_slug=slug, active_topic_ids=active,
            )
            # Filter on the card's own bucket rather than recomputing it.
            # Two copies of this rule had already drifted apart once.
            if bucket is not None and card.bucket != bucket:
                continue
            cards.append(card)
            if crow["saved_at"]:
                states[card.id] = "saved"
            elif crow["dismissed_at"]:
                states[card.id] = "dismissed"

        # Stamp shown_at for the cards we're returning from this profile.
        shown_ids = [c.id for c in cards if c.profile == slug]
        if shown_ids:
            candidates_repo.mark_shown_bulk(conn, pid, shown_ids)

    cards.sort(key=lambda c: c.score, reverse=True)
    cards = cards[:_DAILY_LIMIT]

    fetched_at, scored = _last_run_metrics(conn, profile_ids)
    fetch_ms = int(round((time.perf_counter() - t0) * 1000))

    return DailyRadarResponse(
        date=_today_label(),
        fetchedAt=fetched_at,
        fetchMs=fetch_ms,
        candidatesScored=scored,
        cards=cards,
        states=states,
    )


def _resolve_card(
    conn: sqlite3.Connection, user_id: int, card_id: str
) -> tuple[int, str] | None:
    """Find ``(profile_id, openalex_id)`` for a card belonging to this user.

    Cards are addressed by their ``openalex_id``. The same paper can be
    a candidate in multiple profiles; we resolve to the most recently
    fetched row owned by this user so a save lands somewhere sensible.
    """
    row = conn.execute(
        """
        SELECT pc.profile_id, pc.openalex_id
        FROM profile_candidates pc
        JOIN profiles p ON p.id = pc.profile_id
        WHERE p.user_id = ? AND pc.openalex_id = ?
        ORDER BY pc.fetched_at DESC
        LIMIT 1
        """,
        (user_id, card_id),
    ).fetchone()
    if row is None:
        return None
    return int(row["profile_id"]), row["openalex_id"]


def save(
    conn: sqlite3.Connection, user_id: int, card_id: str
) -> CardActionResponse | None:
    resolved = _resolve_card(conn, user_id, card_id)
    if resolved is None:
        return None
    profile_id, openalex_id = resolved
    state = candidates_repo.mark_saved(conn, profile_id, openalex_id)
    return CardActionResponse(id=card_id, state=state)


def dismiss(
    conn: sqlite3.Connection, user_id: int, card_id: str
) -> CardActionResponse | None:
    resolved = _resolve_card(conn, user_id, card_id)
    if resolved is None:
        return None
    profile_id, openalex_id = resolved
    state = candidates_repo.mark_dismissed(conn, profile_id, openalex_id)
    return CardActionResponse(id=card_id, state=state)


def card_feedback_context(
    conn: sqlite3.Connection, user_id: int, card_id: str
) -> dict | None:
    """Look up everything ``feedback.log_event`` needs for a card.

    Returns ``None`` when the card doesn't belong to ``user_id``. When
    found, returns a dict with: ``profile_id``, ``profile_slug``,
    ``openalex_id``, ``doi``, ``score_pct`` (falls back to the legacy
    ``score`` column when ``score_pct`` is NULL), ``selector`` class
    name (parsed from ``selector_config_json.type``), and
    ``selector_config_hash`` (sha256[:16] of the JSON config).
    """
    row = conn.execute(
        """
        SELECT pc.profile_id, pc.openalex_id, pc.score, pc.score_pct,
               p.doi,
               pr.slug             AS profile_slug,
               pr.selector_config_json
        FROM profile_candidates pc
        JOIN profiles pr   ON pr.id = pc.profile_id
        JOIN papers   p    USING (openalex_id)
        WHERE pr.user_id = ? AND pc.openalex_id = ?
        ORDER BY pc.fetched_at DESC
        LIMIT 1
        """,
        (user_id, card_id),
    ).fetchone()
    if row is None:
        return None

    sel_name: str | None = None
    sel_hash: str | None = None
    cfg_raw = row["selector_config_json"]
    if cfg_raw:
        try:
            cfg = json.loads(cfg_raw)
        except (TypeError, ValueError):
            cfg = None
        if isinstance(cfg, dict):
            type_str = cfg.get("type")
            if isinstance(type_str, str) and type_str:
                # Map the persisted selector type to its class name.
                # Centroid -> CentroidSelector; max_seed -> MaxSeedSelector.
                parts = [p.capitalize() for p in type_str.split("_") if p]
                sel_name = "".join(parts) + "Selector" if parts else None
            sel_hash = selector_config_hash(cfg)

    score_pct = row["score_pct"]
    if score_pct is None:
        score_pct = row["score"]

    return {
        "profile_id": int(row["profile_id"]),
        "profile_slug": row["profile_slug"],
        "openalex_id": row["openalex_id"],
        "doi": row["doi"],
        "score_pct": float(score_pct) if score_pct is not None else None,
        "selector": sel_name,
        "selector_config_hash": sel_hash,
    }
