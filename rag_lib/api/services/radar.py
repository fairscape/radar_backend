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
from rag_lib.openalex_tiers import enabled_topic_ids

from ..mappers import candidate_row_to_card
from ..schemas import (
    Bucket,
    Card,
    CardActionResponse,
    CardState,
    DailyRadarResponse,
)


_DAILY_LIMIT = 200


def fair_shares(sizes: list[int], limit: int) -> list[int]:
    """How many cards each interest gets of ``limit``, given how many it has.

    An equal share each, and what an interest with fewer cards leaves
    unused goes to the others, so the feed is full whenever there are
    ``limit`` cards in all. A flat ``limit // n`` left most of it empty
    when a few interests had many cards.
    """
    shares = [0] * len(sizes)
    left = limit
    open_ = [i for i, n in enumerate(sizes) if n > 0]
    while left > 0 and open_:
        each = max(1, left // len(open_))
        nxt = []
        for i in open_:
            if left <= 0:
                break
            take = min(each, sizes[i] - shares[i], left)
            shares[i] += take
            left -= take
            if shares[i] < sizes[i]:
                nxt.append(i)
        open_ = nxt
    return shares


def _today_label() -> str:
    """Mock-api formats date as 'THU 16 APR 2026'. Match that on the API."""
    return datetime.now(timezone.utc).strftime("%a %d %b %Y").upper()


def _active_topic_ids(topic_filters: dict | None) -> list[str]:
    """Topics the card should highlight a match against.

    "Active" has to mean the same thing here as it does at gather time,
    so this defers to ``enabled_topic_ids``. It used to return every
    topic in the blob: a card was credited with matching a topic the
    user had switched off, and ``topicMatch`` — the fraction of the
    card's topics inside the active set — was inflated by topics that
    could not have contributed a single result.
    """
    return enabled_topic_ids(topic_filters)


# Runs that scanned for the feed: not a wizard trial, not a seed import.
_SCAN_RUN = "(tier_used IS NULL OR (tier_used != 'dry_run' AND tier_used NOT LIKE '%import'))"


def _last_run_metrics(
    conn: sqlite3.Connection,
    profile_ids: list[int],
) -> tuple[str, int]:
    """``(fetchedAt, candidatesScored)`` over each interest's latest scan.

    ``fetchedAt`` is when the most recent of them finished, as ISO-8601
    UTC: it used to be the time of day alone, which the feed rendered as
    "Last scan finished" with no date. ``candidatesScored`` adds up each
    interest's latest scan, not every run it ever had (it read 9,000+ for
    a week of nightly scans), and trials and imports are not scans.
    """
    if not profile_ids:
        return "", 0
    placeholders = ",".join("?" for _ in profile_ids)
    row = conn.execute(
        f"""
        WITH latest AS (
          SELECT profile_id, MAX(id) AS id FROM gather_runs
          WHERE profile_id IN ({placeholders}) AND finished_at IS NOT NULL
            AND error IS NULL AND {_SCAN_RUN}
          GROUP BY profile_id
        )
        SELECT MAX(g.finished_at) AS finished_at, SUM(COALESCE(g.n_fetched, 0)) AS scored
        FROM gather_runs g JOIN latest l ON l.id = g.id
        """,
        profile_ids,
    ).fetchone()
    finished_at = row["finished_at"] or ""
    if finished_at and "T" not in finished_at:
        # SQLite's datetime('now'): "YYYY-MM-DD HH:MM:SS", UTC.
        finished_at = finished_at.replace(" ", "T", 1) + "Z"
    return finished_at, int(row["scored"] or 0)


def daily(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    profile_slug: str | None = None,
    bucket: Bucket | None = None,
    year_from: int | None = None,
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

    states: dict[str, CardState | None] = {}
    profile_ids: list[int] = []
    per_profile: list[tuple[int, list[Card], list[Card]]] = []

    for prow in profile_rows:
        pid = int(prow["id"])
        profile_ids.append(pid)
        slug = prow["slug"] or prow["name"]
        topic_filters = profiles_repo.topic_filters(conn, pid)
        active = _active_topic_ids(topic_filters)
        # The whole limit from each interest, filtered, and only then cut to
        # its share. Cutting first (to 200 / interests) and filtering the
        # bucket after dropped matching cards: across 15 interests "Above
        # the bar" showed 19 cards of 41.
        threshold = prow["threshold"]
        try:
            seed_sim_min = prow["seed_sim_min"]
        except (IndexError, KeyError):
            seed_sim_min = None
        band = float(seed_sim_min) if seed_sim_min is not None and threshold is not None else None

        def cards_for(triaged: bool, limit: int) -> list[Card]:
            rows = candidates_repo.top_for_profile(
                conn, pid, limit=limit, year_from=year_from, below=bucket == "low",
                sim_at_least=band if bucket == "high" else None,
                sim_below=band if bucket == "medium" else None,
                triaged=triaged,
            )
            out: list[Card] = []
            for crow in rows:
                card = candidate_row_to_card(
                    crow, profile_slug=slug, active_topic_ids=active,
                    threshold=float(threshold) if threshold is not None else None,
                    seed_sim_min=float(seed_sim_min) if seed_sim_min is not None else None,
                )
                # Filter on the card's own bucket rather than recomputing it.
                # Two copies of this rule had already drifted apart once.
                if bucket is not None and card.bucket != bucket:
                    continue
                card.state = (
                    "saved" if crow["saved_at"] else
                    "dismissed" if crow["dismissed_at"] else None
                )
                out.append(card)
            return out

        # Papers still to look at and papers already saved or dismissed are
        # shared out separately: the feed hides the second kind by default,
        # and when they shared one budget they crowded the first out.
        # One row over the limit for the papers still to look at, so an
        # interest past the cap alone still reports papers not shown.
        per_profile.append((pid, cards_for(False, _DAILY_LIMIT + 1), cards_for(True, _DAILY_LIMIT)))

    cards: list[Card] = []
    n_unshown = 0
    fresh_shares = fair_shares([len(f) for _, f, _ in per_profile], _DAILY_LIMIT)
    done_shares = fair_shares([len(d) for _, _, d in per_profile], _DAILY_LIMIT)
    for (pid, fresh, done), n_fresh, n_done in zip(per_profile, fresh_shares, done_shares):
        kept = fresh[:n_fresh]
        n_unshown += len(fresh) - len(kept)
        cards.extend(kept)
        for card in done[:n_done]:
            cards.append(card)
            states[card.id] = card.state
        # Stamp shown_at for the cards we're returning from this profile.
        # Held open deliberately: one commit for all profiles, below.
        if kept:
            candidates_repo.mark_shown_bulk(conn, pid, [c.id for c in kept], commit=False)

    conn.commit()

    # Below the bar, closest to it first: ``score`` there is a rank from the
    # paper's own scan, not its similarity to the seeds now.
    if bucket == "low":
        cards.sort(key=lambda c: c.similarity if c.similarity is not None else -1.0, reverse=True)
    else:
        cards.sort(key=lambda c: c.score, reverse=True)

    fetched_at, scored = _last_run_metrics(conn, profile_ids)
    fetch_ms = int(round((time.perf_counter() - t0) * 1000))

    return DailyRadarResponse(
        date=_today_label(),
        fetchedAt=fetched_at,
        fetchMs=fetch_ms,
        candidatesScored=scored,
        cards=cards,
        states=states,
        # Papers to review left out by the shared cap (at least: each
        # interest is read up to the cap). Without it "All interests" showed
        # ~13 papers of each interest and nothing said there were more.
        untriagedNotShown=n_unshown,
    )


def _resolve_card(
    conn: sqlite3.Connection, user_id: int, card_id: str,
    profile_slug: str | None = None,
) -> tuple[int, str] | None:
    """Find ``(profile_id, openalex_id)`` for a card belonging to this user.

    A card is one paper in one interest, so with ``profile_slug`` the row
    is that interest's and nothing else. Without it -- older clients --
    the paper alone is ambiguous when two interests gathered it, and this
    falls back to the most recently fetched row: a save could then land
    on an interest that was not on screen, and since save toggles, un-save
    a paper there.
    """
    if profile_slug:
        row = conn.execute(
            """
            SELECT pc.profile_id, pc.openalex_id
            FROM profile_candidates pc
            JOIN profiles p ON p.id = pc.profile_id
            WHERE p.user_id = ? AND pc.openalex_id = ? AND p.slug = ?
            """,
            (user_id, card_id, profile_slug),
        ).fetchone()
        return None if row is None else (int(row["profile_id"]), row["openalex_id"])
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


def _set(
    conn: sqlite3.Connection, user_id: int, card_id: str, column: str,
    profile_slug: str | None, on: bool | None,
) -> tuple[CardActionResponse, bool] | None:
    resolved = _resolve_card(conn, user_id, card_id, profile_slug)
    if resolved is None:
        return None
    profile_id, openalex_id = resolved
    res = candidates_repo.set_state(conn, profile_id, openalex_id, column, on)
    if res is None:
        return None
    state, changed = res
    slug = profile_slug
    if slug is None:
        row = conn.execute("SELECT slug FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        slug = row["slug"] if row is not None else None
    return CardActionResponse(id=card_id, state=state, profile=slug), changed


def save(
    conn: sqlite3.Connection, user_id: int, card_id: str,
    profile_slug: str | None = None, on: bool | None = None,
) -> tuple[CardActionResponse, bool] | None:
    """Save (``on=True``), un-save (``False``) or toggle (None). ``(response, changed)``."""
    return _set(conn, user_id, card_id, "saved_at", profile_slug, on)


def dismiss(
    conn: sqlite3.Connection, user_id: int, card_id: str,
    profile_slug: str | None = None, on: bool | None = None,
) -> tuple[CardActionResponse, bool] | None:
    """Dismiss / un-dismiss / toggle, as ``save``."""
    return _set(conn, user_id, card_id, "dismissed_at", profile_slug, on)


def card_feedback_context(
    conn: sqlite3.Connection, user_id: int, card_id: str,
    profile_slug: str | None = None,
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
          AND (? IS NULL OR pr.slug = ?)
        ORDER BY pc.fetched_at DESC
        LIMIT 1
        """,
        (user_id, card_id, profile_slug, profile_slug),
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
