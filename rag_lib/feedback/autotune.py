"""Threshold from feedback.

Reads an interest's saved and dismissed papers and recommends the
threshold that best separates them: an F1-like blend of precision (saves
above the threshold / everything triaged above it) and coverage (saves
above / all saves). Surfaced on the Threshold tab
(``GET /api/profiles/{key}/feedback-threshold``) and as ``?autotune=1`` on
``POST /api/profiles/{key}/refit``.

It reads the papers' current state and raw cosine from
``profile_candidates``. It used to read the feedback log, whose ``score``
is the card's rank percentile: those were swept as if they were cosines
against 0.50..0.95, and an un-save stayed counted as a save.
"""

from __future__ import annotations

import math
import sqlite3

from rag_lib.db.repos import profiles as profiles_repo

MIN_EVENTS = 30


def triaged_scores(db: sqlite3.Connection, profile_id: int) -> tuple[list[float], list[float]]:
    """``(saved cosines, dismissed cosines)`` for an interest's papers now."""
    from rag_lib.db.repos.candidates import NOT_SEED

    rows = db.execute(
        f"""
        SELECT COALESCE(pc.score_raw, pc.score) AS s, pc.saved_at IS NOT NULL AS saved
        FROM profile_candidates pc
        WHERE pc.profile_id = ? AND (pc.saved_at IS NOT NULL OR pc.dismissed_at IS NOT NULL)
          AND COALESCE(pc.score_raw, pc.score) IS NOT NULL AND {NOT_SEED}
        """,
        (profile_id,),
    ).fetchall()
    saves = [float(r[0]) for r in rows if r[1]]
    dismisses = [float(r[0]) for r in rows if not r[1]]
    return saves, dismisses


def recommend_threshold(
    db: sqlite3.Connection,
    profile_id: int,
    *,
    min_events: int = MIN_EVENTS,
) -> float | None:
    """Return the F1-optimal threshold over the triaged papers, or ``None``.

    ``None`` when fewer than ``min_events`` papers are saved or dismissed,
    or none is saved. The cut points tried are the saved papers' own
    cosines (a threshold between two of them scores the same as the
    upper one); the winner is rounded *down* to 3 decimals, as the slider
    is, so it still keeps the save it was cut at. Ties pick the *lower*
    threshold (more permissive).
    """
    saves, dismisses = triaged_scores(db, profile_id)
    if len(saves) + len(dismisses) < min_events or not saves:
        return None
    total_saves = len(saves)

    best_th: float | None = None
    best_score = -1.0
    # Cut at the saves' exact scores, so every cut keeps at least one save.
    # Cutting at round(s, 3) put the top save below its own cut whenever it
    # rounded up (0.9126 -> 0.913): no save above it, and a division by zero.
    for th in sorted(set(saves)):
        s_above = sum(1 for s in saves if s >= th)
        d_above = sum(1 for s in dismisses if s >= th)
        precision = s_above / (s_above + d_above)
        coverage = s_above / total_saves
        f1 = 2 * precision * coverage / (precision + coverage)
        # Lower threshold wins ties (strictly greater wins outright).
        if f1 > best_score:
            best_score = f1
            best_th = th
    # Three decimals, as the slider; down, so the save it was cut at stays in.
    return None if best_th is None else math.floor(best_th * 1000 + 1e-9) / 1000


def apply_recommended(
    db: sqlite3.Connection,
    profile_id: int,
    *,
    min_events: int = MIN_EVENTS,
) -> dict:
    """Persist the recommended threshold on the profile row.

    Returns ``{"old": float|None, "new": float|None, "n_events": int,
    "applied": bool}``. ``applied=False`` when there is not enough data
    or the recommendation matches the current value.
    """
    saves, dismisses = triaged_scores(db, profile_id)
    n_events = len(saves) + len(dismisses)
    row = db.execute(
        "SELECT threshold FROM profiles WHERE id = ?",
        (profile_id,),
    ).fetchone()
    old = float(row["threshold"]) if row and row["threshold"] is not None else None

    new = recommend_threshold(db, profile_id, min_events=min_events)
    if new is None or new == old:
        return {"old": old, "new": new, "n_events": n_events, "applied": False}

    profiles_repo.update_threshold(db, profile_id, new)
    from rag_lib.scheduler.jobs import rescore_pool  # the feed's pool moved
    rescore_pool(db, profile_id)
    return {"old": old, "new": new, "n_events": n_events, "applied": True}
