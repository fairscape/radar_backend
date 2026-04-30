"""Threshold autotune sketch.

Reads the recent saved/dismissed events for a profile and recommends a
score threshold that maximises an F1-like blend of precision (saves
above the threshold / total above) and coverage (saves above /
total saves). Opt-in only — surfaced via ``?autotune=1`` on
``POST /api/profiles/{key}/refit``.

Phase 11 may surface ``recommend_threshold`` as a read-only preview in
the wizard's Step 4.
"""

from __future__ import annotations

import sqlite3

from rag_lib.db.repos import feedback as feedback_repo
from rag_lib.db.repos import profiles as profiles_repo


_SWEEP_LO = 0.50
_SWEEP_HI = 0.95
_SWEEP_STEP = 0.01


def _sweep_thresholds() -> list[float]:
    out: list[float] = []
    n = int(round((_SWEEP_HI - _SWEEP_LO) / _SWEEP_STEP)) + 1
    for i in range(n):
        out.append(round(_SWEEP_LO + i * _SWEEP_STEP, 2))
    return out


def recommend_threshold(
    db: sqlite3.Connection,
    profile_id: int,
    *,
    min_events: int = 30,
) -> float | None:
    """Return the F1-optimal threshold over recent feedback, or ``None``.

    ``None`` when fewer than ``min_events`` scored save+dismiss events
    exist (insufficient signal to make a recommendation). The "F1-like"
    metric is ``2 * precision * coverage / (precision + coverage)``
    where:
        precision = saves above th / (saves+dismisses above th)
        coverage  = saves above th / total saves
    Ties pick the *lower* threshold (more permissive).
    """
    rows = feedback_repo.scored_save_dismiss_for_profile(db, profile_id)
    if len(rows) < min_events:
        return None

    saves = [float(r["score"]) for r in rows if r["action"] == "saved"]
    dismisses = [float(r["score"]) for r in rows if r["action"] == "dismissed"]
    total_saves = len(saves)
    if total_saves == 0:
        return None

    best_th: float | None = None
    best_score = -1.0
    for th in _sweep_thresholds():
        s_above = sum(1 for s in saves if s >= th)
        d_above = sum(1 for s in dismisses if s >= th)
        denom = s_above + d_above
        if denom == 0:
            continue
        precision = s_above / denom
        coverage = s_above / total_saves
        if precision + coverage == 0:
            continue
        f1 = 2 * precision * coverage / (precision + coverage)
        # Lower threshold wins ties (strictly greater wins outright).
        if f1 > best_score:
            best_score = f1
            best_th = th
    return best_th


def apply_recommended(
    db: sqlite3.Connection,
    profile_id: int,
    *,
    min_events: int = 30,
) -> dict:
    """Persist the recommended threshold on the profile row.

    Returns ``{"old": float|None, "new": float|None, "n_events": int,
    "applied": bool}``. ``applied=False`` when there is not enough data
    or the recommendation matches the current value.
    """
    n_events = feedback_repo.total_for_profile(db, profile_id)
    row = db.execute(
        "SELECT threshold FROM profiles WHERE id = ?",
        (profile_id,),
    ).fetchone()
    old = float(row["threshold"]) if row and row["threshold"] is not None else None

    new = recommend_threshold(db, profile_id, min_events=min_events)
    if new is None or new == old:
        return {"old": old, "new": new, "n_events": n_events, "applied": False}

    profiles_repo.update_threshold(db, profile_id, new)
    return {"old": old, "new": new, "n_events": n_events, "applied": True}
