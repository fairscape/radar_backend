"""Radar orchestration helpers.

``dry_run`` runs a profile's gatherer over a window of recent OpenAlex
output and scores the result at a sweep of thresholds, reporting
candidate counts and top titles for each. It is the Phase 1B manual-
validation tool: given a freshly-built Profile, does the selector
surface relevant work? What threshold feels right by eye? The
researcher inspects, decides, and commits a threshold back into the
Profile.

``centroid_drift`` is a CentroidSelector-specific drift metric for the
profile health dashboard (Phase 4C) — the cosine between two centroids
across time, used to flag when a seed corpus has drifted enough that
a refit is warranted.

Neither function is on the Selector / Gatherer protocol surface; they
operate above both.
"""

from __future__ import annotations

import datetime

import numpy as np

from .gatherer import Gatherer
from .profile import Profile
from .selector import Selector


def dry_run(
    selector: Selector,
    profile: Profile,
    gatherer: Gatherer,
    *,
    thresholds: list[float] | None = None,
    days: int = 30,
    limit: int | None = None,
) -> dict:
    """Fetch the last ``days`` of candidates via ``gatherer`` and score
    them with ``selector`` at each threshold in ``thresholds`` (defaults
    to [0.80, 0.85, 0.90]). Returns a report dict keyed by threshold::

        {
          "since": "2026-03-24",
          "fetched": 842,
          "selector_cost": {...},
          "gatherer_cost": {...},
          "results": {
             "0.80": {"count": 134, "top_titles": [...]},
             "0.85": {"count": 58,  "top_titles": [...]},
             "0.90": {"count": 12,  "top_titles": [...]},
          },
        }

    The thresholds are placeholders inherited from SPECTER2's
    title+abstract regime; recalibrate on a real seed corpus before
    trusting the defaults (see PHASE1_TRACK1_PROGRESS.md).
    """
    if thresholds is None:
        thresholds = [0.80, 0.85, 0.90]
    since = _days_ago_iso(days)
    candidates = gatherer.fetch(profile, since, limit=limit)
    gatherer_cost = dict(gatherer.cost())

    # Score once with no threshold, then bucket by each cutoff. Avoids
    # re-running the embedder on candidates per threshold.
    all_scored = selector.select(candidates, profile, threshold=None)
    selector_cost = dict(selector.cost())

    # Selectors return 3-tuples since Phase 3 (score, paper, breakdown);
    # accept the legacy 2-tuple shape too so any older fixtures keep working.
    def _entry(t):
        return (t[0], t[1]) if len(t) >= 2 else t

    results: dict[str, dict] = {}
    for thr in thresholds:
        passing = [_entry(t) for t in all_scored if t[0] >= thr]
        top = [p.title for _, p in passing[:10]]
        results[f"{thr:.2f}"] = {"count": len(passing), "top_titles": top}

    return {
        "since": since,
        "fetched": len(candidates),
        "selector_cost": selector_cost,
        "gatherer_cost": gatherer_cost,
        "results": results,
    }


def centroid_drift(v1, v2) -> float:
    """Cosine between two centroids, mapped to ``[0, 1]`` same as
    CentroidSelector scoring. Returns 1.0 when identical, 0.0 when
    antipodal."""
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    c = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, (c + 1.0) / 2.0))


def _days_ago_iso(days: int) -> str:
    d = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    return d.strftime("%Y-%m-%d")
