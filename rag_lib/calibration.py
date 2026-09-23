"""Score-scale calibration for the selector's embedding regime.

Everything here was measured on 2026-09-21 with SPECTER2 + the proximity
adapter, on two real four-paper seed folders (FAIR data stewardship and
neonatal vital-sign monitoring) plus 200 OpenAlex candidates gathered
for each over the previous 30 days. ``docs/CALIBRATION.md`` has the
full table. The numbers that matter:

    random pairs, same OpenAlex topics       median 0.85   p95 0.90
    random pairs, different fields           median 0.82   p95 0.87
    random 4-paper sets, same topics         coherence median 0.85, max 0.89
    the two real seed sets                   coherence median 0.915 / 0.926
    the two real sets deliberately mixed     coherence median 0.878, IQR 0.07
    seeds' leave-one-out cosine to centroid  0.927 – 0.962
    other field's seeds vs a centroid        0.84 – 0.90
    30-day candidates vs centroid            p50 0.89  p90 0.92  max 0.96

So the whole working range of this model sits between about 0.80 and
0.96. The earlier bands (coherence "tight" above 0.75, a threshold cap
at 0.85, card colours at 0.95 / 0.925 on a raw score) came from the
title+abstract regime SPECTER2 was described in, and on this scale they
either call everything focused or let everything through. The
functions below replace those constants with ones anchored to the
measurements, and express results in terms a person can act on.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

EMBEDDING_REGIME = "specter2-proximity"

# Pairwise cosine among random papers that share the profile's topics.
NULL_SAME_FIELD_MEDIAN = 0.85
NULL_SAME_FIELD_P95 = 0.90
NULL_CROSS_FIELD_MEDIAN = 0.82

# Seed-set coherence bands (median pairwise cosine among seeds).
# "focused" sits above the ceiling random same-field sets reach (0.89);
# "broad" covers the band where a mix of two real topics lands (0.878).
COHERENCE_FOCUSED = 0.90
COHERENCE_BROAD = 0.86
# An interquartile range this wide means two groups, not one loose one:
# the focused real sets had IQR 0.02–0.03, the mixed set 0.073. A real
# five-paper set with median 0.934 (agreement 88) had IQR 0.058 and was
# wrongly demoted at the earlier 0.05, so the gate sits at 0.07 and is
# skipped altogether once agreement is 80+: that high a median is one
# topic whatever the spread says.
COHERENCE_SPREAD_MIXED = 0.07
AGREEMENT_OVERRIDES_SPREAD = 80

# Plain-language "agreement" score: 0 at cross-field noise, 100 at the
# similarity of near-duplicate papers.
AGREEMENT_FLOOR = 0.82
AGREEMENT_CEIL = 0.95

# How far below the least typical seed the suggested threshold sits.
THRESHOLD_MARGIN = 0.01
# Where the suggestion is allowed to land within the candidate scores:
# at least the top quarter must fail, at least the top 2% must pass.
THRESHOLD_MIN_PERCENTILE = 75
THRESHOLD_MAX_PERCENTILE = 98
THRESHOLD_FALLBACK = 0.90

CoherenceLabel = str  # "focused" | "broad" | "mixed" | "single" | "none"


def agreement_score(median: float | None) -> int | None:
    """Map a coherence median onto 0–100 for display.

    Linear between ``AGREEMENT_FLOOR`` and ``AGREEMENT_CEIL``: random
    same-field sets score about 20, the mixed real set 45, the focused
    real sets 73 and 82.
    """
    if median is None or not np.isfinite(median):
        return None
    x = (float(median) - AGREEMENT_FLOOR) / (AGREEMENT_CEIL - AGREEMENT_FLOOR)
    return int(round(100 * min(1.0, max(0.0, x))))


def coherence_label(median: float | None, iqr: float | None, n: int) -> CoherenceLabel:
    if n < 1:
        return "none"
    if n < 2:
        return "single"
    if median is None or not np.isfinite(median):
        return "none"
    m = float(median)
    spread = float(iqr) if iqr is not None and np.isfinite(iqr) else 0.0
    agreement = agreement_score(m) or 0
    if m >= COHERENCE_FOCUSED and (
        spread < COHERENCE_SPREAD_MIXED or agreement >= AGREEMENT_OVERRIDES_SPREAD
    ):
        return "focused"
    if m >= COHERENCE_BROAD:
        return "broad"
    return "mixed"


def describe_coherence(label: CoherenceLabel, n: int, agreement: int | None) -> str:
    """One sentence a non-specialist can act on."""
    if label == "none":
        return "No seed papers yet."
    if label == "single":
        return (
            "One seed paper. Radar will look for papers like it; agreement "
            "between seeds can't be measured until there are two."
        )
    a = f"{agreement}/100" if agreement is not None else "—"
    if label == "focused":
        return (
            f"Your {n} seed papers agree with each other ({a}). They describe "
            "one clear topic, so matches should be on target."
        )
    if label == "broad":
        return (
            f"Your {n} seed papers only loosely agree ({a}). They may cover "
            "two related topics; results will lean towards whichever group "
            "is larger. Removing the odd ones out, or splitting into two "
            "interests, would sharpen it."
        )
    return (
        f"Your {n} seed papers don't share a topic ({a}) — about as similar "
        "as random papers from the same field. Split them into separate "
        "interests or remove the ones that don't belong."
    )


def health_for(median: float | None, n_seed: int) -> str:
    """Profile health from coherence alone: ok / warn / err."""
    label = coherence_label(median, None, n_seed)
    if label == "focused":
        return "ok"
    if label == "mixed":
        return "err"
    return "warn"


def _unit_rows(vecs: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vecs, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-12)


def seed_similarity_band(vecs: Sequence[Sequence[float]] | np.ndarray) -> dict | None:
    """Leave-one-out cosine of each seed to the centroid of the others.

    This is the honest answer to "how similar is a paper like my seeds
    to the centroid?" — the number a candidate has to reach to be as
    close as the seeds themselves. Returns ``{min, median, max, values}``
    or ``None`` with fewer than two seeds.
    """
    u = _unit_rows(vecs)
    n = u.shape[0]
    if n < 2:
        return None
    vals: list[float] = []
    for i in range(n):
        others = np.delete(u, i, axis=0).mean(axis=0)
        others = others / (np.linalg.norm(others) + 1e-12)
        vals.append(float(u[i] @ others))
    return {
        "min": float(min(vals)),
        "median": float(np.median(vals)),
        "max": float(max(vals)),
        "values": vals,
    }


def least_similar_pair(vecs: Sequence[Sequence[float]] | np.ndarray) -> tuple[int, int, float] | None:
    """Indices of the two seeds that agree least, and their cosine."""
    u = _unit_rows(vecs)
    n = u.shape[0]
    if n < 2:
        return None
    sim = u @ u.T
    iu = np.triu_indices(n, k=1)
    k = int(np.argmin(sim[iu]))
    i, j = int(iu[0][k]), int(iu[1][k])
    return i, j, float(sim[i, j])


def suggest_threshold(
    seed_band_min: float | None,
    scores: Sequence[float] | None,
) -> float:
    """A starting threshold: just below the least typical seed, kept
    inside the range where it separates the candidate pool.

    ``seed_band_min`` is ``seed_similarity_band()["min"]``. ``scores``
    are the candidate cosines from a trial scan. With both, the
    suggestion is the seed floor minus a small margin, clamped so at
    least the top 2% of candidates pass and at least the top quarter
    does not. With only one of them, the other rule applies; with
    neither, a fixed fallback.
    """
    arr = np.asarray([s for s in (scores or []) if np.isfinite(s)], dtype=float)
    if seed_band_min is not None and np.isfinite(seed_band_min):
        thr = float(seed_band_min) - THRESHOLD_MARGIN
        if arr.size >= 10:
            lo = float(np.percentile(arr, THRESHOLD_MIN_PERCENTILE))
            hi = float(np.percentile(arr, THRESHOLD_MAX_PERCENTILE))
            thr = min(max(thr, lo), hi)
        return round(thr, 3)
    if arr.size >= 10:
        return round(float(np.percentile(arr, 90)), 3)
    return THRESHOLD_FALLBACK


def score_range(scores: Sequence[float] | None, seed_band: dict | None) -> tuple[float, float] | None:
    """The axis a threshold slider should cover: the observed scores
    plus the seed band, with a little padding. ``None`` when there is
    nothing to show."""
    vals = [float(s) for s in (scores or []) if np.isfinite(s)]
    if seed_band:
        vals.extend([seed_band["min"], seed_band["max"]])
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    pad = max(0.005, (hi - lo) * 0.05)
    return round(max(0.0, lo - pad), 3), round(min(1.0, hi + pad), 3)
