"""Seed-corpus coherence metrics.

``coherence(vecs)`` inspects the pairwise cosine-similarity distribution
across a seed corpus and returns a compact summary:

    {
      "median": float,   # median pairwise cosine
      "iqr":    float,   # interquartile range (Q3 - Q1)
      "bimodal": bool,   # True when the distribution looks bimodal
      "n":      int,     # number of seed vectors
    }

Interpretation (SPECTER2 baseline, per build spec):
    median > 0.75   tight — single clear topic
    0.60–0.75       acceptable — topic is real but loose
    < 0.60          the profile is two or more sub-topics; split before
                    using (CentroidSelector will be dominated by whichever
                    cluster happens to pull the mean)

The 0.70 / 0.60 cutoffs come from SPECTER2's title+abstract training
regime. The Phase 1B expanded embedding input (adding MeSH / keywords /
substances / body) shifts the distributions by an unknown amount;
thresholds need recalibration on a real seed corpus before Phase 2
(noted in PHASE1_TRACK1_PROGRESS.md).

Bimodality detection. We use a simple histogram-gap heuristic: if the
two tallest histogram bins are separated by a clear empty or near-empty
gap, the distribution is flagged bimodal. Cheap, interpretable, and
stable on small samples (10-15 seed papers yields 45-105 pairwise
scores). Phase 2+ can swap in Hartigan's dip test if the heuristic
proves noisy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def coherence(
    vecs: Sequence[Sequence[float]] | np.ndarray,
    *,
    bimodal_min_n: int = 20,
    bimodal_gap_threshold: float = 0.1,
) -> dict:
    """Compute coherence metrics over a seed-corpus embedding set.

    Inputs may be a list of lists or an ndarray of shape (n, d).

    ``bimodal_min_n`` — below this many pairwise comparisons the
    bimodality flag is always False (too few samples for the histogram
    to be informative). 20 pairwise scores ≈ 7 seed papers.

    ``bimodal_gap_threshold`` — minimum relative gap between a
    histogram trough and the smaller of its surrounding peaks for the
    distribution to be flagged bimodal. 0.1 means the trough must be
    less than 10% the height of the lower peak.
    """
    arr = _as_array(vecs)
    if arr.shape[0] < 2:
        return {"median": float("nan"), "iqr": float("nan"),
                "bimodal": False, "n": int(arr.shape[0])}

    cos = _pairwise_cosines(arr)
    q1, med, q3 = np.quantile(cos, [0.25, 0.5, 0.75])
    bim = False
    if cos.size >= bimodal_min_n:
        bim = _looks_bimodal(cos, gap_threshold=bimodal_gap_threshold)
    return {
        "median": float(med),
        "iqr": float(q3 - q1),
        "bimodal": bool(bim),
        "n": int(arr.shape[0]),
    }


def _as_array(vecs) -> np.ndarray:
    arr = np.asarray(vecs, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, d); got {arr.shape}")
    return arr


def _pairwise_cosines(arr: np.ndarray) -> np.ndarray:
    """Vectorised upper-triangle pairwise cosine similarities."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    normed = arr / (norms + 1e-12)
    sim = normed @ normed.T
    iu = np.triu_indices(arr.shape[0], k=1)
    return sim[iu]


def _looks_bimodal(cos: np.ndarray, *, gap_threshold: float) -> bool:
    """Very simple histogram-based bimodality check.

    We bin the cosines into 10 bins across their observed range, find
    the two tallest bins, and look for a clear trough between them.
    Returns True when the trough's count is meaningfully smaller than
    the shorter of the two peak bins AND the peaks are separated by at
    least one bin.
    """
    hist, _ = np.histogram(cos, bins=10)
    # Find two tallest bins by count.
    order = np.argsort(hist)[::-1]
    top_two = sorted(order[:2].tolist())  # ascending position
    left, right = top_two
    if right - left < 2:  # peaks adjacent or same → unimodal
        return False
    between = hist[left + 1:right]
    if between.size == 0:
        return False
    trough = int(between.min())
    peak_low = int(min(hist[left], hist[right]))
    if peak_low == 0:
        return False
    return (trough / peak_low) < gap_threshold
