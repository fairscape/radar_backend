"""Per-batch rank percentile.

Selectors emit ``(score, paper, breakdown)`` ranked descending by score.
``attach_percentile`` walks the list in that order and writes
``breakdown["score_pct"]`` for each entry: top entry gets 1.0, bottom
gets 0.0, single-entry batches collapse to 1.0.

Cosine scores cluster in narrow bands once a topic prefilter has
already narrowed the pool, and the absolute numbers don't carry between
profiles (a 0.97 in fair_data is not the same target as a 0.97 in
neonatal_vitals). The percentile lets downstream consumers threshold or
compare across profiles without reasoning about raw cosine.

Tied raw scores are *not* given identical percentiles in this
implementation — we use rank order from the sorted list directly. At
demo scale this is fine; if it bites we can switch to averaged-rank
without changing the call site.
"""

from __future__ import annotations

from typing import Sequence

from ..paper import Paper

Ranked = list[tuple[float, Paper, dict]]


def attach_percentile(ranked: Ranked) -> Ranked:
    """Mutate each ``breakdown`` to include ``score_pct``. Returns the same list.

    Assumes ``ranked`` is already sorted descending by primary score. If
    you pass an unsorted list the percentile becomes meaningless — sort
    first.
    """
    n = len(ranked)
    for i, (score, paper, breakdown) in enumerate(ranked):
        pct = 1.0 if n <= 1 else (n - 1 - i) / (n - 1)
        breakdown["score_pct"] = float(pct)
    return ranked


def percentile_of(rank: int, n: int) -> float:
    """Standalone helper for callers that have an index but not the full list."""
    if n <= 1:
        return 1.0
    return (n - 1 - rank) / (n - 1)
