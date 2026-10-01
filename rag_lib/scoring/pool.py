"""Rescore an interest's whole candidate pool on one scale.

The reranker min-max normalizes the selector and reranker scores *within
one gather batch* and blends them, and the percentile is a rank within that
batch too. Both were persisted and then ranked together across every batch
the interest ever had: the feed orders by ``score_blended`` and labels each
card with ``score_pct`` as "top X% of this interest's pool". So every
batch's best paper scored ~1.0 however weak it was. Measured on 2026-10-01:
the feed's first card came from a 5-paper nightly batch with cosine 0.915,
above a paper from a 165-paper batch with cosine 0.930. A batch of one gets
span 0, so its paper blended to 0.0 and sank while labelled top 0%.

The fix keeps the per-batch values as the reranker wrote them and, after
every write, recomputes the pool-wide ones from the inputs that *are*
comparable across batches: the selector's raw cosine (``score_raw``) and
the reranker's raw logit (``score_reranker_raw``) -- each a function of one
paper and the interest's seeds, not of whatever else was in the batch.

The pool is the papers that clear the interest's threshold. Gathers keep
the ones below it too (so lowering the threshold can show them), but those
are not in the feed and get no blend or percentile; the threshold route
rescores when it moves.
"""

from __future__ import annotations

import sqlite3

from .percentile import percentile_of


def _span(values: list[float]) -> tuple[float, float]:
    """Offset and span for ``(x - lo) / span`` -> [0, 1]; a flat pool -> span 1."""
    lo, hi = min(values), max(values)
    return lo, (hi - lo) if hi > lo else 1.0


def pool_scores(
    rows: list[dict], *, alpha: float, beta: float,
) -> dict[str, dict]:
    """Pool-wide ``score_reranker_norm``, ``score_blended`` and ``score_pct``.

    ``rows`` carry ``openalex_id``, ``score``, ``score_raw`` and
    ``score_reranker_raw``. Pure, so it can be tested without a database.

    When no row has a reranker score the interest was never reranked:
    ``score_blended`` stays None and the percentile ranks the selector
    score. When some do, a row without one (a CLI gather, a day the
    reranker was off, rows from before it existed) is blended with the
    pool's *median* reranker score standing in for its own. Giving it the
    selector score alone, as a first version did, let it reach 1.0 where a
    reranked row needs both stages high -- so unreranked rows took the top
    of the feed. The median says "the reranker has no opinion", which is
    the truth about that row.
    """
    if not rows:
        return {}
    sel = [r["score_raw"] if r["score_raw"] is not None else r["score"] for r in rows]
    sel_lo, sel_span = _span([s for s in sel if s is not None] or [0.0])
    rr_vals = [r["score_reranker_raw"] for r in rows if r["score_reranker_raw"] is not None]
    reranked = bool(rr_vals)
    if reranked:
        rr_lo, rr_span = _span(rr_vals)
        norms = sorted((v - rr_lo) / rr_span for v in rr_vals)
        mid = len(norms) // 2
        rr_median = norms[mid] if len(norms) % 2 else (norms[mid - 1] + norms[mid]) / 2

    out: dict[str, dict] = {}
    keys: list[tuple[float, str]] = []
    for r, s in zip(rows, sel):
        sel_norm = 0.0 if s is None else (s - sel_lo) / sel_span
        rr_norm = blended = None
        if reranked:
            if r["score_reranker_raw"] is not None:
                rr_norm = (r["score_reranker_raw"] - rr_lo) / rr_span
                blended = alpha * sel_norm + beta * rr_norm
            else:
                blended = alpha * sel_norm + beta * rr_median
        out[r["openalex_id"]] = {"score_reranker_norm": rr_norm, "score_blended": blended}
        keys.append((blended if blended is not None else sel_norm, r["openalex_id"]))

    keys.sort(key=lambda k: k[0], reverse=True)
    n = len(keys)
    for rank, (_, oa) in enumerate(keys):
        out[oa]["score_pct"] = percentile_of(rank, n)
    return out


# Pass as ``threshold=`` to ``Selector.select`` to keep every paper.
# ``threshold=None`` does not: a selector rebuilt from its stored config
# falls back to the threshold it was fitted with (``self.threshold``).
KEEP_ALL = float("-inf")


def passes(score_raw: float | None, threshold: float | None) -> bool:
    """Whether a stored candidate is in the feed: the rule ``candidates.PASSES`` spells in SQL."""
    return score_raw is None or threshold is None or score_raw >= threshold


def rescore_profile_pool(
    conn: sqlite3.Connection, profile_id: int, *, alpha: float, beta: float,
) -> int:
    """Rewrite one interest's pool-wide scores in place. Returns rows in the pool.

    Leaves ``score``, ``score_raw`` and ``score_reranker_raw`` -- the
    per-paper inputs -- untouched, so running it twice changes nothing.
    Rows below the threshold get NULL blend and percentile.
    """
    prow = conn.execute("SELECT threshold FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    threshold = None if prow is None or prow[0] is None else float(prow[0])
    rows = [dict(r) for r in conn.execute(
        """
        SELECT openalex_id, score, score_raw, score_reranker_raw
        FROM profile_candidates WHERE profile_id = ?
        """,
        (profile_id,),
    ).fetchall()]
    pool = [r for r in rows if passes(r["score_raw"], threshold)]
    scores = pool_scores(pool, alpha=alpha, beta=beta)
    empty = {"score_reranker_norm": None, "score_blended": None, "score_pct": None}
    conn.executemany(
        """
        UPDATE profile_candidates
        SET score_reranker_norm = ?, score_blended = ?, score_pct = ?
        WHERE profile_id = ? AND openalex_id = ?
        """,
        [
            (v["score_reranker_norm"], v["score_blended"], v["score_pct"], profile_id, oa)
            for oa, v in ((r["openalex_id"], scores.get(r["openalex_id"], empty)) for r in rows)
        ],
    )
    conn.commit()
    return len(scores)
