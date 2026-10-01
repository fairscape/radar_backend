"""Pool-wide rescoring (rag_lib/scoring/pool.py).

The blend and percentile the reranker writes are relative to one gather
batch, and the feed ranked them across all batches: a 5-paper batch's best
(cosine 0.915) came out above a 165-paper batch's paper at cosine 0.930.
"""

from rag_lib.scoring.pool import pool_scores


def _row(oa, cos, rr=None):
    return {"openalex_id": oa, "score": cos, "score_raw": cos, "score_reranker_raw": rr}


def test_the_pool_is_ranked_on_one_scale_whatever_batch_a_row_came_from():
    # Per batch, both of these were the best of their batch and blended to 1.0.
    rows = [
        _row("small_batch_best", 0.915, rr=1.0),
        _row("big_batch_best", 0.930, rr=3.0),
        _row("big_batch_worst", 0.850, rr=-2.0),
    ]
    out = pool_scores(rows, alpha=0.4, beta=0.6)
    order = sorted(out, key=lambda k: out[k]["score_blended"], reverse=True)
    assert order == ["big_batch_best", "small_batch_best", "big_batch_worst"]
    assert out["big_batch_best"]["score_pct"] == 1.0
    assert out["big_batch_worst"]["score_pct"] == 0.0


def test_a_lone_row_is_not_sunk_to_zero():
    """A batch of one had span 0: blended 0.0 yet percentile 1.0."""
    out = pool_scores([_row("only", 0.97, rr=2.0)], alpha=0.4, beta=0.6)
    assert out["only"]["score_pct"] == 1.0


def test_a_row_the_reranker_skipped_stays_on_the_same_scale():
    rows = [_row("a", 0.95, rr=2.0), _row("b", 0.90, rr=-1.0), _row("skipped", 0.93)]
    out = pool_scores(rows, alpha=0.4, beta=0.6)
    assert 0.0 <= out["skipped"]["score_blended"] <= 1.0   # not a raw 0.93 among blends
    assert out["skipped"]["score_reranker_norm"] is None


def test_an_unreranked_row_cannot_outrank_by_skipping_the_reranker():
    """With the selector alone it could reach 1.0; a reranked row needs both."""
    rows = [_row("reranked_best", 0.95, rr=3.0), _row("mid", 0.92, rr=0.0),
            _row("low", 0.90, rr=-3.0), _row("cli_row", 0.95)]
    out = pool_scores(rows, alpha=0.4, beta=0.6)
    assert out["cli_row"]["score_blended"] < out["reranked_best"]["score_blended"]


def test_a_never_reranked_pool_keeps_blended_unset():
    """The reranker-comparison view reads 'blended is set' as 'reranked'."""
    out = pool_scores([_row("a", 0.95), _row("b", 0.90)], alpha=0.4, beta=0.6)
    assert all(v["score_blended"] is None for v in out.values())
    assert out["a"]["score_pct"] == 1.0 and out["b"]["score_pct"] == 0.0
