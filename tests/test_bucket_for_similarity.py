"""Card buckets: the threshold decides "low" before anything else."""

from rag_lib.api.mappers import bucket_for_similarity


def test_under_the_threshold_is_low_even_when_like_the_seeds():
    # A threshold set above the least typical seed (0.85) used to make a
    # 0.88 paper under a 0.9 bar "high", so "Below the bar" dropped it.
    assert bucket_for_similarity(0.88, threshold=0.9, seed_sim_min=0.85, percentile=None) == "low"
    assert bucket_for_similarity(0.92, threshold=0.9, seed_sim_min=0.85, percentile=None) == "high"
    assert bucket_for_similarity(0.86, threshold=0.8, seed_sim_min=0.9, percentile=None) == "medium"
    assert bucket_for_similarity(0.75, threshold=0.8, seed_sim_min=0.9, percentile=None) == "low"
