"""merge_umls_topics — ranking and near-duplicate rejection.

The merge decides which UMLS-derived topics get appended to a profile's
``topic_filters``, and those topics feed the OpenAlex gatherer's tier-2
query. A bad pick doesn't just look wrong in the UI — it actively pulls
off-topic papers into the candidate pool, so the selection rules are
worth pinning down.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_lib.umls import topic_mapper
from rag_lib.umls.topic_mapper import MappedTopic, merge_umls_topics


def _mt(tid: str, name: str, sim: float) -> MappedTopic:
    return MappedTopic(
        topic_id=tid,
        display_name=name,
        similarity=sim,
        source_cui="C0000000",
        source_name="concept",
    )


def _base(*names: str) -> dict:
    return {
        "topics": [
            {"id": f"https://openalex.org/T{100 + i}", "display_name": n}
            for i, n in enumerate(names)
        ],
        "subfields": [],
    }


class _FakeIndex:
    """Topic index stub: vectors supplied per topic id."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._v = {
            k: np.array(v, dtype=np.float32) / (np.linalg.norm(v) + 1e-12)
            for k, v in vectors.items()
        }

    def vector_for(self, topic_id: str):
        return self._v.get(topic_id)


def _umls_ids(filters: dict) -> list[str]:
    return [t["display_name"] for t in filters["topics"] if t.get("source") == "umls"]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_similarity_outranks_paper_count():
    """A high-confidence match beats a generic term seen in every paper.

    With a handful of seeds the count is a coarse 1/2/3 with almost no
    discriminating power, so ranking on it first let a term that merely
    appeared in each paper's reference list win a slot over a strong
    domain match.
    """
    generic = _mt("https://openalex.org/T900", "Proteins in Food Systems", 0.42)
    precise = _mt("https://openalex.org/T901", "Retinal Diseases and Treatments", 0.89)

    # generic appears in all three papers, precise in only one
    per_paper = [[generic, precise], [generic], [generic]]

    out = merge_umls_topics(_base("Diabetes Management"), per_paper, max_additions=1)
    assert _umls_ids(out) == ["Retinal Diseases and Treatments"]


def test_count_breaks_ties_on_equal_similarity():
    a = _mt("https://openalex.org/T900", "Alpha", 0.80)
    b = _mt("https://openalex.org/T901", "Beta", 0.80)
    per_paper = [[a, b], [b]]  # b seen twice

    out = merge_umls_topics(_base("Base"), per_paper, max_additions=1)
    assert _umls_ids(out) == ["Beta"]


def test_max_additions_is_respected():
    per_paper = [[
        _mt(f"https://openalex.org/T{900 + i}", f"Topic {i}", 0.9 - i / 100)
        for i in range(10)
    ]]
    out = merge_umls_topics(_base("Base"), per_paper, max_additions=3)
    assert len(_umls_ids(out)) == 3


# ---------------------------------------------------------------------------
# Near-duplicate rejection
# ---------------------------------------------------------------------------


def test_exact_id_match_against_base_is_excluded():
    base = _base("Diabetes Management")
    existing_id = base["topics"][0]["id"]
    per_paper = [[_mt(existing_id, "Diabetes Management", 0.99)]]

    out = merge_umls_topics(base, per_paper, max_additions=4)
    assert _umls_ids(out) == []


def test_exact_name_match_against_base_is_excluded():
    per_paper = [[_mt("https://openalex.org/T900", "diabetes management", 0.99)]]
    out = merge_umls_topics(_base("Diabetes Management"), per_paper, max_additions=4)
    assert _umls_ids(out) == []


def test_word_overlap_misses_synonyms_without_the_index():
    """Documents the fallback's weakness — motivates the cosine path.

    "Renal Diseases and Glomerulopathies" vs "Renal and Vascular
    Pathologies" share only "renal" and the stopword "and", so Jaccard
    scores 0.33 and both slip through.
    """
    per_paper = [[
        _mt("https://openalex.org/T900", "Renal Diseases and Glomerulopathies", 0.86),
        _mt("https://openalex.org/T901", "Renal and Vascular Pathologies", 0.85),
    ]]
    out = merge_umls_topics(_base("Base"), per_paper, max_additions=4)
    assert len(_umls_ids(out)) == 2


def test_candidates_are_not_deduplicated_against_each_other_by_cosine(monkeypatch):
    """Deliberate: both survive, even at cosine 0.95.

    This used to collapse to one. The cosine test is still applied
    against topics the profile already has, but no longer between
    candidates, because no threshold separates the two cases it has to
    tell apart. It was calibrated on synonym pairs (0.806-0.854) versus
    unrelated pairs (0.512-0.687); sibling topics inside one field were
    not in that set and measure 0.719-0.815, overlapping the synonyms.
    Keeping the test cost a type-2 diabetes profile both "Diabetes,
    Cardiovascular Risks and Lipoproteins" (0.761 against the accepted
    "Diabetes and associated disorders") and "Diabetes Treatment and
    Management" (0.815) — distinct topics that retrieve distinct papers.

    The candidates are a list the user prunes in the wizard, so a
    redundant neighbour costs one toggle while a discarded one can never
    be chosen at all.
    """
    renal_a = "https://openalex.org/T900"
    renal_b = "https://openalex.org/T901"
    index = _FakeIndex({
        renal_a: [1.0, 0.0, 0.0],
        renal_b: [0.95, 0.31, 0.0],   # cosine ~0.95 with renal_a
    })
    monkeypatch.setattr(topic_mapper, "get_topic_index", lambda _d: index)

    per_paper = [[
        _mt(renal_a, "Renal Diseases and Glomerulopathies", 0.86),
        _mt(renal_b, "Renal and Vascular Pathologies", 0.85),
    ]]
    out = merge_umls_topics(
        _base("Base"), per_paper, max_additions=4, cache_dir="/nonexistent",
    )
    assert _umls_ids(out) == [
        "Renal Diseases and Glomerulopathies",
        "Renal and Vascular Pathologies",
    ]


def test_a_candidate_matching_an_existing_topic_is_still_dropped(monkeypatch):
    """The other direction of the same test is unchanged — it has a
    reason the candidate-vs-candidate one lacked: the profile already
    carries that topic, so adding it again is pure duplication."""
    base_id = "https://openalex.org/T800"
    cand = "https://openalex.org/T900"
    index = _FakeIndex({
        base_id: [1.0, 0.0, 0.0],
        cand: [0.95, 0.31, 0.0],
    })
    monkeypatch.setattr(topic_mapper, "get_topic_index", lambda _d: index)

    base = {
        "topics": [{"id": base_id, "display_name": "Renal Diseases and Glomerulopathies"}],
        "subfields": [], "fields": [], "domains": [],
    }
    per_paper = [[_mt(cand, "Renal and Vascular Pathologies", 0.9)]]
    out = merge_umls_topics(
        base, per_paper, max_additions=4, cache_dir="/nonexistent",
    )
    assert _umls_ids(out) == []


def test_identically_named_candidates_still_collapse():
    """OpenAlex carries "Medical research and treatments" and "Medical
    Research and Treatments" under different ids; one is enough. Checked
    before the cut, so the collision does not consume a slot."""
    per_paper = [[
        _mt("https://openalex.org/T900", "Medical research and treatments", 0.9),
        _mt("https://openalex.org/T901", "Medical Research and Treatments", 0.88),
        _mt("https://openalex.org/T902", "Diabetes Treatment and Management", 0.7),
    ]]
    out = merge_umls_topics(_base("Base"), per_paper, max_additions=2)
    assert _umls_ids(out) == [
        "Medical research and treatments",
        "Diabetes Treatment and Management",
    ]


def test_cosine_dedup_rejects_candidate_close_to_existing_topic(monkeypatch):
    """A UMLS topic that merely renames an existing one is not added."""
    base = _base("Renal and Vascular Pathologies")
    existing_id = base["topics"][0]["id"]
    candidate_id = "https://openalex.org/T900"
    index = _FakeIndex({
        existing_id: [1.0, 0.0, 0.0],
        candidate_id: [0.96, 0.28, 0.0],
    })
    monkeypatch.setattr(topic_mapper, "get_topic_index", lambda _d: index)

    per_paper = [[_mt(candidate_id, "Renal Diseases and Glomerulopathies", 0.86)]]
    out = merge_umls_topics(
        base, per_paper, max_additions=4, cache_dir="/nonexistent",
    )
    assert _umls_ids(out) == []


def test_distinct_topics_survive_cosine_dedup(monkeypatch):
    a, b = "https://openalex.org/T900", "https://openalex.org/T901"
    index = _FakeIndex({a: [1.0, 0.0, 0.0], b: [0.0, 1.0, 0.0]})  # orthogonal
    monkeypatch.setattr(topic_mapper, "get_topic_index", lambda _d: index)

    per_paper = [[_mt(a, "Retinal Diseases", 0.9), _mt(b, "Heart Rate", 0.88)]]
    out = merge_umls_topics(
        _base("Base"), per_paper, max_additions=4, cache_dir="/nonexistent",
    )
    assert _umls_ids(out) == ["Retinal Diseases", "Heart Rate"]


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_missing_index_falls_back_instead_of_raising(monkeypatch):
    """A missing/corrupt index must not break topic aggregation."""
    def _boom(_d):
        raise FileNotFoundError("topic index not built")

    monkeypatch.setattr(topic_mapper, "get_topic_index", _boom)
    per_paper = [[_mt("https://openalex.org/T900", "Retinal Diseases", 0.9)]]
    out = merge_umls_topics(
        _base("Base"), per_paper, max_additions=4, cache_dir="/nonexistent",
    )
    assert _umls_ids(out) == ["Retinal Diseases"]


def test_empty_input_returns_base_unchanged():
    base = _base("Diabetes Management")
    out = merge_umls_topics(base, [], max_additions=4)
    assert out == base
    assert out is not base  # copy, not the caller's dict


def test_base_filters_not_mutated():
    base = _base("Diabetes Management")
    before = len(base["topics"])
    per_paper = [[_mt("https://openalex.org/T900", "Retinal Diseases", 0.9)]]
    merge_umls_topics(base, per_paper, max_additions=4)
    assert len(base["topics"]) == before
