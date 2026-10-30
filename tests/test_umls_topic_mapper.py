"""merge_umls_topics — which UMLS topics an interest gets (FUNCTIONAL_SPEC U.2).

The merge decides which UMLS-derived topics get appended to a profile's
``topic_filters``; switched on, they feed the OpenAlex gatherer's query,
and a bad pick pulls off-topic papers into the candidate pool. They are
chosen on the concepts' confidence alone: from each seed its concepts of
0.9 or more, five of them, at most twenty for the interest; added off.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_lib.umls import topic_mapper
from rag_lib.umls.topic_mapper import MappedTopic, merge_umls_topics


def _mt(tid: str, name: str, conf: float, concept: str = "concept", sim: float = 0.5) -> MappedTopic:
    return MappedTopic(
        topic_id=tid if tid.startswith("https://") else f"https://openalex.org/{tid}",
        display_name=name,
        similarity=sim,
        source_cui="C0000000",
        source_name=concept,
        confidence=conf,
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


def _umls(filters: dict) -> list[dict]:
    return [t for t in filters["topics"] if t.get("source") == "umls"]


# ---------------------------------------------------------------------------
# Choosing (U-1 .. U-5, U-8)
# ---------------------------------------------------------------------------


def test_u1_only_sure_concepts_five_a_seed_by_confidence():
    paper = [_mt(f"T9{i:02d}", f"Topic {i}", c) for i, c in
             enumerate([0.95, 0.89, 1.0, 0.92, 0.7, 0.99, 0.9, 0.91, 0.96])]
    out = merge_umls_topics(_base("Base"), [paper])
    # 0.89 and 0.7 are under 0.9; of the seven left, the five surest
    assert _umls_ids(out) == ["Topic 2", "Topic 5", "Topic 8", "Topic 0", "Topic 3"]


def test_u1_settings_are_honoured():
    paper = [_mt(f"T9{i:02d}", f"Topic {i}", c) for i, c in enumerate([0.95, 0.85, 0.8])]
    out = merge_umls_topics(_base("Base"), [paper], min_confidence=0.8, per_paper=2)
    assert _umls_ids(out) == ["Topic 0", "Topic 1"]


def test_u2_a_concept_whose_topic_the_interest_has_is_skipped_and_the_next_takes_its_place():
    base = _base("Microglia and Neuroinflammation")
    paper = [_mt(base["topics"][0]["id"], "Microglia and Neuroinflammation", 1.0, "Microglia")] + \
        [_mt(f"T9{i:02d}", f"Topic {i}", 0.99 - i / 100) for i in range(6)]
    out = merge_umls_topics(base, [paper])
    assert _umls_ids(out) == [f"Topic {i}" for i in range(5)]


def test_u3_two_concepts_one_topic_count_once_and_the_surer_concept_stays():
    paper = [_mt("T900", "Immune Response", 0.99, "Immune response"),
             _mt("T900", "Immune Response", 0.95, "Inflammation"),
             *[_mt(f"T91{i}", f"Topic {i}", 0.94 - i / 100) for i in range(5)]]
    out = merge_umls_topics(_base("Base"), [paper])
    topics = _umls(out)
    assert [t["display_name"] for t in topics] == ["Immune Response", *[f"Topic {i}" for i in range(4)]]
    assert topics[0]["concept"] == "Immune response" and topics[0]["confidence"] == 0.99


def test_u4_two_seeds_choosing_one_topic_list_it_once_with_the_best_confidence():
    a = [_mt("T900", "Shared", 0.93, "from a")]
    b = [_mt("T900", "Shared", 0.98, "from b"), _mt("T901", "Only b", 0.95)]
    topics = _umls(merge_umls_topics(_base("Base"), [a, b]))
    assert [t["display_name"] for t in topics] == ["Shared", "Only b"]
    assert topics[0]["concept"] == "from b" and topics[0]["count"] == 2 and topics[0]["papers"] == [0, 1]
    assert topics[1]["papers"] == [1]


def test_u4_identically_named_topics_under_two_ids_are_one():
    """OpenAlex carries "Medical research and treatments" and "Medical
    Research and Treatments" under different ids; one is enough."""
    a = [_mt("T900", "Medical research and treatments", 0.95)]
    b = [_mt("T901", "Medical Research and Treatments", 0.99)]
    assert _umls_ids(merge_umls_topics(_base("Base"), [a, b])) == ["Medical Research and Treatments"]


def test_u5_six_seeds_of_five_topics_keep_the_twenty_surest():
    seeds = [[_mt(f"T{s}{i}", f"S{s} T{i}", round(0.9 + (s * 5 + i) / 400, 4)) for i in range(5)]
             for s in range(6)]
    out = _umls(merge_umls_topics(_base("Base"), seeds))
    assert len(out) == 20
    every = sorted((t for p in seeds for t in p), key=lambda m: -m.confidence)
    assert [t["display_name"] for t in out] == [m.display_name for m in every[:20]]


def test_max_additions_is_respected():
    paper = [_mt(f"T9{i:02d}", f"Topic {i}", 0.99 - i / 100) for i in range(5)]
    assert len(_umls_ids(merge_umls_topics(_base("Base"), [paper], max_additions=3))) == 3


def test_u6_added_topics_are_off_and_say_where_they_came_from():
    out = _umls(merge_umls_topics(_base("Base"), [[_mt("T900", "Topic", 0.97, "Adaptive Immunity")]]))
    assert out == [{"id": "https://openalex.org/T900", "display_name": "Topic", "count": 1, "source": "umls",
                    "on": False, "concept": "Adaptive Immunity", "confidence": 0.97, "papers": [0]}]


def test_u8_a_seed_without_a_sure_concept_adds_nothing():
    base = _base("Base")
    out = merge_umls_topics(base, [[_mt("T900", "Topic", 0.89)], [_mt("T901", "Other", 0.5)]])
    assert out == base


def test_a_mapping_without_confidence_adds_nothing():
    old = MappedTopic(topic_id="https://openalex.org/T900", display_name="Old", similarity=0.9,
                      source_cui="C1", source_name="old")
    assert _umls_ids(merge_umls_topics(_base("Base"), [[old]])) == []


def test_similarity_does_not_choose():
    """A concept mapped to a topic far from its name still counts on its
    confidence; the similarity is not a threshold or a rank."""
    near = _mt("T900", "Near", 0.91, sim=0.95)
    far = _mt("T901", "Far", 0.99, sim=0.10)
    assert _umls_ids(merge_umls_topics(_base("Base"), [[near, far]], per_paper=1)) == ["Far"]


def test_stale_mapping_is_recognised():
    assert topic_mapper.stale_mapping([{"topic_id": "T1", "similarity": 0.5}])
    assert not topic_mapper.stale_mapping([_mt("T1", "x", 0.9).to_dict()])
    assert not topic_mapper.stale_mapping([])


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
        _mt("https://openalex.org/T900", "Renal Diseases and Glomerulopathies", 0.96),
        _mt("https://openalex.org/T901", "Renal and Vascular Pathologies", 0.95),
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

    The topics are a list the user switches on in the wizard, so a
    redundant neighbour costs one look while a discarded one can never
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
        _mt(renal_a, "Renal Diseases and Glomerulopathies", 0.96),
        _mt(renal_b, "Renal and Vascular Pathologies", 0.95),
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

    per_paper = [[_mt(candidate_id, "Renal Diseases and Glomerulopathies", 0.96)]]
    out = merge_umls_topics(
        base, per_paper, max_additions=4, cache_dir="/nonexistent",
    )
    assert _umls_ids(out) == []


def test_distinct_topics_survive_cosine_dedup(monkeypatch):
    a, b = "https://openalex.org/T900", "https://openalex.org/T901"
    index = _FakeIndex({a: [1.0, 0.0, 0.0], b: [0.0, 1.0, 0.0]})  # orthogonal
    monkeypatch.setattr(topic_mapper, "get_topic_index", lambda _d: index)

    per_paper = [[_mt(a, "Retinal Diseases", 0.99), _mt(b, "Heart Rate", 0.98)]]
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


# --------------------------------------------------------------------------
# The index must come from the embedder the mapper uses
# --------------------------------------------------------------------------


def _write_index(d, dim: int, model: str | None):
    import json

    from rag_lib.umls.cache import META_FILE

    topics = [{"id": f"https://openalex.org/T{i}", "display_name": f"Topic {i}"} for i in range(3)]
    (d / "openalex_topics.json").write_text(json.dumps(topics))
    np.savez_compressed(d / "openalex_topic_embeddings.npz", embeddings=np.eye(3, dim, dtype=np.float32))
    if model:
        (d / META_FILE).write_text(json.dumps({"embedding_model": model, "dim": dim}))


@pytest.fixture()
def fresh_index(monkeypatch):
    from rag_lib import embedders
    from rag_lib.umls import cache

    monkeypatch.setattr(cache, "_index", None)
    dims = {"model-1024": 1024, "model-768": 768}
    monkeypatch.setattr(embedders, "get_embedder",
                        lambda name: (lambda text: [1.0] + [0.0] * (dims[name] - 1)))


def _concept():
    from rag_lib.umls.extractor import UmlsConcept
    return UmlsConcept(cui="C1", name="diabetes", tui="T047", semantic_type="Disease", confidence=0.9)


def test_an_index_built_with_the_mappers_model_maps(tmp_path, fresh_index):
    _write_index(tmp_path, 1024, "model-1024")
    out = topic_mapper.map_concepts_to_topics([_concept()], embedding_model="model-1024", cache_dir=str(tmp_path))
    assert [m.topic_id for m in out] == ["https://openalex.org/T0"]


def test_an_index_built_with_another_model_is_refused_by_name(tmp_path, fresh_index):
    from rag_lib.umls.cache import TopicIndexMismatch

    # Same dimension, different model: used to match nonsense silently.
    _write_index(tmp_path, 1024, "other-1024")
    with pytest.raises(TopicIndexMismatch, match="build_topic_index.py --embedding-model model-1024"):
        topic_mapper.map_concepts_to_topics([_concept()], embedding_model="model-1024", cache_dir=str(tmp_path))


def test_an_old_index_without_meta_is_refused_on_dimension(tmp_path, fresh_index):
    from rag_lib.umls.cache import TopicIndexMismatch

    # The default rebuild wrote 768-d specter2 vectors; mxbai gives 1024-d.
    _write_index(tmp_path, 768, None)
    with pytest.raises(TopicIndexMismatch, match="768-d"):
        topic_mapper.map_concepts_to_topics([_concept()], embedding_model="model-1024", cache_dir=str(tmp_path))


def test_each_concept_gets_its_one_nearest_topic_however_far(tmp_path, fresh_index):
    from rag_lib.umls.extractor import UmlsConcept

    _write_index(tmp_path, 1024, "model-1024")
    concepts = [UmlsConcept(cui="C1", name="a", tui="T1", semantic_type="x", confidence=0.95),
                UmlsConcept(cui="C2", name="b", tui="T1", semantic_type="x", confidence=0.72)]
    out = topic_mapper.map_concepts_to_topics(concepts, embedding_model="model-1024", cache_dir=str(tmp_path))
    assert [(m.source_cui, m.confidence) for m in out] == [("C1", 0.95), ("C2", 0.72)]
    assert all(m.topic_id == "https://openalex.org/T0" for m in out)    # one topic each, kept twice


def test_a_topic_the_profile_wants_is_already_the_interests():
    wanted = [{"id": "https://openalex.org/T900", "display_name": "Wanted"}]
    paper = [_mt("T900", "Wanted", 0.99)] + [_mt(f"T9{i}{i}", f"Topic {i}", 0.98 - i / 100) for i in range(1, 6)]
    out = merge_umls_topics(_base("Base"), [paper], also_have=wanted)
    assert _umls_ids(out) == [f"Topic {i}" for i in range(1, 6)]
