"""merge_umls_topics — which UMLS topics an interest gets (FUNCTIONAL_SPEC U.2).

The merge decides which UMLS-derived topics get appended to a profile's
``topic_filters``; switched on, they feed the OpenAlex gatherer's query,
and a bad pick pulls off-topic papers into the candidate pool. Since CTX
(records/35) they are chosen on how near each concept, embedded in the
sentences it was read from, is to its topic: from each seed its concepts
of 0.9 or more, the five nearest; at most twenty for the interest, those
more seeds chose first, then the nearer; added off. There is no
similarity bar any more (SIMILARITY 0.0).
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_lib.umls import topic_mapper
from rag_lib.umls.topic_mapper import MappedTopic, merge_umls_topics


def _mt(tid: str, name: str, conf: float, concept: str = "concept", sim: float = 0.7,
        mention: str | None = None, cui: str | None = None, rank: int = 1) -> MappedTopic:
    # Each entry its own concept unless ``cui`` says two are one concept's
    # neighbours (a concept gives one topic, CTX-3b).
    return MappedTopic(
        topic_id=tid if tid.startswith("https://") else f"https://openalex.org/{tid}",
        display_name=name,
        similarity=sim,
        source_cui=cui or f"C-{concept}-{tid}",
        rank=rank,
        source_name=concept,
        confidence=conf,
        mention=mention,
        mapping=topic_mapper.MAPPING,
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
# Choosing (U-1 .. U-5, U-8; CTX-3)
# ---------------------------------------------------------------------------


def test_u1_only_sure_concepts_five_a_seed_by_similarity():
    # (confidence, similarity): the 0.89 and the 0.7 are nearest but unsure
    pairs = [(0.95, 0.60), (0.89, 0.99), (1.0, 0.55), (0.92, 0.75), (0.7, 0.98),
             (0.99, 0.70), (0.9, 0.80), (0.91, 0.65), (0.96, 0.50)]
    paper = [_mt(f"T9{i:02d}", f"Topic {i}", c, sim=s) for i, (c, s) in enumerate(pairs)]
    out = merge_umls_topics(_base("Base"), [paper])
    # of the seven sure ones, the five nearest, nearest first; not the surest
    assert _umls_ids(out) == ["Topic 6", "Topic 3", "Topic 5", "Topic 7", "Topic 0"]


def test_u1_settings_are_honoured():
    paper = [_mt(f"T9{i:02d}", f"Topic {i}", c, sim=s) for i, (c, s) in
             enumerate([(0.95, 0.5), (0.85, 0.7), (0.8, 0.6), (0.79, 0.9)])]
    out = merge_umls_topics(_base("Base"), [paper], min_confidence=0.8, per_paper=2)
    assert _umls_ids(out) == ["Topic 1", "Topic 2"]


def test_equal_similarity_falls_back_to_confidence_then_name():
    paper = [_mt("T901", "B topic", 0.91), _mt("T902", "A topic", 0.91), _mt("T903", "C topic", 0.99)]
    assert _umls_ids(merge_umls_topics(_base("Base"), [paper])) == ["C topic", "A topic", "B topic"]


def test_u2_a_concept_whose_topic_the_interest_has_is_skipped_and_the_next_takes_its_place():
    base = _base("Microglia and Neuroinflammation")
    paper = [_mt(base["topics"][0]["id"], "Microglia and Neuroinflammation", 1.0, "Microglia", sim=0.99)] + \
        [_mt(f"T9{i:02d}", f"Topic {i}", 0.95, sim=0.9 - i / 100) for i in range(6)]
    out = merge_umls_topics(base, [paper])
    assert _umls_ids(out) == [f"Topic {i}" for i in range(5)]


def test_ctx3b_a_concept_whose_nearest_the_interest_has_gives_its_next():
    """User 11-11 (option B): the Immunity paper's concepts were all nearest
    its own Alzheimer topics, so nothing was added."""
    base = _base("Alzheimer's disease research")
    own = base["topics"][0]["id"]
    paper = [_mt(own, "Alzheimer's disease research", 0.95, "Adaptive Immunity", sim=0.75, cui="C1", rank=1),
             _mt("T901", "Immune responses and vaccinations", 0.95, "Adaptive Immunity", sim=0.69, cui="C1", rank=2),
             _mt("T902", "T-cell and B-cell Immunology", 0.95, "Adaptive Immunity", sim=0.68, cui="C1", rank=3)]
    out = merge_umls_topics(base, [paper])
    assert _umls_ids(out) == ["Immune responses and vaccinations"]
    assert _umls(out)[0]["concept"] == "Adaptive Immunity"


def test_ctx3b_a_concept_gives_one_topic():
    paper = [_mt(f"T90{k}", f"Topic {k}", 0.95, "one", sim=0.9 - k / 100, cui="C1", rank=k + 1) for k in range(3)] + \
        [_mt("T910", "Other", 0.95, "two", sim=0.5, cui="C2")]
    assert _umls_ids(merge_umls_topics(_base("Base"), [paper])) == ["Topic 0", "Other"]


def test_ctx3b_a_topic_a_nearer_concept_took_sends_the_concept_to_its_next():
    paper = [_mt("T901", "Shared", 0.95, "near", sim=0.80, cui="C1"),
             _mt("T901", "Shared", 0.95, "far", sim=0.70, cui="C2", rank=1),
             _mt("T902", "Second of far", 0.95, "far", sim=0.65, cui="C2", rank=2)]
    out = merge_umls_topics(_base("Base"), [paper])
    assert [(t["display_name"], t["concept"]) for t in _umls(out)] == [("Shared", "near"), ("Second of far", "far")]


def test_ctx3b_neighbours_equal_once_rounded_go_nearest_first():
    """Review B3-R2: similarity is stored to 4 places; a tie went by name."""
    paper = [_mt("T901", "Zeta", 0.95, "one", sim=0.8, cui="C1", rank=1),
             _mt("T902", "Alpha", 0.95, "one", sim=0.8, cui="C1", rank=2)]
    assert _umls_ids(merge_umls_topics(_base("Base"), [paper])) == ["Zeta"]


def test_u3_two_concepts_one_topic_count_once_and_the_nearer_concept_stays():
    paper = [_mt("T900", "Immune Response", 0.95, "Inflammation", sim=0.80, mention="inflamed"),
             _mt("T900", "Immune Response", 0.99, "Immune response", sim=0.78),
             *[_mt(f"T91{i}", f"Topic {i}", 0.94, sim=0.70 - i / 100) for i in range(5)]]
    out = merge_umls_topics(_base("Base"), [paper])
    topics = _umls(out)
    # the topic takes one of the five places, not two
    assert [t["display_name"] for t in topics] == ["Immune Response", *[f"Topic {i}" for i in range(4)]]
    assert topics[0]["concept"] == "Inflammation" and topics[0]["similarity"] == 0.80
    assert topics[0]["mention"] == "inflamed" and topics[0]["count"] == 1


def test_u4_two_seeds_choosing_one_topic_list_it_once_with_the_nearest_concept():
    a = [_mt("T900", "Shared", 0.93, "from a", sim=0.71)]
    b = [_mt("T900", "Shared", 0.98, "from b", sim=0.62), _mt("T901", "Only b", 0.95, sim=0.90)]
    topics = _umls(merge_umls_topics(_base("Base"), [a, b]))
    # two seeds before one, however near the one is
    assert [t["display_name"] for t in topics] == ["Shared", "Only b"]
    assert topics[0]["concept"] == "from a" and topics[0]["similarity"] == 0.71
    assert topics[0]["count"] == 2 and topics[0]["papers"] == [0, 1]
    assert topics[1]["papers"] == [1]


def test_u4_identically_named_topics_under_two_ids_are_one():
    """OpenAlex carries "Medical research and treatments" and "Medical
    Research and Treatments" under different ids; one is enough."""
    a = [_mt("T900", "Medical research and treatments", 0.95, sim=0.6)]
    b = [_mt("T901", "Medical Research and Treatments", 0.99, sim=0.7)]
    topics = _umls(merge_umls_topics(_base("Base"), [a, b]))
    assert [t["display_name"] for t in topics] == ["Medical Research and Treatments"]
    assert topics[0]["count"] == 2


def test_u5_more_seeds_first_then_the_nearer_twenty_kept():
    # topic k is chosen by k % 4 + 1 seeds, at similarity 0.5 + k / 100
    seeds: list[list[MappedTopic]] = [[] for _ in range(4)]
    for k in range(30):
        for s in range(k % 4 + 1):
            seeds[s].append(_mt(f"T{k:03d}", f"Topic {k}", 0.95, sim=round(0.5 + k / 100, 4)))
    out = _umls(merge_umls_topics(_base("Base"), seeds, per_paper=30))
    expected = sorted(range(30), key=lambda k: (-(k % 4 + 1), -k))[:20]
    assert [t["display_name"] for t in out] == [f"Topic {k}" for k in expected]
    assert [t["count"] for t in out] == [k % 4 + 1 for k in expected]


def test_max_additions_is_respected():
    paper = [_mt(f"T9{i:02d}", f"Topic {i}", 0.99, sim=0.9 - i / 100) for i in range(5)]
    assert _umls_ids(merge_umls_topics(_base("Base"), [paper], max_additions=3)) == ["Topic 0", "Topic 1", "Topic 2"]


def test_u6_added_topics_are_off_and_say_where_they_came_from():
    out = _umls(merge_umls_topics(_base("Base"), [[_mt("T900", "Topic", 0.97, "Adaptive Immunity",
                                                        sim=0.6543, mention="adaptive immune")]]))
    assert out == [{"id": "https://openalex.org/T900", "display_name": "Topic", "count": 1, "source": "umls",
                    "on": False, "concept": "Adaptive Immunity", "confidence": 0.97,
                    "mention": "adaptive immune", "similarity": 0.6543, "papers": [0]}]


def test_u8_a_seed_without_a_sure_concept_adds_nothing():
    base = _base("Base")
    out = merge_umls_topics(base, [[_mt("T900", "Topic", 0.89, sim=0.99)], [_mt("T901", "Other", 0.5)]])
    assert out == base


def test_a_mapping_without_confidence_adds_nothing():
    old = MappedTopic(topic_id="https://openalex.org/T900", display_name="Old", similarity=0.9,
                      source_cui="C1", source_name="old")
    assert _umls_ids(merge_umls_topics(_base("Base"), [[old]])) == []


def test_similarity_ranks_and_is_no_bar_unless_set():
    """CTX-3: a sentence against a description measures 0.6-0.8 for the
    right topic and the wrong ones alike, so no bar by default; the
    nearer wins whatever the confidence. The setting can still set one."""
    assert topic_mapper.SIMILARITY == 0.0
    near = _mt("T900", "Near", 0.91, sim=0.75)
    far = _mt("T901", "Far", 0.99, sim=0.31)
    assert _umls_ids(merge_umls_topics(_base("Base"), [[far, near]], per_paper=1)) == ["Near"]
    assert _umls_ids(merge_umls_topics(_base("Base"), [[far]])) == ["Far"]
    assert _umls_ids(merge_umls_topics(_base("Base"), [[far, near]], min_similarity=0.5)) == ["Near"]


def test_a_mapping_without_similarity_is_not_used():
    bare = MappedTopic(topic_id="https://openalex.org/T900", display_name="Bare",
                       similarity=None, source_cui="C1", source_name="x", confidence=0.99)
    assert _umls_ids(merge_umls_topics(_base("Base"), [[bare]])) == []


def test_the_setting_reaches_the_merge():
    from rag_lib.api.settings import Settings

    assert Settings.model_fields["RADAR_UMLS_TOPIC_MIN_SIMILARITY"].default == topic_mapper.SIMILARITY == 0.0


def test_stale_mapping_is_recognised():
    # made before CTX: no mapping marker (or another one)
    assert topic_mapper.stale_mapping([{"topic_id": "T1", "similarity": 0.5}])
    assert topic_mapper.stale_mapping([{"topic_id": "T1", "similarity": 0.5, "mapping": "name-0"}])
    # one topic a concept (CTX before 3b): mapped again for its neighbours
    assert topic_mapper.stale_mapping([{"topic_id": "T1", "similarity": 0.5, "mapping": "context-1"}])
    assert topic_mapper.stale_mapping([_mt("T1", "x", 0.9).to_dict(), {"topic_id": "T2"}])
    assert not topic_mapper.stale_mapping([_mt("T1", "x", 0.9).to_dict()])
    assert not topic_mapper.stale_mapping([])


def test_stale_extraction_is_recognised():
    fresh = {"cui": "C1", "name": "x", "mentions": [{"text": "x", "sentence": "x is."}]}
    mapped = [_mt("T1", "x", 0.9).to_dict()]
    assert not topic_mapper.stale_extraction([fresh], mapped)
    # concepts stored without their sentences
    assert topic_mapper.stale_extraction([fresh, {"cui": "C2", "name": "y"}], mapped)
    # a mapping by the earlier rules
    assert topic_mapper.stale_extraction([fresh], [{"topic_id": "T1"}])
    # a concept with no usable mention still has the key: not stale
    assert not topic_mapper.stale_extraction([{"cui": "C3", "name": "z", "mentions": []}], [])
    assert not topic_mapper.stale_extraction([], [])
    assert not topic_mapper.stale_extraction(["junk"], [None])


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


def _write_index(d, dim: int, model: str | None, *, rich: bool = True):
    import json

    from rag_lib.umls import cache

    names = ((cache.RICH_TOPICS_FILE, cache.RICH_EMBEDDINGS_FILE, cache.RICH_META_FILE) if rich
             else (cache.TOPICS_FILE, cache.EMBEDDINGS_FILE, cache.META_FILE))
    topics = [{"id": f"https://openalex.org/T{i}", "display_name": f"Topic {i}"} for i in range(3)]
    (d / names[0]).write_text(json.dumps(topics))
    np.savez_compressed(d / names[1], embeddings=np.eye(3, dim, dtype=np.float32))
    if model:
        (d / names[2]).write_text(json.dumps({"embedding_model": model, "dim": dim}))


@pytest.fixture()
def fresh_index(monkeypatch):
    """Fresh index caches and a stub embedder: a text naming "Topic k" points
    at topic k; the texts embedded are in the list it returns."""
    from rag_lib import embedders
    from rag_lib.umls import cache

    monkeypatch.setattr(cache, "_index", None)
    monkeypatch.setattr(cache, "_rich_index", None)
    dims = {"model-1024": 1024, "model-768": 768}
    seen: list[str] = []

    def embed(name):
        def _e(text):
            seen.append(text)
            v = [0.0] * dims[name]
            v[next((k for k in range(3) if f"Topic {k}" in text), 0)] = 1.0
            return v
        return _e

    monkeypatch.setattr(embedders, "get_embedder", embed)
    return seen


def _concept(cui="C1", name="diabetes", confidence=0.95, mentions=(("diabetic", "Diabetic mice were fed."),)):
    from rag_lib.umls.extractor import UmlsConcept
    return UmlsConcept(cui=cui, name=name, tui="T047", semantic_type="Disease", confidence=confidence,
                       mentions=tuple(mentions))


def test_an_index_built_with_the_mappers_model_maps(tmp_path, fresh_index):
    _write_index(tmp_path, 1024, "model-1024")
    out = topic_mapper.map_concepts_to_topics([_concept()], embedding_model="model-1024", cache_dir=str(tmp_path))
    assert [(m.mention, m.mapping, m.rank) for m in out] == [
        ("diabetic", topic_mapper.MAPPING, k + 1) for k in range(3)]
    assert out[0].topic_id == "https://openalex.org/T0"
    assert {m.topic_id for m in out} == {f"https://openalex.org/T{k}" for k in range(3)}
    # the concept embedded in its sentence, not its name alone
    assert fresh_index == ["diabetes: Diabetic mice were fed."]


def test_the_rich_index_is_searched_not_the_name_one(tmp_path, fresh_index):
    _write_index(tmp_path, 1024, "model-1024", rich=False)
    with pytest.raises(FileNotFoundError, match="build_topic_index.py --rich"):
        topic_mapper.map_concepts_to_topics([_concept()], embedding_model="model-1024", cache_dir=str(tmp_path))


def test_an_index_built_with_another_model_is_refused_by_name(tmp_path, fresh_index):
    from rag_lib.umls.cache import TopicIndexMismatch

    # Same dimension, different model: used to match nonsense silently.
    _write_index(tmp_path, 1024, "other-1024")
    with pytest.raises(TopicIndexMismatch, match="build_topic_index.py --rich --embedding-model model-1024"):
        topic_mapper.map_concepts_to_topics([_concept()], embedding_model="model-1024", cache_dir=str(tmp_path))


def test_an_old_index_without_meta_is_refused_on_dimension(tmp_path, fresh_index):
    from rag_lib.umls.cache import TopicIndexMismatch

    # The default rebuild wrote 768-d specter2 vectors; mxbai gives 1024-d.
    _write_index(tmp_path, 768, None)
    with pytest.raises(TopicIndexMismatch, match="768-d"):
        topic_mapper.map_concepts_to_topics([_concept()], embedding_model="model-1024", cache_dir=str(tmp_path))


def test_each_concept_gets_its_nearest_topics_however_far(tmp_path, fresh_index):
    _write_index(tmp_path, 1024, "model-1024")
    concepts = [_concept("C1", "a", 0.95, [("a", "About Topic 2.")]),
                _concept("C2", "b", 0.91, [("b", "Nothing near.")])]
    out = topic_mapper.map_concepts_to_topics(concepts, embedding_model="model-1024", cache_dir=str(tmp_path),
                                              neighbours=1)
    assert [(m.source_cui, m.topic_id) for m in out] == [("C1", "https://openalex.org/T2"),
                                                        ("C2", "https://openalex.org/T0")]


def test_ctx3b_each_concept_keeps_its_neighbours_nearest_first(tmp_path, fresh_index):
    """CTX-3b: NEIGHBOURS (3) topics a concept, ranked 1, 2, 3, nearest first."""
    assert topic_mapper.NEIGHBOURS == 3
    _write_index(tmp_path, 1024, "model-1024")
    c = _concept(mentions=[("x", "On Topic 2."), ("x", "On Topic 2 again."), ("x", "On Topic 1.")])
    out = topic_mapper.map_concepts_to_topics([c], embedding_model="model-1024", cache_dir=str(tmp_path))
    assert [(m.topic_id[-2:], m.rank) for m in out] == [("T2", 1), ("T1", 2), ("T0", 3)]
    assert out[0].similarity > out[1].similarity > out[2].similarity


def test_the_mentions_are_averaged(tmp_path, fresh_index):
    _write_index(tmp_path, 1024, "model-1024")
    # by its first sentence alone Topic 0; two of its three say Topic 1
    c = _concept(mentions=[("x", "On Topic 0."), ("x", "On Topic 1."), ("x", "Again Topic 1.")])
    out = topic_mapper.map_concepts_to_topics([c], embedding_model="model-1024", cache_dir=str(tmp_path))
    assert [m.topic_id for m in out if m.rank == 1] == ["https://openalex.org/T1"]
    assert out[0].similarity == round(2 / 5 ** 0.5, 4)
    assert len(fresh_index) == 3


def test_unsure_concepts_and_ones_without_a_mention_are_not_mapped(tmp_path, fresh_index):
    _write_index(tmp_path, 1024, "model-1024")
    concepts = [_concept("C1", "sure", 0.9), _concept("C2", "unsure", 0.89),
                _concept("C3", "no mention", 0.99, mentions=())]
    out = topic_mapper.map_concepts_to_topics(concepts, embedding_model="model-1024", cache_dir=str(tmp_path))
    assert {m.source_cui for m in out} == {"C1"}
    assert fresh_index == ["sure: Diabetic mice were fed."]       # nothing embedded for the others
    out = topic_mapper.map_concepts_to_topics(concepts, embedding_model="model-1024", cache_dir=str(tmp_path),
                                              min_confidence=0.8)
    assert [m.source_cui for m in out if m.rank == 1] == ["C1", "C2"]


def test_every_map_passes_the_confidence_setting():
    """Class check: each map_concepts_to_topics call in rag_lib passes
    min_confidence, so RADAR_UMLS_TOPIC_MIN_CONFIDENCE also decides what is
    embedded, not only what is chosen."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    missing = []
    for path in root.joinpath("rag_lib").rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in ("rag_lib/umls/__init__.py", "rag_lib/umls/topic_mapper.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and (getattr(node.func, "id", None)
                                               or getattr(node.func, "attr", None)) == "map_concepts_to_topics":
                if not any(k.arg == "min_confidence" for k in node.keywords):
                    missing.append(f"{rel}:{node.lineno}")
    assert missing == []


def test_a_topic_the_profile_wants_is_already_the_interests():
    wanted = [{"id": "https://openalex.org/T900", "display_name": "Wanted"}]
    paper = [_mt("T900", "Wanted", 0.99, sim=0.9)] + [_mt(f"T9{i}{i}", f"Topic {i}", 0.95, sim=0.8 - i / 100)
                                                      for i in range(1, 6)]
    out = merge_umls_topics(_base("Base"), [paper], also_have=wanted)
    assert _umls_ids(out) == [f"Topic {i}" for i in range(1, 6)]


def test_every_merge_passes_the_similarity_setting():
    """Class check (UT-d): each merge_umls_topics call in rag_lib passes
    min_similarity, so the setting is not silently the library default."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    missing = []
    for path in root.joinpath("rag_lib").rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in ("rag_lib/umls/__init__.py", "rag_lib/umls/topic_mapper.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and (getattr(node.func, "id", None)
                                               or getattr(node.func, "attr", None)) == "merge_umls_topics":
                if not any(k.arg == "min_similarity" for k in node.keywords):
                    missing.append(f"{rel}:{node.lineno}")
    assert missing == []


@pytest.mark.parametrize("field", ["confidence", "similarity"])
def test_nan_does_not_pass_a_bar(field):
    """Review L3: ``NaN < bar`` is False, so NaN used to pass."""
    import dataclasses

    nan = dataclasses.replace(_mt("T900", "Nan", 0.99), **{field: float("nan")})
    good = _mt("T901", "Good", 0.95)
    assert _umls_ids(merge_umls_topics(_base("Base"), [[nan, good]], per_paper=1)) == ["Good"]


SCORE_BARS = ["RADAR_UMLS_MIN_CONFIDENCE", "RADAR_UMLS_TOPIC_MIN_CONFIDENCE", "RADAR_UMLS_TOPIC_MIN_SIMILARITY",
              "RADAR_RERANKER_MIN_UMLS_CONFIDENCE", "RADAR_LLM_TOPIC_MIN_SIMILARITY",
              "RADAR_LLM_TOPIC_EXTRA_SIMILARITY"]


@pytest.mark.parametrize("field", SCORE_BARS)
@pytest.mark.parametrize("value,ok", [("80", False), ("-0.1", False), ("1.01", False),
                                      ("0", True), ("1", True), ("0.8", True)])
def test_a_score_bar_outside_0_to_1_stops_the_start(monkeypatch, field, value, ok):
    """Review L4: 80 for 0.80 quietly dropped everything."""
    from pydantic import ValidationError

    from rag_lib.api.settings import Settings

    monkeypatch.setenv("RADAR_REQUIRE_AUTH", "false")
    monkeypatch.setenv(field, value)
    if ok:
        assert getattr(Settings(_env_file=None), field) == float(value)
    else:
        with pytest.raises(ValidationError, match="between 0 and 1"):
            Settings(_env_file=None)


def test_every_score_bar_setting_is_checked():
    """Class check: a float setting named like a bar (CONFIDENCE / SIMILARITY)
    is in the 0-1 check."""
    from rag_lib.api.settings import Settings

    bars = sorted(n for n, f in Settings.model_fields.items()
                  if f.annotation is float and ("CONFIDENCE" in n or "SIMILARITY" in n))
    assert bars == sorted(SCORE_BARS)


def test_the_shown_word_is_the_surest_mention(tmp_path, fresh_index):
    """Mentions come surest first (extractor); the topic shows that one
    (review CTX-R5: the last one passed every test)."""
    _write_index(tmp_path, 1024, "model-1024")
    c = _concept(mentions=[("surest word", "On Topic 0."), ("least sure", "On Topic 0 too.")])
    out = topic_mapper.map_concepts_to_topics([c], embedding_model="model-1024", cache_dir=str(tmp_path))
    assert {m.mention for m in out} == {"surest word"}


def test_a_zero_vector_is_near_nothing(tmp_path, fresh_index, monkeypatch):
    """Review CTX-R6: with no similarity bar a zero vector would still
    bring in whichever topic came first."""
    from rag_lib import embedders

    _write_index(tmp_path, 1024, "model-1024")
    inner = embedders.get_embedder

    def embed(name):
        real = inner(name)
        return lambda text: [0.0] * 1024 if "ZERO" in text else (
            [float("nan")] * 1024 if "NAN" in text else real(text))
    monkeypatch.setattr(embedders, "get_embedder", embed)
    only_zero = _concept("C1", "a", 0.95, [("a", "ZERO here.")])
    nan_and_good = _concept("C2", "b", 0.95, [("b", "NAN here."), ("b", "On Topic 2.")])
    out = topic_mapper.map_concepts_to_topics([only_zero, nan_and_good], embedding_model="model-1024",
                                              cache_dir=str(tmp_path))
    assert [(m.source_cui, m.topic_id, m.similarity) for m in out if m.rank == 1] == [
        ("C2", "https://openalex.org/T2", 1.0)]
    assert {m.source_cui for m in out} == {"C2"}


def test_the_confidence_class_check_sees_every_caller():
    """The class check above walks the real callers, not none: vault and the
    extractor's two walkthroughs."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    found = [p.relative_to(root).as_posix() for p in root.joinpath("rag_lib").rglob("*.py")
             for node in ast.walk(ast.parse(p.read_text(encoding="utf-8")))
             if isinstance(node, ast.Call)
             and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "map_concepts_to_topics"]
    assert sorted(found) == ["rag_lib/api/services/vault.py", "rag_lib/umls/extractor.py", "rag_lib/umls/extractor.py"]
