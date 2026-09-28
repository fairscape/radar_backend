"""Unit tests for the Researcher Profile import service (no HTTP, no ollama)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rag_lib.api.services import rp_profile_import as rp

FIXTURE = Path(__file__).parent / "fixtures" / "sheffield_profile.jsonld"


def _doc(**overrides) -> dict:
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    d.update(overrides)
    return d


# --------------------------------------------------------------------------- parse


def test_parse_profile_reads_identity_and_signal_fields():
    p = rp.parse_profile(FIXTURE.read_text(encoding="utf-8"))
    assert p.name == "Nathan C. Sheffield"
    assert p.orcid == "0000-0001-5643-4068" and p.rid == "0000-0001-5643-4068"
    assert p.openalex_author_id == "https://openalex.org/A5055872618"
    assert p.level == "full" and p.provenance == "self_published"
    assert p.affiliation == "University of Virginia" and p.field == "Computational Biology"
    assert p.summary.startswith("Nathan Sheffield is a computational biologist")
    assert len(p.expertise) == 9 and "atac-seq-pipelines" in p.expertise
    assert p.not_interests == ["phylogenomics", "insect-evolutionary-biology",
                               "population-genetics", "wet-lab-protocol-development"]
    assert "Christoph Bock" in p.collaborators and len(p.collaborators) == 7
    assert p.paper_stats["total"] == 83
    assert p.same_as == ["https://scholar.google.com/citations?user=DwYCmoIAAAAJ"]
    assert p.warnings == []
    meta = p.meta()
    assert meta["source"] == "rp_profile" and meta["orcid"] == p.orcid and "mapped" not in meta


def test_parse_profile_orcid_from_id_iri_when_rid_is_local():
    d = _doc(rid="local:someone-ab12cd", **{"@id": "https://orcid.org/0000-0002-1825-0097"})
    p = rp.parse_profile_dict(d)
    assert p.orcid == "0000-0002-1825-0097" and p.rid == "local:someone-ab12cd"


def test_parse_profile_without_orcid_keeps_going_with_a_warning():
    d = _doc(rid="local:someone-ab12cd", **{"@id": "#me"})
    p = rp.parse_profile_dict(d)
    assert p.orcid is None
    assert any("local:" in w for w in p.warnings)


@pytest.mark.parametrize("bad", [
    "", "   ", "not json", "[1, 2]",
    json.dumps({"@type": "Person"}),                       # no name
    json.dumps({"@type": "Organization", "name": "X"}),   # not a person
])
def test_parse_profile_rejects_unusable_documents(bad):
    with pytest.raises(rp.RpProfileError):
        rp.parse_profile(bad)


def test_parse_profile_collaborators_accept_strings_and_objects():
    d = _doc(collaborators=["Alice Smith", {"name": "Bob Jones", "relationship": "coauthor"}, {"url": "x"}, 42])
    assert rp.parse_profile_dict(d).collaborators == ["Alice Smith", "Bob Jones"]


def test_parse_profile_warns_on_foreign_context():
    d = _doc(**{"@context": "https://schema.org", "conformsTo": "https://schema.org"})
    p = rp.parse_profile_dict(d)
    assert any("@context" in w for w in p.warnings)


def test_deslug():
    assert rp.deslug("atac-seq-pipelines") == "atac seq pipelines"
    assert rp.deslug("  vector_embeddings-for-genomics ") == "vector embeddings for genomics"


# --------------------------------------------------------------------------- mapping (stubbed index)


class _Index:
    def __init__(self, topics, vectors):
        self.topics = topics
        self._v = np.asarray(vectors, dtype=np.float32)

    def search(self, q, *, top_k, min_similarity):
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self._v @ q
        order = np.argsort(-sims)[:top_k]
        return [(self.topics[i], float(sims[i])) for i in order if sims[i] >= min_similarity]


def test_map_phrases_to_topics_uses_the_topic_index(monkeypatch):
    topics = [
        {"id": "https://openalex.org/T1", "display_name": "Chromatin", "field": {"display_name": "Bio"}},
        {"id": "https://openalex.org/T2", "display_name": "Insects", "field": {"display_name": "Eco"}},
    ]
    index = _Index(topics, [[1, 0], [0, 1]])
    seen = []

    def fake_embedder(text):
        seen.append(text)
        return [1.0, 0.1] if "chromatin" in text else [0.1, 1.0]

    monkeypatch.setattr("rag_lib.embedders.get_embedder", lambda name: fake_embedder)
    monkeypatch.setattr("rag_lib.umls.cache.get_topic_index", lambda d: index)

    class S:
        RADAR_UMLS_EMBEDDING_MODEL = "stub"
        RADAR_UMLS_CACHE_DIR = "/nowhere"

    hits, warnings = rp.map_phrases_to_topics(
        ["chromatin-accessibility-analysis", "insect-evolutionary-biology"], settings=S(), min_sim=0.5, top_k=2,
    )
    assert warnings == []
    assert seen == ["chromatin accessibility analysis", "insect evolutionary biology"]
    assert [(h.phrase, h.topic_id) for h in hits] == [
        ("chromatin-accessibility-analysis", "https://openalex.org/T1"),
        ("insect-evolutionary-biology", "https://openalex.org/T2"),
    ]
    assert hits[0].field == "Bio" and hits[0].similarity > 0.9


def test_map_phrases_to_topics_degrades_when_the_index_is_missing(monkeypatch):
    def boom(d):
        raise FileNotFoundError("no index")
    monkeypatch.setattr("rag_lib.umls.cache.get_topic_index", boom)
    monkeypatch.setattr("rag_lib.embedders.get_embedder", lambda name: (lambda t: [1.0]))

    class S:
        RADAR_UMLS_EMBEDDING_MODEL = "stub"
        RADAR_UMLS_CACHE_DIR = "/nowhere"

    hits, warnings = rp.map_phrases_to_topics(["x"], settings=S(), min_sim=0.5, top_k=2)
    assert hits == [] and warnings and "FileNotFoundError" in warnings[0]
    assert rp.map_phrases_to_topics([], settings=S(), min_sim=0.5, top_k=2) == ([], [])


# --------------------------------------------------------------------------- apply_rp_signals


def _hit(phrase, tid, name, sim=0.8):
    return rp.PhraseHit(phrase=phrase, topic_id=f"https://openalex.org/{tid}", display_name=name, similarity=sim)


BASE = {
    "topics": [
        {"id": "https://openalex.org/T1", "display_name": "Genomics and Chromatin Dynamics", "count": 12, "on": True, "seed_papers": 3, "source": "orcid"},
        {"id": "https://openalex.org/T2", "display_name": "Genomics and Phylogenetic Studies", "count": 9, "on": True, "seed_papers": 1, "source": "orcid"},
        {"id": "https://openalex.org/T3", "display_name": "Gene expression and cancer classification", "count": 2, "on": False, "seed_papers": 0, "source": "orcid"},
    ],
    "subfields": [], "fields": [], "domains": [],
}


def test_not_interest_switches_a_seeded_concept_off():
    out, summary = rp.apply_rp_signals(BASE, expertise_hits=[], not_interest_hits=[_hit("phylogenomics", "T2", "Genomics and Phylogenetic Studies")])
    t2 = next(t for t in out["topics"] if t["id"].endswith("T2"))
    assert t2["on"] is False and t2["rp_off_by"] == ["phylogenomics"]
    assert [x["id"] for x in summary["switched_off"]] == ["https://openalex.org/T2"]
    assert BASE["topics"][1]["on"] is True  # input untouched


def test_expertise_switches_an_unseeded_concept_on_and_adds_a_new_one():
    out, summary = rp.apply_rp_signals(BASE, expertise_hits=[
        _hit("dna-methylation-analysis", "T3", "Gene expression and cancer classification", 0.7),
        _hit("vector-embeddings-for-genomics", "T9", "Vector Embeddings for Biology", 0.66),
        _hit("chromatin-accessibility-analysis", "T1", "Genomics and Chromatin Dynamics", 0.9),
    ], not_interest_hits=[])
    by = {t["id"].rsplit("/", 1)[-1]: t for t in out["topics"]}
    assert by["T3"]["on"] is True and by["T3"]["rp_on_by"] == ["dna-methylation-analysis"]
    assert by["T1"]["on"] is True and by["T1"]["rp_on_by"] == ["chromatin-accessibility-analysis"]
    assert by["T9"]["source"] == "rp_expertise" and by["T9"]["count"] == 0 and by["T9"]["on"] is True
    assert [x["id"] for x in summary["switched_on"]] == ["https://openalex.org/T3"]  # T1 was already on
    assert [x["id"] for x in summary["added"]] == ["https://openalex.org/T9"]
    assert out["rp_signals"]["expertise_hits"][0]["phrase"] == "dna-methylation-analysis"


def test_conflict_goes_to_the_higher_cosine_and_additions_are_capped():
    ex = [_hit("p", f"T{i}", f"New {i}", 0.9 - i * 0.01) for i in range(10, 16)]  # six hits for one phrase
    ex.append(_hit("chromatin-accessibility-analysis", "T2", "Genomics and Phylogenetic Studies", 0.70))
    ni = [_hit("phylogenomics", "T2", "Genomics and Phylogenetic Studies", 0.86)]
    out, summary = rp.apply_rp_signals(BASE, expertise_hits=ex, not_interest_hits=ni, max_added_per_phrase=3)
    t2 = next(t for t in out["topics"] if t["id"].endswith("T2"))
    assert t2["on"] is False and "rp_on_by" not in t2          # 0.86 not_interest beats 0.70 expertise
    assert len(summary["added"]) == 3 and [x["id"].rsplit("/", 1)[-1] for x in summary["added"]] == ["T10", "T11", "T12"]

    # The other way round: a stronger expertise hit keeps the concept on and reports the overrule.
    ni2 = [_hit("phylogenomics", "T1", "Genomics and Chromatin Dynamics", 0.758)]
    ex2 = [_hit("chromatin-accessibility-analysis", "T1", "Genomics and Chromatin Dynamics", 0.778)]
    out2, summary2 = rp.apply_rp_signals(BASE, expertise_hits=ex2, not_interest_hits=ni2)
    t1 = next(t for t in out2["topics"] if t["id"].endswith("T1"))
    assert t1["on"] is True and summary2["switched_off"] == []
    assert summary2["overruled"][0]["kept_by_expertise"] == 0.778


def test_additions_respect_the_add_floor_but_switch_on_does_not():
    ex = [_hit("dna-methylation-analysis", "T3", "Gene expression and cancer classification", 0.62),
          _hit("reference-genome-management", "T7", "Genome Rearrangement Algorithms", 0.753),
          _hit("dna-methylation-analysis", "T8", "Epigenetics and DNA Methylation", 0.786)]
    out, summary = rp.apply_rp_signals(BASE, expertise_hits=ex, not_interest_hits=[], add_min_sim=0.78)
    ids = {t["id"].rsplit("/", 1)[-1] for t in out["topics"]}
    assert "T8" in ids and "T7" not in ids                 # only the >= 0.78 hit is added
    assert next(t for t in out["topics"] if t["id"].endswith("T3"))["on"] is True  # existing: no add floor


def test_apply_rp_signals_tolerates_an_empty_filter_dict():
    out, summary = rp.apply_rp_signals({}, expertise_hits=[_hit("x", "T5", "Five")], not_interest_hits=[])
    assert [t["id"] for t in out["topics"]] == ["https://openalex.org/T5"] and len(summary["added"]) == 1
