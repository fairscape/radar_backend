"""Profile unit tests — JSON round-trip, file I/O, seed embedding access,
topic-filter aggregation."""

from __future__ import annotations

import json

import pytest

from rag_lib.paper import Paper, Topic, TopicNode
from rag_lib.profile import Profile


def _sample_profile() -> Profile:
    return Profile(
        name="neonatal_vitals",
        papers=[
            Paper(doi="10.1/a", openalex_id="W1", title="A", abstract="alpha",
                  embeddings={"placeholder-v1": [0.1, 0.2, 0.3]},
                  local_path="/tmp/a.pdf"),
            Paper(doi="10.1/b", openalex_id="W2", title="B", abstract="beta",
                  embeddings={"placeholder-v1": [0.4, 0.5, 0.6]},
                  local_path="/tmp/b.pdf"),
        ],
        topic_filters={
            "topics": [{"id": "T10123", "display_name": "HRV", "count": 2}],
            "subfields": [], "fields": [], "domains": [],
        },
        embedding_model="placeholder-v1",
        selector_config={"type": "centroid", "threshold": 0.82},
        gatherer_config={"type": "openalex"},
        threshold=0.82,
        created="2026-04-24T10:00:00Z",
    )


def test_profile_dict_round_trip():
    p = _sample_profile()
    p2 = Profile.from_dict(p.to_dict())
    assert p2 == p


def test_profile_json_round_trip(tmp_path):
    p = _sample_profile()
    path = tmp_path / "profile.json"
    p.to_json(path)
    p2 = Profile.from_json(path)
    assert p2 == p
    # Sanity check: JSON is valid and contains expected keys.
    data = json.loads(path.read_text())
    assert data["name"] == "neonatal_vitals"
    assert len(data["papers"]) == 2
    assert data["papers"][0]["local_path"] == "/tmp/a.pdf"
    assert data["papers"][0]["embeddings"]["placeholder-v1"] == [0.1, 0.2, 0.3]


def test_profile_json_has_no_pdf_bytes(tmp_path):
    """Profile JSON must be portable without the source PDFs. local_path
    strings are fine; raw bytes are not."""
    p = _sample_profile()
    path = tmp_path / "profile.json"
    p.to_json(path)
    # local_path is a string, not bytes
    data = json.loads(path.read_text())
    for paper in data["papers"]:
        assert isinstance(paper["local_path"], str)


def test_seed_embeddings_returns_stored_vectors():
    p = _sample_profile()
    vecs = p.seed_embeddings()
    assert len(vecs) == 2
    assert vecs[0] == [0.1, 0.2, 0.3]


def test_seed_embeddings_explicit_model_override():
    p = _sample_profile()
    vecs = p.seed_embeddings(model="placeholder-v1")
    assert len(vecs) == 2


def test_seed_embeddings_raises_when_model_absent():
    p = _sample_profile()
    with pytest.raises(ValueError, match="specter2"):
        p.seed_embeddings(model="specter2")


def test_empty_profile_round_trips(tmp_path):
    p = Profile(name="empty")
    path = tmp_path / "empty.json"
    p.to_json(path)
    p2 = Profile.from_json(path)
    assert p2 == p


# ----------------------------------------------------------------------
# Profile.aggregate_topic_filters (staticmethod)
# ----------------------------------------------------------------------


def _paper_with_topic(doi: str, topic_id: str, subfield_id: str,
                      field_id: str, domain_id: str) -> Paper:
    t = Topic(
        id=topic_id, display_name=f"T:{topic_id}",
        subfield=TopicNode(id=subfield_id, display_name=f"SF:{subfield_id}"),
        field=TopicNode(id=field_id, display_name=f"F:{field_id}"),
        domain=TopicNode(id=domain_id, display_name=f"D:{domain_id}"),
        score=0.9,
    )
    return Paper(doi=doi, openalex_id=None, title=doi,
                 primary_topic=t, topics=[t])


def test_aggregate_topic_filters_counts_at_every_level():
    papers = [
        _paper_with_topic("a", "T1", "SF1", "F1", "D1"),
        _paper_with_topic("b", "T1", "SF1", "F1", "D1"),
        _paper_with_topic("c", "T2", "SF2", "F1", "D1"),
    ]
    f = Profile.aggregate_topic_filters(papers)
    # Each paper contributes primary + topics[] (same here), so T1 counts 4.
    topic_ids = {t["id"] for t in f["topics"]}
    assert topic_ids == {"T1", "T2"}
    field_ids = {fl["id"] for fl in f["fields"]}
    assert field_ids == {"F1"}
    domain_ids = {d["id"] for d in f["domains"]}
    assert domain_ids == {"D1"}
    # Counts preserved.
    t1 = next(t for t in f["topics"] if t["id"] == "T1")
    assert t1["count"] == 4
    # Display name plumbed through.
    assert t1["display_name"] == "T:T1"


def test_aggregate_topic_filters_ignores_topicless_papers():
    papers = [Paper(doi="x", openalex_id=None, title="x")]
    f = Profile.aggregate_topic_filters(papers)
    assert f == {"topics": [], "subfields": [], "fields": [], "domains": []}


def test_aggregate_topic_filters_respects_top_k():
    papers = [_paper_with_topic(f"p{i}", f"T{i}", f"SF{i}", f"F{i}", f"D{i}")
              for i in range(20)]
    f = Profile.aggregate_topic_filters(papers, top_k_each=3)
    assert len(f["topics"]) == 3
    assert len(f["subfields"]) == 3
