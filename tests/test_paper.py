"""Paper / Topic / TopicNode unit tests.

Coverage priority: JSON round-trip (no data loss) and the OpenAlex topic
hierarchy. local_path and embeddings dict get their own focused tests
because both are load-bearing for Profile JSON portability.
"""

from __future__ import annotations

from rag_lib.paper import Paper, Topic, TopicNode


def test_topicnode_round_trip():
    n = TopicNode(id="https://openalex.org/fields/27", display_name="Medicine")
    assert TopicNode.from_dict(n.to_dict()) == n


def test_topicnode_from_none():
    assert TopicNode.from_dict(None) is None


def test_topic_full_hierarchy_round_trip():
    t = Topic(
        id="T10123",
        display_name="Heart Rate Variability and Autonomic Control",
        subfield=TopicNode(id="SF1", display_name="Pediatrics"),
        field=TopicNode(id="F1", display_name="Medicine"),
        domain=TopicNode(id="D1", display_name="Health Sciences"),
        score=0.87,
    )
    d = t.to_dict()
    t2 = Topic.from_dict(d)
    assert t2 == t


def test_topic_partial_hierarchy_round_trip():
    t = Topic(id="T1", display_name="x")  # all hierarchy fields None
    t2 = Topic.from_dict(t.to_dict())
    assert t2 == t
    assert t2.subfield is None and t2.field is None and t2.domain is None


def test_paper_minimal_round_trip():
    p = Paper(doi="10.1/a", openalex_id=None, title="A", abstract="b")
    p2 = Paper.from_dict(p.to_dict())
    assert p2 == p


def test_paper_full_round_trip_preserves_local_path_and_embeddings():
    t = Topic(
        id="T1", display_name="Topic 1",
        subfield=TopicNode(id="SF1", display_name="Subfield 1"),
        field=TopicNode(id="F1", display_name="Field 1"),
        domain=TopicNode(id="D1", display_name="Domain 1"),
        score=0.9,
    )
    p = Paper(
        doi="10.1234/abcd",
        openalex_id="https://openalex.org/W123",
        title="A Paper",
        abstract="An abstract.",
        year=2024,
        venue="Nature",
        mesh=["Humans", "Infant"],
        keywords=["sepsis", "hrv"],
        substances=["Caffeine"],
        embeddings={
            "placeholder-v1": [0.1, 0.2, 0.3],
            "specter2": [0.5, 0.5, 0.5],
        },
        primary_topic=t,
        topics=[t],
        local_path="/tmp/pdfs/paper.pdf",
        source="openalex",
        added="2026-04-24T10:00:00Z",
    )
    p2 = Paper.from_dict(p.to_dict())
    assert p2 == p
    assert p2.local_path == "/tmp/pdfs/paper.pdf"
    assert p2.embeddings["specter2"] == [0.5, 0.5, 0.5]
    assert p2.embedding_for("placeholder-v1") == [0.1, 0.2, 0.3]
    assert p2.embedding_for("missing-model") is None


def test_paper_default_fields_are_empty_not_none():
    p = Paper(doi=None, openalex_id=None, title="t")
    assert p.abstract == ""
    assert p.mesh == [] and p.keywords == [] and p.substances == []
    assert p.embeddings == {}
    assert p.topics == []
    assert p.primary_topic is None
    assert p.local_path is None
    assert p.source == "unknown"


def test_paper_to_dict_is_json_serializable():
    import json
    p = Paper(doi="10.1/a", openalex_id=None, title="A",
              embeddings={"m1": [0.1, 0.2]})
    json.dumps(p.to_dict())  # no TypeError
