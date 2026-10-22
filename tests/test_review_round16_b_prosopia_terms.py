"""Review 16 (b): prosopia_terms with the concept-shaped research_interests."""

from __future__ import annotations

from tests.test_researcher_topics import _topic


def _found(monkeypatch, fields):
    from rag_lib import interest_terms

    class Resp:
        status_code = 200

        def json(self):
            return {"fields": fields}

    monkeypatch.setattr(interest_terms.requests, "get", lambda url, timeout=0: Resp())
    return interest_terms.prosopia_terms("https://prosopia.example", "x")


def test_hyphenated_not_interest_still_filters_hyphenated_expertise(monkeypatch):
    # Old shape: labels() de-hyphenates both sides, so this is filtered.
    # New shape: the not_interests come from researcher_topics.read (hyphens
    # kept) while expertise goes through labels() (hyphens -> spaces).
    found = _found(monkeypatch, {
        "research_interests": [_topic("T10222", "Genomics", weight=0.5, method="declared")],
        "expertise": ["wet-lab-protocol-development", "Statistics"],
        "not_interests": ["wet-lab-protocol-development"],
    })
    labels = [t["label"].lower() for t in found["terms"]]
    assert "wet lab protocol development" not in labels, labels


def test_not_interests_in_top_shape_survive_the_new_shape(monkeypatch):
    # {"top": [...]} is a shape labels() reads (see test_interest_terms);
    # researcher_topics.read's _strings drops it.
    found = _found(monkeypatch, {
        "research_interests": [_topic("T10222", "Genomics", weight=0.5, method="declared")],
        "expertise": ["phylogenomics", "Statistics"],
        "not_interests": {"top": ["phylogenomics"]},
    })
    labels = [t["label"].lower() for t in found["terms"]]
    assert "phylogenomics" not in labels, found
    assert found["not_interests"] == ["phylogenomics"] or "phylogenomics" in found["not_interests"], found


def test_expertise_as_text_is_not_split_into_characters(monkeypatch):
    # The profile's expertise is a markdown text elsewhere in the API
    # (ApiProfile.expertise = text_part("expertise")); a string here is
    # iterated character by character.
    found = _found(monkeypatch, {
        "research_interests": [_topic("T10222", "Genomics", weight=0.5, method="declared")],
        "expertise": "ATAC-seq",
    })
    labels = [t["label"] for t in found["terms"]]
    assert all(len(l) > 1 for l in labels), labels


def test_topic_concepts_with_a_code_but_another_id_spelling_are_dropped():
    from rag_lib import researcher_topics as rt

    for cid in ("http://openalex.org/T10222", "https://openalex.org/topics/T10222",
                "https://api.openalex.org/topics/T10222", "openalex:topic/T10222"):
        view = rt.read({"research_interests": [{"method": "declared", "weight": 0.5,
                                                "concept": {"@id": cid, "code": "T10222", "display": "G"}}]})
        assert [t["id"] for t in view["wanted"]] == ["https://openalex.org/T10222"], (cid, view)
    # Some other identifier with no topic code is still skipped.
    view = rt.read({"research_interests": [{"concept": {"@id": "https://openalex.org/W1", "display": "a work"}}]})
    assert view["wanted"] == [] and view["wanted_labels"] == []


def test_read_takes_not_interests_in_top_shape():
    from rag_lib import researcher_topics as rt

    view = rt.read({"not_interests": {"top": ["phylogenomics", {"label": "Ecology"}]}})
    assert view["unwanted_labels"] == ["phylogenomics", "Ecology"]


def test_unwanted_topic_name_with_hyphens_filters_the_expertise(monkeypatch):
    # The unwanted name comes from the concept as written; expertise goes
    # through labels(), hyphens to spaces.
    found = _found(monkeypatch, {
        "research_interests": [_topic("T10222", "Genomics", weight=0.5, method="declared"),
                               _topic("T10333", "Single-cell sequencing", weight=-0.8, method="declared")],
        "expertise": ["single-cell sequencing", "Statistics"],
    })
    labels = [t["label"].lower() for t in found["terms"]]
    assert "single cell sequencing" not in labels, labels
    assert "statistics" in labels
