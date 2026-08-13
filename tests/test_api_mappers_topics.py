"""topic_filters_to_topics must report the stored flags, not assume them."""
from rag_lib.api.mappers import topic_filters_to_topics


def _tf(*entries):
    return {"topics": list(entries), "subfields": [], "fields": [], "domains": []}


def test_a_switched_off_topic_is_reported_as_off():
    """The gather already skips it; the profile page said otherwise, so a
    profile with ten of fourteen topics off read as fourteen active."""
    out = topic_filters_to_topics(_tf(
        {"id": "T1", "display_name": "On one", "count": 3, "on": True},
        {"id": "T2", "display_name": "Off one", "count": 1, "on": False},
    ))
    assert [t.on for t in out] == [True, False]


def test_absent_on_still_means_on():
    """Same rule openalex_tiers.is_enabled applies: entries written
    before the flag existed keep participating."""
    out = topic_filters_to_topics(_tf({"id": "T1", "display_name": "n", "count": 1}))
    assert out[0].on is True


def test_the_source_survives():
    out = topic_filters_to_topics(_tf(
        {"id": "T1", "display_name": "from openalex", "count": 1},
        {"id": "T2", "display_name": "from umls", "count": 1, "source": "umls"},
    ))
    assert [t.source for t in out] == [None, "umls"]


def test_it_agrees_with_the_wizard_endpoint_on_the_same_blob():
    """Two mappers built the same shape and disagreed; that divergence is
    why the profile page was wrong while step 3 looked right."""
    entries = [
        {"id": "T1", "display_name": "a", "count": 2, "on": True},
        {"id": "T2", "display_name": "b", "count": 1, "on": False, "source": "umls"},
        {"id": "T3", "display_name": "c", "count": 1},
    ]
    mine = topic_filters_to_topics(_tf(*entries))
    theirs = [
        (e.get("id"), e.get("display_name"), int(e.get("count") or 0),
         bool(e.get("on", True)), e.get("source"))
        for e in entries
    ]
    assert [(t.id, t.name, t.count, t.on, t.source) for t in mine] == theirs


def test_entries_without_an_id_are_dropped():
    out = topic_filters_to_topics(_tf({"display_name": "no id", "count": 1}))
    assert out == []


def test_empty_and_missing_filters():
    assert topic_filters_to_topics(None) == []
    assert topic_filters_to_topics({}) == []
