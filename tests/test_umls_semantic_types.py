"""Semantic-type filtering — which UMLS concepts reach the topic mapper.

The module had no tests. It decides what ``extract_umls_concepts`` keeps,
so its behaviour is upstream of the profile's topic filters and therefore
of what the daily gather queries OpenAlex for.
"""

from __future__ import annotations

import pytest

from rag_lib.umls.semantic_types import (
    _UNINFORMATIVE_TUIS,
    all_types,
    get_type_group,
    get_type_name,
    is_relevant,
    relevant_tuis,
    relevant_types,
)


def test_the_data_file_carries_the_full_nlm_registry():
    assert len(all_types()) == 127


@pytest.mark.parametrize("tui,name", [
    ("T047", "Disease or Syndrome"),
    ("T121", "Pharmacologic Substance"),
    ("T023", "Body Part, Organ, or Organ Component"),
])
def test_subject_matter_types_are_relevant(tui, name):
    assert is_relevant(tui)
    assert get_type_name(tui) == name


@pytest.mark.parametrize("tui,name", [
    ("T062", "Research Activity"),
    ("T041", "Mental Process"),
    ("T070", "Natural Phenomenon or Process"),
])
def test_uninformative_types_are_excluded(tui, name):
    """These describe that a paper is research, not what it is about.

    Their concepts — "Clinical Research", "research study",
    "Self-Assessment", "Rain" — map onto OpenAlex's generic topic names
    at a higher cosine than a specific concept matches a specific topic,
    so with a capped candidate list they crowd out the topics that
    actually describe the paper.
    """
    assert get_type_name(tui) == name          # still in the registry
    assert not is_relevant(tui)                # but not offered to the mapper
    assert tui not in relevant_tuis()
    assert tui not in relevant_types()


def test_excluded_types_are_a_strict_subset_of_the_published_relevant_set():
    """The data file is generated from the NLM release and left alone;
    the exclusion is applied on top of it."""
    import json
    from pathlib import Path

    import rag_lib.umls.semantic_types as st

    published = set(json.loads(Path(st._DATA_PATH).read_text())["relevant_tuis"])
    assert _UNINFORMATIVE_TUIS < published
    assert relevant_tuis() == published - _UNINFORMATIVE_TUIS


def test_unknown_tui_is_not_relevant():
    assert not is_relevant("T999")
    assert get_type_name("T999") is None
    assert get_type_group("T999") is None


def test_relevant_types_agrees_with_is_relevant():
    assert all(is_relevant(t) for t in relevant_types())
    assert relevant_types().keys() == relevant_tuis()


def test_relevant_tuis_returns_a_mutable_copy():
    """Callers get a set they can modify without corrupting the cache."""
    first = relevant_tuis()
    first.add("T999")
    assert "T999" not in relevant_tuis()
