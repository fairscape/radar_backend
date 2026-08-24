"""extract_umls_concepts — the filtering, dedup and ordering it applies.

The module had no tests. Loading en_core_sci_lg and the UMLS knowledge
base costs about eighty seconds and a gigabyte, which is not something a
unit suite can carry, so ``_get_nlp`` is replaced by a stub. What is
under test here is everything the function does *around* the linker:
which entities it keeps, how it resolves several semantic types on one
concept, how it handles the same CUI appearing twice, and what order the
result comes back in. The linker's own accuracy is a separate question,
answered by ``probe_umls_consistency.py`` against real papers.
"""

from __future__ import annotations

import pytest

from rag_lib.umls import extractor
from rag_lib.umls.extractor import UmlsConcept, extract_umls_concepts


class _KbEntry:
    def __init__(self, canonical_name: str, types: list[str]):
        self.canonical_name = canonical_name
        self.types = types


class _Underscore:
    def __init__(self, kb_ents):
        self.kb_ents = kb_ents


class _Ent:
    """One recognised mention. ``kb_ents`` is scispacy's (cui, score) list."""

    def __init__(self, kb_ents):
        self._ = _Underscore(kb_ents)


class _Doc:
    def __init__(self, ents):
        self.ents = ents


class _Linker:
    def __init__(self, kb):
        self.kb = type("KB", (), {"cui_to_entity": kb})()


class _Nlp:
    def __init__(self, ents, kb):
        self._ents = ents
        self._linker = _Linker(kb)

    def get_pipe(self, name):
        assert name == "scispacy_linker"
        return self._linker

    def __call__(self, text):
        return _Doc(self._ents)


@pytest.fixture
def linked(monkeypatch):
    """Return a setter: ``linked(mentions, kb)`` installs a stub pipeline.

    ``mentions`` is a list of (cui, score) lists — one per recognised
    entity, ordered as scispacy orders candidates. ``kb`` maps a CUI to
    (canonical name, semantic type ids).
    """
    def install(mentions, kb):
        entries = {
            cui: _KbEntry(name, types) for cui, (name, types) in kb.items()
        }
        nlp = _Nlp([_Ent(m) for m in mentions], entries)
        monkeypatch.setattr(extractor, "_get_nlp", lambda *a, **k: nlp)
    return install


# T047 Disease or Syndrome and T121 Pharmacologic Substance are relevant;
# T062 Research Activity is excluded as uninformative about a subject.
DISEASE, DRUG, RESEARCH = "T047", "T121", "T062"


def test_a_linked_disease_comes_back(linked):
    linked(
        [[("C0011860", 0.98)]],
        {"C0011860": ("Diabetes Mellitus, Non-Insulin-Dependent", [DISEASE])},
    )
    out = extract_umls_concepts("anything")
    assert out == [UmlsConcept(
        cui="C0011860",
        name="Diabetes Mellitus, Non-Insulin-Dependent",
        tui=DISEASE,
        semantic_type="Disease or Syndrome",
        confidence=0.98,
    )]


def test_only_the_linker_s_first_candidate_is_considered(linked):
    """The pipeline is configured with max_entities_per_mention=1, so a
    mention resolves to one concept and the runners-up are not scanned
    for a better semantic type."""
    linked(
        [[("C_RESEARCH", 0.99), ("C0011860", 0.98)]],
        {
            "C_RESEARCH": ("Clinical Research", [RESEARCH]),
            "C0011860": ("Diabetes Mellitus", [DISEASE]),
        },
    )
    assert extract_umls_concepts("anything") == []


def test_a_mention_below_the_confidence_floor_is_dropped(linked):
    linked([[("C1", 0.69)]], {"C1": ("Borderline", [DISEASE])})
    assert extract_umls_concepts("anything", min_confidence=0.7) == []
    assert len(extract_umls_concepts("anything", min_confidence=0.6)) == 1


def test_a_cui_missing_from_the_knowledge_base_is_skipped(linked):
    linked([[("C_UNKNOWN", 0.99)]], {})
    assert extract_umls_concepts("anything") == []


def test_a_concept_with_no_relevant_semantic_type_is_dropped(linked):
    """"Clinical Research" is a real, high-confidence link. It is also
    true of every paper, so it says nothing about this one."""
    linked([[("C1", 0.99)]], {"C1": ("Clinical Research", [RESEARCH])})
    assert extract_umls_concepts("anything") == []


def test_the_first_relevant_type_wins_when_a_concept_has_several(linked):
    """Concepts carry a list of semantic types. The first *relevant* one
    is reported, so an irrelevant type listed ahead of a relevant one
    does not disqualify the concept."""
    linked([[("C1", 0.99)]], {"C1": ("Metformin", [RESEARCH, DRUG])})
    out = extract_umls_concepts("anything")
    assert [(c.tui, c.semantic_type) for c in out] == [
        (DRUG, "Pharmacologic Substance"),
    ]


def test_the_same_cui_twice_keeps_the_higher_score(linked):
    """A concept mentioned in several places links once per mention."""
    linked(
        [[("C1", 0.80)], [("C1", 0.95)], [("C1", 0.70)]],
        {"C1": ("Insulin", [DRUG])},
    )
    out = extract_umls_concepts("anything")
    assert len(out) == 1
    assert out[0].confidence == 0.95


def test_results_are_ordered_by_confidence(linked):
    linked(
        [[("C1", 0.80)], [("C2", 0.99)], [("C3", 0.90)]],
        {c: (c, [DISEASE]) for c in ("C1", "C2", "C3")},
    )
    out = extract_umls_concepts("anything")
    assert [c.cui for c in out] == ["C2", "C3", "C1"]


def test_max_concepts_keeps_the_most_confident(linked):
    linked(
        [[(f"C{i}", 0.70 + i / 100)] for i in range(10)],
        {f"C{i}": (f"C{i}", [DISEASE]) for i in range(10)},
    )
    out = extract_umls_concepts("anything", max_concepts=3)
    assert [c.cui for c in out] == ["C9", "C8", "C7"]


def test_confidence_is_rounded_for_stable_storage(linked):
    """Concepts are serialised to JSON on the paper row; four decimals
    keeps a re-extraction comparable to what was stored."""
    linked([[("C1", 0.9876543)]], {"C1": ("X", [DISEASE])})
    assert extract_umls_concepts("anything")[0].confidence == 0.9877


def test_a_mention_the_linker_could_not_resolve_is_skipped(linked):
    linked([[], [("C1", 0.99)]], {"C1": ("Insulin", [DRUG])})
    assert [c.cui for c in extract_umls_concepts("anything")] == ["C1"]


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_blank_input_returns_nothing_without_loading_the_model(text, monkeypatch):
    """Guarded before ``_get_nlp``, so an empty abstract does not pull a
    gigabyte of knowledge base into memory."""
    def _boom(*a, **k):
        raise AssertionError("the model must not be loaded for blank input")

    monkeypatch.setattr(extractor, "_get_nlp", _boom)
    assert extract_umls_concepts(text) == []
