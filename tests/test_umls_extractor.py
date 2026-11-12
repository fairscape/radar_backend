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


# ---------------------------------------------------------------------------
# Mentions: the words a concept was read from, in their sentence (CTX-1)
# ---------------------------------------------------------------------------
#
# On a real spaCy Doc (blank English with a sentencizer), so the sentence
# boundaries are spaCy's own; only the linker is stubbed.


@pytest.fixture
def real_doc(monkeypatch):
    """Return ``install(text, mentions, kb, abbreviations=())``: ``mentions``
    is a list of (the mention's text as in ``text``, its (cui, score) list);
    ``abbreviations`` the mention texts the text defines as short forms."""
    spacy = pytest.importorskip("spacy")
    from spacy.tokens import Doc, Span

    if not Span.has_extension("kb_ents"):
        Span.set_extension("kb_ents", default=[])
    if not Doc.has_extension("abbreviations"):
        Doc.set_extension("abbreviations", default=[])
    blank = spacy.blank("en")
    blank.add_pipe("sentencizer")

    def install(text, mentions, kb, abbreviations=()):
        doc = blank(text)
        ents, at = [], 0
        for words, _ in mentions:
            start = text.index(words, at)
            at = start + len(words)
            span = doc.char_span(start, at, label="ENTITY")
            assert span is not None, words
            ents.append(span)
        doc.ents = ents
        # Set on doc.ents' own spans: a span's extension value is keyed by
        # its label among others, as the linker sets it.
        for ent, (_, cands) in zip(doc.ents, mentions):
            ent._.kb_ents = cands
        doc._.abbreviations = [e for e in doc.ents if e.text in abbreviations]
        entries = {cui: _KbEntry(name, types) for cui, (name, types) in kb.items()}
        linker = _Linker(entries)
        nlp = type("Nlp", (), {"get_pipe": lambda self, n: linker, "__call__": lambda self, t: doc})()
        monkeypatch.setattr(extractor, "_get_nlp", lambda *a, **k: nlp)
    return install


LONG = " and the cohort was followed for twelve years in four hospitals across the region."


def test_a_concept_carries_its_word_and_sentence(real_doc):
    text = "Insulin resistance rose in obese adults" + LONG + " Nothing else."
    real_doc(text, [("Insulin resistance", [("C1", 0.97)])], {"C1": ("Insulin Resistance", [DISEASE])})
    [c] = extract_umls_concepts(text)
    assert c.mentions == (("Insulin resistance", "Insulin resistance rose in obese adults" + LONG),)
    assert c.to_dict()["mentions"] == [{"text": "Insulin resistance", "sentence": c.mentions[0][1]}]


def test_a_short_sentence_takes_the_next_one_with_it(real_doc):
    text = "Ticks bite. They carry Lyme disease" + LONG + " A third sentence."
    real_doc(text, [("Ticks", [("C1", 0.95)])], {"C1": ("Ixodida", [DISEASE])})
    [c] = extract_umls_concepts(text)
    assert c.mentions == (("Ticks", "Ticks bite. They carry Lyme disease" + LONG),)


def test_a_short_last_sentence_stays_alone(real_doc):
    text = "They carry Lyme disease" + LONG + " Ticks bite."
    real_doc(text, [("Ticks", [("C1", 0.95)])], {"C1": ("Ixodida", [DISEASE])})
    assert extract_umls_concepts(text)[0].mentions == (("Ticks", "Ticks bite."),)


def test_a_sentence_is_cut_at_its_limit(real_doc):
    text = "Malaria spreads " + "widely " * 400 + "today."
    real_doc(text, [("Malaria", [("C1", 0.95)])], {"C1": ("Malaria", [DISEASE])})
    [(_, sentence)] = extract_umls_concepts(text)[0].mentions
    assert len(sentence) == extractor.SENTENCE_MAX_CHARS and sentence.startswith("Malaria spreads widely")


def test_the_surest_three_mentions_each_sentence_once(real_doc):
    s = [f"Sentence {i} says diabetes again{LONG}" for i in range(5)]
    text = " ".join(s[:2]) + " Diabetes and diabetes twice in this one" + LONG + " " + " ".join(s[2:])
    mentions = [("diabetes", [("C1", 0.80)]), ("diabetes", [("C1", 0.85)]),
                ("Diabetes", [("C1", 0.99)]), ("diabetes", [("C1", 0.98)]),
                ("diabetes", [("C1", 0.90)]), ("diabetes", [("C1", 0.95)]), ("diabetes", [("C1", 0.70)])]
    real_doc(text, mentions, {"C1": ("Diabetes Mellitus", [DISEASE])})
    [c] = extract_umls_concepts(text)
    assert c.confidence == 0.99
    # 0.99 and 0.98 are one sentence: the next surest takes the second place
    assert [(w, sent.split(" ")[1]) for w, sent in c.mentions] == [("Diabetes", "and"), ("diabetes", "3"),
                                                                   ("diabetes", "2")]


def test_a_short_word_is_no_mention_unless_the_text_defines_it(real_doc):
    text = ("Amyotrophic lateral sclerosis (ALS) progressed" + LONG +
            " The D term in the model of Asp" + LONG)
    kb = {"C1": ("Amyotrophic Lateral Sclerosis", [DISEASE]), "C2": ("Aspartic Acid", [DRUG]),
          "C3": ("Asparagine", [DRUG])}
    real_doc(text, [("ALS", [("C1", 0.93)]), ("D", [("C2", 1.0)]), ("Asp", [("C3", 0.96)])], kb,
             abbreviations=("ALS",))
    out = {c.cui: c for c in extract_umls_concepts(text)}
    # the concepts and their confidence are as before; only the mentions differ
    assert {cui: c.confidence for cui, c in out.items()} == {"C1": 0.93, "C2": 1.0, "C3": 0.96}
    # three letters, but the text defines it
    assert [w for w, _ in out["C1"].mentions] == ["ALS"]
    assert out["C2"].mentions == () and out["C3"].mentions == ()


def test_a_mention_in_a_formula_is_no_mention(real_doc):
    text = "y = 3.2 x + 0.4 glucose , p < 0.05 , n = 12 , r2 = 0.91 . Glucose uptake was measured" + LONG
    real_doc(text, [("glucose", [("C1", 0.97)]), ("Glucose", [("C1", 0.92)])], {"C1": ("Glucose", [DRUG])})
    [c] = extract_umls_concepts(text)
    assert c.confidence == 0.97
    assert [s.split()[0] for _, s in c.mentions] == ["Glucose"]


@pytest.mark.parametrize("sentence,formula", [
    ("The cohort was followed for years.", False),
    ("x = 3 + y ( 2 )", True),
    ("Glucose 3.2 4.1 5.0 6.6", True),
    ("Glucose rose 3.2 to 4.1 mmol.", False),
    ("", True),
])
def test_formula_like(sentence, formula):
    assert extractor._formula_like(sentence) is formula


def test_the_spec_numbers():
    """FUNCTIONAL_SPEC CTX-1: three mentions, the next sentence under 120
    characters, at most 1000 (review CTX-R5: changing them went unseen)."""
    assert (extractor.MAX_MENTIONS, extractor._SHORT_SENTENCE, extractor.SENTENCE_MAX_CHARS) == (3, 120, 1000)


def test_a_sentence_just_under_the_short_limit_takes_the_next(real_doc):
    first = "Ticks transmit several bacterial pathogens to humans and to dogs in rural parts of the state."
    assert 60 < len(first) < 120
    text = first + " They carry Lyme disease" + LONG
    real_doc(text, [("Ticks", [("C1", 0.95)])], {"C1": ("Ixodida", [DISEASE])})
    assert extract_umls_concepts(text)[0].mentions[0][1] == text


def test_a_long_sentence_is_cut_around_its_word(real_doc):
    """Review CTX-R2: a PDF paragraph with no full stop; the word well past
    the first 1000 characters."""
    text = "Cohorts were " + "followed and " * 150 + "Borrelia was found in " + "many and " * 150 + "more."
    real_doc(text, [("Borrelia", [("C1", 0.95)])], {"C1": ("Borrelia", [DISEASE])})
    [(word, sentence)] = extract_umls_concepts(text)[0].mentions
    assert len(sentence) == 1000 and "Borrelia" in sentence
    at = sentence.index("Borrelia")
    assert 400 < at < 600                       # centred on it
