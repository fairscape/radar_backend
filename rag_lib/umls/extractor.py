"""SciSpacy NER + UMLS linker for biomedical concept extraction.

Lazy-loads the spaCy model + UMLS linker on first call (same pattern as
``rag_lib.embedders._load_specter2``). Subsequent calls reuse the cached
pipeline.

The ``SCISPACY_CACHE`` environment variable controls where the ~1 GB
UMLS knowledge base is stored on disk. Set it before importing this
module to avoid writing to ``~/.scispacy/``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from threading import Lock

from .semantic_types import is_relevant, get_type_name

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UmlsConcept:
    """A single extracted UMLS concept."""

    cui: str  # e.g. "C0011849"
    name: str  # preferred name, e.g. "Diabetes Mellitus, Type 2"
    tui: str  # semantic type unique identifier, e.g. "T047"
    semantic_type: str  # human-readable, e.g. "Disease or Syndrome"
    confidence: float  # linker similarity score in [0, 1]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_nlp = None
_nlp_lock = Lock()


def _get_nlp(spacy_model: str, cache_dir: str | None, threshold: float):
    """Load spaCy model + UMLS linker, cached globally."""
    global _nlp

    if _nlp is not None:
        return _nlp

    with _nlp_lock:
        if _nlp is not None:
            return _nlp

        # Set cache dir before importing scispacy
        if cache_dir:
            os.environ["SCISPACY_CACHE"] = str(cache_dir)

        import spacy
        import scispacy  # noqa: F401
        from scispacy.linking import EntityLinker  # noqa: F401 — registers 'scispacy_linker' factory

        log.info("Loading SciSpacy model: %s", spacy_model)
        nlp = spacy.load(spacy_model)

        nlp.add_pipe("scispacy_linker", config={
            "linker_name": "umls",
            "resolve_abbreviations": True,
            "threshold": threshold,
            "no_definition_threshold": 0.95,
            "filter_for_definitions": True,
            "max_entities_per_mention": 1,
            "k": 30,
        })
        log.info("SciSpacy + UMLS linker loaded")

        _nlp = nlp
        return _nlp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_umls_concepts(
    text: str,
    *,
    min_confidence: float = 0.7,
    spacy_model: str = "en_core_sci_lg",
    cache_dir: str | None = None,
    max_concepts: int = 30,
) -> list[UmlsConcept]:
    """Extract UMLS concepts from biomedical text.

    Parameters
    ----------
    text:
        Input text (abstract preferred, body_text as fallback).
    min_confidence:
        Minimum linker similarity score to keep a concept.
    spacy_model:
        SciSpacy model name (must be installed).
    cache_dir:
        Directory for UMLS KB cache. If None, uses ~/.scispacy/.
    max_concepts:
        Maximum number of concepts to return (sorted by confidence desc).

    Returns
    -------
    List of UmlsConcept, deduplicated by CUI, filtered by relevant
    semantic types and min_confidence, sorted by confidence descending.
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp(spacy_model, cache_dir, min_confidence)
    linker = nlp.get_pipe("scispacy_linker")

    doc = nlp(text)

    # Collect concepts, dedup by CUI (keep highest confidence)
    seen: dict[str, UmlsConcept] = {}

    for ent in doc.ents:
        if not ent._.kb_ents:
            continue

        # Take the top match only (max_entities_per_mention=1)
        cui, score = ent._.kb_ents[0]

        if score < min_confidence:
            continue

        # Look up full entity from KB
        kb_entry = linker.kb.cui_to_entity.get(cui)
        if kb_entry is None:
            continue

        # Filter by relevant semantic types
        tuis = kb_entry.types  # list of TUI strings
        relevant_tui = None
        for t in tuis:
            if is_relevant(t):
                relevant_tui = t
                break

        if relevant_tui is None:
            continue

        # Dedup: keep highest confidence per CUI
        if cui in seen and seen[cui].confidence >= score:
            continue

        seen[cui] = UmlsConcept(
            cui=cui,
            name=kb_entry.canonical_name,
            tui=relevant_tui,
            semantic_type=get_type_name(relevant_tui) or "",
            confidence=round(score, 4),
        )

    # Sort by confidence descending, truncate
    results = sorted(seen.values(), key=lambda c: c.confidence, reverse=True)
    return results[:max_concepts]
