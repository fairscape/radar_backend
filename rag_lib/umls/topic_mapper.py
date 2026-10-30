"""UMLS concept → OpenAlex topic mapping via embedding similarity.

Embeds each UMLS concept's canonical name with mxbai-embed-large,
computes cosine similarity against the precomputed OpenAlex topic
embedding index, and returns the top matches.

``merge_umls_topics()`` takes the per-paper mapping results and merges
new topics into an existing ``topic_filters`` dict (the format returned
by ``Profile.aggregate_topic_filters()``).
"""

from __future__ import annotations

import copy
import logging
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .cache import get_topic_index
from .extractor import UmlsConcept

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MappedTopic:
    """The OpenAlex topic nearest to one UMLS concept of a paper."""

    topic_id: str  # e.g. "https://openalex.org/T12345"
    display_name: str
    similarity: float  # concept name to topic name; shown, not used to choose
    source_cui: str  # which UMLS concept triggered this match
    source_name: str  # UMLS concept name for traceability
    # The concept's confidence (how surely the text's words are this
    # concept): what the topics are chosen by (merge_umls_topics). None
    # in a mapping stored before it was recorded (stale_mapping).
    confidence: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# How the UMLS topics of an interest are chosen (FUNCTIONAL_SPEC U.2):
# from each seed, its concepts of CONFIDENCE or more, each with its one
# nearest topic, PER_PAPER of them by confidence; at most MAX_TOPICS for
# the interest, by confidence. They are added switched off.
CONFIDENCE = 0.9
PER_PAPER = 5
MAX_TOPICS = 20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_concepts_to_topics(
    concepts: list[UmlsConcept],
    *,
    embedding_model: str = "mxbai-embed-large",
    cache_dir: str | None = None,
) -> list[MappedTopic]:
    """Each concept's nearest OpenAlex topic, by embedding its name.

    One entry per concept, in the order given, however near or far the
    topic is: which of them an interest gets is decided on the concepts'
    confidence (merge_umls_topics), not on this similarity.
    """
    if not concepts or not cache_dir:
        return []

    from ..embedders import get_embedder

    embedder = get_embedder(embedding_model)
    index = get_topic_index(cache_dir)
    index.check_compatible(embedding_model)

    out: list[MappedTopic] = []
    for concept in concepts:
        vec = np.array(embedder(concept.name), dtype=np.float32)
        # An index without META_FILE can still be caught on dimension
        # (a bare matmul error said nothing about why).
        index.check_compatible(embedding_model, dim=int(vec.shape[-1]))
        for topic_dict, sim in index.search(vec, top_k=1, min_similarity=-1.0):
            out.append(MappedTopic(
                topic_id=topic_dict["id"],
                display_name=topic_dict["display_name"],
                similarity=round(sim, 4),
                source_cui=concept.cui,
                source_name=concept.name,
                confidence=round(float(concept.confidence), 4),
            ))
    return out


def stale_mapping(entries: list[dict]) -> bool:
    """A stored ``umls_mapped_topics_json`` made by the earlier mapping
    (three topics a concept, no confidence): it is mapped again."""
    return any("confidence" not in e for e in entries)


def _name_similarity(a: str, b: str) -> float:
    """Word-overlap Jaccard similarity between two topic names.

    Fallback for when the topic embedding index isn't available. Weak:
    it scores "Renal Diseases and Glomerulopathies" vs "Renal and
    Vascular Pathologies" at 0.33 (only "renal" and the stopword "and"
    overlap) even though they're near-synonyms. Prefer
    :func:`_embedding_similarity`.
    """
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


_DEDUP_NAME_THRESHOLD = 0.5  # Jaccard >= 0.5 -> near-duplicate (fallback path)

# Cosine >= this on the OpenAlex topic embedding index -> near-duplicate.
# Calibrated against the live index (mxbai-embed-large, 4516 topics):
#   near-synonyms  0.806 - 0.854   e.g. "Renal Diseases and
#                                  Glomerulopathies" vs "Renal and
#                                  Vascular Pathologies" = 0.854
#   unrelated      0.512 - 0.687   e.g. "Diabetes Management and
#                                  Research" vs "Proteins in Food
#                                  Systems" = 0.548
# 0.75 sits in the gap with margin on both sides.
_DEDUP_COSINE_THRESHOLD = 0.75


def _embedding_similarity(index: Any, a_id: str, b_id: str) -> float | None:
    """Cosine between two OpenAlex topics, or None if either is unknown."""
    if index is None:
        return None
    va = index.vector_for(a_id)
    vb = index.vector_for(b_id)
    if va is None or vb is None:
        return None
    return float(va @ vb)  # both rows are unit-length


def _is_near_duplicate(
    index: Any, a_id: str, a_name: str, b_id: str, b_name: str
) -> bool:
    """Near-duplicate test: embedding cosine, falling back to word overlap."""
    sim = _embedding_similarity(index, a_id, b_id)
    if sim is not None:
        return sim >= _DEDUP_COSINE_THRESHOLD
    return _name_similarity(a_name, b_name) >= _DEDUP_NAME_THRESHOLD


def merge_umls_topics(
    base_filters: dict[str, Any],
    umls_topics_per_paper: list[list[MappedTopic]],
    *,
    min_confidence: float = CONFIDENCE,
    per_paper: int = PER_PAPER,
    max_additions: int = MAX_TOPICS,
    cache_dir: str | None = None,
    also_have: list[dict] | None = None,
) -> dict[str, Any]:
    """``base_filters`` with the UMLS topics of its seeds appended, switched
    off (FUNCTIONAL_SPEC U.2).

    Chosen on the concepts' confidence alone. In each seed, the concepts
    of ``min_confidence`` or more, by confidence: one whose topic the
    interest already has (same id or name, or an embedding cosine of
    _DEDUP_COSINE_THRESHOLD or more to one of ``base_filters``' topics)
    is skipped, and so is one whose topic a concept of the same seed
    already gave; ``per_paper`` are taken. The seeds' topics together:
    each once (same id, or the same name under another id), with its best
    confidence; ``max_additions`` of them, by confidence.

    Each entry carries the concept it came from (``concept``, its
    ``confidence``), how many seeds chose it (``count``) and which
    (``papers``: their indexes in ``umls_topics_per_paper``).

    Parameters
    ----------
    base_filters:
        The existing topic_filters dict with keys "topics", "subfields",
        "fields", "domains".
    umls_topics_per_paper:
        One list of MappedTopic per seed paper (map_concepts_to_topics).
    also_have:
        Topics (``id``, ``display_name``) the interest has that are not in
        ``base_filters``: the researcher profile's wanted ones.
    cache_dir:
        Directory holding the OpenAlex topic embedding index. When given,
        near-duplicate detection compares topic *embeddings* (cosine)
        rather than word overlap, which catches synonyms that share no
        vocabulary. Falls back to word overlap when the index is absent.

    Returns
    -------
    A new topic_filters dict. Does not modify base_filters in place.
    """
    if not umls_topics_per_paper:
        return dict(base_filters)

    index = None
    if cache_dir:
        try:
            index = get_topic_index(cache_dir)
        except Exception as exc:  # missing index, bad shape, ...
            log.warning(
                "Topic index unavailable for dedup (%s); "
                "falling back to word-overlap similarity", exc,
            )

    existing = [
        (t["id"], t.get("display_name") or "")
        for t in [*(base_filters.get("topics") or []), *(also_have or [])]
        if t.get("id")
    ]
    existing_ids = {tid for tid, _ in existing}
    existing_names = {n.lower() for _, n in existing if n}
    known: dict[str, bool] = {}     # topic id -> the interest has it already

    def has_already(mt: MappedTopic) -> bool:
        if mt.topic_id not in known:
            known[mt.topic_id] = (
                mt.topic_id in existing_ids
                or mt.display_name.lower() in existing_names
                # Synonyms such as "Renal Diseases and Glomerulopathies" vs
                # "Renal and Vascular Pathologies". Not applied between
                # the UMLS topics themselves: siblings within one field
                # measure 0.719-0.815 and would be lost as "duplicates".
                or any(_is_near_duplicate(index, mt.topic_id, mt.display_name, eid, ename)
                       for eid, ename in existing)
            )
        return known[mt.topic_id]

    def by_confidence(mt: MappedTopic):
        return (-(mt.confidence or 0.0), mt.display_name.lower(), mt.topic_id)

    best: dict[str, MappedTopic] = {}       # topic name (lowercased) -> its best concept
    papers: dict[str, list[int]] = {}
    for i, paper_topics in enumerate(umls_topics_per_paper):
        chosen: dict[str, MappedTopic] = {}
        for mt in sorted(paper_topics, key=by_confidence):
            if len(chosen) >= per_paper:
                break
            if mt.confidence is None or mt.confidence < min_confidence:
                continue
            name = mt.display_name.lower()
            if name in chosen or has_already(mt):
                continue
            chosen[name] = mt
        for name, mt in chosen.items():
            papers.setdefault(name, []).append(i)
            if name not in best or by_confidence(mt) < by_confidence(best[name]):
                best[name] = mt

    accepted = sorted(best, key=lambda n: by_confidence(best[n]))[:max_additions]
    new_entries = [
        {
            "id": best[n].topic_id,
            "display_name": best[n].display_name,
            "count": len(papers[n]),
            "source": "umls",
            "on": False,
            "concept": best[n].source_name,
            "confidence": best[n].confidence,
            "papers": papers[n],
        }
        for n in accepted
    ]
    if not new_entries:
        return dict(base_filters)

    merged = copy.deepcopy(base_filters)
    merged.setdefault("topics", []).extend(new_entries)
    log.info(
        "Merged %d UMLS topics into topic_filters (total now: %d)",
        len(new_entries),
        len(merged["topics"]),
    )
    return merged
