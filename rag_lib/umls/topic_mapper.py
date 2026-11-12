"""UMLS concept → OpenAlex topic mapping via embedding similarity.

Embeds each UMLS concept together with the sentences it was read from
("Amyloid: Amyloid β accumulation drives ...") with mxbai-embed-large,
computes cosine similarity against the precomputed OpenAlex topic index
built from the topics' names, descriptions and keywords, and returns the
nearest topic (CTX-2). The name alone matched names: "Middle East
Respiratory Syndrome" landed on Inflammatory Bowel Disease, and on
biomedical papers the primary topic was among the chosen five for 0% of
PDF bodies and 19% of abstracts; with the sentences 83% and 78%
(records/35).

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

from .cache import get_rich_topic_index, get_topic_index
from .extractor import UmlsConcept

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MappedTopic:
    """One of the NEIGHBOURS OpenAlex topics nearest to one UMLS concept
    of a paper (``rank`` 1 is the nearest)."""

    topic_id: str  # e.g. "https://openalex.org/T12345"
    display_name: str
    # The concept in its sentences to the topic's description (CTX-2):
    # what a seed's topics are ranked by (merge_umls_topics).
    similarity: float
    source_cui: str  # which UMLS concept triggered this match
    source_name: str  # UMLS concept name for traceability
    # The concept's confidence (how surely the text's words are this
    # concept): under CONFIDENCE it is not chosen. None in a mapping stored
    # before it was recorded.
    confidence: float | None = None
    # The word the text says it with ("Amyloid β", "genome-wide
    # association studies"), shown beside the topic: the UMLS name of a
    # misread word ("Analysis of substances" for "analysis") does not say
    # where the topic came from (CTX-4).
    mention: str | None = None
    # MAPPING for a mapping made the way described here; anything else
    # was made by earlier rules and is mapped again (stale_mapping).
    mapping: str | None = None
    # 1 for the concept's nearest topic, 2 for the next, ... (CTX-3b).
    rank: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


# How the UMLS topics of an interest are chosen (FUNCTIONAL_SPEC U.2, CTX):
# from each seed, its concepts of CONFIDENCE or more, each with one of its
# NEIGHBOURS nearest topics -- the nearest the interest does not have yet
# (CTX-3b) -- PER_PAPER of them by similarity; at most MAX_TOPICS for
# the interest, those most seeds chose first, then by similarity. They are
# added switched off.
#
# SIMILARITY is no bar any more (0.0). It was 0.80 between a concept's name
# and a topic's name (UT-1); a sentence against a description measures
# 0.6-0.8 for the right topic and the wrong ones alike, so the ranking and
# the five a seed are what keep a topic out (records/35).
CONFIDENCE = 0.9
SIMILARITY = 0.0
PER_PAPER = 5
MAX_TOPICS = 20
# A concept whose nearest topic the paper already has gives its 2nd or 3rd
# nearest (user 11-11, option B): the Immunity paper's concepts all land
# on its own two Alzheimer topics, and nothing was added. On 300 abstracts
# the topics added went from 2.91 to 4.58 a paper and the far-off ones from
# 18% to 11%; at 5 neighbours far ones come back in single papers
# ("Immune response" -> Atherosclerosis), records/35.
NEIGHBOURS = 3
MAPPING = "context-2"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_concepts_to_topics(
    concepts: list[UmlsConcept],
    *,
    embedding_model: str = "mxbai-embed-large",
    cache_dir: str | None = None,
    min_confidence: float = CONFIDENCE,
    neighbours: int = NEIGHBOURS,
) -> list[MappedTopic]:
    """Each concept's ``neighbours`` nearest OpenAlex topics, by embedding
    it in the sentences it was read from (CTX-2, CTX-3b).

    Each of its mentions is embedded as "name: sentence" and the vectors
    averaged; the ``neighbours`` nearest topics in the rich index (name,
    description, keywords), however near or far, are its candidates, nearest
    first (``rank`` 1, 2, ...). Concepts in the order given; which topic,
    if any, a concept gives an interest is decided when its topics are
    chosen (merge_umls_topics).

    Not mapped: a concept under ``min_confidence`` (never chosen, and each
    mention is an embedding call), and one without a usable mention (every
    one short or formula-like: placed by its name alone it is where the
    misread words came from).
    """
    if not concepts or not cache_dir:
        return []

    from ..embedders import get_embedder

    embedder = get_embedder(embedding_model)
    index = get_rich_topic_index(cache_dir)
    index.check_compatible(embedding_model)

    out: list[MappedTopic] = []
    for concept in concepts:
        if not _reaches(concept.confidence, min_confidence) or not concept.mentions:
            continue
        vecs = []
        for _word, sentence in concept.mentions:
            vec = np.array(embedder(f"{concept.name}: {sentence}"), dtype=np.float32)
            # An index without its meta file can still be caught on
            # dimension (a bare matmul error said nothing about why).
            index.check_compatible(embedding_model, dim=int(vec.shape[-1]))
            norm = float(np.linalg.norm(vec))
            # A zero or NaN vector is near nothing: with no similarity bar
            # it would still bring in whichever topic came first.
            if np.isfinite(norm) and norm > 0:
                vecs.append(vec / norm)
        if not vecs:
            continue
        mean = np.mean(vecs, axis=0)
        if not np.linalg.norm(mean) > 1e-6:     # mentions pointing opposite ways
            continue
        hits = index.search(mean, top_k=neighbours, min_similarity=-1.0)
        for rank, (topic_dict, sim) in enumerate(hits, start=1):
            out.append(MappedTopic(
                topic_id=topic_dict["id"],
                display_name=topic_dict["display_name"],
                similarity=round(sim, 4),
                source_cui=concept.cui,
                source_name=concept.name,
                confidence=round(float(concept.confidence), 4),
                mention=concept.mentions[0][0],
                mapping=MAPPING,
                rank=rank,
            ))
    return out


def _reaches(value: float | None, bar: float) -> bool:
    """Whether a confidence or similarity is ``bar`` or more. None and NaN
    (a zero-vector embedding) do not: ``NaN < bar`` is False, so a plain
    "under the bar" test let NaN through (review 2026-11-06 L3)."""
    return value is not None and value >= bar


def stale_mapping(entries: list[dict]) -> bool:
    """A stored ``umls_mapped_topics_json`` made by earlier rules (a name
    to a name; one topic a concept, context-1; CTX-5): it is mapped again."""
    return any(e.get("mapping") != MAPPING for e in entries)


def stale_extraction(concepts: list, entries: list) -> bool:
    """A paper's stored UMLS data made by earlier rules: concepts stored
    without where the text says them (the sentences the mapping needs),
    or a mapping ``stale_mapping`` refuses. Both are made again."""
    dicts = [c for c in concepts if isinstance(c, dict)]
    return any("mentions" not in c for c in dicts) or stale_mapping(
        [e for e in entries if isinstance(e, dict)])


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
    min_similarity: float = SIMILARITY,
    per_paper: int = PER_PAPER,
    max_additions: int = MAX_TOPICS,
    cache_dir: str | None = None,
    also_have: list[dict] | None = None,
) -> dict[str, Any]:
    """``base_filters`` with the UMLS topics of its seeds appended, switched
    off (FUNCTIONAL_SPEC U.2).

    Chosen on the similarity of a concept in its sentences to the topic
    (CTX-3). In each seed, the concepts of ``min_confidence`` or more whose
    topic is ``min_similarity`` or more similar (no bar by default), the
    (concept, topic) pairs -- each concept's NEIGHBOURS nearest topics --
    nearest first: one whose topic the interest already has (same id or
    name, or an embedding cosine of _DEDUP_COSINE_THRESHOLD or more to one
    of ``base_filters``' topics) is skipped, and so is one whose topic a
    concept of the same seed already gave, and one whose concept already
    gave a topic (a concept gives one: its nearest the interest lacks,
    CTX-3b); ``per_paper`` are taken. The
    seeds' topics together: each once (same id, or the same name under
    another id), with its nearest concept; ``max_additions`` of them, the
    ones more seeds chose first, then the nearer (records/35: on an
    81-seed interest 7 of its 8 OpenAlex topics are among the twenty when
    it has none yet, against 1 by confidence).

    Each entry carries the concept it came from (``concept``, its
    ``confidence``, the word the text says it with: ``mention``), its
    ``similarity``, how many seeds chose it (``count``) and which
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

    def nearest(mt: MappedTopic):
        # Only mappings that passed _reaches are ranked: both are numbers.
        # ``rank`` before the name: a concept's two neighbours equal once
        # rounded to 4 places go nearest first (review B3-R2).
        return (-mt.similarity, -(mt.confidence or 0.0), mt.rank, mt.display_name.lower(), mt.topic_id)

    best: dict[str, MappedTopic] = {}       # topic name (lowercased) -> its nearest concept
    papers: dict[str, list[int]] = {}
    for i, paper_topics in enumerate(umls_topics_per_paper):
        usable = [mt for mt in paper_topics
                  if _reaches(mt.confidence, min_confidence) and _reaches(mt.similarity, min_similarity)]
        chosen: dict[str, MappedTopic] = {}
        gave: set[str] = set()      # concepts of this seed that gave a topic
        for mt in sorted(usable, key=nearest):
            if len(chosen) >= per_paper:
                break
            name = mt.display_name.lower()
            if mt.source_cui in gave or name in chosen or has_already(mt):
                continue
            chosen[name] = mt
            gave.add(mt.source_cui)
        for name, mt in chosen.items():
            papers.setdefault(name, []).append(i)
            if name not in best or nearest(mt) < nearest(best[name]):
                best[name] = mt

    # The topics more seeds chose first: on a large interest one seed's
    # misread word reaches one seed, the subject reaches many.
    accepted = sorted(best, key=lambda n: (-len(papers[n]), *nearest(best[n])))[:max_additions]
    new_entries = [
        {
            "id": best[n].topic_id,
            "display_name": best[n].display_name,
            "count": len(papers[n]),
            "source": "umls",
            "on": False,
            "concept": best[n].source_name,
            "confidence": best[n].confidence,
            "mention": best[n].mention,
            "similarity": best[n].similarity,
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
