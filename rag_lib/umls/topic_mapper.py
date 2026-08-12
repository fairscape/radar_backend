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
    """An OpenAlex topic matched from a UMLS concept."""

    topic_id: str  # e.g. "https://openalex.org/T12345"
    display_name: str
    similarity: float
    source_cui: str  # which UMLS concept triggered this match
    source_name: str  # UMLS concept name for traceability

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_concepts_to_topics(
    concepts: list[UmlsConcept],
    *,
    min_similarity: float = 0.40,
    top_k: int = 3,
    embedding_model: str = "mxbai-embed-large",
    cache_dir: str | None = None,
) -> list[MappedTopic]:
    """Map UMLS concepts to OpenAlex topics via embedding similarity.

    For each concept, embeds its canonical name, searches the topic
    index for the top-k most similar topics, and returns all matches
    above min_similarity. Results are deduplicated by topic_id (keeping
    the highest similarity).

    Parameters
    ----------
    concepts:
        UmlsConcept list from extract_umls_concepts().
    min_similarity:
        Minimum cosine similarity to keep a mapping.
    top_k:
        Number of top matches per concept to consider.
    embedding_model:
        Embedder key (must be registered in rag_lib.embedders).
    cache_dir:
        Directory containing the topic index files.

    Returns
    -------
    List of MappedTopic, deduplicated by topic_id, sorted by similarity
    descending.
    """
    if not concepts or not cache_dir:
        return []

    from ..embedders import get_embedder

    embedder = get_embedder(embedding_model)
    index = get_topic_index(cache_dir)

    # Collect all mappings, dedup by topic_id (keep highest similarity)
    best: dict[str, MappedTopic] = {}

    for concept in concepts:
        # Embed the UMLS concept name
        vec = np.array(embedder(concept.name), dtype=np.float32)

        # Search topic index
        matches = index.search(vec, top_k=top_k, min_similarity=min_similarity)

        for topic_dict, sim in matches:
            tid = topic_dict["id"]
            if tid in best and best[tid].similarity >= sim:
                continue
            best[tid] = MappedTopic(
                topic_id=tid,
                display_name=topic_dict["display_name"],
                similarity=round(sim, 4),
                source_cui=concept.cui,
                source_name=concept.name,
            )

    results = sorted(best.values(), key=lambda m: m.similarity, reverse=True)
    return results


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
    max_additions: int = 4,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Merge UMLS-mapped topics into existing topic_filters.

    Takes the base topic_filters dict (from Profile.aggregate_topic_filters)
    and a list of per-paper MappedTopic lists. Counts how often each
    UMLS-mapped topic appears across papers, excludes topics already
    present in the base filters (by ID, name, or embedding proximity),
    and appends the top ``max_additions`` new topics.

    Parameters
    ----------
    base_filters:
        The existing topic_filters dict with keys "topics", "subfields",
        "fields", "domains".
    umls_topics_per_paper:
        One list of MappedTopic per seed paper.
    max_additions:
        Maximum number of new topics to add.
    cache_dir:
        Directory holding the OpenAlex topic embedding index. When given,
        near-duplicate detection compares topic *embeddings* (cosine)
        rather than word overlap, which catches synonyms that share no
        vocabulary. Falls back to word overlap when the index is absent.

    Returns
    -------
    A new topic_filters dict with UMLS topics appended to the "topics"
    list. Does not modify base_filters in place.
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

    # Existing topics as (id, name) so we can compare by embedding, not
    # just by exact string match.
    existing = [
        (t["id"], t.get("display_name") or "")
        for t in base_filters.get("topics", [])
        if t.get("id")
    ]
    existing_ids = {tid for tid, _ in existing}
    existing_names_lower = {n.lower() for _, n in existing if n}

    # Count how often each UMLS-mapped topic appears across papers
    topic_counter: Counter[str] = Counter()
    topic_info: dict[str, MappedTopic] = {}  # keep best similarity

    for paper_topics in umls_topics_per_paper:
        seen_this_paper: set[str] = set()
        for mt in paper_topics:
            if mt.topic_id in existing_ids:
                continue
            if mt.display_name.lower() in existing_names_lower:
                continue
            if mt.topic_id not in seen_this_paper:
                topic_counter[mt.topic_id] += 1
                seen_this_paper.add(mt.topic_id)
            # Keep the mapping with highest similarity
            if mt.topic_id not in topic_info or topic_info[mt.topic_id].similarity < mt.similarity:
                topic_info[mt.topic_id] = mt

    if not topic_counter:
        return dict(base_filters)

    # Rank by mapping confidence, breaking ties on how many seed papers
    # mentioned the concept.
    #
    # Similarity leads deliberately. With a handful of seed papers the
    # count can only be 1/2/3, so it has almost no discriminating power —
    # sorting on it first lets a generic term that happens to appear in
    # every paper's reference list (0.4 similarity, count 3) outrank a
    # high-confidence domain match (0.89 similarity, count 1).
    ranked_ids = sorted(
        topic_counter.keys(),
        key=lambda tid: (topic_info[tid].similarity, topic_counter[tid]),
        reverse=True,
    )

    # Second pass: drop candidates that are near-duplicates of a topic
    # already in the profile, or of one we've already accepted. The
    # exact-match checks above only catch identical ids/names; this
    # catches synonyms like "Renal Diseases and Glomerulopathies" vs
    # "Renal and Vascular Pathologies".
    accepted: list[str] = []
    for tid in ranked_ids:
        name = topic_info[tid].display_name
        if any(
            _is_near_duplicate(index, tid, name, eid, ename)
            for eid, ename in existing
        ):
            continue
        if any(
            _is_near_duplicate(
                index, tid, name, aid, topic_info[aid].display_name
            )
            for aid in accepted
        ):
            continue
        accepted.append(tid)
        if len(accepted) >= max_additions:
            break

    # Build new topic entries, dedup by case-insensitive display name
    new_entries = []
    seen_names_lower: set[str] = set()
    for tid in accepted:
        name = topic_info[tid].display_name
        name_lower = name.lower()
        if name_lower in seen_names_lower:
            continue
        seen_names_lower.add(name_lower)
        new_entries.append({
            "id": tid,
            "display_name": name,
            "count": topic_counter[tid],
            "source": "umls",
        })

    # Merge
    merged = copy.deepcopy(base_filters)
    merged.setdefault("topics", []).extend(new_entries)

    log.info(
        "Merged %d UMLS topics into topic_filters (total now: %d)",
        len(new_entries),
        len(merged["topics"]),
    )
    return merged
