"""OpenAlex topics mapped from the research topics an LLM generated.

The research topics (``interest_terms``, generated with the user's own
Claude key) are free text: they rank the feed but no scan searches by them.
Here each label is mapped to the OpenAlex topics nearest to it, the way
UMLS concept names are (``umls.topic_mapper``: the same embedder, the same
topic index), and the new ones join the interest's topic list as
``source: "llm"`` -- switched off, so a scan searches by one only once the
user has switched it on.

No LLM call is made here: the labels were generated (and paid for) already.

Calibrated on the two live interests with Claude-generated topics: a
label's nearest topic is the right one from about 0.70 up (0.701-0.948
over 14 labels), while second and third matches ran from right ("Speech
and Audio Processing" 0.823 for "Speech & Language Analysis") to noise
("Data Analysis and Archiving" 0.698 for "Electronic Health Records
Analysis"). Hence two bars: the best match from ``min_similarity``, more
only from ``extra_similarity``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

SOURCE = "llm"

# The research-topic record sources made by an LLM (interest_terms).
LLM_RECORD_SOURCES = frozenset({"rp-sdk-llm"})


@dataclass(frozen=True)
class LabelMatch:
    topic_id: str
    display_name: str
    similarity: float
    label: str  # the research topic it was mapped from


def map_labels(
    labels: list[str],
    *,
    embedding_model: str,
    cache_dir: str,
    min_similarity: float = 0.65,
    extra_similarity: float = 0.80,
    top_k: int = 3,
) -> list[LabelMatch]:
    """The OpenAlex topics nearest each label, best first, one per topic."""
    from .embedders import get_embedder
    from .umls.cache import get_topic_index

    labels = [s.strip() for s in labels if s and s.strip()]
    if not labels:
        return []
    embedder = get_embedder(embedding_model)
    index = get_topic_index(cache_dir)
    index.check_compatible(embedding_model)
    best: dict[str, LabelMatch] = {}
    for label in labels:
        vec = np.array(embedder(label), dtype=np.float32)
        index.check_compatible(embedding_model, dim=int(vec.shape[-1]))
        matches = index.search(vec, top_k=top_k, min_similarity=min_similarity)
        for rank, (topic, sim) in enumerate(matches):
            if rank > 0 and sim < extra_similarity:
                break
            tid = topic["id"]
            if tid in best and best[tid].similarity >= sim:
                continue
            best[tid] = LabelMatch(tid, topic["display_name"], round(float(sim), 4), label)
    return sorted(best.values(), key=lambda m: m.similarity, reverse=True)


def merge(
    topic_filters: dict[str, Any],
    matches: list[LabelMatch],
    *,
    max_additions: int = 10,
    cache_dir: str | None = None,
) -> tuple[dict[str, Any], list[dict]]:
    """``topic_filters`` with the matches it does not have yet, as LLM topics.

    A mapping run again replaces the LLM topics left off (the user has not
    taken them); one switched on is the user's choice and stays. A match
    that duplicates a topic already listed -- by id, by name, or by its
    embedding (``umls.topic_mapper._is_near_duplicate``) -- is not added.
    Returns the new filters and the topics new to the list: one mapped
    again is listed, not "added" (a second run said "Added 5" for the
    same five).
    """
    from .researcher_topics import code
    from .umls.topic_mapper import _is_near_duplicate

    index = None
    if cache_dir:
        try:
            from .umls.cache import get_topic_index
            index = get_topic_index(cache_dir)
        except Exception:  # noqa: BLE001 -- dedup falls back to word overlap
            index = None

    out = copy.deepcopy(topic_filters or {})
    listed_before = {code(t.get("id")) for t in out.get("topics") or [] if t.get("id")}
    topics = [t for t in out.get("topics") or []
              if not (t.get("source") == SOURCE and not t.get("on", True))]
    existing = [(t["id"], t.get("display_name") or "") for t in topics if t.get("id")]
    have_ids = {code(tid) for tid, _ in existing}
    have_names = {n.lower() for _, n in existing if n}

    added: list[dict] = []
    for m in matches:
        if len(added) >= max_additions:
            break
        name = m.display_name
        if code(m.topic_id) in have_ids or name.lower() in have_names:
            continue
        if any(_is_near_duplicate(index, m.topic_id, name, eid, ename) for eid, ename in existing):
            continue
        entry = {"id": m.topic_id, "display_name": name, "count": 0, "source": SOURCE,
                 "on": False, "llm_label": m.label, "similarity": m.similarity}
        added.append(entry)
        have_ids.add(code(m.topic_id))
        have_names.add(name.lower())
    out["topics"] = topics + added
    return out, [t for t in added if code(t["id"]) not in listed_before]


def keep_stored(stored: dict[str, Any], topic_filters: dict[str, Any]) -> None:
    """Carry the stored LLM topics into a list rebuilt from the seeds: no
    seed brings them back, and a rebuild would drop them (as the adopted
    topics, ``wizard._keep_adopted_topics``). A seed that carries one
    lists it as its own topic instead."""
    from .researcher_topics import code

    topics = topic_filters.setdefault("topics", [])
    have = {code(t.get("id")) for t in topics if t.get("id")}
    for t in stored.get("topics") or []:
        if t.get("id") and t.get("source") == SOURCE and code(t["id"]) not in have:
            topics.append(dict(t))
            have.add(code(t["id"]))
