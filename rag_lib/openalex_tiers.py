"""Tier-based OpenAlex filter construction.

Encodes the three-tier "must-have AND → top-topics OR + subfields AND
→ subfields OR" strategy so both the standalone CLI gather and the
runtime ``OpenAlexGatherer`` (used by the wizard dry-run, scheduled
gathers, and ``gather-now``) inherit the same prevalence-based
narrowing. Without this module the runtime path just hands the raw
aggregated ``profile.topic_filters`` dict to OpenAlex, which AND-joins
every populated hierarchy level and over-constrains.

A tier is a list of ``(oa_filter_key, ids, mode)`` triples consumed by
``build_filter_string``. Modes:

  - ``"or"``  → joined with ``|`` in a single ``key:val1|val2`` clause.
  - ``"and"`` → emitted as the same key repeated, comma-separated:
                ``key:val1,key:val2``. OpenAlex parses commas as AND
                even when the key matches.
"""

from __future__ import annotations

from collections import defaultdict

from .profile import Profile


DEFAULT_MUST_HAVE_PREVALENCE = 0.5
DEFAULT_TOP_TOPICS_N = 5
DEFAULT_TOP_SUBFIELDS_N = 2

FilterPart = tuple[str, list[str], str]  # (oa_filter_key, ids, mode)
TierSpec = tuple[str, list[FilterPart]]


def bare_id(oa_id: str) -> str:
    """Strip the OpenAlex URL prefix.

    The /works filter API accepts both forms but bare IDs keep the
    URL short and dodge a parser quirk that drops AND'd URL-form
    clauses (the colon in ``https://...`` confuses it)."""
    return oa_id.rsplit("/", 1)[-1] if "/" in oa_id else oa_id


def distinct_paper_prevalence(profile: Profile) -> dict[str, float]:
    """For each topic-id, the fraction of seed papers it appears in.

    Counts a paper once even when the topic shows up as both
    ``primary_topic`` and in ``topics[]``. Distinct from
    ``profile.topic_filters['topics'][i]['count']`` which double-counts
    those cases."""
    n = len(profile.papers)
    if n == 0:
        return {}
    topic_to_papers: dict[str, set[int]] = defaultdict(set)
    for i, p in enumerate(profile.papers):
        seen: set[str] = set()
        if p.primary_topic and p.primary_topic.id:
            seen.add(p.primary_topic.id)
        for t in p.topics:
            if t and t.id:
                seen.add(t.id)
        for tid in seen:
            topic_to_papers[tid].add(i)
    return {tid: len(papers) / n for tid, papers in topic_to_papers.items()}


def is_enabled(entry: dict) -> bool:
    """Whether a topic_filters entry is switched on.

    Absent ``on`` means on: entries written before the flag existed, and
    every level other than ``topics`` (only topics are user-toggleable),
    must keep participating.
    """
    return bool(entry.get("on", True))


def enabled_topic_ids(topic_filters: dict | None) -> list[str]:
    """Full OpenAlex ids of the topics the user has left switched on.

    Order follows ``topic_filters`` — the aggregator emits topics by
    descending seed count, so callers that take a prefix get the most
    prevalent ones first.
    """
    return [
        t["id"]
        for t in (topic_filters or {}).get("topics") or []
        if t.get("id") and is_enabled(t)
    ]


def _top_ids(profile_filters: dict, level: str, n: int) -> list[str]:
    items = [it for it in (profile_filters.get(level) or []) if is_enabled(it)]
    return [bare_id(it["id"]) for it in items[:n] if it.get("id")]


def build_filter_string(
    parts: list[FilterPart],
    since: str,
    *,
    extras: str = "type:article,language:en",
) -> str:
    """Assemble an OpenAlex /works filter string from tier parts.

    Output is comma-joined (commas are AND between clauses):
      - mode ``"or"``  → ``key:val1|val2|val3`` (single clause).
      - mode ``"and"`` → ``key:val1,key:val2`` (each clause AND'd).

    ``from_publication_date:<since>`` and ``extras`` are always emitted
    first so the date-bound + article/language filters apply to every
    tier."""
    out: list[str] = [f"from_publication_date:{since}"]
    if extras:
        out.append(extras)
    for oa_key, ids, mode in parts:
        if not ids:
            continue
        if mode == "or":
            out.append(f"{oa_key}:{'|'.join(ids)}")
        else:
            for v in ids:
                out.append(f"{oa_key}:{v}")
    return ",".join(out)


def tier_specs(
    profile: Profile,
    *,
    must_have_prevalence: float = DEFAULT_MUST_HAVE_PREVALENCE,
    top_topics_n: int = DEFAULT_TOP_TOPICS_N,
    top_subfields_n: int = DEFAULT_TOP_SUBFIELDS_N,
) -> list[TierSpec]:
    """Return the three tiers as a list of ``(tier_name, parts)`` pairs.

    Tiers are ordered narrowest → widest:

      1. ``must-have-AND`` — topics whose distinct-paper prevalence is
         ≥ ``must_have_prevalence`` (default 0.5) become AND-required:
         every candidate must contain *all* of them.
      2. ``top-topics-OR-subfields-AND`` — top-N topics OR'd, with the
         top-M subfields each AND'd as additional constraints. Picks
         up adjacent work that shares a core topic and sits in the
         seed's subfield(s).
      3. ``subfields-OR`` — bare OR over the top subfields. Broadest
         fallback.

    Empty tiers (e.g. no topic clears the prevalence floor; no
    subfields aggregated) are dropped so callers can walk the result
    without skipping empties."""
    prev = distinct_paper_prevalence(profile)
    must_have = sorted(
        [tid for tid, p in prev.items() if p >= must_have_prevalence],
        key=lambda t: -prev[t],
    )
    must_have_bare = [bare_id(t) for t in must_have]
    top_topics = _top_ids(profile.topic_filters or {}, "topics", top_topics_n)
    top_subfields = _top_ids(profile.topic_filters or {}, "subfields", top_subfields_n)

    out: list[TierSpec] = []
    if must_have_bare:
        out.append(("must-have-AND", [("topics.id", must_have_bare, "and")]))
    if top_topics or top_subfields:
        out.append((
            "top-topics-OR-subfields-AND",
            [
                ("topics.id", top_topics, "or"),
                ("topics.subfield.id", top_subfields, "and"),
            ],
        ))
    if top_subfields:
        out.append(("subfields-OR", [("topics.subfield.id", top_subfields, "or")]))
    return out
