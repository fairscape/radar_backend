"""A researcher's declared interests, as topics for an interest built from them.

A Prosopia profile carries ``research_interests``: OpenAlex topics the
owner declared (``method: declared``, with a ``weight`` whose sign says
whether they want more of it or none of it) or that were inferred from
their papers (``method: inferred``, no weight). Two plain-text lists,
``interests`` and ``not_interests``, say the same in words for concepts
that map to no topic.

RADAR stores the profile document on the researcher row
(``researchers.document_json``), so this module only reads it. :func:`read`
turns the document into the topics wanted and not wanted; :func:`apply`
folds them into an interest's ``topic_filters``: a wanted topic the seeds
did not bring in is added (``source = "researcher"``), and every topic the
profile has an opinion on is marked ``profile = "interest" | "not_interest"``
so the lists can show it and the aggregation can start a not-wanted topic
switched off. The profile's negatives are a filter, never a deletion: the
topic stays in the list, off, and the user can switch it back on.

Each not-wanted topic also carries the user's choice for it
(``topic_filters["not_interest_choices"]``, by topic code): ``filter`` it out
(off wherever the seeds bring it in), ``keep`` it as the other sources have
it, or ``add`` it as a topic of the interest's own. Until the user picks,
it is ``keep`` when another source lists the same topic -- the seeds, UMLS,
a PDF, Claude, or the profile's own interests -- and ``filter`` otherwise;
that default is recorded when the profile is applied, so a seed that brings
a filtered topic in later finds it off.
"""

from __future__ import annotations

import json
import re
from typing import Any

SOURCE = "researcher"
INTEREST = "interest"
NOT_INTEREST = "not_interest"
FLAG = "researcher_topics"
#: ``source`` of a profile topic kept after the researcher was forgotten:
#: no seed carries it, so a rebuild from the seeds has to carry it over.
ADOPTED = "adopted"          # topic_filters key: the profile's topics are applied
# On a marked topic: its on/off before the profile was applied, restored when
# the profile is removed again.
BEFORE = "on_before_profile"
CHOICES = "not_interest_choices"   # topic_filters key: {topic code: choice}
FILTER = "filter"
KEEP = "keep"
ADD = "add"
CHOICE_VALUES = (FILTER, KEEP, ADD)
#: How the profile's own interests show among a not-wanted topic's other sources.
PROFILE_INTEREST = "profile_interest"

_TOPIC_ID = re.compile(r"^(?:https?://(?:api\.)?openalex\.org/(?:topics/)?)?(T\d+)$", re.IGNORECASE)


def _topic_id(concept: dict) -> str | None:
    # Each identifier in turn: an "@id" in another spelling of the URL
    # hid a valid "code" and dropped the topic.
    for key in ("@id", "id", "code"):
        m = _TOPIC_ID.match(str(concept.get(key) or "").strip())
        if m:
            return f"https://openalex.org/{m.group(1).upper()}"
    return None


def _code(topic_id: Any) -> str:
    """``T10222`` for either spelling of a topic id, for matching."""
    return str(topic_id or "").strip().rsplit("/", 1)[-1].upper()


def _label(concept: dict) -> str:
    for key in ("display", "display_name", "label", "name"):
        value = concept.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _weight(entry: dict) -> float | None:
    try:
        w = entry.get("weight")
        return None if w is None else float(w)
    except (TypeError, ValueError):
        return None


def _strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.get("top")
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in value:
        if isinstance(v, dict):
            v = v.get("label")
        if isinstance(v, str) and v.strip() and v.strip().lower() not in seen:
            seen.add(v.strip().lower())
            out.append(v.strip())
    return out


def read(document: dict | None) -> dict:
    """``{"wanted", "unwanted", "wanted_labels", "unwanted_labels"}`` of a profile.

    ``wanted`` / ``unwanted`` are OpenAlex topics
    (``{"id", "display_name", "weight", "method"}``); the labels are the
    text concepts that map to no topic. One topic asserted several times is
    settled once: a declared assertion beats an inferred one, and at the
    same level a negative beats a positive (the owner saying "not this" is
    the one worth honouring). A negative weight is "not wanted"; no weight
    (an inferred topic) counts as wanted.
    """
    empty = {"wanted": [], "unwanted": [], "wanted_labels": [], "unwanted_labels": []}
    if not isinstance(document, dict):
        return empty
    best: dict[str, tuple[tuple[int, int], dict]] = {}
    wanted_labels: list[str] = []
    unwanted_labels: list[str] = []
    for entry in document.get("research_interests") or []:
        if not isinstance(entry, dict):
            continue
        concept = entry.get("concept") if isinstance(entry.get("concept"), dict) else {}
        weight = _weight(entry)
        negative = weight is not None and weight < 0
        method = str(entry.get("method") or "inferred").lower()
        tid = _topic_id(concept)
        if tid is None:
            # A concept with some other identifier (a work, a MeSH heading)
            # is not a text label; only an unmapped one is.
            if concept.get("@id") or concept.get("id"):
                continue
            label = _label(concept)
            if label:
                (unwanted_labels if negative else wanted_labels).append(label)
            continue
        rank = (1 if method == "declared" else 0, 1 if negative else 0)
        item = {"id": tid, "display_name": _label(concept) or tid.rsplit("/", 1)[-1],
                "weight": weight, "method": method, "negative": negative}
        if tid not in best or rank > best[tid][0]:
            best[tid] = (rank, item)
    wanted_labels = _strings(document.get("interests")) + wanted_labels
    unwanted_labels = _strings(document.get("not_interests")) + unwanted_labels

    def _sorted(items: list[dict]) -> list[dict]:
        return sorted(items, key=lambda t: (t["method"] != "declared", -(abs(t["weight"]) if t["weight"] is not None else 0),
                                            t["display_name"].lower()))

    wanted = _sorted([i for _, i in best.values() if not i["negative"]])
    unwanted = _sorted([i for _, i in best.values() if i["negative"]])
    for i in (*wanted, *unwanted):
        i.pop("negative", None)
    # A label the topics already cover is not repeated in words.
    names = {t["display_name"].lower() for t in (*wanted, *unwanted)}
    return {
        "wanted": wanted,
        "unwanted": unwanted,
        "wanted_labels": _strings([l for l in wanted_labels if l.lower() not in names]),
        "unwanted_labels": _strings([l for l in unwanted_labels if l.lower() not in names]),
    }


def public(view: dict) -> dict:
    """``view`` in the API's spelling: each topic's ``display_name`` as ``name``."""
    def _t(t: dict) -> dict:
        return {"id": t["id"], "name": t.get("display_name") or t["id"], "weight": t.get("weight"),
                "method": t.get("method") or "inferred", "label": t.get("label"),
                "similarity": t.get("similarity")}
    return {
        "wanted": [_t(t) for t in view.get("wanted") or []],
        "unwanted": [_t(t) for t in view.get("unwanted") or []],
        "wanted_labels": list(view.get("wanted_labels") or []),
        "unwanted_labels": list(view.get("unwanted_labels") or []),
        "labels_unmapped": bool(view.get("labels_unmapped")),
        "unmatched_labels": list(view.get("unmatched_labels") or []),
    }


def from_row(row: Any) -> dict:
    """:func:`read` of a researcher row's stored document (empty when there is none)."""
    raw = row["document_json"] if row is not None else None
    if not raw:
        return read(None)
    try:
        return read(json.loads(raw))
    except (ValueError, TypeError):
        return read(None)


def available(view: dict) -> bool:
    """Whether the profile said anything a topic list can use."""
    return bool(view.get("wanted") or view.get("unwanted"))


def is_applied(topic_filters: dict | None) -> bool:
    return bool((topic_filters or {}).get(FLAG))


def other_sources(topic_filters: dict | None, view: dict) -> dict[str, list[str]]:
    """``{topic code: [source, ...]}``: where each topic in the list came
    from other than the profile, plus the profile's own interests
    (:data:`PROFILE_INTEREST`). A seed's OpenAlex topic is ``openalex``."""
    found: dict[str, set[str]] = {}
    for t in (topic_filters or {}).get("topics") or []:
        if t.get("id") and t.get("source") != SOURCE:
            found.setdefault(_code(t["id"]), set()).add(t.get("source") or "openalex")
    for item in view.get("wanted") or []:
        found.setdefault(_code(item["id"]), set()).add(PROFILE_INTEREST)
    return {c: sorted(v) for c, v in found.items()}


def default_choice(overlap: list[str] | set[str]) -> str:
    """What a not-wanted topic is until the user picks: kept when another
    source lists it too (the user decides), filtered out otherwise."""
    return KEEP if overlap else FILTER


def is_legacy(stored: dict | None) -> bool:
    """Applied before choices were recorded: every not-wanted topic was
    switched off, whichever other source listed it."""
    stored = stored or {}
    return is_applied(stored) and not isinstance(stored.get(CHOICES), dict)


def stored_choices(stored: dict | None, view: dict | None = None) -> dict[str, str]:
    """The choices recorded on a stored list. A legacy list (:func:`is_legacy`)
    reads as ``filter`` for every not-wanted topic -- each in ``view`` when
    given, else each it marked -- so it keeps doing what it did, for a
    not-wanted topic a new seed brings in as well."""
    stored = stored or {}
    if isinstance(stored.get(CHOICES), dict):
        return {str(k): str(v) for k, v in stored[CHOICES].items() if v in CHOICE_VALUES}
    if not is_applied(stored):
        return {}
    out = {_code(t["id"]): FILTER for t in stored.get("topics") or []
           if t.get("id") and t.get("profile") == NOT_INTEREST}
    for item in (view or {}).get("unwanted") or []:
        out[_code(item["id"])] = FILTER
    return out


def apply(topic_filters: dict, view: dict, *, basis: dict | None = None) -> dict:
    """``topic_filters`` with the profile's topics folded in; a new dict.

    Wanted topics the seeds did not bring in are appended with
    ``source = "researcher"`` and ``count = 0`` (no seed carries them).
    Each not-wanted topic goes by its choice in
    ``topic_filters[CHOICES]`` (the default recorded there for one with
    none): ``filter`` marks it where it is listed, ``add`` treats it as a
    wanted one, ``keep`` leaves it as the other sources have it. Every
    listed topic the profile has an opinion on gets
    ``profile = "interest" | "not_interest"``. On/off is not touched here:
    the caller settles it (see ``wizard._aggregate_topics``).

    A default is counted against ``basis`` when given (the list rebuilt
    from the seeds, which is what the topic step shows and counts its
    defaults against), else against ``topic_filters``.
    """
    out = dict(topic_filters or {})
    topics = [dict(t) for t in out.get("topics") or []]
    # Matched on the bare code: the seeds' topics carry whatever id OpenAlex
    # gave them, the profile the full URL.
    by_code = {_code(t.get("id")): t for t in topics if t.get("id")}
    for t in topics:
        t.pop("profile", None)
    overlap = other_sources(basis if basis is not None else out, view)
    choices = {str(k): v for k, v in (out.get(CHOICES) or {}).items() if v in CHOICE_VALUES}

    def _want(item: dict) -> None:
        have = by_code.get(_code(item["id"]))
        if have is not None:
            have["profile"] = INTEREST
        else:
            new = {"id": item["id"], "display_name": item["display_name"], "count": 0, "papers": 0,
                   "source": SOURCE, "profile": INTEREST}
            topics.append(new)
            by_code[_code(item["id"])] = new

    for item in view.get("wanted") or []:
        _want(item)
    for item in view.get("unwanted") or []:
        c = _code(item["id"])
        choice = choices.setdefault(c, default_choice(overlap.get(c, [])))
        if choice == ADD:
            _want(item)
        elif choice == FILTER and c in by_code:
            by_code[c]["profile"] = NOT_INTEREST
    out["topics"] = topics
    out[CHOICES] = choices
    out[FLAG] = True
    return out


def not_interest_rows(view: dict, topic_filters: dict | None, *, basis: dict | None = None) -> list[dict]:
    """Each not-wanted topic as the topic step lists it: what it was
    matched from, where else it comes from, its choice (recorded on
    ``topic_filters``, or the default) and its place in the list now.

    ``basis`` is the list the profile is applied to, when that is not
    ``topic_filters`` itself: a draft's is rebuilt from the seeds, so its
    overlaps are counted there -- the defaults shown are then the ones
    applying records."""
    stored = topic_filters or {}
    base = basis if basis is not None else stored
    overlap = other_sources(base, view)
    choices = stored_choices(stored, view)
    listed_in = stored if stored.get("topics") else base
    by_code = {_code(t["id"]): t for t in listed_in.get("topics") or [] if t.get("id")}
    rows = []
    for item in view.get("unwanted") or []:
        c = _code(item["id"])
        listed = by_code.get(c)
        rows.append({
            "id": item["id"], "name": item.get("display_name") or item["id"],
            "method": item.get("method") or "inferred", "label": item.get("label"),
            "weight": item.get("weight"), "similarity": item.get("similarity"),
            "overlap": overlap.get(c, []),
            "choice": choices.get(c) or default_choice(overlap.get(c, [])),
            "in_list": listed is not None,
            "on": bool(listed.get("on", True)) if listed is not None else None,
        })
    return rows


def code(topic_id: Any) -> str:
    """The bare OpenAlex code (``T10001``) of a topic id in either form."""
    return _code(topic_id)


def adopt(topic_filters: dict) -> dict:
    """``topic_filters`` with the profile's topics kept as the interest's
    own (source ``adopted``): the marks and the flag gone, every state as it
    is. For a saved list that ``strip`` would leave with nothing on; a new
    dict."""
    out = dict(topic_filters or {})
    kept = []
    for t in out.get("topics") or []:
        t = dict(t)
        if t.get("source") == SOURCE:
            t["source"] = ADOPTED
        t.pop("profile", None)
        t.pop(BEFORE, None)
        kept.append(t)
    out["topics"] = kept
    out.pop(CHOICES, None)
    out[FLAG] = False
    return out


def strip(topic_filters: dict) -> dict:
    """``topic_filters`` without the profile's topics and marks; a new dict.
    The not-wanted choices are kept for a later apply."""
    out = dict(topic_filters or {})
    kept = []
    for t in out.get("topics") or []:
        if t.get("source") == SOURCE:
            continue
        t = dict(t)
        t.pop("profile", None)
        t.pop(BEFORE, None)
        kept.append(t)
    out["topics"] = kept
    # The not-wanted choices stay: applying the profile again picks them up
    # rather than going back to the defaults.
    out[FLAG] = False
    return out


# --- text labels → topics -----------------------------------------------------

# A label is a concept the profile could not map to a topic itself
# ("phylogenomics", "wet-lab-protocol-development"). Its nearest OpenAlex
# topic name, by the same index the UMLS mapper and the PDF reader use, is
# close enough to act on only above this; a real topic name is usually
# above 0.7 against a paraphrase of itself.
LABEL_MIN_SIMILARITY = 0.6
# The mapping runs inside requests (a researcher page, a topic list). An
# embedder that is down must not hold every one of them for its timeout:
# after a failure nothing is mapped for this long, and a label that mapped
# to nothing is not asked about again for as long either.
LABEL_RETRY_SECONDS = 600.0
_LABEL_CACHE: dict[tuple[str, str, str], tuple[dict | None, float]] = {}   # (model, index dir, label) -> (hit, when)
_DOWN_UNTIL = 0.0


def reset_label_cache() -> None:
    global _DOWN_UNTIL
    _LABEL_CACHE.clear()
    _DOWN_UNTIL = 0.0


def _nearest_topic(label: str, *, settings: Any) -> dict | None:
    import time

    model = str(settings.RADAR_UMLS_EMBEDDING_MODEL)
    cache_dir = str(settings.RADAR_UMLS_CACHE_DIR)
    key = (model, cache_dir, label.strip().lower())
    cached = _LABEL_CACHE.get(key)
    if cached is not None and (cached[0] is not None or time.monotonic() - cached[1] < LABEL_RETRY_SECONDS):
        return cached[0]
    import numpy as np

    from .embedders import get_embedder
    from .umls.cache import get_topic_index

    index = get_topic_index(cache_dir)
    index.check_compatible(model)
    vec = np.array(get_embedder(model)(label.replace("-", " ")), dtype=np.float32)
    index.check_compatible(model, dim=int(vec.shape[-1]))
    hit = None
    for t, sim in index.search(vec, top_k=1, min_similarity=LABEL_MIN_SIMILARITY):
        if t.get("id"):
            hit = {"id": t["id"], "display_name": (t.get("display_name") or "").strip() or t["id"],
                   "similarity": round(float(sim), 4)}
        break
    _LABEL_CACHE[key] = (hit, time.monotonic())
    return hit


def with_labels(view: dict, *, settings: Any) -> dict:
    """``view`` plus the topics its text labels map to (``method = "label"``).

    Best effort: a label that maps nowhere, or an index or embedder that is
    not available, leaves the view as it was. A mapped label joins the
    side it was on (``not_interests`` → ``unwanted``), carrying the label
    in ``label`` and the match's ``similarity``; a topic already on that
    side is not added again. A not-wanted label whose topic is on the
    wanted side as well is kept, not dropped: the two disagree, and
    ``not_interest_rows`` shows it as overlapping so the user settles it.
    A wanted label whose topic the profile rejects outright is left out.

    ``labels_unmapped`` on the result says the labels should have been
    mapped but could not be just now (the mapper failed, or is being left
    alone after a failure): the view is then only part of the profile, and
    an aggregation keeps what its last full read stored of it. With UMLS
    off there is nothing to map against, which is not an outage.
    ``unmatched_labels`` are the not-wanted labels that map to no topic
    (all of them with UMLS off; none is known while the mapper is down).
    """
    import time

    global _DOWN_UNTIL
    labels = bool(view.get("wanted_labels") or view.get("unwanted_labels"))
    # The topic-name index is part of the UMLS cache: without it there is
    # nothing to map against.
    if settings is None or not getattr(settings, "RADAR_UMLS_ENABLED", False):
        return {**view, "labels_unmapped": False, "unmatched_labels": list(view.get("unwanted_labels") or [])}
    if time.monotonic() < _DOWN_UNTIL:
        return {**view, "labels_unmapped": labels}
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in view.items()}
    out["labels_unmapped"] = False
    out["unmatched_labels"] = []
    on_side = {side: {t["id"] for t in out[side]} for side in ("wanted", "unwanted")}
    for side in ("wanted", "unwanted"):
        for label in out.get(f"{side}_labels") or []:
            try:
                hit = _nearest_topic(label, settings=settings)
            except Exception as exc:  # noqa: BLE001 -- the index or embedder is unavailable
                import structlog
                _DOWN_UNTIL = time.monotonic() + LABEL_RETRY_SECONDS
                structlog.get_logger("rag_lib.researcher_topics").warning(
                    "researcher_topics.label_unmapped", label=label, reason=str(exc)[:160],
                    retry_in_s=LABEL_RETRY_SECONDS)
                return {**view, "labels_unmapped": labels}
            if hit is None:
                if side == "unwanted":
                    out["unmatched_labels"].append(label)
                continue
            if hit["id"] in on_side[side]:
                continue
            if side == "wanted" and hit["id"] in on_side["unwanted"]:
                continue
            out[side].append({**hit, "weight": None, "method": "label", "label": label})
            on_side[side].add(hit["id"])
    return out
