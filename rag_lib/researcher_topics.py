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
"""

from __future__ import annotations

import json
import re
from typing import Any

SOURCE = "researcher"
INTEREST = "interest"
NOT_INTEREST = "not_interest"
FLAG = "researcher_topics"          # topic_filters key: the profile's topics are applied

_TOPIC_ID = re.compile(r"^(?:https://openalex\.org/)?(T\d+)$", re.IGNORECASE)


def _topic_id(concept: dict) -> str | None:
    raw = concept.get("@id") or concept.get("id") or concept.get("code") or ""
    m = _TOPIC_ID.match(str(raw).strip())
    return f"https://openalex.org/{m.group(1).upper()}" if m else None


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
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in value:
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
                "method": t.get("method") or "inferred", "label": t.get("label")}
    return {
        "wanted": [_t(t) for t in view.get("wanted") or []],
        "unwanted": [_t(t) for t in view.get("unwanted") or []],
        "wanted_labels": list(view.get("wanted_labels") or []),
        "unwanted_labels": list(view.get("unwanted_labels") or []),
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


def apply(topic_filters: dict, view: dict) -> dict:
    """``topic_filters`` with the profile's topics folded in; a new dict.

    Wanted topics the seeds did not bring in are appended with
    ``source = "researcher"`` and ``count = 0`` (no seed carries them).
    Every listed topic the profile has an opinion on gets
    ``profile = "interest" | "not_interest"``. On/off is not touched here:
    the caller settles it (see ``wizard._aggregate_topics``).
    """
    out = dict(topic_filters or {})
    topics = [dict(t) for t in out.get("topics") or []]
    # Matched on the bare code: the seeds' topics carry whatever id OpenAlex
    # gave them, the profile the full URL.
    by_code = {_code(t.get("id")): t for t in topics if t.get("id")}
    for t in topics:
        t.pop("profile", None)
    for item in view.get("wanted") or []:
        have = by_code.get(_code(item["id"]))
        if have is not None:
            have["profile"] = INTEREST
        else:
            new = {"id": item["id"], "display_name": item["display_name"], "count": 0,
                   "source": SOURCE, "profile": INTEREST}
            topics.append(new)
            by_code[_code(item["id"])] = new
    for item in view.get("unwanted") or []:
        have = by_code.get(_code(item["id"]))
        if have is not None:
            have["profile"] = NOT_INTEREST
    out["topics"] = topics
    out[FLAG] = True
    return out


def strip(topic_filters: dict) -> dict:
    """``topic_filters`` without the profile's topics and marks; a new dict."""
    out = dict(topic_filters or {})
    kept = []
    for t in out.get("topics") or []:
        if t.get("source") == SOURCE:
            continue
        t = dict(t)
        t.pop("profile", None)
        kept.append(t)
    out["topics"] = kept
    out[FLAG] = False
    return out


# --- text labels → topics -----------------------------------------------------

# A label is a concept the profile could not map to a topic itself
# ("phylogenomics", "wet-lab-protocol-development"). Its nearest OpenAlex
# topic name, by the same index the UMLS mapper and the PDF reader use, is
# close enough to act on only above this; a real topic name is usually
# above 0.7 against a paraphrase of itself.
LABEL_MIN_SIMILARITY = 0.6
_LABEL_CACHE: dict[str, dict | None] = {}


def _nearest_topic(label: str, *, settings: Any) -> dict | None:
    key = label.strip().lower()
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]
    import numpy as np

    from .embedders import get_embedder
    from .umls.cache import get_topic_index

    model = settings.RADAR_UMLS_EMBEDDING_MODEL
    index = get_topic_index(str(settings.RADAR_UMLS_CACHE_DIR))
    index.check_compatible(model)
    vec = np.array(get_embedder(model)(label.replace("-", " ")), dtype=np.float32)
    index.check_compatible(model, dim=int(vec.shape[-1]))
    hit = None
    for t, sim in index.search(vec, top_k=1, min_similarity=LABEL_MIN_SIMILARITY):
        if t.get("id"):
            hit = {"id": t["id"], "display_name": (t.get("display_name") or "").strip() or t["id"],
                   "similarity": round(float(sim), 4)}
        break
    _LABEL_CACHE[key] = hit
    return hit


def with_labels(view: dict, *, settings: Any) -> dict:
    """``view`` plus the topics its text labels map to (``method = "label"``).

    Best effort: a label that maps nowhere, or an index or embedder that is
    not available, leaves the view as it was. A mapped label joins the
    side it was on (``not_interests`` → ``unwanted``), carrying the label
    in ``label``; a topic the profile already asserts outright is not
    added again, and a topic a label would add to both sides is left out.
    """
    if settings is None or not getattr(settings, "RADAR_UMLS_ENABLED", False):
        return view
    out = {k: list(v) for k, v in view.items()}
    have = {t["id"] for t in (*out["wanted"], *out["unwanted"])}
    mapped: dict[str, tuple[str, dict]] = {}
    clashes: set[str] = set()
    for side in ("wanted", "unwanted"):
        for label in out.get(f"{side}_labels") or []:
            try:
                hit = _nearest_topic(label, settings=settings)
            except Exception as exc:  # noqa: BLE001 -- the index or embedder is unavailable
                import structlog
                structlog.get_logger("rag_lib.researcher_topics").warning(
                    "researcher_topics.label_unmapped", label=label, reason=str(exc)[:160])
                return view
            if hit is None or hit["id"] in have:
                continue
            if hit["id"] in mapped and mapped[hit["id"]][0] != side:
                clashes.add(hit["id"])
                continue
            mapped.setdefault(hit["id"], (side, {**hit, "weight": None, "method": "label", "label": label}))
    for tid, (side, item) in mapped.items():
        if tid not in clashes:
            out[side].append(item)
    return out
