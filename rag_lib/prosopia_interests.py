"""Prosopia profiles' interests on an interest's topic list (FUNCTIONAL_SPEC M.3).

An interest keeps a copy of each Prosopia profile its papers were imported
from (``profiles.prosopia_docs_json``: ``[{"key", "name", "document"}]``),
whether or not a researcher profile was saved. What each profile says it is
interested in joins the topic list, marked with the profiles it comes from:

- a topic the seeds bring already is only marked; its switch is as it was;
- one they do not is added (``source = "prosopia"``), switched off.

Radar suggests nothing about which to search: the mark says where a topic
comes from, and the user switches it on or off as any other. The list is
rebuilt from the seeds every time, so :func:`apply` runs on every rebuild,
from the copies as stored, and an Edit interest copy carries them.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from . import researcher_topics

#: ``source`` of a topic only a Prosopia interest brought (no seed carries it).
SOURCE = "prosopia"
#: On a listed topic: the Prosopia profiles that list it as an interest, ``[{"profile"}]``.
MARK = "prosopia"


def docs(raw: str | None) -> list[dict]:
    try:
        got = json.loads(raw) if raw else []
    except (ValueError, TypeError):
        return []
    return [d for d in got if isinstance(d, dict) and d.get("key")] if isinstance(got, list) else []


def keep_doc(conn: sqlite3.Connection, profile_id: int, *, key: str, name: str, document: dict | None) -> None:
    """Keep (or refresh) profile ``key``'s copy on interest ``profile_id``. Committed."""
    row = conn.execute("SELECT prosopia_docs_json FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    if row is None:
        return
    kept = [d for d in docs(row[0]) if d["key"] != key]
    kept.append({"key": key, "name": name or key, "document": document})
    conn.execute("UPDATE profiles SET prosopia_docs_json = ? WHERE id = ?", (json.dumps(kept), profile_id))
    conn.commit()


def stated(document: dict | None) -> dict:
    """What the profile itself says it is interested in: its ``interests``,
    in ``researcher_topics.read``'s shape (``wanted`` / ``wanted_labels``).
    Not the topics Prosopia infers from the person's papers
    (``research_interests`` with ``method = "inferred"``): those are not
    the person's word, and listed they swamped the stated few.

    A label names its topic where an entry of ``research_interests`` gives
    that topic under the same name; the rest are left as labels, matched
    to their nearest topic by ``with_labels``. A document without the
    field falls back to its declared, wanted entries."""
    view = researcher_topics.read(document)
    empty = {"wanted": [], "unwanted": [], "wanted_labels": [], "unwanted_labels": []}
    if not isinstance(document, dict):
        return empty
    labels = researcher_topics._strings(document.get("interests"))
    if not labels:
        return {**empty, "wanted": [t for t in view["wanted"] if t.get("method") == "declared"]}
    by_name: dict[str, str] = {}
    for entry in document.get("research_interests") or []:
        concept = entry.get("concept") if isinstance(entry, dict) and isinstance(entry.get("concept"), dict) else {}
        tid = researcher_topics._topic_id(concept)
        name = researcher_topics._label(concept)
        if tid and name:
            by_name.setdefault(name.lower(), tid)
    out = dict(empty)
    out["wanted"], out["wanted_labels"] = [], []
    for label in labels:
        tid = by_name.get(label.lower())
        if tid is None:
            out["wanted_labels"].append(label)
        elif all(t["id"] != tid for t in out["wanted"]):
            out["wanted"].append({"id": tid, "display_name": label, "weight": None, "method": "declared"})
    return out


def interests(raw_docs: str | None, *, settings: Any) -> list[dict]:
    """Every topic each kept profile is interested in, as
    ``{"profile", "profile_name", "id", "name", "label"}``; a text label is
    matched to its nearest topic as the researcher's profile topics are
    (``researcher_topics.with_labels``)."""
    out: list[dict] = []
    for d in docs(raw_docs):
        view = researcher_topics.with_labels(
            stated(d.get("document") if isinstance(d.get("document"), dict) else None), settings=settings)
        for t in view.get("wanted") or []:
            out.append({"profile": d["key"], "profile_name": d.get("name") or d["key"],
                        "id": t["id"], "name": t.get("display_name") or t["id"], "label": t.get("label")})
    return out


def apply(topic_filters: dict, prior_on: dict[str, bool], listed: list[dict]) -> dict:
    """``topic_filters`` with the Prosopia interests ``listed`` (from
    :func:`interests`) on it; a new dict.

    A topic the list has is marked with each profile that lists it; one it
    has not is appended (``source = "prosopia"``), off unless the user
    switched it on before. Nothing already listed is switched."""
    code = researcher_topics.code
    out = dict(topic_filters)
    topics = [{k: v for k, v in t.items() if k != MARK} for t in out.get("topics") or []]
    by_code = {code(t["id"]): t for t in topics if t.get("id")}
    # A topic only a Prosopia interest brings is written as the list writes
    # its ids (OpenAlex URLs from OpenAlex; a bare code where the list has
    # those): one id per topic.
    bare = bool(topics) and all("/" not in str(t.get("id")) for t in topics if t.get("id"))
    for i in listed:
        c = code(i["id"])
        t = by_code.get(c)
        if t is None:
            tid = c if bare else f"https://openalex.org/{c}"
            t = {"id": tid, "display_name": i.get("name") or c, "count": 0, "papers": 0,
                 "source": SOURCE, "on": prior_on.get(tid, False)}
            topics.append(t)
            by_code[c] = t
        name = i.get("profile_name") or i.get("profile")
        if all(m.get("profile") != name for m in t.get(MARK) or []):
            t.setdefault(MARK, []).append({"profile": name})
    out["topics"] = topics
    return out
