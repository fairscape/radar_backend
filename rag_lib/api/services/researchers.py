"""The Researchers library: keep an imported ``profile.jsonld`` whole.

Unlike the wizard's "From Profile" path (which takes the ORCID and the
expertise / not_interests signal and builds a topic), this service stores
the document verbatim and a parsed, display-ready view of *every* section
the Researcher Profile spec defines — identity, summary, expertise,
not_interests, collaborators, training, career, career_stage, paper_stats,
capability flags, and a summary of the file manifest (what the profile
contains, by role and visibility). Nothing is fetched and nothing here
feeds recommendation.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import structlog

from ...db.repos import researchers as repo
from . import rp_profile_import as rp

log = structlog.get_logger(__name__)

PERSONA_ROLES = ("expertise", "soul", "topics")
CAPABILITY_FLAGS = ("hasCitationGraph", "hasEmbeddingIndex", "expertiseCitesPaperIds")


def _as_list_of_dicts(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, dict)]


def _manifest_summary(doc: dict) -> dict:
    """What the profile says it contains, without fetching anything."""
    entries = _as_list_of_dicts(doc.get("hasPart")) + _as_list_of_dicts(doc.get("subjectOf"))
    by_role: dict[str, int] = {}
    by_visibility: dict[str, int] = {}
    bytes_total = 0
    paper_ids: list[str] = []
    seen_ids: set[str] = set()
    persona: list[str] = []
    restricted_roles: set[str] = set()
    for e in entries:
        role = str(e.get("role") or "other")
        vis = str(e.get("visibility") or "public")
        by_role[role] = by_role.get(role, 0) + 1
        by_visibility[vis] = by_visibility.get(vis, 0) + 1
        try:
            bytes_total += int(e.get("bytes") or 0)
        except (TypeError, ValueError):
            pass
        pid = e.get("paperId")
        if isinstance(pid, str) and pid and pid not in seen_ids:
            seen_ids.add(pid)
            paper_ids.append(pid)
        if role in PERSONA_ROLES and role not in persona:
            persona.append(role)
        if vis == "restricted":
            restricted_roles.add(role)
    return {
        "n_entries": len(entries),
        "by_role": by_role,
        "by_visibility": by_visibility,
        "bytes_total": bytes_total,
        "paper_ids": paper_ids,
        "persona_documents": persona,
        "restricted_roles": sorted(restricted_roles),
        "works_url": next((e.get("contentUrl") for e in entries if e.get("role") == "works"), None),
    }


def parse_for_library(text: str) -> tuple[rp.RpProfile, dict, str]:
    """Validate the document and build the library's parsed view.

    Returns ``(profile, parsed, doc_json)``; ``doc_json`` is the document
    re-serialised canonically (sorted keys) so re-imports of the same text
    compare equal. Raises :class:`rp.RpProfileError` on an unusable document.
    """
    profile = rp.parse_profile(text)          # validation + the base fields
    doc = json.loads(text)                     # parse_profile already proved this is a dict

    parsed: dict[str, Any] = profile.meta()    # name, rid, orcid, affiliation, field, summary, expertise, …
    parsed.pop("source", None)
    # The dedupe key. Any document that carries an ORCID (in ``rid`` or ``@id``,
    # bare or as an IRI) is keyed on the normalised ORCID, so the same person
    # imported from two differently-written documents lands on one row.
    parsed["rid"] = profile.orcid or profile.rid
    parsed["n_expertise"] = len(profile.expertise)
    parsed["n_not_interests"] = len(profile.not_interests)
    stats = profile.paper_stats or {}
    parsed["n_papers"] = int(stats.get("total") or 0) if isinstance(stats.get("total"), (int, float)) else None
    parsed["training"] = _as_list_of_dicts(doc.get("training"))
    parsed["career"] = _as_list_of_dicts(doc.get("career"))
    parsed["career_stage"] = doc.get("career_stage") if isinstance(doc.get("career_stage"), dict) else None
    parsed["license"] = doc.get("license") if isinstance(doc.get("license"), str) else None
    parsed["visibility"] = doc.get("visibility") if isinstance(doc.get("visibility"), str) else None
    parsed["capabilities"] = {k: bool(doc.get(k)) for k in CAPABILITY_FLAGS if k in doc}
    parsed["manifest"] = _manifest_summary(doc)
    parsed["warnings"] = list(profile.warnings)
    doc_json = json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return profile, parsed, doc_json


def import_researcher(
    conn: sqlite3.Connection, *, user_id: int, text: str, source_kind: str = "paste",
) -> tuple[sqlite3.Row, bool]:
    """Parse and store (or overwrite) one researcher for ``user_id``."""
    profile, parsed, doc_json = parse_for_library(text)
    rid = parsed["rid"]
    if not rid:
        # No rid and no ORCID: key on the name so the row is still addressable.
        rid = f"local:{rp.deslug(profile.name).replace(' ', '-').lower()}"
        parsed["rid"] = rid
        parsed.setdefault("warnings", []).append("profile has neither rid nor ORCID; keyed on the name")
    row = {
        "rid": rid,
        "orcid": profile.orcid,
        "name": profile.name,
        "affiliation": profile.affiliation,
        "field": profile.field,
        "level": profile.level,
        "provenance": profile.provenance,
        "date_modified": profile.date_modified,
        "source_kind": source_kind if source_kind in ("paste", "file", "url") else "paste",
        "source_url": None,
        "doc_json": doc_json,
        "parsed_json": json.dumps(parsed, ensure_ascii=False),
    }
    rid_id, created = repo.upsert(conn, user_id, row)
    log.info("researchers.imported", user_id=user_id, researcher_id=rid_id, rid=rid, created=created)
    stored = repo.get(conn, user_id, rid_id)
    assert stored is not None
    return stored, created


def list_researchers(conn: sqlite3.Connection, user_id: int) -> list[sqlite3.Row]:
    return repo.list_for_user(conn, user_id)


def get_researcher(conn: sqlite3.Connection, user_id: int, researcher_id: int) -> sqlite3.Row | None:
    return repo.get(conn, user_id, researcher_id)


def delete_researcher(conn: sqlite3.Connection, user_id: int, researcher_id: int) -> bool:
    return repo.delete(conn, user_id, researcher_id)


def summary_dict(row: sqlite3.Row) -> dict:
    """``ResearcherSummary`` fields from a list or full row."""
    keys = row.keys()
    parsed = None
    if "parsed_json" in keys and row["parsed_json"]:
        try:
            parsed = json.loads(row["parsed_json"])
        except json.JSONDecodeError:
            parsed = None

    def _n(col: str) -> int | None:
        if col in keys and row[col] is not None:
            return int(row[col])
        if parsed is not None and parsed.get(col) is not None:
            return int(parsed[col])
        return None

    return {
        "id": int(row["id"]),
        "rid": row["rid"],
        "orcid": row["orcid"],
        "name": row["name"],
        "affiliation": row["affiliation"],
        "field": row["field"],
        "level": row["level"],
        "provenance": row["provenance"],
        "date_modified": row["date_modified"],
        "source_kind": row["source_kind"],
        "imported_at": row["imported_at"],
        "updated_at": row["updated_at"],
        "n_expertise": _n("n_expertise") or 0,
        "n_not_interests": _n("n_not_interests") or 0,
        "n_papers": _n("n_papers"),
    }


def detail_dict(row: sqlite3.Row) -> dict:
    d = summary_dict(row)
    parsed = json.loads(row["parsed_json"]) if row["parsed_json"] else {}
    doc = json.loads(row["doc_json"]) if row["doc_json"] else {}
    d["parsed"] = parsed
    d["doc"] = doc
    d["warnings"] = list(parsed.get("warnings") or [])
    return d


__all__ = [
    "delete_researcher",
    "detail_dict",
    "get_researcher",
    "import_researcher",
    "list_researchers",
    "parse_for_library",
    "summary_dict",
]
