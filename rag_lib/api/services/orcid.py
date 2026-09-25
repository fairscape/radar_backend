"""ORCID import — a researcher's OpenAlex works become a draft's seeds.

The Prosopia import needs the researcher to have published a Prosopia
profile. Most have not, but nearly everyone with papers has an ORCID,
and OpenAlex indexes works by it. This path asks OpenAlex for the
author's works, lets the caller choose which ones to keep, and then
hands the chosen set to the same import machinery the Prosopia path
uses — :func:`services.prosopia.run_import` — so a seed that came in by
ORCID is the same kind of row as one from Prosopia or a PDF.

Two calls, because the choice belongs to the user:

  :func:`list_works`     one OpenAlex query (``author.orcid:``), returned
                         as a slim list the UI can render as checkboxes.
  :func:`prepare_import` re-reads that list, keeps the ids the caller
                         picked, opens the draft and returns the
                         ``ImportPlan`` the background job consumes.

The author's name comes from the authorship that carries the ORCID on
the works themselves, so there is no second request to ``/authors``.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

import structlog

from ...db.repos import profiles as profiles_repo
from ...db.repos import researchers as researchers_repo
from ...openalex_client import OpenAlexClient
from ...prosopia_seeds import normalize_doi
from . import wizard as wizard_service
from .prosopia import ImportPlan


log = structlog.get_logger("rag_lib.api.services.orcid")


__all__ = [
    "MAX_WORKS",
    "WorkRecord",
    "WorksNotFound",
    "list_works",
    "normalize_orcid",
    "prepare_import",
    "records_for",
]


# OpenAlex pages 200 at a time; five pages covers all but the most
# prolific authors, and a seed set beyond that is not a seed set.
MAX_WORKS = 1000

ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", re.IGNORECASE)
_ORCID_PREFIX = re.compile(r"^https?://(www\.)?orcid\.org/", re.IGNORECASE)
_OPENALEX_PREFIX = "https://openalex.org/"


class WorksNotFound(LookupError):
    """None of the requested works belong to this ORCID."""


def normalize_orcid(raw: str | None) -> str | None:
    """Bare upper-case ORCID from an id or an ``orcid.org`` URL; else None."""
    value = (raw or "").strip()
    value = _ORCID_PREFIX.sub("", value).strip("/").strip()
    if not ORCID_RE.match(value):
        return None
    return value.upper()


@dataclass
class WorkRecord:
    """The record shape :func:`prosopia_seeds.resolve_seed` reads.

    A stand-in for the SDK's ``PaperRecord`` carrying only what the
    resolution ladder and the embedding fallback touch. Built from a
    work OpenAlex already handed us, so the ladder's first rung
    (``work_id``) is what catches it.
    """

    paper_id: str
    name: str
    openalex_id: str
    doi: str | None = None
    pmcid: str | None = None
    year: int | None = None
    journal: str | None = None
    venue: str | None = None
    summary: str | None = None
    abstract: str | None = None
    pdf_url: str | None = None
    full_text_link: str | None = None


def _strip_openalex(value: str | None) -> str:
    raw = (value or "").strip()
    if raw.lower().startswith(_OPENALEX_PREFIX):
        raw = raw[len(_OPENALEX_PREFIX):]
    return raw.rsplit("/", 1)[-1].upper()


def _authorship_for(work: dict, orcid: str) -> dict | None:
    for entry in work.get("authorships") or []:
        author = entry.get("author") or {}
        if (author.get("orcid") or "").upper().endswith(orcid):
            return entry
    return None


def _author_name(works: list[dict], orcid: str) -> str | None:
    for work in works:
        entry = _authorship_for(work, orcid)
        name = ((entry or {}).get("author") or {}).get("display_name")
        if name:
            return str(name)
    return None


def _summary(work: dict, orcid: str) -> dict[str, Any]:
    authorships = work.get("authorships") or []
    names = [
        (a.get("author") or {}).get("display_name")
        for a in authorships
    ]
    names = [n for n in names if n]
    mine = _authorship_for(work, orcid) or {}
    venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    return {
        "openalex_id": _strip_openalex(work.get("id")),
        "doi": normalize_doi(work.get("doi")),
        "title": (work.get("title") or work.get("display_name") or "").strip(),
        "year": work.get("publication_year"),
        "venue": venue,
        "type": work.get("type"),
        "cited_by_count": work.get("cited_by_count"),
        "authors": names[:3],
        "n_authors": len(names) or None,
        "author_position": mine.get("author_position"),
    }


def list_works(orcid: str, client: Any, *, limit: int = MAX_WORKS) -> dict[str, Any]:
    """Every OpenAlex work carrying ``orcid`` in its authorships.

    Newest first, then most cited, so the list a user is about to prune
    starts with what they most likely still care about. ``client`` only
    needs ``paginate_filter``.
    """
    works = client.paginate_filter(f"author.orcid:{orcid}", limit=limit) or []
    summaries = [_summary(w, orcid) for w in works]
    summaries = [s for s in summaries if s["openalex_id"] and s["title"]]
    summaries.sort(
        key=lambda s: (-(s["year"] or 0), -(s["cited_by_count"] or 0), s["title"]),
    )
    log.info("orcid.list_works", orcid=orcid, n=len(summaries))
    return {"orcid": orcid, "name": _author_name(works, orcid), "works": summaries}


def prepare_import(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    user_id: int,
    orcid: str,
    openalex_ids: list[str],
    name: str | None = None,
    embedding_model: str | None = None,
    openalex_client: Any | None = None,
) -> ImportPlan:
    """Keep the chosen works, open the draft, return the job's plan.

    Raises ``ValueError`` for a malformed ORCID or an empty selection,
    and :class:`WorksNotFound` when none of the ids are the author's —
    a stale selection from a list that has since changed, or a caller
    guessing. The router maps those to 400 / 404.
    """
    oid = normalize_orcid(orcid)
    if not oid:
        raise ValueError("orcid must be an ORCID iD such as 0000-0001-5643-4068")
    wanted = {_strip_openalex(i) for i in openalex_ids if (i or "").strip()}
    if not wanted:
        raise ValueError("select at least one work to import")

    client = openalex_client or OpenAlexClient(mailto=settings.RADAR_DEFAULT_MAILTO)
    listing = list_works(oid, client)
    records = records_for(listing, wanted)
    if not records:
        raise WorksNotFound(f"none of the selected works are by ORCID {oid}")

    model = embedding_model or settings.RADAR_DEFAULT_EMBEDDING_MODEL
    # Record the researcher too: the works embedded for this draft are
    # then on hand for the next interest built from the same person.
    from . import researchers as researchers_service
    researcher_id = researchers_service.record_orcid_researcher(
        conn, user_id=user_id, orcid=oid, name=listing["name"],
    )
    draft_name = (name or listing["name"] or f"ORCID {oid}").strip()
    draft = wizard_service.create_draft(
        conn,
        user_id=user_id,
        name=draft_name,
        embedding_model=model,
        selector=settings.RADAR_DEFAULT_SELECTOR,
    )
    draft_row = profiles_repo.get_by_slug(conn, user_id, draft["slug"])
    if draft_row is None:  # pragma: no cover — create_draft just wrote it
        raise RuntimeError(f"draft '{draft['slug']}' vanished after create")
    researchers_repo.set_profile_researcher(conn, int(draft_row["id"]), researcher_id)

    log.info(
        "orcid.prepare_import",
        orcid=oid, n_selected=len(wanted), n_kept=len(records),
        draft_slug=draft["slug"],
    )
    return ImportPlan(
        slug=oid,
        draft_slug=draft["slug"],
        draft_name=draft["name"],
        profile_id=int(draft_row["id"]),
        researcher_id=researcher_id,
        user_id=user_id,
        embedding_model=model,
        records=records,
        profile=None,
    )


def records_for(listing: dict[str, Any], wanted: set[str] | None) -> list[WorkRecord]:
    """The listing's works as ladder records; ``wanted`` narrows to a selection."""
    chosen = [
        w for w in listing["works"]
        if wanted is None or w["openalex_id"] in wanted
    ]
    return [
        WorkRecord(
            paper_id=w["openalex_id"],
            name=w["title"],
            openalex_id=w["openalex_id"],
            doi=w["doi"],
            year=w["year"],
            venue=w["venue"],
        )
        for w in chosen
    ]
