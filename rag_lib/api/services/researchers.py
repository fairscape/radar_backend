"""Researchers — stored people, and interests built from their papers.

A researcher is what an import leaves behind: the person (a Prosopia
profile or an ORCID), the metadata the source published about them, and
the papers the import resolved and embedded. Nothing here scores or
recommends; it is bookkeeping so that the expensive part of an import
(OpenAlex resolution, one embedding per paper) is paid once per person
rather than once per interest.

Two ways in, one job:

  :func:`record_prosopia_researcher` / :func:`record_orcid_researcher`
      write the person row. The wizard's own import routes call these
      too, so an interest built by import leaves the researcher behind.

  :func:`prepare_researcher_import` opens a plan for
      ``services.prosopia.run_import`` that targets only the researcher —
      no draft is created. The scheduler job and the status route are
      the ones the wizard already uses.

Building an interest from a stored researcher is synchronous
(:func:`create_interest`): the papers are already embedded, so it is a
draft row plus one ``profile_seeds`` row per chosen paper, and the
wizard can go straight to the coherence check.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import structlog

from ...db.repos import (
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    researchers as researchers_repo,
)
from ...openalex_client import OpenAlexClient
from ...prosopia_client import DEFAULT_BASE_URL
from ..mappers import profile_row_to_profile
from ..schemas import (
    Draft,
    Profile,
    Researcher,
    ResearcherDetail,
    ResearcherPaper,
)
from . import wizard as wizard_service
from .prosopia import ImportPlan, _resolve_profile, select_records
from .profiles import _saves_dismisses


log = structlog.get_logger("rag_lib.api.services.researchers")


SOURCES = ("prosopia", "orcid")


class ResearcherNotFound(LookupError):
    """No such researcher for this user."""


# ---------------------------------------------------------------------------
# Recording a person
# ---------------------------------------------------------------------------


def _read(profile: Any, attr: str, default: Any = None) -> Any:
    """One lazy SDK property, or ``default`` if it is missing or will not load.

    Everything but ``name`` and ``papers`` is optional on a Prosopia
    profile, and each is its own artifact read. A profile whose grants
    file is broken must still import.
    """
    try:
        value = getattr(profile, attr)
    except AttributeError:
        return default
    except Exception as exc:  # noqa: BLE001 — best effort, logged
        log.warning("researchers.read_skipped", attr=attr, reason=str(exc)[:160])
        return default
    return default if value is None else value


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", by_alias=True)
        elif isinstance(value, list):
            value = [
                v.model_dump(mode="json", by_alias=True) if hasattr(v, "model_dump") else v
                for v in value
            ]
        return json.dumps(value, default=str)
    except Exception as exc:  # noqa: BLE001
        log.warning("researchers.json_skipped", reason=str(exc)[:160])
        return None


def record_prosopia_researcher(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    slug: str,
    profile: Any,
    base_url: str | None,
) -> int:
    """Upsert the person behind a Prosopia profile; returns the row id."""
    rid = str(_read(profile, "rid", "") or "")
    orcid = rid if rid and not rid.lower().startswith("local:") else None
    metadata = _read(profile, "metadata")
    url = getattr(metadata, "url", None) if metadata is not None else None
    expertise = _read(profile, "expertise", "") or None
    soul = _read(profile, "soul", "") or None
    grants = _read(profile, "grants", None)
    return researchers_repo.upsert(
        conn,
        user_id=user_id,
        source="prosopia",
        key=slug,
        name=(str(_read(profile, "name", "") or slug)).strip() or slug,
        base_url=(base_url or DEFAULT_BASE_URL).rstrip("/"),
        orcid=orcid,
        affiliation=_read(profile, "affiliation", None) or None,
        url=url or f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/{slug}",
        expertise=str(expertise) if expertise else None,
        soul=str(soul) if soul else None,
        grants_json=_json_or_none(grants) if grants else None,
        document_json=_json_or_none(metadata),
    )


def record_orcid_researcher(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    orcid: str,
    name: str | None,
) -> int:
    """Upsert the person behind an ORCID; returns the row id."""
    return researchers_repo.upsert(
        conn,
        user_id=user_id,
        source="orcid",
        key=orcid,
        name=(name or f"ORCID {orcid}").strip(),
        orcid=orcid,
        url=f"https://orcid.org/{orcid}",
    )


# ---------------------------------------------------------------------------
# Importing into a researcher (no draft)
# ---------------------------------------------------------------------------


def prepare_researcher_import(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    user_id: int,
    source: str,
    ref: str,
    base_url: str | None = None,
    paper_ids: list[str] | None = None,
    embedding_model: str | None = None,
    prosopia_client: Any | None = None,
    openalex_client: Any | None = None,
) -> ImportPlan:
    """Read the person, write the row, return the plan for the job.

    ``source`` is ``prosopia`` (``ref`` a slug, profile URL or ORCID
    published there) or ``orcid`` (``ref`` an ORCID whose works OpenAlex
    lists). ``paper_ids`` narrows the import to a selection — Prosopia
    paper ids or OpenAlex work ids respectively; ``None`` imports
    everything the source has.

    Raises the same errors the two wizard imports raise (``ValueError``
    for bad input, ``ProfileNotFound`` / ``PapersNotFound`` /
    ``WorksNotFound`` for a miss, ``ProsopiaError`` or any other
    exception for upstream trouble), so the router maps them the same
    way.
    """
    from . import orcid as orcid_service

    model = embedding_model or settings.RADAR_DEFAULT_EMBEDDING_MODEL
    if source == "prosopia":
        slug, profile = _resolve_profile(ref, base_url, prosopia_client)
        records = select_records(profile, slug, paper_ids)
        researcher_id = record_prosopia_researcher(
            conn, user_id=user_id, slug=slug, profile=profile,
            base_url=base_url or DEFAULT_BASE_URL,
        )
        return ImportPlan(
            slug=slug, user_id=user_id, embedding_model=model,
            researcher_id=researcher_id, records=records, profile=profile,
        )
    if source == "orcid":
        oid = orcid_service.normalize_orcid(ref)
        if not oid:
            raise ValueError("orcid must be an ORCID iD such as 0000-0001-5643-4068")
        wanted: set[str] | None = None
        if paper_ids is not None:
            wanted = {orcid_service._strip_openalex(i) for i in paper_ids if (i or "").strip()}
            if not wanted:
                raise ValueError("select at least one work to import")
        client = openalex_client or OpenAlexClient(mailto=settings.RADAR_DEFAULT_MAILTO)
        listing = orcid_service.list_works(oid, client)
        records = orcid_service.records_for(listing, wanted)
        if wanted is not None and not records:
            raise orcid_service.WorksNotFound(
                f"none of the selected works are by ORCID {oid}"
            )
        researcher_id = record_orcid_researcher(
            conn, user_id=user_id, orcid=oid, name=listing["name"],
        )
        return ImportPlan(
            slug=oid, user_id=user_id, embedding_model=model,
            researcher_id=researcher_id, records=records, profile=None,
        )
    raise ValueError(f"source must be one of {', '.join(SOURCES)}")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _to_researcher(conn: sqlite3.Connection, row: sqlite3.Row) -> Researcher:
    n_interests = conn.execute(
        "SELECT COUNT(*) FROM profiles WHERE researcher_id = ?", (int(row["id"]),),
    ).fetchone()[0]
    last_run = None
    if row["last_run_id"] is not None:
        last_run = gather_runs_repo.get(conn, int(row["last_run_id"]))
    importing = bool(last_run is not None and not last_run["finished_at"])
    return Researcher(
        id=int(row["id"]),
        source=row["source"],
        key=row["key"],
        name=row["name"],
        orcid=row["orcid"],
        affiliation=row["affiliation"],
        url=row["url"],
        base_url=row["base_url"],
        n_papers=int(row["n_papers"] or 0),
        n_interests=int(n_interests),
        imported_at=row["imported_at"],
        last_run_id=int(row["last_run_id"]) if row["last_run_id"] is not None else None,
        importing=importing,
        last_error=(last_run["error"] if last_run is not None else None),
        created_at=row["created_at"],
    )


def _to_paper(row: sqlite3.Row) -> ResearcherPaper:
    return ResearcherPaper(
        id=row["openalex_id"],
        title=row["title"] or "",
        year=int(row["year"]) if row["year"] is not None else None,
        venue=row["venue"],
        doi=row["doi"],
        authors=papers_repo.decode_authors(row),
        abstract=(row["abstract"] or "")[:600] or None,
        resolved_by=row["resolved_by"],
        summary=row["summary"],
        pdf_url=row["pdf_url"],
        added_at=row["added_at"],
    )


def _to_profile(conn: sqlite3.Connection, row: sqlite3.Row) -> Profile:
    saves30, dismisses30 = _saves_dismisses(conn, int(row["id"]))
    return profile_row_to_profile(row, saves30=saves30, dismisses30=dismisses30)


def list_researchers(conn: sqlite3.Connection, user_id: int) -> list[Researcher]:
    return [_to_researcher(conn, r) for r in researchers_repo.list_for_user(conn, user_id)]


def _require(conn: sqlite3.Connection, user_id: int, researcher_id: int) -> sqlite3.Row:
    row = researchers_repo.get_for_user(conn, user_id, researcher_id)
    if row is None:
        raise ResearcherNotFound(f"researcher {researcher_id} not found")
    return row


def get_researcher_row(
    conn: sqlite3.Connection, user_id: int, researcher_id: int
) -> sqlite3.Row:
    """The raw row, or ``ResearcherNotFound``. For routes that only need ownership."""
    return _require(conn, user_id, researcher_id)


def get_researcher(
    conn: sqlite3.Connection, user_id: int, researcher_id: int
) -> ResearcherDetail:
    row = _require(conn, user_id, researcher_id)
    grants: list[dict[str, Any]] = []
    if row["grants_json"]:
        try:
            loaded = json.loads(row["grants_json"])
            grants = [g for g in loaded if isinstance(g, dict)] if isinstance(loaded, list) else []
        except ValueError:
            grants = []
    return ResearcherDetail(
        researcher=_to_researcher(conn, row),
        expertise=row["expertise"],
        soul=row["soul"],
        grants=grants,
        papers=[_to_paper(p) for p in researchers_repo.list_papers(conn, researcher_id)],
        interests=[_to_profile(conn, p) for p in researchers_repo.interests_for(conn, researcher_id)],
    )


def delete_researcher(
    conn: sqlite3.Connection, user_id: int, researcher_id: int
) -> None:
    """Forget the person. Their papers stay in the vault, and any interest
    built from them keeps its seeds; it just no longer points back."""
    _require(conn, user_id, researcher_id)
    researchers_repo.delete(conn, researcher_id)


# ---------------------------------------------------------------------------
# Building an interest
# ---------------------------------------------------------------------------


def create_interest(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    user_id: int,
    researcher_id: int,
    name: str,
    openalex_ids: list[str] | None = None,
    embedding_model: str | None = None,
) -> tuple[Draft, int]:
    """A draft seeded with the researcher's papers, ready for step 2.

    ``openalex_ids`` picks a subset; ``None`` takes every paper. Returns
    the draft handle and the number of seeds attached. Raises
    ``ValueError`` for an empty name or a selection with nothing on the
    researcher, ``ResearcherNotFound`` for a bad id.
    """
    _require(conn, user_id, researcher_id)
    if not name.strip():
        raise ValueError("interest name must be non-empty")
    have = researchers_repo.paper_ids(conn, researcher_id)
    if openalex_ids is None:
        chosen = sorted(have)
    else:
        wanted = [i.strip() for i in openalex_ids if (i or "").strip()]
        if not wanted:
            raise ValueError("select at least one paper")
        chosen = [i for i in wanted if i in have]
        if not chosen:
            raise ValueError("none of the selected papers belong to this researcher")

    model = embedding_model or settings.RADAR_DEFAULT_EMBEDDING_MODEL
    draft = wizard_service.create_draft(
        conn,
        user_id=user_id,
        name=name.strip(),
        embedding_model=model,
        selector=settings.RADAR_DEFAULT_SELECTOR,
    )
    draft_row = profiles_repo.get_by_slug(conn, user_id, draft["slug"])
    if draft_row is None:  # pragma: no cover
        raise RuntimeError(f"draft '{draft['slug']}' vanished after create")
    profile_id = int(draft_row["id"])
    researchers_repo.set_profile_researcher(conn, profile_id, researcher_id)
    n = 0
    for openalex_id in chosen:
        n += profiles_repo.attach_seed(conn, profile_id, openalex_id)
    log.info(
        "researchers.create_interest",
        researcher_id=researcher_id, draft_slug=draft["slug"], n_seeds=n,
    )
    return Draft(**draft), n


def attach_seeds(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    slug: str,
    openalex_ids: list[str],
) -> tuple[int, list[str]]:
    """Add already-stored papers to a draft as seeds.

    Only papers the user owns (uploaded, imported with a researcher, or
    already a seed elsewhere) are attached; the rest are returned as
    ``rejected``. Raises ``LookupError`` when the draft does not exist.
    """
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None or not row["is_draft"]:
        raise LookupError(f"draft '{slug}' not found for current user")
    wanted = [i.strip() for i in openalex_ids if (i or "").strip()]
    if not wanted:
        raise ValueError("select at least one paper")
    owned = researchers_repo.user_owns_papers(conn, user_id, wanted)
    attached = 0
    rejected: list[str] = []
    for openalex_id in wanted:
        if openalex_id in owned:
            attached += profiles_repo.attach_seed(conn, int(row["id"]), openalex_id)
        else:
            rejected.append(openalex_id)
    return attached, rejected
