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
    users as users_repo,
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
    ResearcherTopics,
)
from ... import researcher_topics
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
    rid = str(_read(profile, "rid", "") or "").strip()
    # Upper-cased as the ORCID import stores it: the in-flight check and
    # the dedup of one person under both sources compare this column.
    orcid = rid.upper() if rid and not rid.lower().startswith("local:") else None
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


class ResearcherImportRunning(RuntimeError):
    """This person's papers are already being imported for this user.

    Carries the run to watch. Two imports of one ORCID at once embedded
    the same papers twice and raced on the researcher's paper list.
    """

    def __init__(self, researcher_id: int, run_id: int):
        super().__init__("an import of this researcher is already running")
        self.researcher_id = researcher_id
        self.run_id = run_id


def _refuse_if_importing(
    conn: sqlite3.Connection, user_id: int, *,
    researcher_id: int | None = None, orcid: str | None = None,
) -> None:
    row = gather_runs_repo.find_inflight_researcher_import(
        conn, user_id, researcher_id=researcher_id, orcid=orcid,
    )
    if row is not None:
        raise ResearcherImportRunning(int(row["researcher_id"]), int(row["run_id"]))


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
        existing = researchers_repo.get_by_key(conn, user_id, "prosopia", slug)
        rid = str(_read(profile, "rid", "") or "")
        _refuse_if_importing(
            conn, user_id,
            researcher_id=int(existing["id"]) if existing is not None else None,
            orcid=rid.upper() if rid and not rid.lower().startswith("local:") else None,
        )
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
        _refuse_if_importing(conn, user_id, orcid=oid)
        wanted: set[str] | None = None
        if paper_ids is not None:
            wanted = {orcid_service._strip_openalex(i) for i in paper_ids if (i or "").strip()}
            if not wanted:
                raise ValueError("select at least one work to import")
        client = openalex_client or OpenAlexClient(mailto=users_repo.contact_mailto(conn, user_id, settings.RADAR_DEFAULT_MAILTO))
        # No registry call: this path builds an import plan and never
        # looks at claimed / duplicate_of. See orcid.prepare_import.
        listing = orcid_service.list_works(oid, client, check_registry=False)
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
    # The staleness cutoff as everywhere else: a run left open by a dead
    # thread (closed at the next boot) showed "Importing" with Forget and
    # Check-for-new-papers disabled until then.
    importing = bool(last_run is not None and not last_run["finished_at"]
                     and gather_runs_repo.is_inflight(conn, int(last_run["id"])))
    # Counted, not the stored column: that is written only when an import
    # finishes cleanly, so one that died half-way read "0 papers" and the
    # New-interest button was disabled over papers that were stored.
    n_papers = conn.execute(
        "SELECT COUNT(*) FROM researcher_papers WHERE researcher_id = ?", (int(row["id"]),),
    ).fetchone()[0]
    return Researcher(
        id=int(row["id"]),
        source=row["source"],
        key=row["key"],
        name=row["name"],
        orcid=row["orcid"],
        affiliation=row["affiliation"],
        url=row["url"],
        base_url=row["base_url"],
        n_papers=int(n_papers),
        n_interests=int(n_interests),
        imported_at=row["imported_at"],
        last_run_id=int(row["last_run_id"]) if row["last_run_id"] is not None else None,
        importing=importing,
        last_error=(last_run["error"] if last_run is not None else None),
        created_at=row["created_at"],
        mark=row["mark"],
        position=int(row["position"]) if row["position"] is not None else None,
    )


def _topics(row: sqlite3.Row) -> ResearcherTopics:
    """What the researcher's stored profile document says they want and do
    not want (rag_lib/researcher_topics.py); empty for an ORCID-only one."""
    from ..settings import get_settings
    view = researcher_topics.with_labels(researcher_topics.from_row(row), settings=get_settings())
    return ResearcherTopics(available=researcher_topics.available(view), **researcher_topics.public(view))


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


def repairable_excluded(saved: list) -> list:
    """The researcher's papers minus those a refresh should offer again: a
    work stored under its OpenAlex id but never read from OpenAlex (the
    lookup failed at import, ``resolved_by`` none). Importing it once more
    fills the row in. A record OpenAlex never held (synthetic id) stays."""
    return [r for r in saved
            if not (r["resolved_by"] == "none" and not str(r["openalex_id"]).startswith("prosopia:"))]


def works_for_refresh(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    user_id: int,
    researcher_id: int,
    prosopia_client: Any | None = None,
    openalex_client: Any | None = None,
) -> dict[str, Any]:
    """Read the researcher's source again and say which works are already saved.

    For "check for new papers": the full re-import re-imported everything
    the source lists -- bringing back papers deselected the first time
    (duplicates, datasets) and resolving every paper again against
    OpenAlex. This lists, the page offers only the new ones, and the
    import that follows carries just the picked ``paper_ids``.

    A work counts as saved when its id matches (OpenAlex id for ORCID, the
    profile's paper id for Prosopia) or, failing that, its DOI does --
    the same paper can come back under a new OpenAlex id after a merge.
    """
    from . import orcid as orcid_service
    from . import prosopia as prosopia_service
    from ...prosopia_seeds import normalize_doi

    row = _require(conn, user_id, researcher_id)
    saved = repairable_excluded(researchers_repo.list_papers(conn, researcher_id))
    saved_oa = {orcid_service._strip_openalex(r["openalex_id"]) for r in saved}
    saved_pid = {r["paper_id"] for r in saved if r["paper_id"]}
    saved_doi = {d for d in (normalize_doi(r["doi"]) for r in saved) if d}

    if row["source"] == "orcid":
        client = openalex_client or OpenAlexClient(mailto=users_repo.contact_mailto(conn, user_id, settings.RADAR_DEFAULT_MAILTO))
        listing = orcid_service.list_works(row["key"], client)
        works = listing["works"]
        known = {
            w["openalex_id"] for w in works
            if orcid_service._strip_openalex(w["openalex_id"]) in saved_oa
            or w["openalex_id"] in saved_pid
            or (normalize_doi(w.get("doi")) or "") in saved_doi
        }
        name = listing.get("name")
    else:
        listing = prosopia_service.list_works(
            row["key"], base_url=row["base_url"] or DEFAULT_BASE_URL,
            prosopia_client=prosopia_client,
        )
        works = listing["works"]
        known = {
            w["id"] for w in works
            if w["id"] in saved_pid
            or (w.get("openalex_id") and orcid_service._strip_openalex(w["openalex_id"]) in saved_oa)
            or (normalize_doi(w.get("doi")) or "") in saved_doi
        }
        name = listing.get("name")
    known.discard("")
    return {
        "source": row["source"], "key": row["key"], "name": name or row["name"],
        "works": works, "saved_ids": sorted(known),
    }


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
        topics=_topics(row),
    )


def set_mark(
    conn: sqlite3.Connection, user_id: int, researcher_id: int, mark: str | None
) -> Researcher:
    """Mark the person as the user (``self``), as followed, or neither; they
    move to the top of that group (see the repo). ``ResearcherNotFound``
    for another user's or an unknown researcher, ``ValueError`` for a mark
    that is neither."""
    _require(conn, user_id, researcher_id)
    try:
        researchers_repo.set_mark(conn, user_id, researcher_id, mark)
    except LookupError as exc:   # forgotten between the check and the write
        raise ResearcherNotFound(str(exc)) from exc
    row = researchers_repo.get(conn, researcher_id)
    if row is None:
        raise ResearcherNotFound(f"researcher {researcher_id} not found")
    return _to_researcher(conn, row)


def reorder(conn: sqlite3.Connection, user_id: int, ids: list[int]) -> list[Researcher]:
    """Store the order the user arranged their researchers in (every id,
    once; ``ValueError`` otherwise) and return the list in that order."""
    researchers_repo.reorder(conn, user_id, [int(i) for i in ids])
    return list_researchers(conn, user_id)


def delete_researcher(
    conn: sqlite3.Connection, user_id: int, researcher_id: int
) -> None:
    """Forget the person. Any interest built from them keeps its seeds (and
    so those papers stay in the vault); it just no longer points back.
    Their other papers leave the vault with them."""
    _require(conn, user_id, researcher_id)
    # The import's run row hangs off the researcher: forgetting them mid-
    # import deleted it, the job then wrote to nothing, and the draft it was
    # filling lost its "still importing" state and could be committed half
    # done.
    _refuse_if_importing(conn, user_id, researcher_id=researcher_id)
    interests = [(int(p["id"]), not p["is_draft"]) for p in researchers_repo.interests_for(conn, researcher_id)]
    researchers_repo.delete(conn, researcher_id)
    # Their profile topics, where applied, go with them: the saved list is
    # what every scan reads, and with the researcher gone nothing else
    # would rebuild it (the box that offers "Remove" needs the researcher).
    # After the delete, so a delete that fails changes no list; a drop that
    # fails is met by the next aggregation, which finds the researcher gone.
    for profile_id, live in interests:
        wizard_service.drop_profile_topics(conn, profile_id, live=live)


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
    # The draft would be seeded with what has arrived so far, have no import
    # run of its own to wait for, and the rest would go to the researcher
    # only. The stored paper count was 0 until the import finished, which
    # hid this; the count is live now.
    _refuse_if_importing(conn, user_id, researcher_id=researcher_id)
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
    # As attach_seeds: the papers were embedded when imported, under the
    # model of that day; a draft under another model needs its own vectors.
    # Before the draft exists: an embedder failure then is a plain error,
    # not a half-built draft that a retry would duplicate as "Name (2)".
    wizard_service.embed_missing(conn, model, list(chosen))
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
) -> tuple[int, list[str], int | None]:
    """Add already-stored papers to a draft or a live profile as seeds.

    Returns ``(attached, rejected, rescored)``: ``rescored`` is how many
    stored candidates a live interest's refit re-scored, None for a draft.

    Only papers the user owns (uploaded, imported with a researcher, a
    seed elsewhere, or a candidate in one of their interests) are
    attached; the rest are returned as ``rejected``. Raises
    ``LookupError`` when the profile does not exist.

    Any attached paper without a vector is embedded here: the fit leaves
    out a seed it cannot place, and a paper taken from the feed has none.
    A live profile is then re-fitted, so the seeds it scores against are
    the seeds it shows.
    """
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None:
        raise LookupError(f"interest '{slug}' not found for current user")
    if gather_runs_repo.has_unfinished_import(conn, int(row["id"])):
        # As coherence, the trial and commit: the import is filling the seeds.
        from .wizard import ImportInProgress
        raise ImportInProgress(slug)
    wanted = [i.strip() for i in openalex_ids if (i or "").strip()]
    if not wanted:
        raise ValueError("select at least one paper")
    owned = researchers_repo.user_owns_papers(conn, user_id, wanted)
    attached = 0
    rejected: list[str] = []
    accepted: list[str] = []
    try:
        for openalex_id in wanted:
            if openalex_id in owned:
                attached += profiles_repo.attach_seed(conn, int(row["id"]), openalex_id)
                accepted.append(openalex_id)
            else:
                rejected.append(openalex_id)
    except sqlite3.IntegrityError as exc:
        # Deleted in another tab since it was read: the seed row's foreign
        # key fails (OR IGNORE does not cover that). A 404, not a 500.
        conn.rollback()
        if profiles_repo.get_by_slug(conn, user_id, slug) is not None:
            raise
        raise LookupError(f"interest '{slug}' not found for current user") from exc
    if accepted:
        # Over every accepted id, not just the newly attached ones: the
        # seed row commits before this runs, so if embedding or the refit
        # failed, a retry found the seed "already attached", did nothing,
        # and the interest kept scoring against the old centroid. Both
        # steps are idempotent, so running them again is the repair.
        from . import wizard as wizard_service

        wizard_service.embed_missing(conn, row["embedding_model"], accepted)
        if not row["is_draft"]:
            try:
                fit = wizard_service.refit_profile(conn, user_id=user_id, slug=slug)
            except wizard_service.ScanRunning as exc:
                # The seeds are attached; the running scan's rescore is on
                # the old centroid, and the next refit picks them up.
                log.warning("researchers.add_seeds.refit_skipped", slug=slug, reason=str(exc)[:160])
                return attached, rejected, None
            return attached, rejected, fit.get("n_rescored")
    return attached, rejected, None
