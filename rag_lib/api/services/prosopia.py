"""Prosopia import service — a published profile becomes a draft's seeds.

RADAR's other route to a seed corpus is the vault: upload a PDF, extract
its text, resolve it through OpenAlex, embed, attach. That is one paper
per click. A researcher with a Prosopia profile has already published
the same corpus in machine-readable form, so importing it should cost
one call rather than eighty.

The shape of the work mirrors ``services.vault.upload`` deliberately —
resolve through OpenAlex, fall back to a synthetic id, upsert, embed,
attach — because the two paths must produce interchangeable rows. A seed
that came from Prosopia and a seed that came from a PDF are the same
kind of thing to coherence, topic aggregation, and the selector fit.

The work splits in two, and the split is the reason this reads the way
it does:

  :func:`prepare_import` is cheap and synchronous — one SDK read of the
  profile and one INSERT. It runs inside the request so a slug nobody
  has published is a 404 on the POST rather than an error the caller
  only discovers by polling a run row.

  :func:`run_import` is the expensive half — an OpenAlex round trip per
  batch and an embedding per paper, which for the reference profile is
  82 papers behind a cold SPECTER2 load. That cannot live in a request:
  it runs for minutes, and a gateway or a browser will give up on it
  long before it finishes. It runs on the scheduler against a
  ``gather_runs`` row, the same way the wizard's dry-run does, and
  reports progress through the same reporter duck.

What is different from a gather is the reporting. An import is a bulk
operation whose quality is invisible from the outside: eighty rows land
either way. The per-rung ``resolved_by`` counts are the diagnostic — a
profile whose papers all came in on ``work_id`` was imported; one
carried by the ``title`` rung was reconstructed, and its seeds deserve
a look.
"""

from __future__ import annotations

from ..schemas import MAX_NAME

import sqlite3
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import structlog

from ...db.repos import (
    draft_owned as draft_owned_repo,
    embeddings as embeddings_repo,
    paper_rows as paper_rows_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    researchers as researchers_repo,
    users as users_repo,
    versions as versions_repo,
)
from ...embed import build_embedding_input
from ...embedders import get_embedder
from ...openalex_client import OpenAlexClient
from ...paper import Paper
from ...prosopia_client import (
    DEFAULT_BASE_URL,
    ProfileNotFound,
    ProfileNotPublic,
    ProsopiaFormatChanged,
    ProsopiaClient,
    ProsopiaError,
    allowed_base_url,
    read_summary,
    looks_like_orcid,
    normalize_rid,
)
from ...prosopia_seeds import RUNGS, Person, _normalize_openalex_id, _paper_id, current_works, normalize_doi, prefetch_works, resolve_seed
from . import wizard as wizard_service


log = structlog.get_logger("rag_lib.api.services.prosopia")


__all__ = [
    "DEFAULT_BASE_URL",
    "ImportPlan",
    "PapersNotFound",
    "ProfileEmpty",
    "ProfileNotFound",
    "ProfileNotPublic",
    "ProsopiaFormatChanged",
    "ProsopiaError",
    "list_works",
    "prepare_import",
    "run_import",
    "normalize_ref",
    "select_records",
]


class PapersNotFound(LookupError):
    """None of the requested paper ids are on this profile."""


class ProfileEmpty(ProsopiaError):
    """The profile lists no papers: nothing to import, so nothing is
    created (FUNCTIONAL_SPEC O.2 C)."""

    def __init__(self, slug: str, *, refresh: bool = False):
        super().__init__(
            f"The Prosopia profile '{slug}' lists no papers any more, so there is nothing to add. Nothing was changed."
            if refresh else
            f"The Prosopia profile '{slug}' lists no papers, so there is nothing to import. Nothing was created.")


# How often the per-paper loop writes a progress tick. Every paper would
# be one COMMIT per embedding against the same SQLite file the API is
# serving from; every tenth keeps the poll responsive without turning
# the import into a write-lock generator.
_TICK_EVERY = 5


@dataclass
class ImportPlan:
    """Everything the background job needs, resolved inside the request.

    The SDK profile rides along because the job still needs its lazy
    ``summaries`` mapping for the papers that fail to resolve — and
    because re-fetching it in the job would mean a slug could 404 twice,
    once where the caller can see it and once where they cannot.

    Two targets, either optional: ``profile_id`` is the draft the papers
    become seeds of (the wizard's import), ``researcher_id`` the stored
    researcher they are recorded under. The wizard's import sets both,
    so an interest built by import also leaves the researcher behind
    for the next interest; a plain researcher import sets only the
    second.
    """

    slug: str
    user_id: int
    embedding_model: str
    draft_slug: str | None = None
    draft_name: str | None = None
    profile_id: int | None = None
    researcher_id: int | None = None
    records: list[Any] = field(default_factory=list)
    profile: Any = None
    # Made by the request that planned this import: an import that fails
    # removes them again (``undo_import``), so nothing half-built stays.
    new_draft: bool = False
    new_researcher: bool = False
    # Every work id the source offered for this import (the pick list), so
    # the ones left unticked are remembered (FUNCTIONAL_SPEC O.2 E). None:
    # nothing to remember (a caller that did not list).
    listed_ids: set[str] | None = None
    # Whose papers these are, for checking a title match (O.2 B).
    person: Any = None
    # What the seeds say they came from: the researcher they were saved
    # under, or the source itself when no researcher profile was saved
    # (FUNCTIONAL_SPEC M.2).
    seed_source: str = "researcher"
    # Whom a wizard import imports, recorded on its run (``gather_runs.
    # import_orcid`` / ``import_key``) so a second request finds it.
    import_orcid: str | None = None
    import_key: str | None = None
    # The Prosopia profile's interests for the draft (FUNCTIONAL_SPEC M.3),
    # ``{"key", "name", "document"}``: written only once the import has
    # stored its seeds (CR.7). Written while planning, a failed import left
    # them on the draft or Edit copy, and saving the copy kept them.
    prosopia_doc: dict | None = None
    # The researcher an existing, still empty draft is to be marked as
    # built from (``profiles.researcher_id``), written by ``_persist`` once
    # the import succeeds (review B3-1, the user's choice B): marked while
    # planning, a failed import left the mark -- the drafts list named the
    # person and step 3 showed their profile topics over no papers of theirs.
    link_researcher: int | None = None


def normalize_ref(raw: str) -> str:
    """Accept a bare slug, any Prosopia URL the user copied, or an ORCID.

    All of these reduce to ``sheffield-nathan``:

    - ``sheffield-nathan`` (a bare slug; a rid works the same way)
    - ``https://prosopia.databio.org/sheffield-nathan[/...]`` (a site URL
      names the slug first)
    - ``https://prosopia.databio.org/manage/sheffield-nathan/view`` (any
      ``/manage/<slug>/...`` page, with or without the scheme)
    - ``https://prosopia.databio.org/privacy?profile=sheffield-nathan``
    - ``https://prosopia.databio.org/p?u=<encoded document URL>`` (the
      public profile page, whose ``u`` is the document URL below)
    - ``.../api/v1/profiles/sheffield-nathan[/content/profile.jsonld]``
      (the slug right after ``profiles``, however deep the URL goes)

    ``https://orcid.org/0000-0001-5643-4068`` reduces to the bare ORCID,
    which :func:`prepare_import` then resolves to a slug. A known page
    that names no profile (``/privacy``, ``/p`` without ``u``) reduces to
    ``""``; anything else is returned trimmed and left to 404 on its own.
    """
    value = (raw or "").strip()
    if not value:
        return ""
    from_url = "://" in value
    if not from_url and "/" not in value and "?" not in value:
        return unquote(value)
    if not from_url:
        first = value.lstrip("/").split("/", 1)[0]
        # ``prosopia.databio.org/manage/x`` (a host) vs ``/manage/x``.
        if "." in first:
            value, from_url = f"https://{value}", True
        else:
            value = f"https://host/{value.lstrip('/')}"
    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    if query.get("profile"):
        return query["profile"][0].strip()
    if query.get("u"):
        inner = _slug_named_in([unquote(p) for p in urlparse(query["u"][0]).path.split("/") if p])
        if inner:
            return inner
    segments = [unquote(p) for p in parsed.path.split("/") if p]
    if not segments:
        return ""
    named = _slug_named_in(segments)
    if named:
        return named
    if len(segments) == 1 and segments[0] in _NON_PROFILE_PAGES:
        return ""
    return segments[0] if from_url else segments[-1]


# SPA pages that name no profile by themselves.
_NON_PROFILE_PAGES = {"p", "privacy", "manage", "me", "account", "lenses", "admin"}


def _slug_named_in(segments: list[str]) -> str:
    """The slug a path names outright: ``manage/<slug>/...``, or the
    segment after ``profiles`` (``api/v1/profiles/<slug>/...``)."""
    if len(segments) >= 2 and segments[0] == "manage":
        return segments[1]
    if "profiles" in segments:
        idx = segments.index("profiles")
        if idx + 1 < len(segments):
            return segments[idx + 1]
    return ""


def _resolve_profile(ref: str, base_url: str | None, prosopia_client: Any | None):
    """``(slug, profile)`` for a ref, with the ORCID indirection applied."""
    slug = normalize_ref(ref)
    if not slug:
        raise ValueError("ref must be a non-empty slug, profile URL or ORCID")

    if prosopia_client is None:
        from ..settings import get_settings

        settings = get_settings()
        base_url = allowed_base_url(
            base_url,
            [settings.RADAR_PROSOPIA_BASE_URL, *settings.RADAR_PROSOPIA_ALLOWED_BASE_URLS],
        )
    client = prosopia_client or ProsopiaClient(base_url or DEFAULT_BASE_URL)
    # An ORCID (bare, or as an orcid.org URL, which normalize_ref has
    # already reduced to the bare id) names a researcher, not a profile;
    # the client turns it into the slug of the profile they published.
    if looks_like_orcid(slug):
        slug = client.resolve_orcid(slug)
    profile = client.fetch_profile(slug)
    # Every Prosopia profile RADAR reads comes through here: its rid is
    # made one form once (CR.8), whatever client produced it.
    try:
        profile.rid = normalize_rid(getattr(profile, "rid", ""))
    except AttributeError:      # a read-only profile object: its own entry normalized it
        pass
    return slug, profile


def _seed_form(openalex_id: str | None) -> str | None:
    """An OpenAlex work id as seeds store it, ``https://openalex.org/W…``
    (L10): the pick list reads the same whether or not OpenAlex answered.
    None for anything that is not a work id (a DOI, a ``openalex:`` form):
    a made-up openalex.org link is worse than none (review B3-2)."""
    import re

    bare = _normalize_openalex_id(openalex_id)
    return f"https://openalex.org/{bare}" if re.fullmatch(r"W\d+", bare) else None


def _work_summary(record: Any, current: dict[str, dict] | None = None) -> dict[str, Any]:
    """``current``: OpenAlex's work for a record's id (:func:`current_works`);
    the listing carries its id, not a merged-away one Prosopia kept."""
    authors = [a for a in (getattr(record, "authors", None) or []) if a]
    work = (current or {}).get(_normalize_openalex_id(record.openalex_id))
    return {
        "id": _paper_id(record),
        "title": (record.name or "").strip() or _paper_id(record),
        "year": record.year,
        "venue": record.journal or record.venue,
        "doi": normalize_doi(record.doi),
        "openalex_id": _seed_form((work or {}).get("id") or record.openalex_id),
        "cited_by_count": getattr(record, "cited_by_count", None),
        "authors": authors[:3],
        "n_authors": len(authors) or None,
    }


def list_works(
    ref: str,
    *,
    base_url: str | None = None,
    prosopia_client: Any | None = None,
    refresh: bool = False,
    openalex_client: Any | None = None,
) -> dict[str, Any]:
    """The papers on a profile, slimmed for a pick list.

    Same errors as :func:`prepare_import` (400 / 404 / 502 material),
    but nothing is written: this is the read the wizard makes before the
    user decides which papers to keep. ``refresh``: the read is "check for
    new papers" on a stored profile; the result also carries the profile
    itself (``profile``), whose fields the caller stores.

    With ``openalex_client`` each record's OpenAlex id is the one OpenAlex
    has today (FUNCTIONAL_SPEC M.4 rule 6). OpenAlex not answering leaves
    the ids as Prosopia has them: the list is still worth showing, and
    the import stops on its own (F2).
    """
    from ...openalex_status import OpenAlexUnavailable

    slug, profile = _resolve_profile(ref, base_url, prosopia_client)
    if not profile.papers:
        raise ProfileEmpty(slug, refresh=refresh)
    current: dict[str, dict] = {}
    if openalex_client is not None:
        try:
            current, _ = current_works((r.openalex_id for r in profile.papers), openalex_client)
        except OpenAlexUnavailable as exc:
            log.warning("prosopia.list_works_ids_unchecked", slug=slug, reason=str(exc)[:200])
    works = [_work_summary(r, current) for r in profile.papers]
    works.sort(key=lambda w: (-(w["year"] or 0), w["title"]))
    log.info("prosopia.list_works", slug=slug, n=len(works))
    out = {"slug": slug, "name": profile.name or None, "works": works}
    rid = getattr(profile, "rid", "") or ""
    out["orcid"] = rid if looks_like_orcid(rid) else None
    if refresh:
        out["profile"] = profile
    return out


class DraftNotFound(LookupError):
    """``into`` names no draft of this user."""


def target_draft(conn: sqlite3.Connection, user_id: int, into: str | None) -> sqlite3.Row | None:
    """The existing draft an import adds to, or None for a new one.

    Checked before anything is written. Only a draft: a live interest
    takes papers through ``POST /{key}/seeds``, which also refits it.
    """
    if not into:
        return None
    row = profiles_repo.get_by_slug(conn, user_id, into)
    if row is None or not row["is_draft"]:
        raise DraftNotFound(f"no draft '{into}' to add these papers to")
    return row


def open_draft(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    user_id: int,
    target: sqlite3.Row | None,
    name: str,
    embedding_model: str,
    researcher_id: int | None,
) -> sqlite3.Row:
    """The draft an import's papers become seeds of: ``target``, or a new one.

    A new draft is linked to the researcher (it goes with the import if
    that fails). An existing one is returned as it is: whether it is to be
    linked is :func:`_link_for` 's, and the link is written only when the
    import succeeds (B3-1).
    """
    if target is not None:
        return target
    draft = wizard_service.create_draft(
        conn,
        user_id=user_id,
        name=name,
        embedding_model=embedding_model,
        selector=settings.RADAR_DEFAULT_SELECTOR,
    )
    draft_row = profiles_repo.get_by_slug(conn, user_id, draft["slug"])
    if draft_row is None:  # pragma: no cover — create_draft just wrote it
        raise RuntimeError(f"draft '{draft['slug']}' vanished after create")
    researchers_repo.set_profile_researcher(conn, int(draft_row["id"]), researcher_id)
    return draft_row


def open_draft_for_import(
    conn: sqlite3.Connection, settings: Any, *, save_profile: bool, record_researcher, new_researcher: bool,
    **kwargs: Any,
) -> tuple[sqlite3.Row, int | None, int | None]:
    """The draft a wizard import fills, the researcher it saved, if any
    (FUNCTIONAL_SPEC M.2), and the researcher an existing draft is to be
    marked as built from once the import succeeds (:func:`_link_for`).

    Not saving (the default): no researcher; the papers are the draft's
    seeds, and go with it. Saving: ``record_researcher()`` makes the
    researcher profile at once -- a saved one, not the draft's, so it
    stays whatever becomes of the draft. One this request created is
    removed again when the draft cannot be opened, or (``undo_import``)
    when the import fails.
    """
    if not save_profile:
        return open_draft(conn, settings, researcher_id=None, **kwargs), None, None
    researcher_id = record_researcher()
    try:
        link = _link_for(conn, kwargs.get("target"), researcher_id)
        return open_draft(conn, settings, researcher_id=researcher_id, **kwargs), researcher_id, link
    except BaseException:
        conn.rollback()
        if new_researcher:
            researchers_repo.delete(conn, researcher_id)
        raise


def _link_for(conn: sqlite3.Connection, target: sqlite3.Row | None, researcher_id: int | None) -> int | None:
    """The researcher an existing draft is to be marked as built from, once
    the import into it succeeds; None to leave it as it is.

    One built from someone already stays theirs -- so a second
    researcher's papers can join the first's (the wizard's "add papers
    from an ORCID / Prosopia profile" once a draft exists). One that has
    seeds of its own (uploaded PDFs) is not marked either: it was not built
    from this person, and the mark made the wizard, the drafts list and the
    interest page say it was ("4 papers from the profile of X" over two
    PDFs and two of X's papers). Decided now, before the import adds seeds.
    """
    if target is None or researcher_id is None or target["researcher_id"] is not None:
        return None
    return None if _has_seeds(conn, int(target["id"])) else int(researcher_id)


def _has_seeds(conn: sqlite3.Connection, profile_id: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM profile_seeds WHERE profile_id = ? LIMIT 1", (profile_id,)
    ).fetchone() is not None


def prepare_import(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    user_id: int,
    ref: str,
    base_url: str | None = None,
    name: str | None = None,
    embedding_model: str | None = None,
    prosopia_client: Any | None = None,
    paper_ids: list[str] | None = None,
    into: str | None = None,
    save_profile: bool = False,
) -> ImportPlan:
    """Read the profile and open the draft. Fast enough for a request.

    With ``into`` the papers become seeds of that existing draft instead
    of a new one (:class:`DraftNotFound` when it is not one of the user's
    drafts).

    ``paper_ids`` (the ``id`` values from :func:`list_works`) keeps only
    those papers; ``None`` imports the whole profile. Raises
    ``ValueError`` on an empty ref, ``ProfileNotFound`` when the slug is
    unknown to the Prosopia instance, :class:`PapersNotFound` when none
    of ``paper_ids`` are on the profile, and ``ProsopiaError`` for any
    other upstream failure; the router maps those to 400 / 404 / 502.
    """
    # OpenAlex paused: the import could not resolve a paper; nothing is
    # created (FUNCTIONAL_SPEC F2), a 503.
    from ... import openalex_status
    openalex_status.check()
    target = target_draft(conn, user_id, into)
    slug, profile = _resolve_profile(ref, base_url, prosopia_client)
    records = select_records(profile, slug, paper_ids)
    # As the ORCID import: the same person already importing (under either
    # source, by ORCID) is that run, not a second draft and a second job.
    from . import orcid as orcid_service
    from ...db.repos import researchers as researchers_repo
    rid = getattr(profile, "rid", "") or ""
    from . import researchers as researchers_service
    known = researchers_service.saved_person(
        conn, user_id=user_id, source="prosopia", key=slug, orcid=rid if looks_like_orcid(rid) else None,
    )
    orcid_service.refuse_if_importing(
        conn, user_id, rid if looks_like_orcid(rid) else None, into,
        researcher_id=int(known["id"]) if known is not None else None, key=f"prosopia:{slug}",
    )
    # A person already stored (FUNCTIONAL_SPEC M.2a): ticked, the papers go
    # into that profile -- its name and text stay as they are; unticked,
    # the profile is not touched.
    new_researcher = known is None

    model = (target["embedding_model"] if target is not None else None) \
        or embedding_model or settings.RADAR_DEFAULT_EMBEDDING_MODEL
    draft_row, researcher_id, link_researcher = open_draft_for_import(
        conn, settings, save_profile=save_profile, new_researcher=new_researcher,
        record_researcher=lambda: int(known["id"]) if known is not None else researchers_service.record_prosopia_researcher(
            conn, user_id=user_id, slug=slug, profile=profile,
            base_url=base_url or DEFAULT_BASE_URL,
        ),
        user_id=user_id, target=target,
        name=(name or profile.name or slug).strip()[:MAX_NAME], embedding_model=model,
    )
    # The profile's interest / not interest go with the draft, saved
    # profile or not (FUNCTIONAL_SPEC M.3) -- once the import succeeds (CR.7).
    prosopia_doc = {"key": slug, "name": (str(getattr(profile, "name", "") or "") or slug),
                    "document": researchers_service.profile_document(profile)}

    return ImportPlan(
        prosopia_doc=prosopia_doc,
        link_researcher=link_researcher,
        slug=slug,
        draft_slug=draft_row["slug"],
        draft_name=draft_row["name"],
        profile_id=int(draft_row["id"]),
        researcher_id=researcher_id,
        user_id=user_id,
        embedding_model=model,
        records=records,
        profile=profile,
        new_draft=target is None,
        new_researcher=bool(new_researcher and researcher_id is not None),
        listed_ids={_paper_id(r) for r in profile.papers},
        person=person_of(profile),
        seed_source="researcher" if researcher_id is not None else "prosopia",
        import_orcid=rid if looks_like_orcid(rid) else None,
        import_key=f"prosopia:{slug}",
    )


def person_of(profile: Any) -> Person:
    """The person a Prosopia profile is of: their name and ORCID (rid)."""
    return Person.of(getattr(profile, "name", None), str(getattr(profile, "rid", "") or ""))


def select_records(profile: Any, slug: str, paper_ids: list[str] | None) -> list[Any]:
    """The profile's papers, or the subset ``paper_ids`` names.

    Raises ``ValueError`` for an empty selection and
    :class:`PapersNotFound` when none of the ids are on the profile.
    """
    records = list(profile.papers)
    if not records:
        raise ProfileEmpty(slug)
    if paper_ids is not None:
        wanted = {(p or "").strip() for p in paper_ids if (p or "").strip()}
        if not wanted:
            raise ValueError("select at least one paper to import")
        records = [r for r in records if _paper_id(r) in wanted]
        if not records:
            raise PapersNotFound(f"none of the selected papers are on profile '{slug}'")
    return records


def store_resolved_paper(conn: sqlite3.Connection, resolution) -> bool:
    """Write the resolved paper's row; returns whether it was written.

    Not when the record names a work OpenAlex could not be asked about just
    now (rung ``none`` under its real id) and a row for it exists already
    (an uploaded PDF, a gathered copy): the Prosopia-only row -- summary for
    body text, no abstract -- would replace that one's title and text. The
    researcher link is enough; a refresh offers the work again.
    """
    if not resolution.resolved and papers_repo.get_by_openalex_id(conn, resolution.openalex_id) is not None:
        return False
    papers_repo.upsert(conn, dict(resolution.paper))
    return True


def for_user(user_id: int, resolution):
    """``resolution`` keyed so that what came from this user's Prosopia
    profile is stored as theirs (FUNCTIONAL_SPEC M.1).

    A record OpenAlex has nothing for gets the user in its synthetic id:
    two users importing the same profile get a row each, and one's
    refresh never rewrites the other's. A work whose row would hold
    Prosopia's text (``own_text``) becomes the user's copy of it, a
    version like an upload; the public row keeps what OpenAlex says.
    """
    oa_id = resolution.openalex_id
    if oa_id.startswith("prosopia:"):
        own = f"prosopia:{int(user_id)}:{oa_id[len('prosopia:'):]}"
    elif resolution.own_text:
        own = versions_repo.prosopia_copy_id(user_id, oa_id)
        return replace(resolution, openalex_id=own, canonical=oa_id,
                       paper={**resolution.paper, "openalex_id": own})
    else:
        return resolution
    return replace(resolution, openalex_id=own, paper={**resolution.paper, "openalex_id": own})


# What a Prosopia record says of a work OpenAlex could not be reached for
# that may go on the work's public row when there is none yet: what
# identifies it. Its text stays on the user's copy.
_IDENTIFYING = ("title", "doi", "year", "venue")


def store_own_copy(conn: sqlite3.Connection, user_id: int, resolution) -> list[str]:
    """Store ``user_id``'s copy of ``resolution.canonical`` and the public
    row it is a copy of. Returns the rows this added, the public one first
    (so taking them back in reverse removes the copy before its work).

    The public row gets OpenAlex's record, with no abstract when OpenAlex
    has none; a work OpenAlex could not be reached for gets only what
    identifies it, and only when it has no row yet.
    """
    work = resolution.canonical
    added: list[str] = []
    exists = papers_repo.get_by_openalex_id(conn, work) is not None
    if resolution.resolved:
        papers_repo.upsert(conn, {**resolution.paper, "openalex_id": work, "abstract": "", "body_text": None})
    elif not exists:
        papers_repo.upsert(conn, {"openalex_id": work, "source": "prosopia",
                                  **{k: resolution.paper.get(k) for k in _IDENTIFYING}})
    if not exists:
        added.append(work)
    copy = resolution.openalex_id
    if papers_repo.get_by_openalex_id(conn, copy) is None:
        added.append(copy)
    papers_repo.upsert(conn, dict(resolution.paper))
    versions_repo.register(conn, user_id=user_id, paper_id=copy, canonical_id=work,
                           file_hash=f"prosopia:{work.rsplit('/', 1)[-1]}")
    conn.commit()
    return added


def run_import(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    plan: ImportPlan,
    openalex_client: Any | None = None,
    reporter: Any | None = None,
    run_id: int | None = None,
) -> dict:
    """Resolve, embed and attach every work in ``plan``.

    In two halves, so an import that fails leaves nothing behind
    (FUNCTIONAL_SPEC D, F2): every work is first resolved through OpenAlex
    and every missing vector computed, with nothing written; only then is
    it all stored. OpenAlex not answering (``OpenAlexUnavailable``) stops
    the import in the first half -- it is not "this work is unknown" --
    and so does an embedder that fails.

    One paper that will not resolve is a row on the ``none`` rung, not a
    failed import. ``reporter`` is the scheduler's ``_ProgressReporter``
    (or any duck with ``step`` / ``tick``); pass ``None`` to run silent.
    ``run_id`` is recorded on the researcher as its last import.

    A paper this import adds to the database, not just links, is the
    draft's (``draft_owned``) when the import fills one: discarding the
    draft removes it again unless something else uses it by then.
    """
    records = plan.records
    embedder = get_embedder(plan.embedding_model)
    oa = openalex_client or OpenAlexClient(mailto=users_repo.contact_mailto(conn, plan.user_id, settings.RADAR_DEFAULT_MAILTO))

    # Batched identifier resolution up front: one request per 50 DOIs
    # rather than one per paper. A batch OpenAlex refuses (not one it
    # cannot answer) degrades to per-paper lookups, slower but correct.
    _step(reporter, "fetching", total=len(records),
          message=f"Resolving {len(records)} works through OpenAlex")
    cache = prefetch_works(records, oa)

    resolved_by = {rung: 0 for rung in RUNGS}
    unresolved: list[str] = []

    resolved = []
    for i, record in enumerate(records, start=1):
        resolution = resolve_seed(record, oa, slug=plan.slug, cache=cache, person=plan.person)
        if not resolution.resolved:
            # Only now is the summary artifact worth a round trip: it is
            # the embedding input for a paper we could not otherwise
            # describe. ``summaries`` is lazy, so touching it here costs
            # one request per unresolved paper rather than 82 up front.
            resolution.paper["body_text"] = _fallback_body(
                plan.profile, record, resolution.paper_id,
            )
            unresolved.append(resolution.paper_id)
        resolution = for_user(plan.user_id, resolution)
        resolved_by[resolution.rung] += 1
        resolved.append((record, resolution))
        if reporter is not None and (i % _TICK_EVERY == 0 or i == len(records)):
            _tick(reporter, i, f"Resolved {i} / {len(records)} works")

    # A paper already embedded under this model (by an earlier import, or
    # a PDF upload) is not embedded again: that is what makes building a
    # second interest from a researcher free.
    _step(reporter, "embedding", total=len(resolved),
          message=f"Embedding {len(resolved)} seeds")
    vectors: dict[str, Any] = {}
    for i, (_record, resolution) in enumerate(resolved, start=1):
        oa_id = resolution.openalex_id
        if oa_id not in vectors and not embeddings_repo.has(conn, oa_id, plan.embedding_model):
            vectors[oa_id] = embedder(build_embedding_input(_as_paper(dict(resolution.paper))))
        if reporter is not None and (i % _TICK_EVERY == 0 or i == len(resolved)):
            _tick(reporter, i, f"Embedded {i} / {len(resolved)} seeds")

    _step(reporter, "persisting", total=len(resolved), message=f"Saving {len(resolved)} seeds")
    drafted, seeded, linked = _persist(conn, plan, resolved, vectors, run_id=run_id)
    log.info(
        "prosopia.import.done",
        slug=plan.slug, draft_slug=plan.draft_slug,
        researcher_id=plan.researcher_id, drafted=drafted,
        resolved_by=resolved_by, n_unresolved=len(unresolved),
        openalex_calls=getattr(oa, "api_calls", None),
    )

    return {
        "slug": plan.slug,
        "draft_slug": plan.draft_slug,
        "researcher_id": plan.researcher_id,
        "name": plan.draft_name or plan.slug,
        "drafted": drafted,
        # Seeds this import added to the interest: less than ``drafted`` where
        # some already were (the same work, or one with its DOI; M.4).
        "seeded": seeded,
        "resolved_by": resolved_by,
        "unresolved": unresolved,
        # Papers this added to a researcher profile the user already had
        # (FUNCTIONAL_SPEC M.2a), said in the wizard's message.
        **_added_to_saved(conn, plan, linked),
    }


def _adds_to_saved(plan: ImportPlan) -> bool:
    """A wizard / Edit interest import into a profile that was there before."""
    return plan.profile_id is not None and plan.researcher_id is not None and not plan.new_researcher


def _added_to_saved(conn: sqlite3.Connection, plan: ImportPlan, linked: int) -> dict:
    if not _adds_to_saved(plan):
        return {}
    row = researchers_repo.get(conn, int(plan.researcher_id))
    return {"added_to_profile": linked, "profile_name": (row["name"] if row is not None else None)}


def _persist(conn: sqlite3.Connection, plan: ImportPlan, resolved: list, vectors: dict[str, Any], *,
             run_id: int | None) -> tuple[int, int, int]:
    """Write what the first half resolved; returns how many papers, how
    many of them became new seeds of the interest, and how many were
    added to the researcher profile.

    Two records of one paper (the same OpenAlex work) are one paper: the
    first is stored, the rest marked ``duplicate`` (FUNCTIONAL_SPEC O.2 H);
    so is a record of a paper the researcher already has under another of
    the source's ids. Every write the loop makes is listed, and a failure
    part-way takes them back, so a failed import adds nothing even to an
    existing draft or researcher. The repository calls commit one by one,
    which is why this is undone by hand rather than rolled back.
    """
    done: list[tuple] = []        # what to take back, in order
    seen: set[str] = set()
    duplicate: set[str] = set()
    have: dict[str, str | None] = {}
    if plan.researcher_id is not None:
        have = {r["openalex_id"]: r["paper_id"] for r in conn.execute(
            "SELECT openalex_id, paper_id FROM researcher_papers WHERE researcher_id = ?", (plan.researcher_id,))}
    drafted = seeded = 0
    try:
        for record, resolution in resolved:
            oa_id = resolution.openalex_id
            pid = resolution.paper_id
            if oa_id in seen or (oa_id in have and have[oa_id] and pid and have[oa_id] != pid):
                duplicate.add(pid)
                continue
            seen.add(oa_id)
            # No uploaded_by_user_id: an import does not upload anything. The
            # papers row is shared by every user, and upsert keeps the newest
            # non-null owner, so stamping the importer here moved another
            # user's uploaded PDF -- local file, extracted text and all -- out
            # of their vault and into the importer's. The importer's claim is
            # the researcher_papers link written below, which the vault list
            # and the seed ownership check both already read.
            if resolution.canonical is not None:
                added = store_own_copy(conn, plan.user_id, resolution)
            else:
                is_new = papers_repo.get_by_openalex_id(conn, oa_id) is None
                added = [oa_id] if store_resolved_paper(conn, resolution) and is_new else []
            for item in added:
                done.append(("paper", item))
                if plan.profile_id is not None:
                    draft_owned_repo.own(conn, plan.profile_id, "paper", item)
            if added:
                conn.commit()
            if oa_id in vectors:
                if not embeddings_repo.has(conn, oa_id, plan.embedding_model):
                    done.append(("vector", oa_id))
                embeddings_repo.upsert(conn, oa_id, plan.embedding_model, vectors.pop(oa_id))
            if plan.researcher_id is not None:
                if oa_id not in have:
                    done.append(("link", oa_id))
                researchers_repo.attach_paper(
                    conn, plan.researcher_id, oa_id,
                    paper_id=pid,
                    resolved_by=resolution.rung,
                    summary=(getattr(record, "summary", None) or None),
                )
            if plan.profile_id is not None:
                if profiles_repo.attach_seed(conn, plan.profile_id, oa_id, source=plan.seed_source,
                                             researcher_id=plan.researcher_id):
                    done.append(("seed", oa_id))
                    seeded += 1
            drafted += 1

        linked = sum(1 for kind, _ in done if kind == "link")
        if plan.researcher_id is not None and _adds_to_saved(plan):
            # Papers picked for an interest, added to a profile the user
            # already had (FUNCTIONAL_SPEC M.2a): what was taken is marked,
            # nothing else -- the rest of the listing was not turned down
            # for the profile, and it was not checked for new papers.
            researchers_repo.set_work_marks(
                conn, plan.researcher_id, skipped=set(),
                duplicate=duplicate, taken={res.paper_id for _r, res in resolved} - duplicate,
            )
            researchers_repo.count_papers(conn, plan.researcher_id)
        elif plan.researcher_id is not None:
            taken = {res.paper_id for _r, res in resolved}
            # Offered and left unticked: the listing less what was taken and
            # what the researcher already had (a refresh lists those too;
            # marked "skipped", one dropping out of the saved set later
            # would show as "left out before" although it was imported).
            from . import orcid as orcid_service

            had = set(have) | {p for p in have.values() if p} | {orcid_service._strip_openalex(o) for o in have}
            researchers_repo.set_work_marks(
                conn, plan.researcher_id,
                skipped=(plan.listed_ids or set()) - taken - had,
                duplicate=duplicate, taken=taken - duplicate,
            )
            conn.commit()
            researchers_repo.mark_imported(conn, plan.researcher_id, run_id=run_id)
        if plan.link_researcher is not None and plan.profile_id is not None:
            # Marked now the import succeeded (B3-1); not over a mark
            # another import made meanwhile.
            cur = conn.execute(
                "UPDATE profiles SET researcher_id = ?, updated_at = datetime('now') "
                "WHERE id = ? AND researcher_id IS NULL",
                (int(plan.link_researcher), int(plan.profile_id)),
            )
            conn.commit()
            if cur.rowcount:
                done.append(("marked", int(plan.link_researcher)))
        if plan.prosopia_doc is not None and plan.profile_id is not None:
            # Last: everything else is stored (CR.7).
            from ... import prosopia_interests
            prosopia_interests.keep_doc(conn, int(plan.profile_id), **plan.prosopia_doc)
    except BaseException:
        conn.rollback()
        _take_back(conn, plan, done)
        raise
    return drafted, seeded, linked


def _take_back(conn: sqlite3.Connection, plan: ImportPlan, done: list[tuple]) -> None:
    """Undo ``_persist``'s writes after a failure, newest first."""
    for kind, oa_id in reversed(done):
        if kind == "seed":
            conn.execute("DELETE FROM profile_seeds WHERE profile_id = ? AND openalex_id = ?", (plan.profile_id, oa_id))
            conn.execute("UPDATE profiles SET n_seed = (SELECT COUNT(*) FROM profile_seeds WHERE profile_id = ?) "
                         "WHERE id = ?", (plan.profile_id, plan.profile_id))
        elif kind == "link":
            conn.execute("DELETE FROM researcher_papers WHERE researcher_id = ? AND openalex_id = ?",
                         (plan.researcher_id, oa_id))
        elif kind == "marked":
            conn.execute("UPDATE profiles SET researcher_id = NULL WHERE id = ? AND researcher_id = ?",
                         (plan.profile_id, oa_id))
        elif kind == "vector":
            conn.execute("DELETE FROM paper_embeddings WHERE openalex_id = ? AND embedding_model = ?",
                         (oa_id, plan.embedding_model))
        elif kind == "paper":
            if plan.profile_id is not None:
                conn.execute("DELETE FROM draft_owned WHERE profile_id = ? AND kind = 'paper' AND item = ?",
                             (plan.profile_id, oa_id))
            paper_rows_repo.delete_if_unused(conn, oa_id)
    conn.commit()
    if done:
        log.info("prosopia.import.taken_back", n=len(done), profile_id=plan.profile_id,
                 researcher_id=plan.researcher_id)


def import_error(exc: BaseException) -> str:
    """What a failed import's run row says. OpenAlex not answering is said
    in words (with "Nothing was changed."), not as an exception name."""
    from ...openalex_status import OpenAlexUnavailable
    from ...redact import public_error

    if isinstance(exc, OpenAlexUnavailable):
        return str(exc)
    return f"{type(exc).__name__}: {public_error(exc)}"


def undo_import(conn: sqlite3.Connection, plan: ImportPlan, *, run_id: int | None = None) -> None:
    """An import that failed: remove what the request that planned it made.

    A draft it opened goes with everything it brought in (``draft_owned``);
    a researcher it created goes too, unless something else has come to
    use it. What existed before -- a draft added to, a stored researcher
    -- stays as it was. The run row stays, detached from what is deleted,
    so its poll still reports the error.
    """
    rid = plan.researcher_id if plan.new_researcher else None
    if run_id is not None:
        if plan.new_draft:
            conn.execute("UPDATE gather_runs SET profile_id = NULL WHERE id = ?", (run_id,))
        if rid is not None:
            conn.execute("UPDATE gather_runs SET researcher_id = NULL WHERE id = ?", (run_id,))
        conn.commit()
    if plan.new_draft and plan.profile_id is not None:
        row = profiles_repo.get(conn, int(plan.profile_id))
        if row is not None and row["is_draft"]:
            draft_owned_repo.discard(conn, int(plan.profile_id))
    if rid is None or researchers_repo.get(conn, int(rid)) is None:
        return
    if plan.profile_id is not None:
        # The draft added to was marked as built from the new researcher by
        # _persist (B3-1) -- this runs only when the job fails after it.
        conn.execute("UPDATE profiles SET researcher_id = NULL WHERE id = ? AND researcher_id = ?",
                     (int(plan.profile_id), int(rid)))
    draft_owned_repo.disown(conn, "researcher", rid)
    used = conn.execute("SELECT 1 FROM profiles WHERE researcher_id = ?", (int(rid),)).fetchone()
    if used is None:
        researchers_repo.delete(conn, int(rid))
    else:
        conn.commit()
    log.info("prosopia.import.undone", run_id=run_id, profile_id=plan.profile_id,
             researcher_id=rid, new_draft=plan.new_draft)


def _step(reporter: Any | None, name: str, **kwargs: Any) -> None:
    if reporter is None:
        return
    try:
        reporter.step(name, **kwargs)
    except Exception:  # noqa: BLE001 — progress must never sink the job
        log.warning("prosopia.progress_step_failed", step=name)


def _tick(reporter: Any | None, n: int, message: str) -> None:
    if reporter is None:
        return
    try:
        reporter.tick(n_processed=n, message=message)
    except Exception:  # noqa: BLE001
        pass


def _fallback_body(profile: Any, record: Any, paper_id: str) -> str | None:
    """The record's own summary, plus the profile's summary artifact.

    The two are usually not both present — the API's paper index carries
    no ``summary`` field, so in practice this is the artifact — but a
    profile read from files has both, and neither is worth dropping.
    """
    parts: list[str] = []
    summary = (record.summary or "").strip()
    if summary:
        parts.append(summary)
    if profile is not None:
        artifact = (read_summary(profile, paper_id) or "").strip()
        if artifact and artifact not in parts:
            parts.append(artifact)
    return "\n\n".join(parts) or None


def _as_paper(row: dict) -> Paper:
    """The minimum ``Paper`` ``build_embedding_input`` reads."""
    return Paper(
        doi=row.get("doi"),
        openalex_id=row.get("openalex_id"),
        title=row.get("title") or "",
        abstract=row.get("abstract") or "",
        year=row.get("year"),
        venue=row.get("venue"),
        body_text=row.get("body_text"),
    )
