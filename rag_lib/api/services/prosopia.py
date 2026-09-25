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

import sqlite3
from dataclasses import dataclass, field
from typing import Any

import structlog

from ...db.repos import (
    embeddings as embeddings_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    researchers as researchers_repo,
)
from ...embed import build_embedding_input
from ...embedders import get_embedder
from ...openalex_client import OpenAlexClient
from ...paper import Paper
from ...prosopia_client import (
    DEFAULT_BASE_URL,
    ProfileNotFound,
    ProsopiaClient,
    ProsopiaError,
    read_summary,
    looks_like_orcid,
)
from ...prosopia_seeds import RUNGS, _paper_id, normalize_doi, prefetch_works, resolve_seed
from . import wizard as wizard_service


log = structlog.get_logger("rag_lib.api.services.prosopia")


__all__ = [
    "DEFAULT_BASE_URL",
    "ImportPlan",
    "PapersNotFound",
    "ProfileNotFound",
    "ProsopiaError",
    "list_works",
    "prepare_import",
    "run_import",
    "normalize_ref",
    "select_records",
]


class PapersNotFound(LookupError):
    """None of the requested paper ids are on this profile."""


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


def normalize_ref(raw: str) -> str:
    """Accept a bare slug, the profile URL the user copied, or an ORCID.

    ``https://prosopia.databio.org/sheffield-nathan`` and
    ``.../api/v1/profiles/sheffield-nathan`` both reduce to the slug, and
    ``https://orcid.org/0000-0001-5643-4068`` to the bare ORCID (which
    :func:`prepare_import` then resolves to a slug); anything else is
    returned trimmed and left to 404 on its own.
    """
    value = (raw or "").strip()
    if not value:
        return ""
    from_url = "://" in value
    if from_url:
        parts = value.split("://", 1)[1].split("/", 1)
        value = parts[1] if len(parts) > 1 else ""
    value = value.split("?", 1)[0].split("#", 1)[0].strip("/")
    if not value:
        return ""
    segments = [p for p in value.split("/") if p]
    # ``api/v1/profiles/<slug>`` names the slug right after ``profiles``,
    # and so does anything deeper that a user might copy, such as
    # ``.../profiles/<slug>/content/profile.jsonld``. A site URL
    # (``https://host/<slug>[/...]``) names it first. Only a bare value
    # keeps the old last-segment reading.
    if "profiles" in segments:
        idx = segments.index("profiles")
        if idx + 1 < len(segments):
            return segments[idx + 1]
    if from_url:
        return segments[0]
    return segments[-1]


def _resolve_profile(ref: str, base_url: str | None, prosopia_client: Any | None):
    """``(slug, profile)`` for a ref, with the ORCID indirection applied."""
    slug = normalize_ref(ref)
    if not slug:
        raise ValueError("ref must be a non-empty slug, profile URL or ORCID")

    client = prosopia_client or ProsopiaClient(base_url or DEFAULT_BASE_URL)
    # An ORCID (bare, or as an orcid.org URL, which normalize_ref has
    # already reduced to the bare id) names a researcher, not a profile;
    # the client turns it into the slug of the profile they published.
    if looks_like_orcid(slug):
        slug = client.resolve_orcid(slug)
    return slug, client.fetch_profile(slug)


def _work_summary(record: Any) -> dict[str, Any]:
    authors = [a for a in (getattr(record, "authors", None) or []) if a]
    return {
        "id": _paper_id(record),
        "title": (record.name or "").strip() or _paper_id(record),
        "year": record.year,
        "venue": record.journal or record.venue,
        "doi": normalize_doi(record.doi),
        "openalex_id": record.openalex_id or None,
        "cited_by_count": getattr(record, "cited_by_count", None),
        "authors": authors[:3],
        "n_authors": len(authors) or None,
    }


def list_works(
    ref: str,
    *,
    base_url: str | None = None,
    prosopia_client: Any | None = None,
) -> dict[str, Any]:
    """The papers on a profile, slimmed for a pick list.

    Same errors as :func:`prepare_import` (400 / 404 / 502 material),
    but nothing is written: this is the read the wizard makes before the
    user decides which papers to keep.
    """
    slug, profile = _resolve_profile(ref, base_url, prosopia_client)
    works = [_work_summary(r) for r in profile.papers]
    works.sort(key=lambda w: (-(w["year"] or 0), w["title"]))
    log.info("prosopia.list_works", slug=slug, n=len(works))
    return {"slug": slug, "name": profile.name or None, "works": works}


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
) -> ImportPlan:
    """Read the profile and open the draft. Fast enough for a request.

    ``paper_ids`` (the ``id`` values from :func:`list_works`) keeps only
    those papers; ``None`` imports the whole profile. Raises
    ``ValueError`` on an empty ref, ``ProfileNotFound`` when the slug is
    unknown to the Prosopia instance, :class:`PapersNotFound` when none
    of ``paper_ids`` are on the profile, and ``ProsopiaError`` for any
    other upstream failure; the router maps those to 400 / 404 / 502.
    """
    slug, profile = _resolve_profile(ref, base_url, prosopia_client)
    records = select_records(profile, slug, paper_ids)

    model = embedding_model or settings.RADAR_DEFAULT_EMBEDDING_MODEL
    # The researcher is recorded whichever way the import was started,
    # so the papers this draft is about to embed are reusable for the
    # next interest without a second import.
    from . import researchers as researchers_service
    researcher_id = researchers_service.record_prosopia_researcher(
        conn, user_id=user_id, slug=slug, profile=profile,
        base_url=base_url or DEFAULT_BASE_URL,
    )
    draft = wizard_service.create_draft(
        conn,
        user_id=user_id,
        name=(name or profile.name or slug).strip(),
        embedding_model=model,
        selector=settings.RADAR_DEFAULT_SELECTOR,
    )
    draft_row = profiles_repo.get_by_slug(conn, user_id, draft["slug"])
    if draft_row is None:  # pragma: no cover — create_draft just wrote it
        raise RuntimeError(f"draft '{draft['slug']}' vanished after create")
    researchers_repo.set_profile_researcher(conn, int(draft_row["id"]), researcher_id)

    return ImportPlan(
        slug=slug,
        draft_slug=draft["slug"],
        draft_name=draft["name"],
        profile_id=int(draft_row["id"]),
        researcher_id=researcher_id,
        user_id=user_id,
        embedding_model=model,
        records=records,
        profile=profile,
    )


def select_records(profile: Any, slug: str, paper_ids: list[str] | None) -> list[Any]:
    """The profile's papers, or the subset ``paper_ids`` names.

    Raises ``ValueError`` for an empty selection and
    :class:`PapersNotFound` when none of the ids are on the profile.
    """
    records = list(profile.papers)
    if paper_ids is not None:
        wanted = {(p or "").strip() for p in paper_ids if (p or "").strip()}
        if not wanted:
            raise ValueError("select at least one paper to import")
        records = [r for r in records if _paper_id(r) in wanted]
        if not records:
            raise PapersNotFound(f"none of the selected papers are on profile '{slug}'")
    return records


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

    Everything past the batched prefetch is best-effort per paper — one
    paper that will not resolve is a row on the ``none`` rung, not a
    failed import. ``reporter`` is the scheduler's ``_ProgressReporter``
    (or any duck with ``step`` / ``tick``); pass ``None`` to run silent.
    ``run_id`` is recorded on the researcher as its last import.
    """
    records = plan.records
    embedder = get_embedder(plan.embedding_model)
    oa = openalex_client or OpenAlexClient(mailto=settings.RADAR_DEFAULT_MAILTO)

    # Batched identifier resolution up front: one request per 50 DOIs
    # rather than one per paper. A failure here is *not* fatal — the
    # ladder degrades to per-paper lookups, which is slower but correct.
    _step(reporter, "fetching", total=len(records),
          message=f"Resolving {len(records)} works through OpenAlex")
    cache = prefetch_works(records, oa)

    resolved_by = {rung: 0 for rung in RUNGS}
    unresolved: list[str] = []
    drafted = 0

    _step(reporter, "embedding", total=len(records),
          message=f"Embedding {len(records)} seeds")
    for record in records:
        resolution = resolve_seed(record, oa, slug=plan.slug, cache=cache)
        if not resolution.resolved:
            # Only now is the summary artifact worth a round trip: it is
            # the embedding input for a paper we could not otherwise
            # describe. ``summaries`` is lazy, so touching it here costs
            # one request per unresolved paper rather than 82 up front.
            resolution.paper["body_text"] = _fallback_body(
                plan.profile, record, resolution.paper_id,
            )
            unresolved.append(resolution.paper_id)

        resolved_by[resolution.rung] += 1

        row = dict(resolution.paper)
        row["uploaded_by_user_id"] = plan.user_id
        papers_repo.upsert(conn, row)

        # A paper already embedded under this model (by an earlier
        # import, or a PDF upload) is not embedded again: that is what
        # makes building a second interest from a researcher free.
        if not embeddings_repo.has(conn, resolution.openalex_id, plan.embedding_model):
            vector = embedder(build_embedding_input(_as_paper(row)))
            embeddings_repo.upsert(
                conn, resolution.openalex_id, plan.embedding_model, vector,
            )
        if plan.researcher_id is not None:
            researchers_repo.attach_paper(
                conn, plan.researcher_id, resolution.openalex_id,
                paper_id=resolution.paper_id,
                resolved_by=resolution.rung,
                summary=(getattr(record, "summary", None) or None),
            )
        if plan.profile_id is not None:
            profiles_repo.attach_seed(conn, plan.profile_id, resolution.openalex_id)
        drafted += 1
        if reporter is not None and (
            drafted % _TICK_EVERY == 0 or drafted == len(records)
        ):
            _tick(reporter, drafted, f"Imported {drafted} / {len(records)} seeds")

    if plan.researcher_id is not None:
        researchers_repo.mark_imported(conn, plan.researcher_id, run_id=run_id)

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
        "resolved_by": resolved_by,
        "unresolved": unresolved,
    }


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
