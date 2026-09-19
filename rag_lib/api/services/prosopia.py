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
)
from ...prosopia_seeds import RUNGS, prefetch_works, resolve_seed
from . import wizard as wizard_service


log = structlog.get_logger("rag_lib.api.services.prosopia")


__all__ = [
    "DEFAULT_BASE_URL",
    "ImportPlan",
    "ProfileNotFound",
    "ProsopiaError",
    "prepare_import",
    "run_import",
    "normalize_ref",
]


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
    """

    slug: str
    draft_slug: str
    draft_name: str
    profile_id: int
    user_id: int
    embedding_model: str
    records: list[Any] = field(default_factory=list)
    profile: Any = None


def normalize_ref(raw: str) -> str:
    """Accept a bare slug or the profile URL the user copied.

    ``https://prosopia.databio.org/sheffield-nathan`` and
    ``.../api/v1/profiles/sheffield-nathan`` both reduce to the slug;
    anything else is returned trimmed and left to 404 on its own.
    """
    value = (raw or "").strip()
    if not value:
        return ""
    if "://" in value:
        parts = value.split("://", 1)[1].split("/", 1)
        value = parts[1] if len(parts) > 1 else ""
    value = value.split("?", 1)[0].split("#", 1)[0].strip("/")
    if not value:
        return ""
    segments = [p for p in value.split("/") if p]
    # ``api/v1/profiles/<slug>`` and ``profiles/<slug>`` both end in it.
    return segments[-1]


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
) -> ImportPlan:
    """Read the profile and open the draft. Fast enough for a request.

    Raises ``ValueError`` on an empty ref, ``ProfileNotFound`` when the
    slug is unknown to the Prosopia instance, and ``ProsopiaError`` for
    any other upstream failure; the router maps those to 400 / 404 / 502.
    """
    slug = normalize_ref(ref)
    if not slug:
        raise ValueError("ref must be a non-empty slug or profile URL")

    client = prosopia_client or ProsopiaClient(base_url or DEFAULT_BASE_URL)
    profile = client.fetch_profile(slug)

    model = embedding_model or settings.RADAR_DEFAULT_EMBEDDING_MODEL
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

    return ImportPlan(
        slug=slug,
        draft_slug=draft["slug"],
        draft_name=draft["name"],
        profile_id=int(draft_row["id"]),
        user_id=user_id,
        embedding_model=model,
        records=list(profile.papers),
        profile=profile,
    )


def run_import(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    plan: ImportPlan,
    openalex_client: Any | None = None,
    reporter: Any | None = None,
) -> dict:
    """Resolve, embed and attach every work in ``plan``.

    Everything past the batched prefetch is best-effort per paper — one
    paper that will not resolve is a row on the ``none`` rung, not a
    failed import. ``reporter`` is the scheduler's ``_ProgressReporter``
    (or any duck with ``step`` / ``tick``); pass ``None`` to run silent.
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

        vector = embedder(build_embedding_input(_as_paper(row)))
        embeddings_repo.upsert(
            conn, resolution.openalex_id, plan.embedding_model, vector,
        )
        profiles_repo.attach_seed(conn, plan.profile_id, resolution.openalex_id)
        drafted += 1
        if reporter is not None and (
            drafted % _TICK_EVERY == 0 or drafted == len(records)
        ):
            _tick(reporter, drafted, f"Imported {drafted} / {len(records)} seeds")

    log.info(
        "prosopia.import.done",
        slug=plan.slug, draft_slug=plan.draft_slug, drafted=drafted,
        resolved_by=resolved_by, n_unresolved=len(unresolved),
        openalex_calls=getattr(oa, "api_calls", None),
    )

    return {
        "slug": plan.slug,
        "draft_slug": plan.draft_slug,
        "name": plan.draft_name,
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
