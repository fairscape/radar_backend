"""ORCID import — seed a wizard draft from a researcher's own OpenAlex works.

The PDF wizard path turns uploaded files into ``papers`` rows, SPECTER2
vectors and ``profile_seeds``. This module does the same starting from an
ORCID, with no file in sight:

    OpenAlex /authors/orcid:<id>                 -> AuthorInfo (name, institution, topics)
    OpenAlex /works?filter=authorships.author.orcid:<id>  -> raw works, cursor-paginated
    classify_authorship()                         -> the researcher's position on each work
    build_records()                               -> WorkRecord (Paper + authorship + type)
    select_seeds()                                -> lead-author, non-dataset, deduped, capped
    researcher_topic_filters()                    -> weighted top-K topic_filters
    persist_records()                             -> papers / paper_embeddings / profile_seeds
    write_rp_profile()                            -> RP-format profile.jsonld + papers.jsonld

Everything is a plain function over an injected ``client`` (the real
``OpenAlexClient`` or the test fake) so the job, the route's inline test
path and the CLI share one code path. ``run_import`` is the orchestrator;
``reporter`` (a ``_ProgressReporter`` or ``None``) receives phase names
the frontend maps to labels.

Filtering is deliberately what the researcher-profiles SDK itself applies
to an OpenAlex corpus — the type gate, works without title/year, and works
where the ORCID is not among the authorships. No title-based merging of
the corpus: every kept work is stored. Only *seed selection* collapses
copies (preprint + version of record) so one paper does not count twice
in the centroid.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import structlog

from ...db.repos import (
    embeddings as embeddings_repo,
    orcid_works as works_repo,
    papers as papers_repo,
    profiles as profiles_repo,
)
from ...embed import build_embedding_input
from ...embedders import get_embedder
from ...paper import Paper, title_key
from ...profile import Profile
from ..schemas import OrcidAuthor, OrcidImportResult

log = structlog.get_logger("rag_lib.api.services.orcid_import")

#: Canonical ORCID shape after :func:`normalize_orcid`.
ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")

#: OpenAlex ``type`` values that are not scholarly narrative outputs. Same
#: set the researcher-profiles SDK uses for its candidate stream.
EXCLUDED_TYPES = frozenset(
    {"software", "other", "paratext", "libguides", "grant", "peer-review", "book"}
)

#: Types that are stored as corpus members but never count as seeds or
#: topic evidence: OpenAlex files Figshare "Additional file N" deposits as
#: ``dataset`` and its classifier tags them with unrelated topics.
SEED_EXCLUDED_TYPES = frozenset({"dataset"})

#: A lead-author work counts this much more than a middle-author one when
#: aggregating topics.
LEAD_WEIGHT = 2.0

RP_CONTEXT = "https://profiles.databio.org/context/v1.jsonld"
RP_LICENSE = "https://creativecommons.org/licenses/by/4.0/"

#: SPECTER2 runs on one shared GPU (ollama + the reranker live there too).
#: Serialise embedding loops so two imports do not contend for memory.
_GPU_LOCK = threading.Lock()

_PAPER_ID_RE = re.compile(r"^[a-z][a-z0-9]*\d{4}[a-z0-9]+$")
_STOP = {
    "a", "an", "the", "of", "on", "in", "for", "and", "to", "with", "from", "by",
    "at", "is", "are", "as", "into", "via", "using", "toward", "towards", "new",
    "abstract", "poster", "editorial", "correction", "erratum",
}


class ImportAborted(RuntimeError):
    """The draft disappeared (deleted or committed) while the import ran."""


@dataclass
class ClaimedWorks:
    """What the researcher's own ORCID record lists.

    OpenAlex stamps an author entity's ORCID onto every work of that
    entity, and entities are sometimes over-merged (two "Tim W. Clark"s
    in one). The ORCID registry is the person's own claim and is the
    identity authority; OpenAlex is the enrichment source.
    """

    dois: set[str] = field(default_factory=set)        # bare, lowercased
    title_keys: set[str] = field(default_factory=set)  # paper.title_key form
    n_works: int = 0

    def matches(self, doi: str | None, title: str | None) -> bool:
        if doi and _norm_doi(doi) in self.dois:
            return True
        k = title_key(title)
        return bool(k) and k in self.title_keys


ORCID_PUBLIC_API = "https://pub.orcid.org/v3.0"


def _norm_doi(doi: str) -> str:
    """Bare lowercase DOI: strips ``https://doi.org/`` / ``doi:`` prefixes."""
    d = doi.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d[4:] if d.startswith("doi:") else d


def fetch_orcid_claimed(orcid: str, *, timeout: float = 30.0) -> ClaimedWorks | None:
    """Works the person claims on their ORCID record, or ``None`` if unreachable.

    ``None`` (not an empty set) when the registry cannot be read, so the
    caller can tell "no evidence" from "the record is empty".
    """
    import requests

    try:
        r = requests.get(
            f"{ORCID_PUBLIC_API}/{orcid}/works",
            headers={"Accept": "application/json", "User-Agent": "radar-backend orcid-import"},
            timeout=timeout,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001 — registry down: proceed without
        log.warning("orcid_import.orcid_registry_unavailable", orcid=orcid, error=str(exc))
        return None
    out = ClaimedWorks()
    for group in data.get("group") or []:
        out.n_works += 1
        for e in (group.get("external-ids") or {}).get("external-id") or []:
            if (e.get("external-id-type") or "").lower() == "doi" and e.get("external-id-value"):
                out.dois.add(_norm_doi(str(e["external-id-value"])))
        for ws in group.get("work-summary") or []:
            t = ((ws.get("title") or {}).get("title") or {}).get("value")
            k = title_key(t)
            if k:
                out.title_keys.add(k)
    return out


# ---------------------------------------------------------------------------
# ORCID validation
# ---------------------------------------------------------------------------


def normalize_orcid(raw: str) -> str:
    """``https://orcid.org/0000-0001-5643-406x`` -> ``0000-0001-5643-406X``.

    Raises ``ValueError`` when the shape or the ISO 7064 MOD 11-2 check
    digit is wrong. The result is safe to interpolate into an OpenAlex
    filter string and a URL path.
    """
    s = (raw or "").strip()
    for prefix in ("https://orcid.org/", "http://orcid.org/", "orcid.org/"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.strip().strip("/").upper()
    if not ORCID_RE.match(s):
        raise ValueError(f"not an ORCID: {raw!r} (expected 0000-0000-0000-000X)")
    digits = s.replace("-", "")
    total = 0
    for c in digits[:15]:
        total = (total + int(c)) * 2
    remainder = total % 11
    result = (12 - remainder) % 11
    check = "X" if result == 10 else str(result)
    if digits[15] != check:
        raise ValueError(f"ORCID check digit mismatch: {s}")
    return s


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class AuthorInfo:
    orcid: str
    openalex_author_id: str | None
    display_name: str
    institution: str | None = None
    institution_ror: str | None = None
    topics: list[dict] = field(default_factory=list)
    works_count: int | None = None

    def to_schema(self) -> OrcidAuthor:
        return OrcidAuthor(
            orcid=self.orcid,
            openalex_author_id=self.openalex_author_id,
            display_name=self.display_name,
            institution=self.institution,
        )


@dataclass
class WorkRecord:
    raw: dict
    paper: Paper
    wtype: str
    cited_by_count: int
    position: str | None
    is_corresponding: bool
    author_index: int
    total_authors: int
    authors: list[str]
    #: True when the person's ORCID record lists this work; None when the
    #: registry was unavailable (no evidence either way).
    claimed: bool | None = None

    @property
    def is_lead(self) -> bool:
        return self.position in ("first", "last") or self.is_corresponding

    @property
    def seed_eligible(self) -> bool:
        return self.wtype not in SEED_EXCLUDED_TYPES

    @property
    def field_name(self) -> str | None:
        pt = self.raw.get("primary_topic") or {}
        return ((pt.get("field") or {}).get("display_name")) or None

    @property
    def topic_weight(self) -> float:
        if self.wtype in SEED_EXCLUDED_TYPES:
            return 0.0
        w = LEAD_WEIGHT if self.is_lead else 1.0
        # An unclaimed work in an over-merged corpus is weaker evidence.
        return w * (0.5 if self.claimed is False else 1.0)


# ---------------------------------------------------------------------------
# OpenAlex access (with the retry the client deliberately lacks)
# ---------------------------------------------------------------------------


def _with_retry(
    fn: Callable[[], Any],
    *,
    attempts: int = 4,
    base: float = 2.0,
    on_wait: Callable[[str], None] | None = None,
) -> Any:
    """Retry ``fn`` on OpenAlex 429s and transient connection errors.

    ``OpenAlexClient._get`` raises ``RuntimeError("OpenAlex 429: ...")``
    and does not retry by design (the client docstring says so). The
    import runs in a scheduler thread, so sleeping here is fine.
    """
    import requests

    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except RuntimeError as exc:
            if "429" not in str(exc) or i == attempts - 1:
                raise
            last = exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            if i == attempts - 1:
                raise
            last = exc
        wait = base * (2 ** i)
        if on_wait:
            on_wait(f"OpenAlex rate-limited; retrying in {wait:.0f}s")
        log.warning("orcid_import.retry", wait=wait, error=str(last))
        time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


def resolve_author(client, orcid: str) -> AuthorInfo | None:
    raw = _with_retry(lambda: client.get_author_by_orcid(orcid))
    if not raw:
        return None
    insts = raw.get("last_known_institutions") or []
    inst = insts[0] if insts else {}
    return AuthorInfo(
        orcid=orcid,
        openalex_author_id=raw.get("id"),
        display_name=(raw.get("display_name") or "").strip(),
        institution=inst.get("display_name") or None,
        institution_ror=inst.get("ror") or None,
        topics=list(raw.get("topics") or []),
        works_count=raw.get("works_count"),
    )


def fetch_works(
    client,
    orcid: str,
    *,
    limit: int | None = None,
    on_wait: Callable[[str], None] | None = None,
) -> list[dict]:
    return _with_retry(lambda: client.works_by_orcid(orcid, limit=limit), on_wait=on_wait)


# ---------------------------------------------------------------------------
# Work -> WorkRecord
# ---------------------------------------------------------------------------


def classify_authorship(
    work: dict, orcid: str, author_oa_id: str | None = None
) -> tuple[str | None, bool, int, int] | None:
    """Locate the researcher among ``work["authorships"]``.

    Returns ``(author_position, is_corresponding, 1-based index, total)``
    or ``None`` when neither the ORCID nor the OpenAlex author id matches
    (OpenAlex truncates authorships past ~100 authors, so a consortium
    paper can legitimately come back ``None``).
    """
    bare = orcid.lower()
    authorships = work.get("authorships") or []
    total = len(authorships)
    for i, a in enumerate(authorships, start=1):
        au = a.get("author") or {}
        oa_orcid = (au.get("orcid") or "").lower()
        if (bare and oa_orcid.endswith(bare)) or (author_oa_id and au.get("id") == author_oa_id):
            return (a.get("author_position"), bool(a.get("is_corresponding")), i, total)
    return None


def _author_names(work: dict) -> list[str]:
    return [
        (a.get("author") or {}).get("display_name")
        for a in (work.get("authorships") or [])
        if (a.get("author") or {}).get("display_name")
    ]


def build_records(
    raws: list[dict], orcid: str, author_oa_id: str | None, *, client,
    claimed: ClaimedWorks | None = None,
) -> tuple[list[WorkRecord], dict]:
    """Parse and attach authorship; one record per OpenAlex work.

    When the ORCID registry lists works (``claimed``), those are identity
    ground truth and every other OpenAlex work is kept only if its
    primary *field* is one the claimed works occupy. That is what
    separates a merged "wildlife ecologist Tim W. Clark" corpus from the
    data scientist who owns the ORCID, while keeping a researcher's own
    older papers that they never bothered to claim.
    """
    report = {
        "fetched": 0, "dropped_type": 0, "dropped_unusable": 0,
        "dropped_not_author": 0, "duplicate_ids": 0, "kept": 0,
        "orcid_claimed": claimed.n_works if claimed else 0,
        "claimed_matched": 0, "dropped_unclaimed_field": 0,
    }
    by_id: dict[str, WorkRecord] = {}
    for raw in raws:
        report["fetched"] += 1
        wtype = (raw.get("type") or "").lower()
        if wtype in EXCLUDED_TYPES:
            report["dropped_type"] += 1
            continue
        title = (raw.get("title") or "").strip()
        if not title or raw.get("publication_year") is None or not raw.get("id"):
            report["dropped_unusable"] += 1
            continue
        ours = classify_authorship(raw, orcid, author_oa_id)
        if ours is None:
            report["dropped_not_author"] += 1
            continue
        position, corresponding, idx, total = ours
        paper = client.paper_from_work(raw, source="orcid")
        rec = WorkRecord(
            raw=raw,
            paper=paper,
            wtype=wtype,
            cited_by_count=int(raw.get("cited_by_count") or 0),
            position=position,
            is_corresponding=corresponding,
            author_index=idx,
            total_authors=total,
            authors=_author_names(raw),
        )
        if claimed is not None and claimed.n_works:
            rec.claimed = claimed.matches(paper.doi, paper.title)
            if rec.claimed:
                report["claimed_matched"] += 1
        if paper.openalex_id in by_id:
            report["duplicate_ids"] += 1
        by_id[paper.openalex_id] = rec

    records = list(by_id.values())
    if claimed is not None and claimed.n_works and report["claimed_matched"]:
        claimed_fields = {r.field_name for r in records if r.claimed and r.field_name}
        kept = []
        for r in records:
            if r.claimed or r.field_name is None or r.field_name in claimed_fields:
                kept.append(r)
            else:
                report["dropped_unclaimed_field"] += 1
        records = kept
    records.sort(key=lambda r: ((r.paper.year or 0), r.paper.title))
    report["kept"] = len(records)
    return records, report


# ---------------------------------------------------------------------------
# Seeds and topics
# ---------------------------------------------------------------------------


def _dedupe_key(rec: WorkRecord) -> str:
    return title_key(rec.paper.title) or (rec.paper.doi or "").lower() or f"oa:{rec.paper.openalex_id}"


def _version_rank(rec: WorkRecord) -> tuple:
    return (
        1 if rec.paper.doi else 0,
        1 if (rec.paper.abstract or "").strip() else 0,
        rec.cited_by_count,
    )


def select_seeds(records: list[WorkRecord], max_seeds: int) -> tuple[list[WorkRecord], list[str]]:
    """Lead-author, seed-eligible works, one per paper, newest first, capped."""
    seeds, _dup_of, warnings = seed_defaults(records, max_seeds)
    return seeds, warnings


def seed_defaults(
    records: list[WorkRecord], max_seeds: int
) -> tuple[list[WorkRecord], dict[str, str], list[str]]:
    """The default rule's pick plus ``{collapsed_id: kept_id}`` for duplicate copies.

    This is what the seed picker pre-checks; the user may override it.
    """
    warnings: list[str] = []
    pool = [r for r in records if r.is_lead and r.seed_eligible]
    if not pool:
        pool = sorted(
            (r for r in records if r.seed_eligible),
            key=lambda r: -r.cited_by_count,
        )[: max(max_seeds, 1)]
        if pool:
            warnings.append(
                "no first/last/corresponding-author works found; "
                "seeding from the most-cited works instead"
            )
    best: dict[str, WorkRecord] = {}
    groups: dict[str, list[WorkRecord]] = {}
    collapsed = 0
    for r in pool:
        k = _dedupe_key(r)
        groups.setdefault(k, []).append(r)
        prev = best.get(k)
        if prev is None:
            best[k] = r
        else:
            collapsed += 1
            if _version_rank(r) > _version_rank(prev):
                best[k] = r
    dup_of: dict[str, str] = {}
    for k, members in groups.items():
        keep = best[k].paper.openalex_id
        for m in members:
            if m.paper.openalex_id != keep:
                dup_of[m.paper.openalex_id] = keep
    if collapsed:
        warnings.append(f"{collapsed} duplicate copies (preprint/repository versions) collapsed for seeding")
    # Claimed works first (identity certainty), then newest, then most cited.
    seeds = sorted(
        best.values(),
        key=lambda r: (0 if r.claimed else 1, -(r.paper.year or 0), -r.cited_by_count, r.paper.title),
    )
    if max_seeds > 0 and len(seeds) > max_seeds:
        warnings.append(f"{len(seeds)} seed candidates; keeping the {max_seeds} newest")
        seeds = seeds[:max_seeds]
    if not seeds:
        warnings.append("no seed-eligible works")
    return seeds, dup_of, warnings


def researcher_topic_filters(
    records: list[WorkRecord], *, top_k: int, seeds: list[WorkRecord] | None = None,
) -> dict:
    """Weighted OpenAlex topic aggregation over the whole kept corpus.

    A topic starts switched ON only if at least one *seed* paper carries
    it: the seeds are the researcher's recent lead-author work, so a topic
    that only the older / middle-author / unclaimed part of the corpus
    carries is listed for the user to opt into rather than gathered by
    default. ``seed_papers`` records how many seeds carry each topic.
    """
    papers = [r.paper for r in records]
    weights = [r.topic_weight for r in records]
    tf = Profile.aggregate_topic_filters(papers, weights=weights, top_k_topics=top_k)
    seed_topic_counts: dict[str, int] = {}
    for r in seeds or []:
        ids = {t.id for t in [r.paper.primary_topic, *r.paper.topics] if t and t.id}
        for tid in ids:
            seed_topic_counts[tid] = seed_topic_counts.get(tid, 0) + 1
    for t in tf.get("topics") or []:
        n = seed_topic_counts.get(t["id"], 0)
        t["seed_papers"] = n
        t["on"] = (n > 0) if seeds is not None else True
        t["source"] = "orcid"
    return tf


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _paper_dict(rec: WorkRecord) -> dict:
    d = rec.paper.to_dict()
    # papers.upsert COALESCEs on NULL, not on "": an empty abstract must
    # not clobber one an earlier upload or gather stored.
    d["abstract"] = (d.get("abstract") or "").strip() or None
    d["source"] = "orcid"
    d["authors"] = rec.authors
    d["publication_date"] = rec.raw.get("publication_date")
    return d


def persist_records(
    conn: sqlite3.Connection,
    settings,
    *,
    profile_id: int,
    records: list[WorkRecord],
    seeds: list[WorkRecord],
    embedding_model: str,
    on_embed: Callable[[], None] | None = None,
    check_alive: Callable[[], None] | None = None,
) -> int:
    """Upsert every record, embed the seeds, then attach them. Returns n_embedded."""
    for rec in records:
        papers_repo.upsert(conn, _paper_dict(rec))
    if check_alive:
        check_alive()

    n_embedded = 0
    embedder = get_embedder(embedding_model)
    with _GPU_LOCK:
        for i, rec in enumerate(seeds, start=1):
            oa_id = rec.paper.openalex_id
            if not embeddings_repo.has(conn, oa_id, embedding_model):
                vec = embedder(build_embedding_input(rec.paper))
                embeddings_repo.upsert(conn, oa_id, embedding_model, vec)
                n_embedded += 1
            if on_embed:
                on_embed()
            if check_alive and i % 10 == 0:
                check_alive()
    if check_alive:
        check_alive()
    # Attach only once every seed has a vector, so a failed embedding
    # never leaves a half-seeded draft behind.
    for rec in seeds:
        profiles_repo.attach_seed(conn, profile_id, rec.paper.openalex_id)
    conn.commit()
    return n_embedded


def work_rows(records: list[WorkRecord], *, defaults: list[WorkRecord], dup_of: dict[str, str]) -> list[dict]:
    """``orcid_works`` rows for the fetch phase."""
    default_ids = {r.paper.openalex_id for r in defaults}
    return [
        {
            "openalex_id": r.paper.openalex_id,
            "position": r.position,
            "is_corresponding": r.is_corresponding,
            "author_index": r.author_index,
            "total_authors": r.total_authors,
            "work_type": r.wtype,
            "cited_by_count": r.cited_by_count,
            "claimed": r.claimed,
            "seed_eligible": r.seed_eligible,
            "dup_of": dup_of.get(r.paper.openalex_id),
            "default_selected": r.paper.openalex_id in default_ids,
            "selected": None,
        }
        for r in records
    ]


def _record_from_row(row: sqlite3.Row) -> WorkRecord | None:
    """Rebuild a :class:`WorkRecord` from an ``orcid_works`` ⨝ ``papers`` row
    (see ``works_repo.list_for_profile``).

    Enough for phase B — embedding, seeding, topic aggregation, the RP
    seed flags: ``raw`` carries only what ``_make_paper_id`` reads (the
    first author's name) and the stored topic blocks.
    """
    try:
        topics = json.loads(row["topics_json"]) if row["topics_json"] else {}
    except (json.JSONDecodeError, TypeError):
        topics = {}
    try:
        authors = list(json.loads(row["authors_json"])) if row["authors_json"] else []
    except (json.JSONDecodeError, TypeError):
        authors = []
    paper = Paper.from_dict({
        "doi": row["doi"],
        "openalex_id": row["openalex_id"],
        "title": row["title"],
        "abstract": row["abstract"] or "",
        "year": row["year"],
        "venue": row["venue"],
        "primary_topic": topics.get("primary_topic"),
        "topics": topics.get("topics") or [],
        "source": "orcid",
    })
    raw = {
        "authorships": [{"author": {"display_name": authors[0]}}] if authors else [],
        "primary_topic": topics.get("primary_topic"),
        "topics": topics.get("topics") or [],
        "publication_date": row["publication_date"],
    }
    return WorkRecord(
        raw=raw, paper=paper, wtype=row["work_type"] or "article",
        cited_by_count=int(row["cited_by_count"] or 0),
        position=row["position"], is_corresponding=bool(row["is_corresponding"]),
        author_index=int(row["author_index"] or 0), total_authors=int(row["total_authors"] or 0),
        authors=authors,
        claimed=None if row["claimed"] is None else bool(row["claimed"]),
    )


def load_work_records(conn: sqlite3.Connection, profile_id: int) -> list[WorkRecord]:
    """Every fetched work of the draft as records (newest first)."""
    rows = conn.execute(
        """
        SELECT w.*, p.title, p.year, p.venue, p.doi, p.abstract, p.topics_json,
               p.authors_json, p.publication_date
        FROM orcid_works w JOIN papers p ON p.openalex_id = w.openalex_id
        WHERE w.profile_id = ?
        ORDER BY COALESCE(p.year, 0) DESC, w.cited_by_count DESC, p.title
        """,
        (profile_id,),
    ).fetchall()
    out: list[WorkRecord] = []
    for r in rows:
        rec = _record_from_row(r)
        if rec is not None:
            out.append(rec)
    return out


def works_for_draft(conn: sqlite3.Connection, profile_id: int) -> list[dict]:
    """The seed picker's rows (``OrcidWork`` shape)."""
    out: list[dict] = []
    for r in works_repo.list_for_profile(conn, profile_id):
        try:
            authors = list(json.loads(r["authors_json"])) if r["authors_json"] else []
        except (json.JSONDecodeError, TypeError):
            authors = []
        out.append({
            "openalex_id": r["openalex_id"],
            "title": r["title"] or "",
            "year": r["year"],
            "venue": r["venue"],
            "doi": r["doi"],
            "first_author": authors[0] if authors else None,
            "position": r["position"],
            "is_corresponding": bool(r["is_corresponding"]),
            "author_index": r["author_index"],
            "total_authors": r["total_authors"],
            "work_type": r["work_type"],
            "cited_by_count": int(r["cited_by_count"] or 0),
            "claimed": None if r["claimed"] is None else bool(r["claimed"]),
            "seed_eligible": bool(r["seed_eligible"]),
            "dup_of": r["dup_of"],
            "has_abstract": bool(r["has_abstract"]),
            "default_selected": bool(r["default_selected"]),
            "selected": None if r["selected"] is None else bool(r["selected"]),
            "is_seed": bool(r["is_seed"]),
        })
    return out


def store_works(conn: sqlite3.Connection, *, profile_id: int, records: list[WorkRecord],
                defaults: list[WorkRecord], dup_of: dict[str, str]) -> int:
    """Upsert the papers and (re)write the ``orcid_works`` rows. No embedding."""
    for rec in records:
        papers_repo.upsert(conn, _paper_dict(rec))
    return works_repo.replace_all(conn, profile_id, work_rows(records, defaults=defaults, dup_of=dup_of))


def embed_and_attach(
    conn: sqlite3.Connection,
    *,
    profile_id: int,
    seeds: list[WorkRecord],
    embedding_model: str,
    on_embed: Callable[[], None] | None = None,
    check_alive: Callable[[], None] | None = None,
) -> int:
    """Embed the chosen seeds (skipping cached vectors) and attach them,
    replacing whatever was attached before. Returns n_embedded."""
    n_embedded = 0
    embedder = get_embedder(embedding_model)
    with _GPU_LOCK:
        for i, rec in enumerate(seeds, start=1):
            oa_id = rec.paper.openalex_id
            if not embeddings_repo.has(conn, oa_id, embedding_model):
                vec = embedder(build_embedding_input(rec.paper))
                embeddings_repo.upsert(conn, oa_id, embedding_model, vec)
                n_embedded += 1
            if on_embed:
                on_embed()
            if check_alive and i % 10 == 0:
                check_alive()
    if check_alive:
        check_alive()
    # Replace only once every seed has a vector, so a failed embedding
    # never leaves a half-seeded draft behind.
    conn.execute("DELETE FROM profile_seeds WHERE profile_id = ?", (profile_id,))
    for rec in seeds:
        profiles_repo.attach_seed(conn, profile_id, rec.paper.openalex_id)
    conn.commit()
    return n_embedded


def seeds_with_vectors(conn: sqlite3.Connection, profile_id: int, embedding_model: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) FROM profile_seeds ps
        JOIN paper_embeddings pe USING (openalex_id)
        WHERE ps.profile_id = ? AND pe.embedding_model = ?
        """,
        (profile_id, embedding_model),
    ).fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# RP-format by-product
# ---------------------------------------------------------------------------


def _make_paper_id(rec: WorkRecord, taken: set[str]) -> str:
    first = (rec.raw.get("authorships") or [{}])[0]
    first_name = ((first.get("author") or {}).get("display_name") or "").strip()
    last = re.sub(r"[^a-z0-9]", "", (first_name.split()[-1] if first_name else "").lower()) or "anon"
    if not last[0].isalpha():
        last = "x" + last
    words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in (rec.paper.title or "").split()]
    words = [w for w in words if w and w not in _STOP and len(w) >= 3 and not any(c.isdigit() for c in w)]
    word = words[0] if words else "paper"
    base = f"{last}{rec.paper.year}{word}"
    if not _PAPER_ID_RE.match(base):
        base = f"{last}{rec.paper.year}work"
    pid, suffix = base, ""
    while pid in taken:
        suffix = chr(ord(suffix) + 1) if suffix else "b"
        pid = base + suffix
    taken.add(pid)
    return pid


def _first_author_last(rec: WorkRecord) -> str:
    return (rec.authors[0].split()[-1] if rec.authors and rec.authors[0].split() else "")


def _rp_paper_node(rec: WorkRecord, paper_id: str) -> dict:
    p = rec.paper
    raw = rec.raw
    ids = raw.get("ids") or {}
    oa = raw.get("open_access") or {}
    doi = p.doi
    node: dict[str, Any] = {
        "@id": f"https://doi.org/{doi}" if doi else f"https://openalex.org/{p.openalex_id}",
        "@type": "ScholarlyArticle",
        "name": p.title,
        "paper_id": paper_id,
        "openalex_id": p.openalex_id,
        "datePublished": str(p.year) if p.year else None,
        "type": rec.wtype or None,
        "author": [{"@type": "Person", "name": n} for n in rec.authors],
        "first_author": _first_author_last(rec) or None,
        "author_position": rec.position,
        "author_index": rec.author_index,
        "total_authors": rec.total_authors,
        "is_corresponding": rec.is_corresponding,
        "cited_by_count": rec.cited_by_count,
        "abstract": (p.abstract or "").strip() or None,
        "is_oa": oa.get("is_oa"),
        "oa_status": oa.get("oa_status"),
        "oa_url": oa.get("oa_url"),
        "pdf_url": p.pdf_url,
        "access": "open" if oa.get("is_oa") else "closed",
    }
    if doi:
        node["doi"] = doi
    if ids.get("pmid"):
        node["pmid"] = str(ids["pmid"]).rsplit("/", 1)[-1]
    if p.venue:
        node["isPartOf"] = {"@type": "Periodical", "name": p.venue}
    who = node["first_author"] or "Unknown"
    if rec.total_authors > 1:
        who += " et al."
    node["citation"] = f"{who}{', ' + p.venue if p.venue else ''} ({p.year})"
    return {k: v for k, v in node.items() if v is not None}


def _paper_stats(records: list[WorkRecord]) -> dict:
    stats = {"first": 0, "last": 0, "middle": 0, "unknown": 0, "corresponding": 0}
    years = [r.paper.year for r in records if r.paper.year]
    for r in records:
        pos = r.position if r.position in ("first", "last", "middle") else "unknown"
        stats[pos] += 1
        if r.is_corresponding:
            stats["corresponding"] += 1
    stats["total"] = len(records)
    stats["year_min"] = min(years) if years else 0
    stats["year_max"] = max(years) if years else 0
    return stats


def _dump(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_rp_profile(
    root: Path,
    *,
    user_id: int,
    slug: str,
    author: AuthorInfo,
    orcid: str,
    name: str,
    records: list[WorkRecord],
    seeds: list[WorkRecord],
) -> Path:
    """Write an RP-format ``lite`` profile directory. Atomic per directory.

    Plain JSON, no researcher-profiles SDK: the layout mirrors what
    ``rp validate`` accepts (see the rp-lite generator), but nothing here
    validates it. The directory is private to the importing user.
    """
    final_dir = Path(root) / str(user_id) / slug
    final_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{slug}-", dir=final_dir.parent))
    os.chmod(tmp_dir, 0o700)
    try:
        (tmp_dir / "sources").mkdir(mode=0o700)
        (tmp_dir / "meta").mkdir(mode=0o700)

        taken: set[str] = set()
        ids = {r.paper.openalex_id: _make_paper_id(r, taken) for r in records}
        seed_ids = {r.paper.openalex_id for r in seeds}

        papers_doc = {
            "@context": RP_CONTEXT,
            "@type": "Collection",
            "conformsTo": RP_CONTEXT,
            "about": {"@id": f"https://orcid.org/{orcid}"},
            "hasPart": [_rp_paper_node(r, ids[r.paper.openalex_id]) for r in records],
        }
        papers_bytes = _dump(papers_doc)
        (tmp_dir / "sources" / "papers.jsonld").write_bytes(papers_bytes)

        sidecar = [
            {
                "paper_id": ids[r.paper.openalex_id],
                "openalex_id": r.paper.openalex_id,
                "doi": r.paper.doi,
                "type": r.wtype,
                "is_seed": r.paper.openalex_id in seed_ids,
                "orcid_claimed": r.claimed,
                "primary_topic": r.raw.get("primary_topic"),
                "topics": r.raw.get("topics") or [],
                "concepts": [
                    {"id": c.get("id"), "display_name": c.get("display_name"), "score": c.get("score")}
                    for c in (r.raw.get("concepts") or [])[:10]
                ],
                "authorship": {
                    "author_position": r.position,
                    "is_corresponding": r.is_corresponding,
                    "author_index": r.author_index,
                    "total_authors": r.total_authors,
                },
            }
            for r in records
        ]
        (tmp_dir / "meta" / "openalex_topics.json").write_bytes(_dump(sidecar))

        topics = author.topics or []
        fields: dict[str, int] = {}
        subfields: list[str] = []
        for t in topics:
            f = (t.get("field") or {}).get("display_name")
            if f:
                fields[f] = fields.get(f, 0) + 1
            sf = (t.get("subfield") or {}).get("display_name")
            if sf and sf not in subfields:
                subfields.append(sf)
        field_name = max(fields, key=fields.get) if fields else None
        stats = _paper_stats(records)
        profile_doc: dict[str, Any] = {
            "@context": RP_CONTEXT,
            "@id": f"https://orcid.org/{orcid}",
            "@type": "Person",
            "conformsTo": RP_CONTEXT,
            "name": name,
            "rid": orcid,
            "provenance": "third_party",
            "license": RP_LICENSE,
            "level": "lite",
            "dateModified": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sameAs": [f"https://orcid.org/{orcid}"]
            + ([author.openalex_author_id] if author.openalex_author_id else []),
            "expertise": [t["display_name"] for t in topics[:10] if t.get("display_name")],
            "subfields": subfields[:8],
            "paper_stats": stats,
            "expertiseCitesPaperIds": False,
            "hasCitationGraph": False,
            "hasEmbeddingIndex": False,
            "hasPart": [
                {
                    "@type": "Collection",
                    "name": "Works",
                    "visibility": "public",
                    "role": "works",
                    "encodingFormat": "application/ld+json",
                    "contentUrl": "sources/papers.jsonld",
                    "bytes": len(papers_bytes),
                    "sha256": hashlib.sha256(papers_bytes).hexdigest(),
                }
            ],
            "subjectOf": [],
        }
        if author.institution:
            profile_doc["affiliation"] = (
                {"@type": "Organization", "@id": author.institution_ror, "name": author.institution}
                if author.institution_ror
                else author.institution
            )
        if field_name:
            profile_doc["field"] = field_name
            profile_doc["summary"] = (
                f"{name} is a researcher{' at ' + author.institution if author.institution else ''} "
                f"whose OpenAlex record lists {stats['total']} works ({stats['year_min']}-{stats['year_max']}) "
                f"in {field_name}, {stats['first']} as first author and {stats['last']} as last author. "
                "This lite profile was generated automatically from OpenAlex metadata."
            )
        (tmp_dir / "profile.jsonld").write_bytes(_dump(profile_doc))
        for p in tmp_dir.rglob("*"):
            os.chmod(p, 0o700 if p.is_dir() else 0o600)

        # Build sidecar the RP validator expects next to the profile tree
        # (<root>/<user_id>/.build/<slug>/meta/build_state.json): a lite
        # build downloads nothing, so every paper stays ``pending``.
        build_meta = final_dir.parent / ".build" / slug / "meta"
        build_meta.mkdir(parents=True, exist_ok=True, mode=0o700)
        (build_meta / "build_state.json").write_bytes(_dump({
            "schema_version": 1,
            "build": {"completed_phases": ["1"], "mode": "lite", "phase_retries": {}},
            "inputs": {"reporter_supplement": False, "websites": []},
            "papers": {pid: {"status": "pending", "contaminated": False} for pid in ids.values()},
        }))
        (build_meta / "build_report.json").write_bytes(_dump({
            "orcid": orcid, "slug": slug, "openalex_author": author.openalex_author_id,
            "n_records": len(records), "n_seeds": len(seeds),
        }))

        if final_dir.exists():
            stale = final_dir.with_name(f".{slug}-old-{int(time.time())}")
            os.replace(final_dir, stale)
            _rmtree(stale)
        os.replace(tmp_dir, final_dir)
        return final_dir
    except Exception:
        _rmtree(tmp_dir)
        raise


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _step(reporter, name: str, **kw) -> None:
    if reporter is not None:
        reporter.step(name, **kw)


def _check_alive_fn(conn: sqlite3.Connection, profile_id: int) -> Callable[[], None]:
    def check_alive() -> None:
        r = profiles_repo.get(conn, profile_id)
        if r is None or not r["is_draft"]:
            raise ImportAborted("draft was deleted or committed during the import")
    return check_alive


def _embedding_model_for(conn: sqlite3.Connection, settings, profile_id: int) -> str:
    row = profiles_repo.get(conn, profile_id)
    if row is not None and row["embedding_model"]:
        return row["embedding_model"]
    return settings.RADAR_DEFAULT_EMBEDDING_MODEL


def run_fetch(
    conn: sqlite3.Connection,
    settings,
    *,
    user_id: int,
    profile_id: int,
    slug: str,
    orcid: str,
    name: str,
    client,
    reporter=None,
    author: AuthorInfo | None = None,
) -> OrcidImportResult:
    """Phase A: fetch the researcher's works, store them, offer them for seeding.

    Nothing is embedded or attached here. The default rule's pick is
    recorded on ``orcid_works.default_selected`` so the picker can
    pre-check it; :func:`run_seed` does the rest once the user confirms.

    Raises ``LookupError`` when the ORCID is unknown to OpenAlex,
    ``ValueError`` when it has no usable works, ``ImportAborted`` when the
    draft vanished mid-way.
    """
    check_alive = _check_alive_fn(conn, profile_id)

    _step(reporter, "fetch_author", message="Resolving author on OpenAlex")
    if author is None:
        author = resolve_author(client, orcid)
    if author is None:
        raise LookupError(f"ORCID {orcid} not found on OpenAlex")

    _step(reporter, "fetch_works", message="Reading the ORCID record")
    claimed = getattr(client, "orcid_claimed", None)
    if claimed is None and not getattr(client, "skip_orcid_registry", False):
        claimed = fetch_orcid_claimed(orcid)

    _step(reporter, "fetch_works", message="Fetching works from OpenAlex")
    raws = fetch_works(
        client, orcid,
        on_wait=lambda msg: _step(reporter, "fetch_works", message=msg),
    )
    records, report = build_records(
        raws, orcid, author.openalex_author_id, client=client, claimed=claimed,
    )
    log.info("orcid_import.fetched", orcid=orcid, **report)
    if not records:
        raise ValueError(f"no usable works found on OpenAlex for ORCID {orcid}")
    check_alive()

    warnings: list[str] = []
    if claimed is None:
        warnings.append("ORCID registry unavailable; identity not cross-checked against the person's own record")
    elif not claimed.n_works:
        warnings.append("the ORCID record lists no works; every OpenAlex work attributed to this ORCID was kept")
    elif report["dropped_unclaimed_field"]:
        warnings.append(
            f"{report['dropped_unclaimed_field']} OpenAlex works dropped: not on the ORCID record and in a field "
            f"the claimed works never touch (likely a merged same-name author)"
        )

    defaults, dup_of, seed_warnings = seed_defaults(records, settings.RADAR_ORCID_MAX_SEEDS)
    # Reword the rule's notes for a picker: the user can override every one of them.
    n_lead = sum(1 for r in records if r.is_lead and r.seed_eligible)
    if defaults:
        warnings.append(
            f"{n_lead} lead-author works; the newest {len(defaults)} are pre-checked "
            f"(duplicate preprint/repository copies and datasets are listed unchecked)"
        )
    warnings.extend(w for w in seed_warnings if "no first/last" in w or "no seed-eligible" in w)

    _step(reporter, "store_works", total=len(records), message=f"Storing {len(records)} works")
    store_works(conn, profile_id=profile_id, records=records, defaults=defaults, dup_of=dup_of)
    check_alive()

    conn.execute(
        """
        UPDATE profiles
           SET orcid = ?, researcher_name = ?, updated_at = datetime('now')
         WHERE id = ?
        """,
        (orcid, author.display_name or None, profile_id),
    )
    conn.commit()

    rp_dir: str | None = None
    _step(reporter, "write_rp", message="Writing researcher-profile files")
    try:
        rp_dir = str(
            write_rp_profile(
                Path(settings.RADAR_RP_PROFILES_DIR),
                user_id=user_id, slug=slug, author=author, orcid=orcid,
                name=name, records=records, seeds=[],
            )
        )
    except Exception as exc:  # noqa: BLE001 — the by-product must never fail the import
        warnings.append(f"RP profile not written: {type(exc).__name__}: {exc}")
        log.warning("orcid_import.rp_write_failed", error=str(exc))

    return OrcidImportResult(
        phase="fetch",
        n_fetched=report["fetched"],
        n_kept=report["kept"],
        n_works=len(records),
        n_default_seeds=len(defaults),
        n_seeds=0,
        n_embedded=0,
        author=author.to_schema(),
        rp_profile_dir=rp_dir,
        warnings=warnings,
        report=report,
    )


def patch_rp_seed_flags(root: Path, *, user_id: int, slug: str, seed_ids: set[str]) -> None:
    """Rewrite ``is_seed`` in the RP sidecar and ``n_seeds`` in the build report
    after the user's selection. Silent when the files are missing."""
    base = Path(root) / str(user_id)
    sidecar = base / slug / "meta" / "openalex_topics.json"
    if sidecar.exists():
        try:
            rows = json.loads(sidecar.read_text(encoding="utf-8"))
            for r in rows:
                r["is_seed"] = r.get("openalex_id") in seed_ids
            sidecar.write_bytes(_dump(rows))
        except (OSError, ValueError) as exc:
            log.warning("orcid_import.rp_sidecar_patch_failed", error=str(exc))
    report = base / ".build" / slug / "meta" / "build_report.json"
    if report.exists():
        try:
            d = json.loads(report.read_text(encoding="utf-8"))
            d["n_seeds"] = len(seed_ids)
            report.write_bytes(_dump(d))
        except (OSError, ValueError) as exc:
            log.warning("orcid_import.rp_report_patch_failed", error=str(exc))


def run_seed(
    conn: sqlite3.Connection,
    settings,
    *,
    user_id: int,
    profile_id: int,
    slug: str,
    selected_ids: list[str],
    reporter=None,
    after_topics: Callable[[sqlite3.Connection, int, dict], dict] | None = None,
) -> OrcidImportResult:
    """Phase B: embed the chosen works, attach them as seeds, build the concept list.

    ``selected_ids`` must be fetched works of this draft that are
    seed-eligible (``ValueError`` otherwise). Re-running with a different
    selection replaces the seeds; vectors already computed are reused.
    ``after_topics`` lets a caller post-process the concept list (the
    profile import's expertise / not_interests signal).
    """
    check_alive = _check_alive_fn(conn, profile_id)
    embedding_model = _embedding_model_for(conn, settings, profile_id)

    wanted = list(dict.fromkeys(selected_ids))
    if not wanted:
        raise ValueError("select at least one work to seed the topic")
    records = load_work_records(conn, profile_id)
    by_id = {r.paper.openalex_id: r for r in records}
    unknown = [i for i in wanted if i not in by_id]
    if unknown:
        raise ValueError(f"{len(unknown)} selected id(s) are not fetched works of this draft: {unknown[:3]}")
    ineligible = [i for i in wanted if not by_id[i].seed_eligible]
    if ineligible:
        raise ValueError(f"{len(ineligible)} selected work(s) cannot be seeds (datasets): {ineligible[:3]}")
    # Keep the picker's order (newest first) rather than the request order.
    seeds = [r for r in records if r.paper.openalex_id in set(wanted)]
    works_repo.set_selected(conn, profile_id, wanted)

    warnings: list[str] = []
    if len(seeds) > settings.RADAR_ORCID_MAX_SEEDS:
        warnings.append(
            f"{len(seeds)} seeds selected (the default rule keeps {settings.RADAR_ORCID_MAX_SEEDS}); "
            f"gathering and coherence scale with the seed count"
        )
    n_lead = sum(1 for r in seeds if r.is_lead)
    if n_lead < len(seeds):
        warnings.append(f"{len(seeds) - n_lead} selected seed(s) are middle-author works")

    _step(reporter, "embedding", total=len(seeds), message=f"Embedding {len(seeds)} seed papers")
    tick = reporter.make_embed_tick(len(seeds)) if reporter is not None else None
    n_embedded = embed_and_attach(
        conn, profile_id=profile_id, seeds=seeds, embedding_model=embedding_model,
        on_embed=tick, check_alive=check_alive,
    )
    _step(reporter, "attach_seeds", message=f"Attached {len(seeds)} seeds")

    have = seeds_with_vectors(conn, profile_id, embedding_model)
    if have < len(seeds):
        raise RuntimeError(
            f"only {have}/{len(seeds)} seeds have a {embedding_model!r} vector; "
            "embedding model mismatch between the draft and the embedder"
        )

    _step(reporter, "topics", message="Aggregating concepts from the selected seeds")
    # Concepts come from the selected seeds only (user decision 2026-09-22);
    # lead-author works still weigh double.
    topic_filters = researcher_topic_filters(
        seeds, top_k=settings.RADAR_ORCID_TOPIC_TOP_K, seeds=seeds,
    )
    if after_topics is not None:
        _step(reporter, "rp_signals", message="Applying the profile's expertise and not-interests")
        topic_filters = after_topics(conn, profile_id, topic_filters)
    conn.execute(
        """
        UPDATE profiles
           SET topic_filters_json = ?, n_seed = ?, updated_at = datetime('now')
         WHERE id = ?
        """,
        (json.dumps(topic_filters), len(seeds), profile_id),
    )
    conn.commit()

    _step(reporter, "write_rp", message="Recording the seed choice in the researcher-profile files")
    patch_rp_seed_flags(
        Path(settings.RADAR_RP_PROFILES_DIR), user_id=user_id, slug=slug,
        seed_ids={r.paper.openalex_id for r in seeds},
    )

    row = profiles_repo.get(conn, profile_id)
    author = OrcidAuthor(
        orcid=(row["orcid"] if row is not None else "") or "",
        display_name=(row["researcher_name"] if row is not None else "") or "",
    )
    return OrcidImportResult(
        phase="seed",
        n_fetched=0,
        n_kept=len(records),
        n_works=len(records),
        n_default_seeds=len(works_repo.default_ids(conn, profile_id)),
        n_seeds=len(seeds),
        n_embedded=n_embedded,
        author=author,
        rp_profile_dir=str(Path(settings.RADAR_RP_PROFILES_DIR) / str(user_id) / slug),
        warnings=warnings,
        report={"selected": len(wanted), "embedded": n_embedded},
    )


def run_import(
    conn: sqlite3.Connection,
    settings,
    *,
    user_id: int,
    profile_id: int,
    slug: str,
    orcid: str,
    name: str,
    client,
    reporter=None,
    author: AuthorInfo | None = None,
    selection: str | list[str] = "default",
    after_topics: Callable[[sqlite3.Connection, int, dict], dict] | None = None,
) -> OrcidImportResult:
    """One-shot fetch + seed (the CLI path and the inline test path).

    ``selection`` is ``"default"`` (the rule's pick), ``"lead"`` (every
    lead-author work, uncapped), ``"all"`` (every seed-eligible work) or an
    explicit list of OpenAlex ids.
    """
    fetch = run_fetch(
        conn, settings, user_id=user_id, profile_id=profile_id, slug=slug,
        orcid=orcid, name=name, client=client, reporter=reporter, author=author,
    )
    if isinstance(selection, list):
        ids = selection
    else:
        records = load_work_records(conn, profile_id)
        if selection == "all":
            ids = [r.paper.openalex_id for r in records if r.seed_eligible]
        elif selection == "lead":
            defaults, dup_of, _ = seed_defaults(records, 0)
            ids = [r.paper.openalex_id for r in defaults]
        else:
            ids = works_repo.default_ids(conn, profile_id)
    seed = run_seed(
        conn, settings, user_id=user_id, profile_id=profile_id, slug=slug,
        selected_ids=ids, reporter=reporter, after_topics=after_topics,
    )
    seed.phase = "seed"
    seed.n_fetched = fetch.n_fetched
    seed.n_kept = fetch.n_kept
    seed.author = fetch.author
    seed.report = {**fetch.report, **seed.report}
    seed.warnings = fetch.warnings + seed.warnings
    return seed


def has_unfinished_import(conn: sqlite3.Connection, profile_id: int) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM gather_runs
        WHERE profile_id = ? AND tier_used IN ('orcid_import', 'orcid_seed') AND finished_at IS NULL
        LIMIT 1
        """,
        (profile_id,),
    ).fetchone()
    return row is not None


def find_inflight_import(conn: sqlite3.Connection, user_id: int, orcid: str) -> sqlite3.Row | None:
    """An unfinished import of this ORCID by this user: ``(slug, run_id)`` row or None."""
    return conn.execute(
        """
        SELECT p.slug AS slug, g.id AS run_id
        FROM gather_runs g JOIN profiles p ON p.id = g.profile_id
        WHERE p.user_id = ? AND p.orcid = ? AND g.tier_used IN ('orcid_import', 'orcid_seed')
          AND g.finished_at IS NULL
        ORDER BY g.id DESC LIMIT 1
        """,
        (user_id, orcid),
    ).fetchone()


def list_seed_docs(conn: sqlite3.Connection, profile_id: int, slug: str) -> list[dict]:
    """Seeds of a profile in the ``VaultDoc`` shape, regardless of who uploaded them."""
    out: list[dict] = []
    for oa_id in profiles_repo.list_seed_openalex_ids(conn, profile_id):
        prow = papers_repo.get_by_openalex_id(conn, oa_id)
        if prow is None:
            continue
        out.append(
            {
                "id": prow["openalex_id"],
                "title": prow["title"] or prow["openalex_id"],
                "authors": papers_repo.decode_authors(prow),
                "venue": prow["venue"] or "",
                "tags": [slug],
                "pages": int(prow["n_pages"] or 0),
                "chunks": 0,
                "added": prow["first_seen_at"] or "",
                "year": prow["year"],
            }
        )
    out.sort(key=lambda d: (-(d["year"] or 0), d["title"]))
    return out


__all__ = [
    "AuthorInfo",
    "ClaimedWorks",
    "EXCLUDED_TYPES",
    "ImportAborted",
    "LEAD_WEIGHT",
    "SEED_EXCLUDED_TYPES",
    "WorkRecord",
    "build_records",
    "classify_authorship",
    "embed_and_attach",
    "fetch_orcid_claimed",
    "fetch_works",
    "find_inflight_import",
    "load_work_records",
    "patch_rp_seed_flags",
    "run_fetch",
    "run_seed",
    "seed_defaults",
    "store_works",
    "work_rows",
    "works_for_draft",
    "has_unfinished_import",
    "list_seed_docs",
    "normalize_orcid",
    "persist_records",
    "researcher_topic_filters",
    "resolve_author",
    "run_import",
    "select_seeds",
    "write_rp_profile",
]
