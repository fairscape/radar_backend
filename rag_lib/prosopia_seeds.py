"""Prosopia works entry -> RADAR seed paper.

A Prosopia ``hasPart`` entry is a JSON-LD ``ScholarlyArticle``. A fully
built one carries both an ``openalex_id`` and a ``doi``; an entry built
before Prosopia learned to resolve identifiers, or built by someone
else's pipeline, arrives with a title and a URL and nothing else. The
ladder in :func:`resolve_seed` covers both, identifier-first so a
well-built profile costs almost no search calls:

    openalex_id -> doi -> pmcid -> title -> synthetic

The last rung is not a failure mode to be hidden. A paper that resolves
nowhere still gets a row, keyed ``prosopia:{slug}:{paper_id}`` (the same
shape as the vault's ``local:<file_hash>``), and is embedded from its
Prosopia summary rather than dropped. What matters is that the *caller*
can see which rung caught each paper: ``resolved_by`` counts are how you
tell a profile that imported cleanly from one that was reconstructed by
fuzzy title match.

Everything in this module is pure apart from the calls it makes through
the ``OpenAlexClient`` duck it is handed, so the tests stub that client
and never touch the network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable, Iterator

import structlog


log = structlog.get_logger("rag_lib.prosopia_seeds")


# The rungs, narrowest (most trustworthy) first. ``none`` means the
# synthetic id — the paper is still imported, just unlinked.
RUNGS: tuple[str, ...] = ("work_id", "doi", "pmcid", "title", "none")

# OpenAlex caps a filter's OR-list well above this, but 50 is the
# per-page ceiling that keeps one batch to one request.
DOI_BATCH_SIZE = 50

# How close a title-search hit must be before we accept it as the same
# paper. OpenAlex ranks by relevance, so rank 1 for a title it does not
# have is a plausible-looking neighbour; without this check the lowest
# rung quietly injects the wrong paper into the seed corpus.
TITLE_MATCH_MIN_RATIO = 0.90

_BIORXIV_HOSTS = ("biorxiv.org", "medrxiv.org")
# The modern Cold Spring Harbor accession: 2025.11.03.685753.
_BIORXIV_ACCESSION = re.compile(r"\d{4}\.\d{2}\.\d{2}\.\d{6}")
# The pre-2019 form: a bare 6+ digit serial as a path segment.
_BIORXIV_LEGACY = re.compile(r"^(\d{6,})(?:v\d+)?$")
_PMCID = re.compile(r"PMC\d+", re.IGNORECASE)
_DOI_IN_URL = re.compile(r"doi\.org/(10\.\d{4,9}/[^\s?#]+)", re.IGNORECASE)
_OPENALEX_PREFIX = "https://openalex.org/"


@dataclass
class WorkCache:
    """Pre-fetched OpenAlex works, keyed both ways.

    Populated by :func:`prefetch_works` from batched DOI queries. Keying
    by id as well as DOI is what keeps the ``work_id`` rung free: the
    batch that answered a paper's DOI usually also answers its
    ``openalex_id``, so the ladder's first rung is a dict lookup rather
    than a per-paper GET.
    """

    by_doi: dict[str, dict] = field(default_factory=dict)
    by_id: dict[str, dict] = field(default_factory=dict)

    def __len__(self) -> int:  # pragma: no cover - diagnostics only
        return len(self.by_id) or len(self.by_doi)


@dataclass
class Resolution:
    """One resolved seed: the id we will key it by, the row, the rung."""

    paper_id: str
    openalex_id: str
    paper: dict[str, Any]
    rung: str

    @property
    def resolved(self) -> bool:
        return self.rung != "none"


# ----------------------------------------------------------------------
# Extractors
# ----------------------------------------------------------------------


def _candidate_urls(entry: dict) -> Iterator[str]:
    """Every field on an entry that might hold a resolvable URL.

    ``@id`` is a DOI URL on a built entry and an internal
    ``#paper/<id>`` fragment on an unbuilt one, so it is worth reading
    but never worth trusting.
    """
    for key in ("full_text_link", "@id", "url", "sameAs"):
        value = entry.get(key)
        if isinstance(value, str) and value.lower().startswith("http"):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.lower().startswith("http"):
                    yield item


def normalize_doi(doi: str | None) -> str | None:
    """Bare, lowercase DOI from any of the forms sources hand us."""
    raw = (doi or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
                   "http://dx.doi.org/", "doi:"):
        if lowered.startswith(prefix):
            raw = raw[len(prefix):]
            lowered = raw.lower()
            break
    raw = raw.strip().rstrip(".,;")
    if not raw.lower().startswith("10."):
        return None
    return raw.lower()


def _doi_from_url(url: str) -> str | None:
    """Pull a DOI out of a URL, including the ones bioRxiv only implies."""
    m = _DOI_IN_URL.search(url)
    if m:
        return normalize_doi(m.group(1))

    lowered = url.lower()
    if not any(host in lowered for host in _BIORXIV_HOSTS):
        return None

    # Modern accession anywhere in the path:
    #   .../early/2025/11/04/2025.11.03.685753.full.pdf -> 10.1101/2025.11.03.685753
    m = _BIORXIV_ACCESSION.search(url)
    if m:
        return f"10.1101/{m.group(0)}"

    # Legacy serial as its own path segment:
    #   .../early/2016/01/22/037689.full.pdf -> 10.1101/037689
    # Date segments are 2 or 4 digits, so the 6+ floor excludes them.
    path = url.split("?", 1)[0].split("#", 1)[0]
    for segment in path.split("/"):
        stem = segment.split(".", 1)[0]
        m = _BIORXIV_LEGACY.match(stem)
        if m:
            return f"10.1101/{m.group(1)}"
    return None


def _doi_from_entry(entry: dict) -> str | None:
    """Three shapes, all present in live Prosopia data.

    An explicit ``doi`` field; a ``doi.org/`` URL inside a link; and a
    bioRxiv URL whose accession *is* the DOI suffix under ``10.1101``.
    """
    explicit = normalize_doi(entry.get("doi"))
    if explicit:
        return explicit
    for url in _candidate_urls(entry):
        found = _doi_from_url(url)
        if found:
            return found
    return None


def _pmcid_from_link(entry: dict) -> str | None:
    """``PMC\\d+`` anywhere in any of the entry's URLs."""
    for url in _candidate_urls(entry):
        m = _PMCID.search(url)
        if m:
            return m.group(0).upper()
    return None


def _year_hint(entry: dict) -> int | None:
    """``datePublished`` is a string and may be a full date."""
    raw = entry.get("datePublished") or entry.get("year")
    if raw is None:
        return None
    m = re.search(r"\d{4}", str(raw))
    return int(m.group(0)) if m else None


def _venue_from_entry(entry: dict) -> str | None:
    part_of = entry.get("isPartOf")
    if isinstance(part_of, dict):
        name = part_of.get("name")
        if name:
            return str(name)
    if isinstance(part_of, str) and part_of:
        return part_of
    return None


def _paper_id(entry: dict) -> str:
    """A stable per-profile key for the entry.

    ``paper_id`` is the citekey Prosopia builds everything else against
    (summaries, citation graph), so it is the right key. Entries without
    one fall back to a slug of the title so the synthetic id is still
    deterministic across imports.
    """
    pid = entry.get("paper_id") or entry.get("paperId")
    if pid:
        return str(pid)
    title = (entry.get("name") or entry.get("title") or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", title).strip("-")
    return slug[:64] or "unknown"


def _normalize_openalex_id(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith(_OPENALEX_PREFIX):
        raw = raw[len(_OPENALEX_PREFIX):]
    return raw.rsplit("/", 1)[-1].upper()


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _best_title_match(
    title: str,
    works: Iterable[dict] | None,
    *,
    min_ratio: float = TITLE_MATCH_MIN_RATIO,
) -> dict | None:
    """The hit whose title really is this title, or ``None``."""
    target = _norm_title(title)
    if not target:
        return None
    best: dict | None = None
    best_ratio = 0.0
    for work in works or []:
        candidate = _norm_title(work.get("title") or work.get("display_name") or "")
        if not candidate:
            continue
        ratio = SequenceMatcher(None, target, candidate).ratio()
        if ratio > best_ratio:
            best, best_ratio = work, ratio
    if best is not None and best_ratio >= min_ratio:
        return best
    if best is not None:
        log.info(
            "prosopia.title_match_rejected",
            title=title[:120], ratio=round(best_ratio, 3),
            best=(best.get("title") or "")[:120],
        )
    return None


# ----------------------------------------------------------------------
# Batched prefetch
# ----------------------------------------------------------------------


def prefetch_works(
    entries: Iterable[dict],
    client: Any,
    *,
    batch_size: int = DOI_BATCH_SIZE,
) -> WorkCache:
    """One request per 50 DOIs instead of one request per paper.

    Every entry that can produce a DOI — explicitly or from its URL —
    goes into the batch, so the identifier rungs collapse into a handful
    of round trips. A batch that fails is logged and skipped; the ladder
    then falls back to per-paper lookups for those entries, which is
    slower but not wrong.
    """
    dois: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        doi = _doi_from_entry(entry)
        if doi and doi not in seen:
            seen.add(doi)
            dois.append(doi)

    cache = WorkCache()
    for start in range(0, len(dois), batch_size):
        chunk = dois[start:start + batch_size]
        filter_str = "doi:" + "|".join(chunk)
        try:
            works = client.paginate_filter(filter_str, per_page=batch_size)
        except Exception as exc:  # noqa: BLE001 — degrade to per-paper
            log.warning(
                "prosopia.prefetch_batch_failed",
                n=len(chunk), reason=type(exc).__name__, detail=str(exc)[:200],
            )
            continue
        for work in works or []:
            doi = normalize_doi(work.get("doi"))
            if doi:
                cache.by_doi[doi] = work
            wid = _normalize_openalex_id(work.get("id"))
            if wid:
                cache.by_id[wid] = work

    log.info(
        "prosopia.prefetch",
        n_dois=len(dois), n_works=len(cache.by_id), n_batches=-(-len(dois) // batch_size),
    )
    return cache


# ----------------------------------------------------------------------
# The ladder
# ----------------------------------------------------------------------


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run one lookup; a failed rung drops to the next rather than dying.

    An 82-paper import must not be sunk by one 404-shaped surprise. The
    warning carries the reason so a systematic failure (a 429, say) is
    visible in the logs rather than showing up only as a suspiciously
    high ``none`` count.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "prosopia.lookup_failed",
            fn=getattr(fn, "__name__", str(fn)),
            reason=type(exc).__name__, detail=str(exc)[:200],
        )
        return None


def resolve_seed(
    entry: dict,
    client: Any,
    *,
    slug: str = "",
    cache: WorkCache | None = None,
    fallback_text: str | None = None,
) -> Resolution:
    """Prosopia paper entry -> a paper row plus the rung that found it.

    ``client`` only needs the OpenAlexClient surface used here:
    ``get_work``, ``lookup_by_doi``, ``lookup_by_pmcid``,
    ``search_by_title`` and ``paper_from_work``.

    ``cache`` short-circuits the two identifier rungs with the batched
    prefetch; pass ``None`` and each entry pays its own request.

    ``fallback_text`` is the body we embed a *non*-resolving paper from
    — the Prosopia summary, optionally with the profile's public
    summary artifact appended. Defaults to the entry's own ``summary``.
    """
    paper_id = _paper_id(entry)
    work: dict | None = None
    rung = "none"

    oa_id = _normalize_openalex_id(entry.get("openalex_id"))
    if oa_id:
        work = (cache.by_id.get(oa_id) if cache else None)
        if work is None:
            work = _call(client.get_work, oa_id)
        if work is not None:
            rung = "work_id"

    if work is None:
        doi = _doi_from_entry(entry)
        if doi:
            work = (cache.by_doi.get(doi) if cache else None)
            if work is None:
                work = _call(client.lookup_by_doi, doi)
            if work is not None:
                rung = "doi"

    if work is None:
        pmcid = _pmcid_from_link(entry)
        if pmcid:
            work = _call(client.lookup_by_pmcid, pmcid)
            if work is not None:
                rung = "pmcid"

    if work is None:
        title = (entry.get("name") or entry.get("title") or "").strip()
        if title:
            hits = _call(
                client.search_by_title, title, _year_hint(entry), per_page=5,
            ) or []
            work = _best_title_match(title, hits)
            if work is not None:
                rung = "title"

    if work is not None:
        paper = _paper_from_work(client, work, entry)
        return Resolution(
            paper_id=paper_id,
            openalex_id=paper["openalex_id"],
            paper=paper,
            rung=rung,
        )

    paper = _paper_from_entry(entry, slug=slug, fallback_text=fallback_text)
    return Resolution(
        paper_id=paper_id,
        openalex_id=paper["openalex_id"],
        paper=paper,
        rung="none",
    )


def synthetic_id(slug: str, paper_id: str) -> str:
    """``prosopia:{slug}:{paper_id}`` — mirrors the vault's ``local:<hash>``."""
    return f"prosopia:{slug or 'unknown'}:{paper_id}"


def _paper_from_work(client: Any, work: dict, entry: dict) -> dict[str, Any]:
    """OpenAlex wins on every field it has; Prosopia fills the gaps."""
    paper = client.paper_from_work(work, source="prosopia")
    return {
        "openalex_id": paper.openalex_id or _normalize_openalex_id(work.get("id")),
        "doi": paper.doi or _doi_from_entry(entry),
        "title": paper.title or entry.get("name") or "",
        "abstract": paper.abstract or entry.get("abstract") or "",
        "year": paper.year if paper.year is not None else _year_hint(entry),
        "venue": paper.venue or _venue_from_entry(entry),
        "source": "prosopia",
        "primary_topic": (
            paper.primary_topic.to_dict() if paper.primary_topic else None
        ),
        "topics": [t.to_dict() for t in (paper.topics or [])],
        "authors": _authors_from_work(work),
        "pdf_url": paper.pdf_url or entry.get("full_text_link"),
        "oa_status": paper.oa_status,
        "body_text": None,
    }


def _paper_from_entry(
    entry: dict,
    *,
    slug: str,
    fallback_text: str | None = None,
) -> dict[str, Any]:
    """The synthetic row for a paper OpenAlex could not be shown to hold.

    ``abstract`` stays whatever Prosopia had (usually empty), and the
    Prosopia summary goes into ``body_text``. ``build_embedding_input``
    emits a ``BODY:`` section and, when there is no abstract, fills the
    token budget from the body — so a summary-only paper still embeds
    against real text instead of its title alone.
    """
    paper_id = _paper_id(entry)
    body = fallback_text if fallback_text is not None else (entry.get("summary") or "")
    return {
        "openalex_id": synthetic_id(slug, paper_id),
        "doi": _doi_from_entry(entry),
        "title": entry.get("name") or entry.get("title") or paper_id,
        "abstract": entry.get("abstract") or "",
        "year": _year_hint(entry),
        "venue": _venue_from_entry(entry),
        "source": "prosopia",
        "primary_topic": None,
        "topics": [],
        "authors": [],
        "pdf_url": entry.get("full_text_link"),
        "oa_status": None,
        "body_text": body or None,
    }


def _authors_from_work(work: dict) -> list[str]:
    out: list[str] = []
    for entry in (work.get("authorships") or []):
        author = entry.get("author") or {}
        name = author.get("display_name")
        if name:
            out.append(name)
    return out
