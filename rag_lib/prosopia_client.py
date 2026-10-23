"""Read a published Prosopia profile through the researcher-profiles SDK.

A Prosopia instance *is* a ``researcher_profiles`` API server, and RADAR
reads it through the SDK's HTTP storage (``ApiArtifactStorage``), whose
``PaperRecord`` normalizes each work's identifiers. In October 2026
Prosopia changed the profile, works and summary routes in ways rp-sdk 0.3.0
does not read, so ``PagedApiStorage`` overrides those reads and
``ApiProfile`` takes the place of ``ResearcherProfile.from_api``; see
``PagedApiStorage`` for the shapes.

What stays is the part the SDK has no opinion about — RADAR's own failure
vocabulary. ``ProfileNotFound`` is the router's 404 and ``ProsopiaError``
its 502, and every way a read can fail maps onto one of the two. The
SDK signals "no such profile" with a bare ``KeyError(slug)``; letting
that escape as a 500 is the specific thing this module exists to prevent.

Read-only and unauthenticated by design: the reading routes have no
credential gate, so an anonymous caller sees the public tier, which is
all a seed corpus needs.
"""

from __future__ import annotations

import re
from typing import Any

import requests
import structlog

from .db.repos.papers import plausible_year
from researcher_profiles.client import ApiArtifactStorage
from researcher_profiles.profile.storage import LazySummaries
from researcher_profiles.schema import PaperRecord


log = structlog.get_logger("rag_lib.prosopia_client")


DEFAULT_BASE_URL = "https://prosopia.databio.org"

# 0000-0001-5643-4068 — four groups, last char may be X. Matched on the
# whole string so a slug that merely contains digits is never mistaken.
ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", re.IGNORECASE)


def looks_like_orcid(value: str) -> bool:
    """True for a bare ORCID iD (``0000-0001-5643-4068``)."""
    return bool(ORCID_RE.match((value or "").strip()))


def allowed_base_url(base_url: str | None, allowed: list[str]) -> str:
    """``base_url`` normalized, if it is one of ``allowed`` (or the default).

    Raises ``ValueError`` (the router's 400) for anything else. The match
    is on the whole normalized URL, not a prefix, so
    ``https://prosopia.databio.org.evil.example`` does not pass.
    """
    want = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    ok = {DEFAULT_BASE_URL.rstrip("/")} | {a.strip().rstrip("/") for a in allowed if a and a.strip()}
    if want.lower() not in {a.lower() for a in ok}:
        raise ValueError("base_url is not a Prosopia instance this server reads")
    return want


class ProsopiaError(RuntimeError):
    """Any failure talking to a Prosopia instance."""


class ProfileNotFound(ProsopiaError):
    """The slug does not name a profile visible to this caller."""


# The server caps a page of works at 100, and a summaries read at 20 ids.
PAPERS_PAGE_SIZE = 100
SUMMARIES_BATCH = 20
# A runaway cursor (one that never says has_more=false) stops here.
MAX_PAPER_PAGES = 100


class PagedApiStorage(ApiArtifactStorage):
    """The SDK's HTTP storage, reading the API Prosopia serves now.

    rp-sdk 0.3.0 expects the old shapes, and Prosopia changed three of
    them, so every import failed:

    - ``GET /profiles/{slug}`` carries the profile in ``fields`` where it
      used to be ``metadata``;
    - ``GET /profiles/{slug}/papers`` is an envelope (``items``,
      ``next_cursor``, ``has_more``; 20 a page by default) where it was a
      bare list, which the SDK iterated key by key;
    - each work's ``summary`` is ``{"available": bool, ...}`` where it was a
      ``summary_available`` flag, and the bodies come from
      ``GET /summaries?ids=`` (``/summary/{id}`` is gone).

    The old shapes are still read, so a server on either side works.
    """

    def _paper_entries(self) -> list[dict]:
        if self._paper_index_cache is not None:
            return self._paper_index_cache
        entries: list[dict] = []
        cursor: str | None = None
        for _ in range(MAX_PAPER_PAGES):
            params: dict[str, Any] = {"limit": PAPERS_PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", self._api_path("papers"), params=params)
            if isinstance(data, list):  # the unpaged server
                entries = [e for e in data if isinstance(e, dict)]
                break
            if not isinstance(data, dict):
                raise RuntimeError(f"unexpected papers payload: {type(data).__name__}")
            entries.extend(e for e in (data.get("items") or []) if isinstance(e, dict))
            cursor = data.get("next_cursor")
            if not data.get("has_more") or not cursor:
                break
        else:
            raise RuntimeError(f"papers list did not end after {MAX_PAPER_PAGES} pages")
        self._paper_index_cache = entries
        return entries

    def _detail(self) -> dict:
        # ``view=full`` carries the soul and expertise texts inline; the
        # default ``record`` view only says they exist.
        if self._detail_cache is None:
            self._detail_cache = self._request("GET", self._api_path(), params={"view": "full"})
        return self._detail_cache

    def fields(self) -> dict:
        """The profile's own fields (``fields`` now, ``metadata`` before)."""
        d = self._detail()
        if not isinstance(d, dict):
            return {}
        for part in ("fields", "metadata"):
            block = d.get(part)
            if isinstance(block, dict):
                return block
        return {}

    def text_part(self, name: str) -> str:
        value = self._detail().get(name) if isinstance(self._detail(), dict) else None
        return value if isinstance(value, str) else ""

    def load_papers(self) -> list[PaperRecord]:
        # One work at a time: a single malformed entry (a null title, a null
        # in its authors) failed the whole profile with a 502. It is skipped.
        out: list[PaperRecord] = []
        for e in self._paper_entries():
            if not isinstance(e, dict):
                continue
            authors = e.get("authors")
            if isinstance(authors, list):
                authors = [a for a in authors if isinstance(a, str) and a.strip()]
            try:
                out.append(PaperRecord.model_validate(
                    {
                        "paper_id": e.get("paper_id"),
                        "title": e.get("title") or "",
                        "year": plausible_year(e.get("year")),
                        "journal": e.get("journal"),
                        "first_author": e.get("first_author"),
                        "authors": authors,
                        "doi": e.get("doi"),
                        "pmid": e.get("pmid"),
                        "openalex_id": e.get("openalex_id"),
                        "full_text_link": e.get("full_text_link"),
                    }
                ))
            except ValueError as exc:  # pydantic's ValidationError is one
                log.warning("prosopia.paper_skipped", paper_id=e.get("paper_id"), error=str(exc)[:200])
        return out

    def _fetch_summary(self, paper_id: str) -> str:
        data = self._request("GET", self._api_path("summaries"), params={"ids": paper_id})
        if isinstance(data, dict) and isinstance(data.get("summaries"), dict):
            return data["summaries"].get(paper_id) or ""
        return super()._fetch_summary(paper_id)

    def load_summaries(self):
        ids = [
            e["paper_id"]
            for e in self._paper_entries()
            if e.get("paper_id") and _summary_available(e)
        ]
        return LazySummaries(ids, self._fetch_summary)


def _summary_available(entry: dict) -> bool:
    if entry.get("summary_available"):
        return True
    summary = entry.get("summary")
    return isinstance(summary, dict) and bool(summary.get("available"))


class ApiProfile:
    """What RADAR reads of a profile: name, papers, summaries, and for the
    stored researcher its rid, affiliation, fields, expertise and soul.

    ``ResearcherProfile`` would also validate the profile document, whose
    shape Prosopia changed; RADAR never reads the rest of it.
    """

    def __init__(self, storage: PagedApiStorage):
        self._storage = storage
        self.slug = storage.slug
        self.papers = storage.load_papers()
        fields = storage.fields()
        detail = storage._detail()
        self.metadata = fields
        self.name = str(fields.get("name") or "") or storage.slug
        self.rid = str(detail.get("rid") or fields.get("rid") or "") if isinstance(detail, dict) else ""
        self.affiliation = fields.get("affiliation")
        self.expertise = storage.text_part("expertise")
        self.soul = storage.text_part("soul")
        self.grants = None   # not served by the v1 API
        self._summaries = None

    @property
    def summaries(self):
        if self._summaries is None:
            self._summaries = self._storage.load_summaries()
        return self._summaries


class ProsopiaClient:
    """Builds one SDK profile per slug, with RADAR's errors on the way out."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, *, timeout: float = 30.0):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    def fetch_profile(self, slug: str) -> ApiProfile:
        """The profile at ``slug``, with its works already read.

        ``from_api`` is lazy — it builds a storage backend and touches
        nothing — so the reads are forced here. A slug nobody published
        has to 404 inside ``prepare_import`` where the caller can see it,
        not minutes later inside the import job.
        """
        try:
            profile = ApiProfile(
                PagedApiStorage(slug=slug, base_url=self.base_url, timeout=self.timeout)
            )
            papers = profile.papers
            name = profile.name
        except KeyError as exc:
            raise ProfileNotFound(f"prosopia profile '{slug}' not found") from exc
        except Exception as exc:  # noqa: BLE001
            # Everything else is a 502's worth of upstream trouble. The
            # SDK raises RuntimeError for transport failures and for any
            # 4xx/5xx, PermissionError for a 401, and pydantic's
            # ValidationError for a malformed record — and that last one
            # is a ValueError, which the router would otherwise turn into
            # a 400 blaming the caller for the server's bad JSON.
            raise ProsopiaError(
                f"could not read profile '{slug}' from {self.base_url}: {exc}"
            ) from exc

        log.info(
            "prosopia.fetch_profile",
            slug=slug, name=name, n_papers=len(papers), base_url=self.base_url,
        )
        return profile

    def resolve_orcid(self, orcid: str) -> str:
        """The slug of the profile whose researcher id is ``orcid``.

        Prosopia has no lookup-by-ORCID route, but every entry in the
        public profile list carries its ``rid``, which is the ORCID for
        every profile published so far. One list read is cheap and the
        list is short, so this scans it. Raises ``ProfileNotFound`` when
        nobody with that ORCID has published a profile here.
        """
        want = orcid.strip().upper()
        url = f"{self.base_url}/api/v1/profiles"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            entries = resp.json().get("profiles") or []
        except Exception as exc:  # noqa: BLE001
            raise ProsopiaError(
                f"could not list profiles at {self.base_url}: {exc}"
            ) from exc
        for entry in entries:
            rid = str(entry.get("rid") or "").strip().upper()
            slug = entry.get("slug")
            if rid == want and slug:
                log.info("prosopia.resolve_orcid", orcid=want, slug=slug)
                return str(slug)
        raise ProfileNotFound(f"no prosopia profile with ORCID '{want}'")


def read_summary(profile: Any, paper_id: str) -> str:
    """One paper's Prosopia summary, or ``""`` if there is none to read.

    Best-effort on purpose: the summary is the embedding input for a
    paper that resolved nowhere, and a profile that publishes no
    summaries — or one artifact that will not load — must not sink an
    82-paper import.
    """
    try:
        return profile.summaries[paper_id] or ""
    except KeyError:
        return ""
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "prosopia.summary_skipped", paper_id=paper_id, reason=str(exc)[:160],
        )
        return ""
