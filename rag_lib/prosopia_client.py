"""ProsopiaClient — read a published Prosopia researcher profile.

A Prosopia profile is a static, machine-readable bundle: one metadata
document plus a manifest of artifacts, each fetched by path. RADAR only
needs two reads to turn a profile into a seed corpus:

  1. ``GET {base}/api/v1/profiles/{slug}``
     -> ``{slug, rid, metadata, expertise, soul, manifest, withheld, ...}``
     The ``manifest`` is a list of artifact descriptors; each carries
     ``contentUrl``, ``role`` and an ``effective_visibility`` telling us
     whether *this* caller may read it.

  2. ``GET {base}/api/v1/profiles/{slug}/content/sources/papers.jsonld``
     -> a JSON-LD ``Collection`` whose ``hasPart`` array is the works.
     Note ``hasPart``, not ``papers`` — the latter is a different route
     (``/papers``) with a different shape.

Everything here is read-only and unauthenticated: the reading routes have
no credential gate, and an anonymous caller sees the public tier. That is
all RADAR needs — the works list and the public paper summaries.

One deliberate omission: we do not walk ``/papers`` or ``/summary/{id}``.
The manifest already names every public summary artifact, and reading it
through ``/content/{path}`` keeps the client to a single content route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests
import structlog


log = structlog.get_logger("rag_lib.prosopia_client")


DEFAULT_BASE_URL = "https://prosopia.databio.org"

# The manifest entry role that marks the works collection, and the path
# it lives at on every profile built by the Prosopia builder.
WORKS_PATH = "sources/papers.jsonld"
WORKS_ROLE = "works"
SUMMARY_ROLE = "paper_summary"


class ProsopiaError(RuntimeError):
    """Any failure talking to a Prosopia instance."""


class ProfileNotFound(ProsopiaError):
    """The slug does not name a profile visible to this caller."""


@dataclass
class ProsopiaProfile:
    """One fetched profile: who it is, and what works it publishes."""

    slug: str
    name: str
    base_url: str
    metadata: dict[str, Any] = field(default_factory=dict)
    manifest: list[dict[str, Any]] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)

    def summary_paths(self) -> dict[str, str]:
        """``{paper_id: contentUrl}`` for every *readable* paper summary.

        ``effective_visibility`` is the field that matters, not
        ``visibility``: the former is the tier this caller actually got,
        the latter is what the owner asked for. An anonymous read of a
        profile whose summaries are ``internal`` returns
        ``effective_visibility: "withheld"`` (or drops the entry), and
        fetching it would 403.
        """
        out: dict[str, str] = {}
        for item in self.manifest or []:
            if (item.get("role") or "") != SUMMARY_ROLE:
                continue
            paper_id = item.get("paperId") or item.get("paper_id")
            url = item.get("contentUrl")
            if not paper_id or not url:
                continue
            vis = item.get("effective_visibility") or item.get("visibility")
            if vis != "public":
                continue
            out[str(paper_id)] = str(url)
        return out


class ProsopiaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        user_agent: str | None = None,
    ):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault(
            "User-Agent", user_agent or "radar-backend/0.1 (prosopia-import)",
        )
        self._api_calls = 0

    @property
    def api_calls(self) -> int:
        return self._api_calls

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get(self, path: str) -> requests.Response:
        self._api_calls += 1
        url = f"{self.base_url}{path}"
        try:
            r = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ProsopiaError(f"GET {url} failed: {exc}") from exc
        if r.status_code == 404:
            raise ProfileNotFound(f"GET {url} -> 404")
        if r.status_code >= 400:
            raise ProsopiaError(f"GET {url} -> {r.status_code}: {r.text[:200]!r}")
        return r

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_profile(self, slug: str) -> dict:
        """``GET /api/v1/profiles/{slug}`` — metadata + manifest."""
        r = self._get(f"/api/v1/profiles/{slug}")
        try:
            return r.json()
        except ValueError as exc:
            raise ProsopiaError(
                f"profile '{slug}' did not return JSON"
            ) from exc

    def get_content_text(self, slug: str, path: str) -> str:
        """``GET /api/v1/profiles/{slug}/content/{path}`` as raw text."""
        return self._get(f"/api/v1/profiles/{slug}/content/{path}").text

    def get_content_json(self, slug: str, path: str) -> Any:
        r = self._get(f"/api/v1/profiles/{slug}/content/{path}")
        try:
            return r.json()
        except ValueError as exc:
            raise ProsopiaError(
                f"artifact '{path}' on profile '{slug}' is not JSON"
            ) from exc

    def get_works(self, slug: str, path: str = WORKS_PATH) -> list[dict]:
        """The ``hasPart`` array of the works collection.

        Returns ``[]`` rather than raising when the artifact exists but
        carries no parts — an unbuilt profile is empty, not broken.
        """
        doc = self.get_content_json(slug, path)
        if isinstance(doc, list):
            # Some builders emit the bare array. Accept it.
            return [e for e in doc if isinstance(e, dict)]
        if not isinstance(doc, dict):
            raise ProsopiaError(f"works artifact '{path}' has unexpected shape")
        parts = doc.get("hasPart")
        if not isinstance(parts, list):
            return []
        return [e for e in parts if isinstance(e, dict)]

    def fetch_profile(self, slug: str) -> ProsopiaProfile:
        """Both reads, assembled. This is the entry point callers want."""
        doc = self.get_profile(slug)
        metadata = doc.get("metadata") or {}
        manifest = doc.get("manifest") or []
        if not isinstance(manifest, list):
            manifest = []

        works_path = _works_path_from_manifest(manifest) or WORKS_PATH
        entries = self.get_works(slug, works_path)

        profile = ProsopiaProfile(
            slug=str(doc.get("slug") or slug),
            name=str(metadata.get("name") or slug),
            base_url=self.base_url,
            metadata=metadata,
            manifest=manifest,
            entries=entries,
        )
        log.info(
            "prosopia.fetch_profile",
            slug=profile.slug,
            n_entries=len(entries),
            works_path=works_path,
        )
        return profile

    def get_summary(self, slug: str, content_url: str) -> str:
        """One paper-summary artifact, or ``""`` if it cannot be read.

        Best-effort on purpose: the summary is a bonus signal for papers
        that failed to resolve, and a 403 on one artifact must not sink
        an 82-paper import.
        """
        try:
            return self.get_content_text(slug, content_url)
        except ProsopiaError as exc:
            log.warning(
                "prosopia.summary_skipped",
                slug=slug, path=content_url, reason=str(exc)[:160],
            )
            return ""


def _works_path_from_manifest(manifest: list[dict]) -> str | None:
    """Prefer the manifest's own pointer to the works collection.

    Hard-coding ``sources/papers.jsonld`` works today, but the manifest
    is the contract and the path is data. Falls back to the constant
    when no entry carries ``role: "works"``.
    """
    for item in manifest:
        if not isinstance(item, dict):
            continue
        if (item.get("role") or "") == WORKS_ROLE:
            url = item.get("contentUrl")
            if url:
                return str(url)
    return None
