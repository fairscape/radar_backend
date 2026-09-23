"""Read a published Prosopia profile through the researcher-profiles SDK.

A Prosopia instance *is* a ``researcher_profiles`` API server, so the SDK
ships the reader RADAR used to hand-roll. ``ResearcherProfile.from_api``
returns a profile whose ``papers`` are ``PaperRecord`` objects with their
identifiers already normalized, and whose ``summaries`` fetch one body at
a time on access. Manifest walking, visibility filtering and the JSON-LD
works parse all live upstream now, which is the point: an identifier fix
there reaches RADAR on a pin bump instead of as a second implementation
here.

What stays is the part the SDK has no opinion about — RADAR's own failure
vocabulary. ``ProfileNotFound`` is the router's 404 and ``ProsopiaError``
its 502, and every way the SDK can fail maps onto one of the two. The
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
from researcher_profiles import ResearcherProfile


log = structlog.get_logger("rag_lib.prosopia_client")


DEFAULT_BASE_URL = "https://prosopia.databio.org"

# 0000-0001-5643-4068 — four groups, last char may be X. Matched on the
# whole string so a slug that merely contains digits is never mistaken.
ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", re.IGNORECASE)


def looks_like_orcid(value: str) -> bool:
    """True for a bare ORCID iD (``0000-0001-5643-4068``)."""
    return bool(ORCID_RE.match((value or "").strip()))


class ProsopiaError(RuntimeError):
    """Any failure talking to a Prosopia instance."""


class ProfileNotFound(ProsopiaError):
    """The slug does not name a profile visible to this caller."""


class ProsopiaClient:
    """Builds one SDK profile per slug, with RADAR's errors on the way out."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, *, timeout: float = 30.0):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    def fetch_profile(self, slug: str) -> ResearcherProfile:
        """The profile at ``slug``, with its works already read.

        ``from_api`` is lazy — it builds a storage backend and touches
        nothing — so the reads are forced here. A slug nobody published
        has to 404 inside ``prepare_import`` where the caller can see it,
        not minutes later inside the import job.
        """
        url = f"{self.base_url}/api/v1/profiles/{slug}"
        try:
            profile = ResearcherProfile.from_api(url, timeout=self.timeout)
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
