"""The ORCID registry as the identity authority.

OpenAlex is an enrichment source, not an identity one. It stamps an
author entity's ORCID onto every work of that entity, and entities are
sometimes over-merged -- two "Tim W. Clark"s collapsed into one -- so
``author.orcid:<id>`` can hand back papers the person never wrote. The
ORCID record is the person's own claim about what is theirs, so it is
what settles the question.

This module only answers "does the person claim this work". Deciding what
to do about an unclaimed one belongs to the caller: the ORCID picker
shows it unticked with a note rather than hiding it, because the registry
is frequently incomplete -- plenty of researchers never curate it -- and
a work missing from it is a weak signal, not a verdict.

Reading the registry needs no key and no account: ``pub.orcid.org`` is
the public API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from .paper import title_key


log = structlog.get_logger("rag_lib.orcid_registry")

ORCID_PUBLIC_API = "https://pub.orcid.org/v3.0"


def _norm_doi(doi: str) -> str:
    """Bare lowercase DOI: strips ``https://doi.org/`` / ``doi:`` prefixes."""
    d = doi.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d[4:] if d.startswith("doi:") else d


@dataclass
class ClaimedWorks:
    """What the researcher's own ORCID record lists.

    Matched on DOI first and on a normalized title second, because a
    registry entry often carries one identifier or none -- a DOI-less
    conference paper still has a title, and that is all there is to go on.
    """

    dois: set[str] = field(default_factory=set)        # bare, lowercased
    title_keys: set[str] = field(default_factory=set)  # paper.title_key form
    n_works: int = 0

    def matches(self, doi: str | None, title: str | None) -> bool | None:
        """True / False / ``None`` for "nothing here is comparable".

        The third case is not pedantry. ``title_key`` returns "" for any
        title under 30 alphanumerics, and a registry group often carries a
        non-DOI external id or none at all (measured on one record: 84
        DOIs across 69 groups). So a short-titled, DOI-less work has no
        key on either side, and reporting that as False would tell a
        researcher their own paper is not on their own record.
        """
        d = _norm_doi(doi) if doi else ""
        if d and d in self.dois:
            return True
        k = title_key(title)
        if k and k in self.title_keys:
            return True
        # A DOI we hold and did not match is real evidence, because the
        # registry indexes DOIs exhaustively when it has them at all.
        if d and self.dois:
            return False
        if k and self.title_keys:
            return False
        return None


@dataclass
class Affiliation:
    """One employment or education entry on the person's ORCID record."""

    name: str
    ror: str | None = None    # bare ROR id (``0153tk833``) when the record gives one


@dataclass
class RegistryRecord:
    """What the public ORCID record says, read once (``/activities``).

    ``found`` is False only on a 404 (no such ORCID iD), None when the
    registry could not be asked. ``claimed`` follows ``fetch_claimed``:
    None whenever the works list is not evidence (unreadable, private,
    empty). ``affiliations`` are the employments and educations the
    person lists, every institution they were at.
    """

    found: bool | None = None
    claimed: ClaimedWorks | None = None
    affiliations: list[Affiliation] = field(default_factory=list)


_ROR = re.compile(r"^(?:https?://)?ror\.org/", re.IGNORECASE)


def bare_ror(value: str | None) -> str | None:
    """``https://ror.org/0153tk833`` → ``0153tk833``; None for anything empty."""
    v = _ROR.sub("", (value or "").strip()).strip("/").lower()
    return v or None


def _claimed_from(works: dict | None) -> ClaimedWorks | None:
    out = ClaimedWorks()
    for group in (works or {}).get("group") or []:
        out.n_works += 1
        for e in (group.get("external-ids") or {}).get("external-id") or []:
            if (e.get("external-id-type") or "").lower() == "doi" \
                    and e.get("external-id-value"):
                out.dois.add(_norm_doi(str(e["external-id-value"])))
        for ws in group.get("work-summary") or []:
            t = ((ws.get("title") or {}).get("title") or {}).get("value")
            k = title_key(t)
            if k:
                out.title_keys.add(k)
    return out if out.n_works else None


def _affiliations_from(data: dict) -> list[Affiliation]:
    out: list[Affiliation] = []
    for section in ("employments", "educations"):
        for group in (data.get(section) or {}).get("affiliation-group") or []:
            for summary in group.get("summaries") or []:
                for entry in (summary or {}).values():
                    org = (entry or {}).get("organization") or {}
                    name = str(org.get("name") or "").strip()
                    if not name:
                        continue
                    dis = org.get("disambiguated-organization") or {}
                    ror = bare_ror(dis.get("disambiguated-organization-identifier")) \
                        if (dis.get("disambiguation-source") or "").upper() == "ROR" else None
                    out.append(Affiliation(name=name, ror=ror))
    return out


def fetch_record(
    orcid: str, *, timeout: tuple[float, float] = (3.0, 5.0)
) -> RegistryRecord:
    """The person's public ORCID record: claimed works and affiliations.

    One request (``/activities`` carries works, employments and
    educations). Never raises: the registry being down is not a reason to
    fail a listing OpenAlex can serve on its own. A 404 is the one answer
    that is evidence of something -- the iD does not exist.

    The timeout is a ``(connect, read)`` pair and deliberately short: this
    runs inside a request handler that is already paying for OpenAlex
    pagination, and the answer is advisory.
    """
    import requests

    try:
        r = requests.get(
            f"{ORCID_PUBLIC_API}/{orcid}/activities",
            headers={"Accept": "application/json",
                     "User-Agent": "radar-backend orcid-import"},
            timeout=timeout,
        )
        if r.status_code == 404:
            log.info("orcid_registry.not_found", orcid=orcid)
            return RegistryRecord(found=False)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001 — registry down: proceed without
        log.warning("orcid_registry.unavailable", orcid=orcid, error=str(exc))
        return RegistryRecord()

    if not isinstance(data, dict):
        log.warning("orcid_registry.unexpected_body", orcid=orcid, kind=type(data).__name__)
        return RegistryRecord()
    claimed = _claimed_from(data.get("works"))
    affiliations = _affiliations_from(data)
    if claimed is None:
        # A works list that is uncurated or visible only to trusted parties
        # comes back empty; reporting that as an empty claim would mark
        # every one of the author's works unclaimed.
        log.info("orcid_registry.empty", orcid=orcid)
    else:
        log.info("orcid_registry.read", orcid=orcid, n_groups=claimed.n_works,
                 n_dois=len(claimed.dois), n_titles=len(claimed.title_keys),
                 n_affiliations=len(affiliations))
    return RegistryRecord(found=True, claimed=claimed, affiliations=affiliations)


def fetch_claimed(
    orcid: str, *, timeout: tuple[float, float] = (3.0, 5.0)
) -> ClaimedWorks | None:
    """Works the person claims on their ORCID record, or ``None``.

    ``None`` rather than an empty ``ClaimedWorks`` whenever the answer
    would not be evidence, so a caller can tell "the registry does not
    list this paper" from "we never got to ask": the request failed, the
    record does not exist (404), or the record is readable but lists
    nothing (uncurated, or visible only to trusted parties). Never raises.
    """
    return fetch_record(orcid, timeout=timeout).claimed
