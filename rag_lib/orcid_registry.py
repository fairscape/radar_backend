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


def fetch_claimed(
    orcid: str, *, timeout: tuple[float, float] = (3.0, 5.0)
) -> ClaimedWorks | None:
    """Works the person claims on their ORCID record, or ``None``.

    ``None`` rather than an empty ``ClaimedWorks`` whenever the answer
    would not be evidence, so a caller can tell "the registry does not
    list this paper" from "we never got to ask". Three cases give
    ``None``: the request failed, the record does not exist (404), and --
    the one that matters most in practice -- the record is readable but
    lists nothing, because a works list that is uncurated or visible only
    to trusted parties returns 200 with an empty group set. Reporting that
    as an empty claim would mark every one of the author's works
    unclaimed, which is the opposite of useful.

    Never raises. The registry being down is not a reason to fail an
    import that OpenAlex can serve on its own.

    The timeout is a ``(connect, read)`` pair and deliberately short: this
    runs inside a request handler that is already paying for OpenAlex
    pagination, and the answer is advisory. A scalar would apply to each
    phase separately, so a black-holed host could cost twice the number
    written.
    """
    import requests

    try:
        r = requests.get(
            f"{ORCID_PUBLIC_API}/{orcid}/works",
            headers={"Accept": "application/json",
                     "User-Agent": "radar-backend orcid-import"},
            timeout=timeout,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001 — registry down: proceed without
        log.warning("orcid_registry.unavailable", orcid=orcid, error=str(exc))
        return None

    out = ClaimedWorks()
    for group in data.get("group") or []:
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
    if out.n_works == 0:
        log.info("orcid_registry.empty", orcid=orcid)
        return None
    log.info("orcid_registry.read", orcid=orcid, n_groups=out.n_works,
             n_dois=len(out.dois), n_titles=len(out.title_keys))
    return out
