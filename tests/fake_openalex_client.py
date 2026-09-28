"""FakeOpenAlexClient — in-memory stand-in for tests.

Implements the OpenAlexClient surface used by Profile.from_csv/from_pdfs
and OpenAlexGatherer. Canned responses keyed by DOI and by normalized
title for lookups; ``search_results`` is a preset list returned by
``search_works`` regardless of filter. No network, deterministic.

Usage::

    client = FakeOpenAlexClient(
        canned_works={"10.1/a": {...raw OpenAlex work...}},
        search_results=[{...work...}, {...work...}],
    )
    Profile.from_csv(..., openalex_client=client)
    OpenAlexGatherer(client=client).fetch(profile, since="...")
"""

from __future__ import annotations

from rag_lib.openalex_client import OpenAlexClient
from rag_lib.paper import Paper


class FakeOpenAlexClient:
    """Duck-typed to satisfy the call sites profile_builder uses:
    lookup_by_doi, lookup_by_title, paper_from_work.
    """

    def __init__(
        self,
        canned_works: dict[str, dict] | None = None,
        search_results: list[dict] | None = None,
        tier_results: list[list[dict]] | None = None,
        authors: dict[str, dict] | None = None,
        orcid_works: list[dict] | None = None,
    ):
        # "From ORCID" path: author records keyed by canonical ORCID and a
        # flat list of raw works returned for any works_by_orcid call.
        self.authors = dict(authors or {})
        self.orcid_works = list(orcid_works or [])
        # Identity cross-check: tests never reach pub.orcid.org. Set
        # ``orcid_claimed`` to a ClaimedWorks to exercise the filter.
        self.skip_orcid_registry = True
        self.orcid_claimed = None
        self.orcid_calls: list[str] = []
        # keyed by doi (lowercased)
        self.canned_works = {k.lower(): v for k, v in (canned_works or {}).items()}
        self.search_results = list(search_results or [])
        # Optional per-call result list for paginate_filter — pop one
        # batch per call so a tier-walking test can simulate "tier 1
        # returns 0, tier 2 returns 100". When exhausted, falls back to
        # search_results.
        self._tier_results = list(tier_results) if tier_results is not None else None
        self.api_calls = 0
        self.lookups_by_doi: list[str] = []
        self.lookups_by_title: list[tuple[str, int | None]] = []
        self.searches: list[dict] = []
        self.paginate_calls: list[dict] = []

    def lookup_by_doi(self, doi: str) -> dict | None:
        self.api_calls += 1
        self.lookups_by_doi.append(doi)
        return self.canned_works.get(doi.lower())

    def lookup_by_title(self, title: str, year_hint: int | None = None) -> dict | None:
        self.api_calls += 1
        self.lookups_by_title.append((title, year_hint))
        # Naive: return the first canned work whose title substring-matches.
        title_lc = title.lower()
        for w in self.canned_works.values():
            if title_lc in (w.get("title") or "").lower():
                return w
        return None

    def search_works(
        self,
        topic_filters: dict,
        since: str,
        *,
        extras: str = "type:article,language:en",
        limit: int | None = None,
        per_page: int = 200,
    ) -> list[dict]:
        self.api_calls += 1
        self.searches.append({
            "topic_filters": topic_filters, "since": since, "limit": limit,
        })
        out = list(self.search_results)
        if limit is not None:
            out = out[:limit]
        return out

    def paginate_filter(
        self,
        filter_str: str,
        *,
        limit: int | None = None,
        per_page: int = 200,
    ) -> list[dict]:
        """In-memory stand-in for ``OpenAlexClient.paginate_filter``.

        Returns the next batch from ``tier_results`` if configured, else
        falls back to ``search_results``. Records the call so tests can
        assert on the rendered filter string."""
        self.api_calls += 1
        self.paginate_calls.append({"filter_str": filter_str, "limit": limit})
        if self._tier_results is not None:
            batch = self._tier_results.pop(0) if self._tier_results else []
            out = list(batch)
        else:
            out = list(self.search_results)
        if limit is not None:
            out = out[:limit]
        return out

    def get_author_by_orcid(self, orcid: str) -> dict | None:
        self.api_calls += 1
        return self.authors.get(orcid)

    def get_author(self, author_id: str) -> dict | None:
        """By OpenAlex id: any canned author whose ``id`` matches."""
        self.api_calls += 1
        bare = author_id.rsplit("/", 1)[-1]
        for a in self.authors.values():
            if str(a.get("id") or "").rsplit("/", 1)[-1] == bare:
                return a
        return None

    def works_by_orcid(
        self, orcid: str, *, limit: int | None = None, per_page: int = 200,
    ) -> list[dict]:
        self.api_calls += 1
        self.orcid_calls.append(orcid)
        out = list(self.orcid_works)
        if limit is not None:
            out = out[:limit]
        return out

    def paper_from_work(self, work: dict, **kwargs) -> Paper:
        # Delegate to the real client's conversion logic — it's pure and
        # has no network dependency. Reusing it keeps fake vs. real Paper
        # shape identical.
        return OpenAlexClient.paper_from_work(self, work, **kwargs)  # type: ignore[arg-type]

    # Methods called by OpenAlexClient.paper_from_work when passed `self`:
    def slim_work(self, work: dict | None) -> dict | None:
        return OpenAlexClient.slim_work(self, work)  # type: ignore[arg-type]

    @staticmethod
    def reconstruct_abstract(inv_index: dict | None) -> str:
        return OpenAlexClient.reconstruct_abstract(inv_index)


def canned_openalex_work(
    *,
    doi: str,
    openalex_id: str,
    title: str,
    year: int | None = None,
    venue: str | None = None,
    abstract_words: list[str] | None = "default",  # type: ignore[assignment]
    primary_topic_id: str = "T10001",
    primary_topic_name: str = "Test Topic",
    subfield_id: str = "SF1",
    subfield_name: str = "Test Subfield",
    field_id: str = "F1",
    field_name: str = "Test Field",
    domain_id: str = "D1",
    domain_name: str = "Test Domain",
    work_type: str = "article",
    cited_by_count: int = 0,
    authorships: list[dict] | None = None,
) -> dict:
    """Build a minimal OpenAlex work dict that paper_from_work can consume.

    Mirrors the shape OpenAlex actually returns, just trimmed.
    ``authorships`` entries look like
    ``{"author": {"id", "display_name", "orcid"}, "author_position",
    "is_corresponding"}``; see ``canned_authorship``."""
    # The gatherer drops abstract-less works by default (they are mostly
    # data-portal exports), so canned works carry a stub abstract unless
    # the caller passes ``abstract_words=None`` explicitly.
    if abstract_words == "default":
        abstract_words = ["canned", "abstract", "for", title.split(" ")[0].lower() if title else "work"]
    inv_index: dict[str, list[int]] = {}
    for i, w in enumerate(abstract_words or []):
        inv_index.setdefault(w, []).append(i)
    topic_block = {
        "id": primary_topic_id,
        "display_name": primary_topic_name,
        "subfield": {"id": subfield_id, "display_name": subfield_name},
        "field": {"id": field_id, "display_name": field_name},
        "domain": {"id": domain_id, "display_name": domain_name},
        "score": 0.9,
    }
    return {
        "id": openalex_id,
        "doi": f"https://doi.org/{doi}",
        "title": title,
        "publication_year": year,
        "type": work_type,
        "cited_by_count": cited_by_count,
        "authorships": list(authorships or []),
        "ids": {"openalex": openalex_id, "doi": f"https://doi.org/{doi}"},
        "primary_location": {"source": {"display_name": venue}} if venue else None,
        "open_access": {"is_oa": False},
        "abstract_inverted_index": inv_index or None,
        "primary_topic": topic_block,
        "topics": [topic_block],
        "concepts": [],
    }


def canned_authorship(
    name: str,
    *,
    position: str = "middle",
    orcid: str | None = None,
    author_id: str | None = None,
    corresponding: bool = False,
) -> dict:
    """One ``authorships[]`` entry in OpenAlex's shape."""
    return {
        "author_position": position,
        "is_corresponding": corresponding,
        "author": {
            "id": author_id or f"https://openalex.org/A{abs(hash(name)) % 10**9}",
            "display_name": name,
            "orcid": f"https://orcid.org/{orcid}" if orcid else None,
        },
    }
