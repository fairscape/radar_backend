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
    ):
        # keyed by doi (lowercased)
        self.canned_works = {k.lower(): v for k, v in (canned_works or {}).items()}
        self.search_results = list(search_results or [])
        self.api_calls = 0
        self.lookups_by_doi: list[str] = []
        self.lookups_by_title: list[tuple[str, int | None]] = []
        self.searches: list[dict] = []

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
    abstract_words: list[str] | None = None,
    primary_topic_id: str = "T10001",
    primary_topic_name: str = "Test Topic",
    subfield_id: str = "SF1",
    subfield_name: str = "Test Subfield",
    field_id: str = "F1",
    field_name: str = "Test Field",
    domain_id: str = "D1",
    domain_name: str = "Test Domain",
) -> dict:
    """Build a minimal OpenAlex work dict that paper_from_work can consume.

    Mirrors the shape OpenAlex actually returns, just trimmed."""
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
        "type": "article",
        "cited_by_count": 0,
        "primary_location": {"source": {"display_name": venue}} if venue else None,
        "open_access": {"is_oa": False},
        "abstract_inverted_index": inv_index or None,
        "primary_topic": topic_block,
        "topics": [topic_block],
        "concepts": [],
    }
