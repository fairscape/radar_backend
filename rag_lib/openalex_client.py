"""OpenAlexClient — thin wrapper around the OpenAlex /works API.

Lifts the working per-paper lookup logic from ``radar/openalex_lookup.py``
into a reusable class. Used by two callers with different intents:

  - rag_lib.profile_builder: enrich known seed papers (DOI -> OpenAlex
    record -> Paper). This is lookup_by_doi / lookup_by_title / slim_work
    / paper_from_work. Already working in the original script; moved here
    so the Paper-building code is deduplicated.

  - rag_lib.gatherers.openalex.OpenAlexGatherer: search for new candidates
    by topic filter since a given date. This is search_works, which is
    *not* in the original script and stays stubbed until Phase 1B. The
    cursor pagination, OR-joined topic filters, and polite-pool mailto
    pattern are all specified in the build spec; the stub keeps the
    integration surface present while the body is written.

All requests carry ``mailto`` for the OpenAlex polite pool. mailto is
required at construction — no baked-in default.
"""

from __future__ import annotations

import time

import requests

from .paper import Paper, Topic, TopicNode


DEFAULT_BASE_URL = "https://api.openalex.org"


class OpenAlexClient:
    def __init__(
        self,
        mailto: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        user_agent: str | None = None,
        rate_limit_sleep: float = 0.15,
        session: requests.Session | None = None,
    ):
        if not mailto:
            raise ValueError("OpenAlexClient requires mailto for the polite pool.")
        self.mailto = mailto
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent or f"radar-backend/0.1 (mailto:{mailto})"
        self.rate_limit_sleep = rate_limit_sleep
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", self.user_agent)
        self._api_calls = 0

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict:
        self._api_calls += 1
        p = dict(params or {})
        p["mailto"] = self.mailto
        url = f"{self.base_url}{path}"
        r = self.session.get(url, params=p, timeout=30)
        # Single attempt. Auto-retry on 429 wastes quota and extends the
        # cooldown when many simultaneous requests share an IP. Surface
        # whatever the server told us instead so the caller (or human)
        # decides what to do.
        if getattr(r, "status_code", 200) == 429:
            headers = getattr(r, "headers", {}) or {}
            remaining = headers.get("X-RateLimit-Remaining", "?")
            reset = headers.get("X-RateLimit-Reset", "?")
            body = ""
            try:
                body = r.text[:200]
            except Exception:
                pass
            raise RuntimeError(
                f"OpenAlex 429: rate-limited. "
                f"X-RateLimit-Remaining={remaining}, X-RateLimit-Reset={reset}s. "
                f"Body: {body!r}"
            )
        r.raise_for_status()
        if self.rate_limit_sleep:
            time.sleep(self.rate_limit_sleep)
        return r.json()

    @property
    def api_calls(self) -> int:
        return self._api_calls

    def reset_api_calls(self) -> None:
        self._api_calls = 0

    # ------------------------------------------------------------------
    # Per-paper lookup (used by profile_builder). Lifted from the original
    # openalex_lookup.py logic.
    # ------------------------------------------------------------------

    def lookup_by_doi(self, doi: str) -> dict | None:
        """Return the raw OpenAlex work record for a DOI, or None if 404."""
        try:
            return self._get(f"/works/doi:{doi}")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise

    def lookup_by_title(self, title: str, year_hint: int | None = None) -> dict | None:
        """Fall back to title search. Returns the top hit or None."""
        params: dict = {"search": title, "per-page": 5}
        if year_hint:
            params["filter"] = (
                f"publication_year:{year_hint - 1}|{year_hint}|{year_hint + 1}"
            )
        j = self._get("/works", params=params)
        results = j.get("results") or []
        return results[0] if results else None

    # ------------------------------------------------------------------
    # Candidate search (used by OpenAlexGatherer). Phase 1B deliverable.
    # ------------------------------------------------------------------

    def search_works(
        self,
        topic_filters: dict,
        since: str,
        *,
        extras: str = "type:article,language:en",
        limit: int | None = None,
        per_page: int = 200,
    ) -> list[dict]:
        """Cursor-paginated ``/works`` query.

        ``topic_filters`` keys follow OpenAlex's hierarchy. Each value may
        be a bare list of IDs or a list of ``{"id": ..., ...}`` dicts
        (the shape ``Profile.aggregate_topic_filters`` returns):

            {
              "topics":    [{"id": "T10123", ...}, ...]  # or ["T10123", ...]
              "subfields": [...], "fields": [...], "domains": [...],
              "primary_topic": [...],  # optional — more restrictive
            }

        All IDs within a level are OR-joined (``|``); levels are
        AND-joined (``,``) with each other, with ``from_created_date``,
        and with ``extras``. ``mailto`` is appended on every request for
        the polite pool.

        Pagination: ``per_page=200`` (OpenAlex max), ``cursor=*`` on the
        first call, then ``cursor=<meta.next_cursor>`` until the cursor
        is null. Stops early when ``limit`` is reached.

        Fallback: if the full topic_filters returns zero hits, retry with
        progressively wider hierarchy levels (topics → subfields → fields
        → domains), keeping only one level at a time. Stops at the first
        non-empty result. Profiles built from very narrow seed sets can
        otherwise produce filters that AND-overconstrain to zero matches.
        """
        for variant in self._filter_variants(topic_filters):
            filter_str = _build_filter_string(variant, since, extras)
            results = self.paginate_filter(filter_str, limit=limit, per_page=per_page)
            if results:
                return results
        return []

    def paginate_filter(
        self,
        filter_str: str,
        *,
        limit: int | None = None,
        per_page: int = 200,
    ) -> list[dict]:
        out: list[dict] = []
        cursor: str | None = "*"
        while cursor:
            params = {
                "filter": filter_str,
                "per-page": per_page,
                "cursor": cursor,
            }
            j = self._get("/works", params=params)
            results = j.get("results") or []
            if limit is not None:
                remaining = limit - len(out)
                if remaining <= 0:
                    break
                if len(results) > remaining:
                    results = results[:remaining]
            out.extend(results)
            if limit is not None and len(out) >= limit:
                break
            cursor = (j.get("meta") or {}).get("next_cursor")
        return out

    @staticmethod
    def _filter_variants(topic_filters: dict) -> list[dict]:
        """Yield successively-broader variants of topic_filters.

        First the full filter as supplied. Then, on miss, fall back to
        one level at a time, narrowest → widest. Stops emitting once it
        runs out of populated levels."""
        variants: list[dict] = [dict(topic_filters or {})]
        order = ("primary_topic", "topics", "subfields", "fields", "domains")
        for key in order:
            ids = _normalize_ids((topic_filters or {}).get(key))
            if ids:
                variants.append({key: ids})
        # De-dupe consecutive identical variants (e.g. when only one
        # level is populated, the full filter == the single-level one).
        deduped: list[dict] = []
        for v in variants:
            if not deduped or v != deduped[-1]:
                deduped.append(v)
        return deduped

    # ------------------------------------------------------------------
    # Shape conversion: OpenAlex work -> Paper.
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruct_abstract(inv_index: dict | None) -> str:
        """OpenAlex serves abstracts as an inverted index. Reassemble to
        running text. Returns '' when the index is empty or missing."""
        if not inv_index:
            return ""
        words = {
            pos: word
            for word, positions in inv_index.items()
            for pos in positions
        }
        return " ".join(words[i] for i in sorted(words))

    def slim_work(self, work: dict | None) -> dict | None:
        """Project the fields we care about out of a raw OpenAlex work.
        Shape matches the canonical form used in radar/openalex_lookup.py,
        extended to carry subfield/field/domain IDs alongside their names
        so hierarchical filtering works."""
        if not work:
            return None
        primary = work.get("primary_topic") or {}
        topics = work.get("topics") or []
        concepts = work.get("concepts") or []
        primary_location = work.get("primary_location") or {}
        best_oa_location = work.get("best_oa_location") or {}
        open_access = work.get("open_access") or {}
        host_venue = ((primary_location.get("source") or {}).get("display_name"))
        # Prefer best_oa_location.pdf_url (covers preprints/repositories) and
        # fall back to the publisher's primary_location.pdf_url. open_access
        # .oa_url is often a landing page rather than a PDF, so we don't use
        # it here — but oa_status is the canonical OA-color label.
        pdf_url = best_oa_location.get("pdf_url") or primary_location.get("pdf_url")
        return {
            "id": work.get("id"),
            "doi": work.get("doi"),
            "title": work.get("title"),
            "publication_year": work.get("publication_year"),
            "type": work.get("type"),
            "cited_by_count": work.get("cited_by_count"),
            "host_venue": host_venue,
            "open_access": open_access,
            "pdf_url": pdf_url,
            "oa_status": open_access.get("oa_status"),
            "abstract_inverted_index": work.get("abstract_inverted_index"),
            "primary_topic": _topic_with_ids(primary) if primary else None,
            "topics": [_topic_with_ids(t) for t in topics[:5]],
            "top_concepts": [
                {
                    "display_name": c.get("display_name"),
                    "score": c.get("score"),
                    "level": c.get("level"),
                }
                for c in concepts[:8]
            ],
        }

    def paper_from_work(
        self,
        work: dict,
        *,
        source: str = "openalex",
        local_path: str | None = None,
        added: str | None = None,
    ) -> Paper:
        """Convert a raw OpenAlex work dict into a Paper record."""
        slim = self.slim_work(work) or {}
        primary_topic = _topic_from_slim(slim.get("primary_topic")) if slim.get("primary_topic") else None
        topics = [_topic_from_slim(t) for t in (slim.get("topics") or [])]
        abstract = self.reconstruct_abstract(slim.get("abstract_inverted_index"))
        # OpenAlex DOIs come back as full URLs ("https://doi.org/10.x/y").
        # Strip to the bare DOI for consistency with user-provided DOIs.
        doi = slim.get("doi")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]
        return Paper(
            doi=doi,
            openalex_id=slim.get("id"),
            title=slim.get("title") or "",
            abstract=abstract,
            year=slim.get("publication_year"),
            venue=slim.get("host_venue"),
            mesh=[],
            keywords=[],
            substances=[],
            embeddings={},
            primary_topic=primary_topic,
            topics=topics,
            local_path=local_path,
            source=source,
            added=added,
            pdf_url=slim.get("pdf_url"),
            oa_status=slim.get("oa_status"),
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


# OpenAlex filter keys per hierarchy level. Maps our internal
# topic_filter keys to OpenAlex /works filter parameter names.
_FILTER_KEYS: dict[str, str] = {
    "topics":        "topics.id",
    "primary_topic": "primary_topic.id",
    "subfields":     "topics.subfield.id",
    "fields":        "topics.field.id",
    "domains":       "topics.domain.id",
}


_OPENALEX_ID_PREFIX = "https://openalex.org/"


def _strip_id_prefix(s: str) -> str:
    # OpenAlex's /works filter parser silently returns 0 results when the
    # filter string ANDs together multiple URL-form ID clauses (the colon
    # in https://... appears to confuse it). Bare IDs work in every
    # combination, so normalize to bare on the way in.
    if s.startswith(_OPENALEX_ID_PREFIX):
        tail = s[len(_OPENALEX_ID_PREFIX):]
        return tail.rsplit("/", 1)[-1]
    return s


def _normalize_ids(values) -> list[str]:
    """Accept either ["T10123", ...] or [{"id": "T10123", ...}, ...].
    Drop any empty / None / non-string entries."""
    if not values:
        return []
    out: list[str] = []
    for v in values:
        if isinstance(v, str) and v:
            out.append(_strip_id_prefix(v))
        elif isinstance(v, dict):
            vid = v.get("id")
            if isinstance(vid, str) and vid:
                out.append(_strip_id_prefix(vid))
    return out


def _build_filter_string(
    topic_filters: dict,
    since: str,
    extras: str,
) -> str:
    """Assemble the comma-joined filter string for /works.

    Uses ``from_publication_date`` (free tier). ``from_created_date`` is
    paid-tier only as of 2026; it would cover backfilled older works that
    publication-date filtering misses, but most "what's new" radar use
    cases are fine with publication-date semantics."""
    parts: list[str] = [f"from_publication_date:{since}"]
    if extras:
        parts.append(extras)
    for key, oa_key in _FILTER_KEYS.items():
        ids = _normalize_ids(topic_filters.get(key))
        if ids:
            parts.append(f"{oa_key}:{'|'.join(ids)}")
    return ",".join(parts)


def _topic_with_ids(t: dict) -> dict:
    """Slimmer for an OpenAlex topic subdict. Preserves subfield/field/domain
    IDs (the original radar/openalex_lookup.py dropped them)."""
    sub = t.get("subfield") or {}
    fld = t.get("field") or {}
    dom = t.get("domain") or {}
    return {
        "id": t.get("id"),
        "display_name": t.get("display_name"),
        "subfield": {"id": sub.get("id"), "display_name": sub.get("display_name")} if sub else None,
        "field": {"id": fld.get("id"), "display_name": fld.get("display_name")} if fld else None,
        "domain": {"id": dom.get("id"), "display_name": dom.get("display_name")} if dom else None,
        "score": t.get("score"),
    }


def _topic_from_slim(st: dict) -> Topic:
    return Topic(
        id=st.get("id"),
        display_name=st.get("display_name") or "",
        subfield=TopicNode.from_dict(st.get("subfield")),
        field=TopicNode.from_dict(st.get("field")),
        domain=TopicNode.from_dict(st.get("domain")),
        score=st.get("score"),
    )
