"""OpenAlexGatherer — Phase 1B real body, tier-aware.

Walks the three-tier strategy from ``rag_lib.openalex_tiers`` against
``OpenAlexClient.paginate_filter``: must-have-AND (topics with ≥50%
seed prevalence) → top-topics-OR-subfields-AND → subfields-OR.
Accumulates results across tiers as a dedup'd union (by
``openalex_id``) and stops once the cumulative pile reaches
``min_results``; tighter AND tiers run first so their hits are always
preserved even when a broader tier was needed to reach the floor. The
last tier reached + its filter string are exposed via
``last_tier_used`` / ``last_filter_str`` — since the tier order is
fixed, that's enough to know which tiers ran.

Without the tier walk the runtime path was handing the raw aggregated
``profile.topic_filters`` dict to OpenAlex, which AND-joins every
populated hierarchy level (topics + subfields + fields + domains) and
over-constrained — none of the prevalence semantics from
``cli/gather.py`` reached the wizard or scheduler. Lifting the
strategy here keeps the CLI and the runtime in sync.

The client is resolved lazily: if no client is injected at
construction, ``fetch()`` builds one from the persisted config fields
(mailto / base_url). This keeps the Gatherer JSON-serializable without
carrying a live HTTP session.
"""

from __future__ import annotations

import time

import structlog

from ..openalex_client import OpenAlexClient
from ..openalex_tiers import (
    DEFAULT_MUST_HAVE_PREVALENCE,
    DEFAULT_TOP_SUBFIELDS_N,
    DEFAULT_TOP_TOPICS_N,
    build_filter_string,
    tier_specs,
)
from ..paper import Paper
from ..profile import Profile


log = structlog.get_logger("rag_lib.gatherers.openalex")


DEFAULT_MIN_RESULTS = 50


class OpenAlexGatherer:
    name = "openalex"

    def __init__(
        self,
        mailto: str | None = None,
        extras: str = "type:article,language:en",
        base_url: str = "https://api.openalex.org",
        client: OpenAlexClient | None = None,
        *,
        must_have_prevalence: float = DEFAULT_MUST_HAVE_PREVALENCE,
        top_topics_n: int = DEFAULT_TOP_TOPICS_N,
        top_subfields_n: int = DEFAULT_TOP_SUBFIELDS_N,
        min_results: int = DEFAULT_MIN_RESULTS,
    ):
        # mailto is required for the polite pool at fetch time. We accept
        # None at construction so the non-invoking compliance tests can
        # instantiate without one; fetch() raises if it's still unset.
        self.mailto = mailto
        self.extras = extras
        self.base_url = base_url
        self._client = client
        self.must_have_prevalence = must_have_prevalence
        self.top_topics_n = top_topics_n
        self.top_subfields_n = top_subfields_n
        self.min_results = min_results
        self._last_cost: dict = {"wall_seconds": 0.0, "api_calls": 0}
        # Populated by fetch() so the scheduler can persist tier_used and
        # the wizard can show "filter that actually ran".
        self.last_tier_used: str | None = None
        self.last_filter_str: str | None = None

    # ------------------------------------------------------------------

    def fetch(
        self,
        profile: Profile,
        since: str,
        *,
        limit: int | None = None,
    ) -> list[Paper]:
        client = self._resolve_client()
        t0 = time.time()
        calls_before = client.api_calls

        tiers = tier_specs(
            profile,
            must_have_prevalence=self.must_have_prevalence,
            top_topics_n=self.top_topics_n,
            top_subfields_n=self.top_subfields_n,
        )

        chosen_papers: list[Paper] = []
        seen_ids: set[str] = set()
        chosen_tier: str | None = None
        chosen_filter: str | None = None

        if not tiers:
            log.info(
                "openalex_gatherer.no_tiers",
                profile=profile.name,
                reason="topic_filters empty and no must-have topics",
            )

        for name, parts in tiers:
            filter_str = build_filter_string(parts, since, extras=self.extras)
            log.info(
                "openalex_gatherer.tier_attempt",
                profile=profile.name, tier=name, filter=filter_str, since=since,
            )
            works = client.paginate_filter(filter_str, limit=limit)
            papers = [
                client.paper_from_work(w, source="openalex_gatherer")
                for w in works
            ]
            new_count = 0
            for p in papers:
                oid = p.openalex_id
                if oid is not None and oid in seen_ids:
                    continue
                if oid is not None:
                    seen_ids.add(oid)
                chosen_papers.append(p)
                new_count += 1
            chosen_tier, chosen_filter = name, filter_str
            log.info(
                "openalex_gatherer.tier_result",
                profile=profile.name, tier=name, n=len(papers),
                new=new_count, cumulative=len(chosen_papers),
                meets_min=len(chosen_papers) >= self.min_results,
            )
            if len(chosen_papers) >= self.min_results:
                break

        self.last_tier_used = chosen_tier
        self.last_filter_str = chosen_filter
        self._last_cost = {
            "wall_seconds": time.time() - t0,
            "api_calls": int(client.api_calls - calls_before),
        }
        return chosen_papers

    # ------------------------------------------------------------------

    def _resolve_client(self) -> OpenAlexClient:
        if self._client is not None:
            return self._client
        if not self.mailto:
            raise ValueError(
                "OpenAlexGatherer.fetch requires mailto for the polite pool."
            )
        self._client = OpenAlexClient(
            mailto=self.mailto,
            base_url=self.base_url,
        )
        return self._client

    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        return {
            "status": "ready" if self.mailto else "unconfigured",
            "mailto_set": self.mailto is not None,
            "extras": self.extras,
            "base_url": self.base_url,
            "must_have_prevalence": self.must_have_prevalence,
            "top_topics_n": self.top_topics_n,
            "top_subfields_n": self.top_subfields_n,
            "min_results": self.min_results,
            "last_tier_used": self.last_tier_used,
            "last_filter_str": self.last_filter_str,
            **self._last_cost,
        }

    def config(self) -> dict:
        return {
            "type": "openalex",
            "mailto": self.mailto,
            "extras": self.extras,
            "base_url": self.base_url,
            "must_have_prevalence": self.must_have_prevalence,
            "top_topics_n": self.top_topics_n,
            "top_subfields_n": self.top_subfields_n,
            "min_results": self.min_results,
        }

    @classmethod
    def from_config(cls, config: dict) -> "OpenAlexGatherer":
        return cls(
            mailto=config.get("mailto"),
            extras=config.get("extras", "type:article,language:en"),
            base_url=config.get("base_url", "https://api.openalex.org"),
            must_have_prevalence=config.get(
                "must_have_prevalence", DEFAULT_MUST_HAVE_PREVALENCE,
            ),
            top_topics_n=config.get("top_topics_n", DEFAULT_TOP_TOPICS_N),
            top_subfields_n=config.get("top_subfields_n", DEFAULT_TOP_SUBFIELDS_N),
            min_results=config.get("min_results", DEFAULT_MIN_RESULTS),
        )

    def cost(self) -> dict:
        return self._last_cost
