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
    bare_id,
    build_filter_string,
    enabled_topic_ids,
    tier_specs,
)
from ..paper import Paper, title_key
from ..profile import Profile


log = structlog.get_logger("rag_lib.gatherers.openalex")


DEFAULT_MIN_RESULTS = 50


class _Deduper:
    """Tracks which papers have already been taken, by id and by title.

    Deduplicating on ``openalex_id`` alone is not enough: OpenAlex
    frequently carries the preprint, the version of record and the
    conference copy of one paper as separate works. Each then consumes a
    slot of its topic's quota and each can surface in the ranked output —
    the same paper appeared twice in the top ten on every trial run.

    Catching it here rather than at persistence keeps the fetch counts
    honest (the copy never enters the list, so nothing has to be
    subtracted afterwards), frees the quota slot for a different paper,
    and makes the wizard's dry-run agree with the gather it is previewing
    — the dry-run does not persist, so a duplicate filtered downstream
    would still be visible there.
    """

    __slots__ = ("_ids", "_titles")

    def __init__(self) -> None:
        self._ids: set[str] = set()
        self._titles: set[str] = set()

    def take(self, paper: Paper) -> bool:
        """Claim ``paper``; False if an equivalent one was already taken."""
        oid = paper.openalex_id
        if oid and oid in self._ids:
            return False
        key = title_key(paper.title)
        if key and key in self._titles:
            return False
        if oid:
            self._ids.add(oid)
        if key:
            self._titles.add(key)
        return True


# Per-topic quota gathering. Each enabled topic gets its own OpenAlex
# query and contributes at most this many papers, so a single prolific
# topic can no longer fill the whole batch and starve the rest.
DEFAULT_PER_TOPIC_QUOTA = 10
# Topics overlap, so a topic's first N hits are often already claimed by
# an earlier topic. Over-fetch this multiple of the quota to leave the
# round-robin enough unclaimed material to reach the quota.
DEFAULT_PER_TOPIC_OVERSAMPLE = 3


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
        per_topic_quota: int = DEFAULT_PER_TOPIC_QUOTA,
        per_topic_oversample: int = DEFAULT_PER_TOPIC_OVERSAMPLE,
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
        self.per_topic_quota = per_topic_quota
        self.per_topic_oversample = per_topic_oversample
        self._last_cost: dict = {"wall_seconds": 0.0, "api_calls": 0}
        # Populated by fetch() so the scheduler can persist tier_used and
        # the wizard can show "filter that actually ran".
        self.last_tier_used: str | None = None
        self.last_filter_str: str | None = None
        # openalex_id -> the topic whose quota claimed it. Persisted to
        # profile_candidates.sourced_by_topic_id so the UI can report what
        # each topic toggle is actually pulling in.
        self.last_source_topics: dict[str, str] = {}

    # ------------------------------------------------------------------

    def fetch(
        self,
        profile: Profile,
        since: str,
        *,
        limit: int | None = None,
    ) -> list[Paper]:
        """Gather candidates, giving every enabled topic its own quota.

        Three cases, in order:

        * some topics enabled -> one query per topic, round-robin merge.
        * topics exist but all switched off -> the user said "none of
          these", so gather nothing. Falling through to the tier walk
          here would quietly ignore that: tier 1 builds its filter from
          seed-paper metadata rather than ``topic_filters``, so it would
          keep returning papers for topics the user just disabled.
        * no topics at all (never aggregated, e.g. a CLI profile) ->
          legacy tier walk.
        """
        client = self._resolve_client()
        t0 = time.time()
        calls_before = client.api_calls

        self.last_source_topics = {}
        self.last_tier_used = None
        self.last_filter_str = None

        topics = (profile.topic_filters or {}).get("topics") or []
        topic_ids = enabled_topic_ids(profile.topic_filters)

        if topic_ids:
            papers = self._fetch_per_topic(
                client, profile, topic_ids, since, limit=limit,
            )
            self._last_cost = {
                "wall_seconds": time.time() - t0,
                "api_calls": int(client.api_calls - calls_before),
            }
            return papers

        if topics:
            log.info(
                "openalex_gatherer.all_topics_disabled",
                profile=profile.name, n_topics=len(topics),
            )
            self.last_tier_used = "no-enabled-topics"
            self._last_cost = {
                "wall_seconds": time.time() - t0,
                "api_calls": int(client.api_calls - calls_before),
            }
            return []

        log.info(
            "openalex_gatherer.no_topics",
            profile=profile.name,
            reason="topic_filters never aggregated; falling back to tier walk",
        )
        return self._fetch_tiered(
            client, profile, since, limit=limit, t0=t0, calls_before=calls_before,
        )

    # ------------------------------------------------------------------

    def _fetch_per_topic(
        self,
        client: OpenAlexClient,
        profile: Profile,
        topic_ids: list[str],
        since: str,
        *,
        limit: int | None = None,
    ) -> list[Paper]:
        """One query per enabled topic, then a fair round-robin merge.

        Papers routinely carry several OpenAlex topics, so the per-topic
        result sets overlap heavily. Draining them in order would let
        whichever topic runs first bank all the shared papers and leave
        later topics with nothing — the allocation would encode list
        order rather than relevance. Instead each topic is over-fetched,
        then we deal papers out one topic at a time until every topic has
        filled its quota or run dry, which makes each topic's
        contribution independent of where it sits in the list.

        The quota is a *floor*, not a ceiling. ``per_topic_quota`` says
        what share a topic is guaranteed no matter how many topics
        compete; it must not also decide how much the gather returns in
        total, or the caller's ``limit`` becomes unreachable. A profile
        with two enabled topics would cap at 20 papers however much
        OpenAlex has, and the wizard's dry-run — which asks for
        ``DRY_RUN_FETCH_LIMIT`` papers over 30 days precisely so the
        threshold sweep has a distribution to work with — would calibrate
        on ``10 x n_topics`` instead. So the share scales with what the
        caller asked for, and the constant only takes over when that
        share would fall below it. Equal shares, hence the fairness
        property, are preserved either way.
        """
        target = limit if limit is not None else self.min_results
        per_topic = max(
            self.per_topic_quota,
            -(-target // len(topic_ids)),  # ceil division
        )
        pools: dict[str, list[Paper]] = {}
        filters: list[str] = []

        for tid in topic_ids:
            parts = [("topics.id", [bare_id(tid)], "or")]
            filter_str = build_filter_string(parts, since, extras=self.extras)
            filters.append(filter_str)
            works = client.paginate_filter(
                filter_str, limit=per_topic * self.per_topic_oversample,
            )
            pools[tid] = [
                client.paper_from_work(w, source="openalex_gatherer")
                for w in works
            ]
            log.info(
                "openalex_gatherer.topic_fetch",
                profile=profile.name, topic=tid, n=len(pools[tid]),
            )

        chosen: list[Paper] = []
        seen = _Deduper()
        credited = {tid: 0 for tid in topic_ids}
        cursor = {tid: 0 for tid in topic_ids}

        for _ in range(per_topic):
            progressed = False
            for tid in topic_ids:
                if credited[tid] >= per_topic:
                    continue
                pool = pools[tid]
                while cursor[tid] < len(pool):
                    p = pool[cursor[tid]]
                    cursor[tid] += 1
                    oid = p.openalex_id
                    if not oid or not seen.take(p):
                        continue
                    chosen.append(p)
                    self.last_source_topics[oid] = tid
                    credited[tid] += 1
                    progressed = True
                    break
            if not progressed:
                break  # every pool exhausted

            if limit is not None and len(chosen) >= limit:
                break

        if limit is not None and len(chosen) > limit:
            # A round can overshoot ``limit`` by up to one paper per
            # topic. Drop the excess, then re-derive the attribution from
            # what actually survived — otherwise last_source_topics keeps
            # entries for papers we never returned, and the per-topic
            # tallies overstate what each topic contributed.
            chosen = chosen[:limit]
            kept = {p.openalex_id for p in chosen}
            self.last_source_topics = {
                oid: tid for oid, tid in self.last_source_topics.items()
                if oid in kept
            }
            credited = {tid: 0 for tid in topic_ids}
            for tid in self.last_source_topics.values():
                credited[tid] += 1

        self.last_tier_used = "per-topic-quota"
        self.last_filter_str = " | ".join(filters)
        log.info(
            "openalex_gatherer.per_topic_result",
            profile=profile.name,
            n_topics=len(topic_ids),
            per_topic=per_topic,
            total=len(chosen),
            per_topic_yield={t: credited[t] for t in topic_ids},
        )
        return chosen

    # ------------------------------------------------------------------

    def _fetch_tiered(
        self,
        client: OpenAlexClient,
        profile: Profile,
        since: str,
        *,
        limit: int | None,
        t0: float,
        calls_before: int,
    ) -> list[Paper]:
        tiers = tier_specs(
            profile,
            must_have_prevalence=self.must_have_prevalence,
            top_topics_n=self.top_topics_n,
            top_subfields_n=self.top_subfields_n,
        )

        chosen_papers: list[Paper] = []
        seen_ids = _Deduper()
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
                if not seen_ids.take(p):
                    continue
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
