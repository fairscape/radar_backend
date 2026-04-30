"""Gatherer protocol — frozen at Phase 1A exit.

A Gatherer is the source of candidate papers for a Profile. It reads the
profile's topic_filters + gatherer_config and returns new candidate Papers
since a given date. The protocol is intentionally parallel to Selector.

Intentional split from profile-building. Deriving topic_filters from a
seed corpus is NOT the gatherer's job — it lives in rag_lib.profile_builder
(which uses the OpenAlexClient for topic lookups). The Gatherer only
fetches candidates against a fully-built Profile. This keeps sources
swappable (OpenAlex today, arXiv/bioRxiv/PubMed later) without entangling
each one with profile-construction semantics.

Invariant — one gatherer per profile, not per selector.
    All selectors running against a profile (including in benchmark mode)
    see the same gatherer output. This is what makes the benchmark an
    apples-to-apples ranking comparison: selectors are distinguished by
    how they rank a shared pool, not by which papers they fetch. Profile
    schema (Phase 2) enforces single-gatherer-per-profile; don't route
    around it.

Adding or removing methods on this Protocol is a breaking change for
every phase downstream and for Track 2's future work. Do not modify
after 1A exit without explicit cross-track re-agreement.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .paper import Paper
from .profile import Profile


@runtime_checkable
class Gatherer(Protocol):
    name: str  # "openalex" | "fixture" | ...

    def fetch(
        self,
        profile: Profile,
        since: str,
        *,
        limit: int | None = None,
    ) -> list[Paper]:
        """Fetch candidate papers that have appeared since the given
        ISO-8601 date, filtered by the profile's topic_filters and
        gatherer_config. Returns Paper records (not dicts). Deduplication
        against the profile's seed corpus or the local vault is a
        radar-layer concern, not a gatherer one: gatherers return
        everything the source knows about; the radar decides what is
        already seen.

        limit: optional cap on the number of candidates returned. Gatherers
        may still paginate source-side as needed."""

    def diagnostics(self) -> dict:
        """Gatherer-specific health metrics (e.g., last-response latency,
        rate-limit headroom, empty-result streak)."""

    def config(self) -> dict:
        """Serializable config for persistence in the profile's
        gatherer_config field."""

    @classmethod
    def from_config(cls, config: dict) -> "Gatherer":
        """Reconstruct from persisted config."""

    def cost(self) -> dict:
        """{'wall_seconds': float, 'api_calls': int} for the last fetch()
        call. api_calls matters for rate-limited sources like the OpenAlex
        polite pool."""
