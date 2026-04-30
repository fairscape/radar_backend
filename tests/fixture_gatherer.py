"""FixtureGatherer — deterministic seeded reference.

Lives under tests/, not rag_lib/gatherers/. Not a product gatherer. Serves
as a deterministic reference for the Gatherer compliance test and for any
test in Phase 2+ that needs a candidate stream without hitting a real API.

Behavior: ignores the profile's topic_filters beyond noting them in each
synthetic candidate's keywords, and returns a seeded list of Paper
records. Deterministic per (seed, n_candidates, profile.name, since).
"""

from __future__ import annotations

import time

from rag_lib.paper import Paper
from rag_lib.profile import Profile


class FixtureGatherer:
    name = "fixture"

    def __init__(self, seed: int | None = None, n_candidates: int = 5):
        self._seed = seed
        self.n_candidates = n_candidates
        self._last_cost: dict = {"wall_seconds": 0.0, "api_calls": 0}

    def fetch(
        self,
        profile: Profile,
        since: str,
        *,
        limit: int | None = None,
    ) -> list[Paper]:
        t0 = time.time()
        topic_ids = [t.get("id") for t in profile.topic_filters.get("topics", [])]
        n = self.n_candidates if limit is None else min(self.n_candidates, limit)
        out = [
            Paper(
                doi=f"10.fix/{self._seed}-{i}",
                openalex_id=f"W{self._seed or 0}{i:04d}",
                title=f"Fixture paper {i} for profile={profile.name}",
                abstract=f"Synthetic abstract {i}. Topics: {topic_ids}",
                keywords=list(topic_ids),
                source="fixture",
                added=since,
            )
            for i in range(n)
        ]
        self._last_cost = {"wall_seconds": time.time() - t0, "api_calls": 0}
        return out

    def diagnostics(self) -> dict:
        return {
            "note": "fixture gatherer, for testing only",
            "n_candidates": self.n_candidates,
        }

    def config(self) -> dict:
        return {
            "type": "fixture",
            "seed": self._seed,
            "n_candidates": self.n_candidates,
        }

    @classmethod
    def from_config(cls, config: dict) -> "FixtureGatherer":
        return cls(
            seed=config.get("seed"),
            n_candidates=config.get("n_candidates", 5),
        )

    def cost(self) -> dict:
        return self._last_cost
