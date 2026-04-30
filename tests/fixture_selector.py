"""FixtureSelector — deterministic random-scoring reference.

Lives under tests/, not rag_lib/selectors/. Not a product selector. Serves
as a deterministic reference for the protocol compliance test and for any
test that needs a selector without SPECTER2 / Chroma / Ollama deps.
"""

from __future__ import annotations

import time

import numpy as np

from rag_lib.paper import Paper
from rag_lib.profile import Profile


class FixtureSelector:
    name = "fixture"

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self._last_cost: dict = {"wall_seconds": 0.0}
        self._n_seed = 0

    def fit(self, profile: Profile) -> None:
        t0 = time.time()
        self._n_seed = len(profile.papers)
        self._last_cost = {"wall_seconds": time.time() - t0}

    def select(
        self,
        candidates: list[Paper],
        profile: Profile,
        threshold: float | None = None,
    ) -> list[tuple[float, Paper, dict]]:
        # profile is accepted and ignored: a random scorer has nothing to
        # say about fit to a specific profile. Accepting it keeps the call
        # surface uniform with selectors that do consume it.
        t0 = time.time()
        scored: list[tuple[float, Paper, dict]] = [
            (float(self.rng.random()), p, {}) for p in candidates
        ]
        if threshold is not None:
            scored = [r for r in scored if r[0] >= threshold]
        self._last_cost = {"wall_seconds": time.time() - t0}
        return sorted(scored, key=lambda r: r[0], reverse=True)

    def diagnostics(self) -> dict:
        return {
            "note": "random selector, for testing only",
            "n_seed": self._n_seed,
        }

    def config(self) -> dict:
        return {"type": "fixture", "seed": self._seed}

    @classmethod
    def from_config(cls, config: dict) -> "FixtureSelector":
        return cls(seed=config.get("seed"))

    def cost(self) -> dict:
        return self._last_cost
