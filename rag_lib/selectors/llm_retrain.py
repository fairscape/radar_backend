"""LLMRetrainSelector — Phase 1A typed stub, Track 2 handoff artifact.

Track 2 (the retrainable-LLM research effort) owns the body of this class.
This stub exists so Track 1 can integrate a second selector against the
protocol from day one: imports work, instantiation works, config
round-trips, diagnostics returns a dict, and the non-invoking compliance
tests pass.

Track 2 replaces the bodies of fit() and select() without changing the
surface. Once fit/select are implemented the full compliance suite
becomes required for this selector; benchmark enrollment gates on
full-suite pass.
"""

from __future__ import annotations

from ..paper import Paper
from ..profile import Profile


class LLMRetrainSelector:
    name = "llm_retrain"

    def __init__(self, model_path: str | None = None, **kwargs):
        self.model_path = model_path
        self._last_cost: dict = {"wall_seconds": 0.0}

    def fit(self, profile: Profile) -> None:
        raise NotImplementedError("Track 2 deliverable")

    def select(
        self,
        candidates: list[Paper],
        profile: Profile,
        threshold: float | None = None,
    ) -> list[tuple[float, Paper]]:
        raise NotImplementedError("Track 2 deliverable")

    def diagnostics(self) -> dict:
        return {"status": "stub", "model_path": self.model_path}

    def config(self) -> dict:
        return {"type": "llm_retrain", "model_path": self.model_path}

    @classmethod
    def from_config(cls, config: dict) -> "LLMRetrainSelector":
        return cls(model_path=config.get("model_path"))

    def cost(self) -> dict:
        return self._last_cost
