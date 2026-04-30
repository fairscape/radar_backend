"""Selector protocol — frozen at Phase 1A exit.

The Selector decides which candidate papers to surface. Every selector
satisfies the same interface, which makes swapping, benchmarking, and
comparing selectors trivial at every layer above this one. This protocol
is the integration contract between Track 1 and Track 2.

A Selector is fit against a Profile (which carries seed papers with
embeddings + OpenAlex topics + the chosen embedding_model key) and then
scores candidate Papers against that same Profile. Both fit and select
receive the Profile so hybrid selectors can reason about the target
beyond their fitted internal state — for example, matching a candidate's
tags against profile.topic_filters, or prompting an LLM with the profile's
human-readable name.

Adding or removing methods on this Protocol is a breaking change for
every phase downstream and for both tracks. Do not modify after 1A exit
without explicit cross-track re-agreement.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .paper import Paper
from .profile import Profile


@runtime_checkable
class Selector(Protocol):
    name: str  # "centroid" | "llm_retrain" | "fixture"

    def fit(self, profile: Profile) -> None:
        """Fit from the profile's seed papers and their stored embeddings
        (keyed by profile.embedding_model). May be a no-op for stateless
        selectors."""

    def select(
        self,
        candidates: list[Paper],
        profile: Profile,
        threshold: float | None = None,
    ) -> list[tuple[float, Paper, dict]]:
        """Score candidates against the profile.

        Returns ``(score, paper, breakdown)`` triples sorted descending by
        ``score``. ``score`` is the selector's primary normalized signal
        in ``[0, 1]`` (tolerance +/-1e-6) — by convention the per-batch
        rank percentile, so absolute thresholds compare across profiles.

        ``breakdown`` is a selector-specific dict of secondary diagnostics
        (e.g., ``{"score_raw": raw_cosine, "score_max_seed": float,
        "score_pct": float}``). It may be empty for selectors that have
        nothing extra to say.

        If ``threshold`` is given, drop entries with ``score < threshold``.

        Phase 3 widened the legacy ``(score, paper)`` shape to a 3-tuple
        with the breakdown side-channel; selectors that don't compute a
        breakdown should still return an empty dict so consumers can
        unpack uniformly."""

    def diagnostics(self) -> dict:
        """Selector-specific health metrics."""

    def config(self) -> dict:
        """Serializable config for persistence in the profile's
        selector_config field."""

    @classmethod
    def from_config(cls, config: dict) -> "Selector":
        """Reconstruct from persisted config."""

    def cost(self) -> dict:
        """{'wall_seconds': float} for the last fit() or select() call."""
