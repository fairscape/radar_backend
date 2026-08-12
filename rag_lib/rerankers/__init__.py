"""Reranker registry — symmetric to ``rag_lib.selectors``.

Maps a stable string key to a reranker class so the CLI, scheduler, and
API can resolve a reranker from configuration without importing the
implementation directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..reranker import Reranker

from .noop import NoopReranker
from .medcpt import MedCPTReranker


RERANKERS: dict[str, type] = {
    "noop": NoopReranker,
    "medcpt": MedCPTReranker,
}


def get_reranker(name: str) -> type:
    """Resolve a reranker class by its registry key."""
    if name not in RERANKERS:
        raise ValueError(
            f"Unknown reranker '{name}'. Known: {sorted(RERANKERS)}"
        )
    return RERANKERS[name]


def register_reranker(name: str, cls: type) -> None:
    """Plug in a custom reranker class."""
    RERANKERS[name] = cls


def reranker_from_config(config: dict) -> "Reranker":
    """Hydrate a reranker from a persisted config dict."""
    t = config.get("type")
    if not t:
        raise ValueError("reranker config missing 'type' key")
    cls = get_reranker(t)
    return cls.from_config(config)


__all__ = [
    "RERANKERS",
    "get_reranker",
    "register_reranker",
    "reranker_from_config",
    "NoopReranker",
    "MedCPTReranker",
]
