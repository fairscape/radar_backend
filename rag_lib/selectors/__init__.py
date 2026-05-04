"""Selector registry — symmetric to ``rag_lib.embedders``.

A selector is a class implementing the ``Selector`` protocol
(``rag_lib.selector.Selector``): a ``name``, plus ``fit``, ``select``,
``diagnostics``, ``config``, ``from_config``, and ``cost``. The registry
maps a stable string key to the class so the API, scheduler, and CLI can
resolve a selector from a profile's persisted configuration without
importing the implementation themselves.

Key vs class.
  ``SELECTORS["centroid"] == CentroidSelector``
The string is what flows through the database (``selector_config_json``,
field ``type``) and over the wire (the API's ``selector`` request param).
The class is what the consumer instantiates.

Plug-in registration. Outside teams call ``register_selector("foo", FooSelector)``
once at package import time — typically from their package's ``__init__.py``.
The same shape ``register_embedder`` uses, so the two extension points feel
identical.

Reconstructing from config. ``selector_from_config(cfg)`` reads ``cfg["type"]``,
looks up the class, and calls ``cls.from_config(cfg)``. Use this in any code
path that hydrates a persisted selector — it replaces the old
``CentroidSelector.from_config(cfg)`` hard-coding."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..selector import Selector

from .centroid import CentroidSelector
from .full_text import FullTextSelector
from .max_seed import MaxSeedSelector


SELECTORS: dict[str, type] = {
    "centroid": CentroidSelector,
    "max_seed": MaxSeedSelector,
    "full_text": FullTextSelector,
}


def get_selector(name: str) -> type:
    """Resolve a selector class by its registry key."""
    if name not in SELECTORS:
        raise ValueError(
            f"Unknown selector '{name}'. Known: {sorted(SELECTORS)}"
        )
    return SELECTORS[name]


def register_selector(name: str, cls: type) -> None:
    """Plug in a custom selector class. Overrides an existing key without
    warning, mirroring ``register_embedder``."""
    SELECTORS[name] = cls


def selector_from_config(config: dict) -> "Selector":
    """Hydrate a selector from a persisted ``selector_config_json`` dict.

    The dict must carry ``type`` (the registry key); other fields are
    selector-specific and consumed by the class's ``from_config``.
    """
    t = config.get("type")
    if not t:
        raise ValueError("selector config missing 'type' key")
    cls = get_selector(t)
    return cls.from_config(config)


__all__ = [
    "SELECTORS",
    "get_selector",
    "register_selector",
    "selector_from_config",
    "CentroidSelector",
    "MaxSeedSelector",
    "FullTextSelector",
]
