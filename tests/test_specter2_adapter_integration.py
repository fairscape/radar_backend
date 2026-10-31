"""Opt-in: proves the real SPECTER2 proximity adapter is applied.
Run with RADAR_RUN_SLOW=1 (needs the HF cache with real weights)."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RADAR_RUN_SLOW") != "1",
        reason="needs real model weights; set RADAR_RUN_SLOW=1",
    ),
]

TEXT = "Heart rate variability monitoring for early detection of neonatal sepsis."


def test_proximity_adapter_changes_embedding():
    pytest.importorskip("adapters")
    from rag_lib.embedders import _load_specter2, specter2_embed

    current = np.array(specter2_embed(TEXT))
    _, model, _ = _load_specter2()
    assert "proximity" in model.active_adapters.flatten()
    try:
        model.set_active_adapters(None)
        base = np.array(specter2_embed(TEXT))
    finally:
        from adapters import AdapterSetup

        with AdapterSetup("proximity"):
            model.set_active_adapters("proximity")
    assert "proximity" in model.active_adapters.flatten()
    cos = float(current @ base / (np.linalg.norm(current) * np.linalg.norm(base)))
    assert cos < 0.99
