"""NoopReranker — passthrough when reranking is disabled."""

from __future__ import annotations

import sqlite3

from ..paper import Paper
from ..profile import Profile


class NoopReranker:
    name = "noop"

    def rerank(
        self,
        ranked: list[tuple[float, Paper, dict]],
        profile: Profile,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> list[tuple[float, Paper, dict]]:
        return ranked

    def config(self) -> dict:
        return {"type": "noop"}

    @classmethod
    def from_config(cls, config: dict) -> "NoopReranker":
        return cls()

    def cost(self) -> dict:
        return {"wall_seconds": 0.0}

    def diagnostics(self) -> dict:
        return {"status": "noop"}
