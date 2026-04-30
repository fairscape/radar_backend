"""OpenAlexGatherer — Phase 1B real body.

Delegates to ``OpenAlexClient.search_works`` for the cursor-paginated
candidate fetch, then maps each hit through ``paper_from_work`` to a
Paper. Reads the profile's ``topic_filters`` (produced by
``Profile.aggregate_topic_filters`` at build time) and passes them
through to the underlying query.

The client is resolved lazily: if no client is injected at
construction, ``fetch()`` builds one from the persisted config fields
(mailto / base_url). This keeps the Gatherer JSON-serializable without
carrying a live HTTP session.
"""

from __future__ import annotations

import time

from ..openalex_client import OpenAlexClient
from ..paper import Paper
from ..profile import Profile


class OpenAlexGatherer:
    name = "openalex"

    def __init__(
        self,
        mailto: str | None = None,
        extras: str = "type:article,language:en",
        base_url: str = "https://api.openalex.org",
        client: OpenAlexClient | None = None,
    ):
        # mailto is required for the polite pool at fetch time. We accept
        # None at construction so the non-invoking compliance tests can
        # instantiate without one; fetch() raises if it's still unset.
        self.mailto = mailto
        self.extras = extras
        self.base_url = base_url
        self._client = client
        self._last_cost: dict = {"wall_seconds": 0.0, "api_calls": 0}

    # ------------------------------------------------------------------

    def fetch(
        self,
        profile: Profile,
        since: str,
        *,
        limit: int | None = None,
    ) -> list[Paper]:
        client = self._resolve_client()
        t0 = time.time()
        calls_before = client.api_calls

        topic_filters = profile.topic_filters or {}
        works = client.search_works(
            topic_filters,
            since=since,
            extras=self.extras,
            limit=limit,
        )
        papers = [
            client.paper_from_work(w, source="openalex_gatherer")
            for w in works
        ]

        self._last_cost = {
            "wall_seconds": time.time() - t0,
            "api_calls": int(client.api_calls - calls_before),
        }
        return papers

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
            **self._last_cost,
        }

    def config(self) -> dict:
        return {
            "type": "openalex",
            "mailto": self.mailto,
            "extras": self.extras,
            "base_url": self.base_url,
        }

    @classmethod
    def from_config(cls, config: dict) -> "OpenAlexGatherer":
        return cls(
            mailto=config.get("mailto"),
            extras=config.get("extras", "type:article,language:en"),
            base_url=config.get("base_url", "https://api.openalex.org"),
        )

    def cost(self) -> dict:
        return self._last_cost
