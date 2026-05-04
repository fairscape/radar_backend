"""Pydantic models mirroring ``radar-website/src/types/radar.ts``.

The frontend's TypeScript file is the contract; field names and types
here must stay aligned. Diff the two files at the end of Phase 5 and
again at Phase 10 — drift between them silently breaks the frontend
swap from mock-api to real API.

Bucket thresholds are constants here so Phase 5 can import them when
assigning ``bucket`` from ``score`` instead of duplicating numbers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# Bucket thresholds — Phase 5 reads these when assigning ``bucket`` from ``score``.
BUCKET_HIGH = 0.95
BUCKET_MEDIUM = 0.925


HealthStatus = Literal["ok", "warn", "err"]
Bucket = Literal["high", "medium", "low"]
CardState = Literal["saved", "dismissed"]
ChatRole = Literal["user", "assistant"]


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class Health(_Model):
    status: HealthStatus = "ok"
    version: str


class Profile(_Model):
    key: str
    name: str
    hue: int
    health: HealthStatus
    threshold: float
    coherence: float
    seeds: int
    saves30: int
    dismisses30: int
    isDraft: bool = False


class Card(_Model):
    id: str
    title: str
    authors: list[str]
    venue: str
    date: str
    doi: str | None
    openalex: str
    profile: str
    score: float
    bucket: Bucket
    abstract: str
    mesh: list[str]
    terms: list[str]
    matched: list[str]
    topicMatch: float
    centroidCos: float
    noveltyDelta: float
    mins: int


class VaultDoc(_Model):
    id: str
    title: str
    authors: list[str]
    venue: str
    tags: list[str]
    pages: int
    chunks: int
    added: str


class Seed(_Model):
    id: str
    idx: int
    title: str
    year: int
    coh: float


class Topic(_Model):
    id: str
    name: str
    count: int
    on: bool


class SweepRow(_Model):
    th: float
    n: int
    top: str


class ChatSource(_Model):
    n: int
    title: str
    score: float


class ChatTurn(_Model):
    who: ChatRole
    t: str
    body: str | list[str]
    sources: list[ChatSource] | None = None


class ChatRequest(_Model):
    """Body of ``POST /api/chat``.

    ``scope`` is the list of profile slugs retrieval is restricted to;
    an empty list searches across the whole user's vault.
    """

    query: str
    scope: list[str] = Field(default_factory=list)


class DailyRadarFilters(_Model):
    profile: str | None = None
    bucket: Bucket | None = None


class DailyRadarResponse(_Model):
    date: str
    fetchedAt: str
    fetchMs: int
    candidatesScored: int
    cards: list[Card]
    states: dict[str, CardState | None] = Field(default_factory=dict)


class VaultStats(_Model):
    docs: int
    pages: int
    chunks: int
    lastIngest: str


class VaultMeta(_Model):
    """Vault provenance shown in the UI's metadata panel.

    Mirrors the mock-api ``VaultMeta`` shape:
      - rootPath: filesystem root for this user's PDFs.
      - indexPath: vector-index location for chat retrieval. Carries a
        ``(not yet indexed)`` suffix until Phase 9 wires Chroma.
      - chunkSize: human-readable chunk policy string.
      - lastIngest: ISO datetime of the most recent upload.
    """

    rootPath: str
    indexPath: str
    chunkSize: str
    lastIngest: str


class CoherenceStats(_Model):
    min: float
    max: float


class ProfileDetail(_Model):
    """Composite returned by ``GET /api/profiles/{key}/detail``.

    Mirrors ``ProfileDetail`` in the mock-api ``profiles.ts`` endpoint:
    the base ``Profile`` plus seeds, topic mix, threshold sweep,
    coherence histogram, and the feedback log preview. The
    ``feedbackLog`` / ``feedbackMoreCount`` fields are populated by
    Phase 8; Phase 5 returns empties.
    """

    profile: Profile
    seeds: list[Seed] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    sweep: list[SweepRow] = Field(default_factory=list)
    coherenceBins: list[int] = Field(default_factory=list)
    coherenceStats: CoherenceStats
    feedbackLog: list[str] = Field(default_factory=list)
    feedbackMoreCount: int = 0


class CardActionResponse(_Model):
    """Return shape for ``POST /api/radar/cards/{id}/{save|dismiss}``."""

    id: str
    state: CardState | None = None


class FeedbackEventOut(_Model):
    """One row from ``feedback_events`` shaped for API consumers (Phase 8)."""

    id: int
    profile_id: int
    openalex_id: str
    doi: str | None = None
    action: str
    score: float | None = None
    selector: str | None = None
    selector_config_hash: str | None = None
    benchmark_run_id: int | None = None
    ts: str


class RefitResponse(_Model):
    ok: bool = True
    key: str
    cost: str


class DryRunResponse(_Model):
    ok: bool = True
    key: str
    n: int
    # Per-candidate raw cosine scores from the profile's persisted
    # candidate set. The UI uses this list to render a slider-driven
    # histogram showing "how many of the N fetched would pass at θ".
    scores: list[float] = Field(default_factory=list)


class GatherRun(_Model):
    """One row from ``gather_runs`` shaped for API consumers."""

    id: int
    profile_id: int
    started_at: str
    finished_at: str | None = None
    since_date: str | None = None
    filter_string: str | None = None
    tier_used: str | None = None
    n_fetched: int | None = None
    n_new: int | None = None
    n_redup: int | None = None
    api_calls: int | None = None
    error: str | None = None


class GatherNowResponse(_Model):
    ok: bool = True
    run_id: int


class Schedule(_Model):
    """One row from ``profile_schedules``."""

    profile_id: int
    cron: str
    tz: str
    enabled: bool
    updated_at: str | None = None


class ScheduleUpdate(_Model):
    """Body of ``PATCH /api/profiles/{key}/schedule`` — all fields optional."""

    cron: str | None = None
    tz: str | None = None
    enabled: bool | None = None


class ProfileThresholdUpdate(_Model):
    """Body of ``PATCH /api/profiles/{key}`` — for inline threshold edits."""

    threshold: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Phase 11 — profile-build wizard
# ---------------------------------------------------------------------------


class Draft(_Model):
    """Lightweight handle returned by ``POST /api/profiles/draft``."""

    slug: str
    name: str


class DraftCreateRequest(_Model):
    name: str
    # Plugin keys. Both default to ``None`` so the router can fall back
    # to the configured ``RADAR_DEFAULT_EMBEDDING_MODEL`` and the
    # built-in ``"centroid"`` selector. Any registered key from
    # ``rag_lib.embedders.EMBEDDERS`` / ``rag_lib.selectors.SELECTORS``
    # is accepted; unknown keys raise 400.
    embedding_model: str | None = None
    selector: str | None = None


class DraftCoherence(_Model):
    """Output of ``POST /api/profiles/draft/{slug}/coherence``.

    Mirrors ``rag_lib.coherence.coherence`` plus the 16-bin histogram
    the UI renders. ``n`` is the number of staged seed embeddings; the
    histogram + median + IQR are NaN-free (NaNs collapse to 0.0) so the
    JSON is renderable without front-end guards.
    """

    bins: list[int] = Field(default_factory=list)
    median: float = 0.0
    iqr: float = 0.0
    bimodal: bool = False
    n: int = 0


class DraftDryRunRequest(_Model):
    """Body of ``POST /api/profiles/draft/{slug}/dry-run``.

    ``days`` capped at 30 server-side regardless of input. ``thresholds``
    defaults to a 0.50–0.95 sweep when omitted.
    """

    days: int = 30
    thresholds: list[float] | None = None


class DraftDryRun(_Model):
    """Wizard Step-4 payload — sweep rows + preview cards + raw scores.

    ``scores`` is the full list of selector raw cosines (one entry per
    fetched candidate); the UI uses it to drive a slider-driven
    histogram so the user can see how many candidates would pass at any
    chosen θ. ``sweep`` is retained for callers that want pre-bucketed
    counts at the canonical thresholds.
    """

    sweep: list[SweepRow] = Field(default_factory=list)
    preview: list[Card] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)


class WizardOption(_Model):
    """One row in the wizard's embedder/selector dropdowns.

    ``key`` is the registry key the create-draft body expects.
    ``label`` and ``description`` are human-friendly strings the UI
    renders. ``default`` is true on exactly one row per list — the value
    the dropdown should pre-select.
    """

    key: str
    label: str
    description: str = ""
    default: bool = False


class WizardOptions(_Model):
    """Output of ``GET /api/profiles/wizard/options``."""

    embedders: list[WizardOption] = Field(default_factory=list)
    selectors: list[WizardOption] = Field(default_factory=list)


class CommitDraftRequest(_Model):
    """Body of ``POST /api/profiles`` (commit-draft)."""

    slug: str
    threshold: float
    selected_topic_ids: list[str] = Field(default_factory=list)
    cron: str | None = None
    tz: str | None = None


# ---------------------------------------------------------------------------
# Phase 12 — auth / users
# ---------------------------------------------------------------------------


class User(_Model):
    """Public shape of a row in ``users``.

    ``mailto`` is the OpenAlex polite-pool address used by the gatherer
    for this user; defaults to ``email`` on first upsert and is editable
    via ``PATCH /api/users/me``.
    """

    id: int
    email: str
    mailto: str | None = None
    created_at: str | None = None


class UserUpdateRequest(_Model):
    """Body of ``PATCH /api/users/me``."""

    mailto: str | None = None
