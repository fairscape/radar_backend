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
LLMProvider = Literal["ollama", "anthropic", "openai"]


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class Health(_Model):
    status: HealthStatus = "ok"
    version: str
    # ``ollama_model`` is retained for back-compat with frontends shipped
    # before /api/chat grew provider support — populate only when the
    # active provider is Ollama. New clients should read ``llm_provider``
    # and ``llm_model`` instead.
    ollama_model: str | None = None
    llm_provider: LLMProvider | None = None
    llm_model: str | None = None


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
    #: Researcher Profile metadata when the topic was seeded "From Profile"
    #: (summary, affiliation, expertise, not_interests, collaborators, …)
    rp_meta: dict | None = None


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
    # Seeds imported from OpenAlex have no PDF (pages=0); the wizard shows
    # the year instead.
    year: int | None = None


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
    source: str | None = None
    #: ORCID profiles only: how many seed papers carry this topic (drives the default ``on``)
    seed_papers: int | None = None
    #: "From Profile" imports: which expertise / not_interests phrases switched it on / off
    rp_on_by: list[str] | None = None
    rp_off_by: list[str] | None = None


class SweepRow(_Model):
    th: float
    n: int
    top: str


class ChatSource(_Model):
    n: int
    title: str
    score: float
    # The retrieved chunk text the LLM saw. Optional so older
    # assistant rows persisted before this field existed still
    # decode cleanly — they render with an empty evidence panel.
    text: str | None = None


class ChatTurn(_Model):
    who: ChatRole
    t: str
    body: str | list[str]
    sources: list[ChatSource] | None = None


class ChatRequest(_Model):
    """Body of ``POST /api/chat``.

    ``scope`` is the list of profile slugs retrieval is restricted to;
    an empty list searches across the whole user's vault. ``provider``
    overrides ``RADAR_LLM_PROVIDER`` for this one request; omit it (or
    pass ``null``) to use the server default.
    """

    query: str
    scope: list[str] = Field(default_factory=list)
    provider: LLMProvider | None = None


class ProviderInfo(_Model):
    id: LLMProvider
    # The model identifier the operator has configured for this provider.
    # Exposed so the UI can label the dropdown without a second round
    # trip. Never includes the API key or any portion of it.
    model: str
    # True when the operator has supplied everything this provider
    # needs (API key for Anthropic/OpenAI, URL for Ollama). Frontends
    # render un-configured options as disabled.
    configured: bool


class ProvidersResponse(_Model):
    default: LLMProvider
    available: list[ProviderInfo]


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
    # Live-progress fields for in-flight runs. Populated incrementally
    # by gather_for_profile via gather_runs.set_step / .tick; the
    # frontend polls /runs and reads these to render a per-step
    # progress chip instead of a bare spinner.
    current_step: str | None = None
    n_processed: int | None = None
    n_total: int | None = None
    last_message: str | None = None
    # When the job last wrote progress; the wizard uses it to tell a slow
    # step from a job that died with a server restart.
    progress_updated_at: str | None = None


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


class DraftSummary(_Model):
    """One row of ``GET /api/profiles/drafts`` — an unfinished wizard draft."""

    slug: str
    name: str
    created_at: str | None = None
    updated_at: str | None = None
    n_seeds: int = 0
    orcid: str | None = None
    researcher_name: str | None = None
    #: an ORCID import job is still running for this draft
    importing: bool = False
    #: latest ORCID import run for this draft (to resume polling / show its result)
    import_run_id: int | None = None
    #: seeded from a Researcher Profile document
    rp: bool = False
    #: fetched works waiting in the seed picker (ORCID / profile drafts)
    n_works: int = 0
    #: ORCID / profile drafts: fetching | selecting | seeding | seeded; None for PDF drafts
    phase: str | None = None


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


class DraftDryRunStart(_Model):
    """Kickoff response from ``POST /api/profiles/draft/{slug}/dry-run``.

    The dry-run is dispatched async and writes progress + result to a
    ``gather_runs`` row keyed by ``run_id``. The wizard polls
    ``GET /api/profiles/draft/{slug}/dry-run/{run_id}`` until ``run.
    finished_at`` is set, then reads ``result``.
    """

    ok: bool = True
    run_id: int


class DraftDryRunStatus(_Model):
    """Status-poll response: in-flight progress + final result if done."""

    run: GatherRun
    result: DraftDryRun | None = None


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


class OrcidDraftCreateRequest(_Model):
    """Body of ``POST /api/profiles/draft/from-orcid``.

    ``orcid`` accepts the bare id or the ``https://orcid.org/…`` URL; it
    is normalised and check-digit validated by the route. ``name``
    defaults to OpenAlex's display name for the author. ``mailto`` joins
    the OpenAlex polite pool (defaults to the user's email).
    """

    orcid: str
    name: str | None = None
    mailto: str | None = None
    embedding_model: str | None = None
    selector: str | None = None


class OrcidDraftStart(_Model):
    """Kickoff response: the draft handle plus the run to poll."""

    slug: str
    name: str
    run_id: int


class OrcidAuthor(_Model):
    orcid: str
    openalex_author_id: str | None = None
    display_name: str
    institution: str | None = None


class OrcidImportResult(_Model):
    """``result_json`` of an ``orcid_import`` (phase "fetch") or ``orcid_seed``
    (phase "seed") gather_runs row."""

    phase: str = "seed"
    n_fetched: int
    n_kept: int
    #: works offered to the seed picker / what the default rule pre-checks
    n_works: int = 0
    n_default_seeds: int = 0
    n_seeds: int
    n_embedded: int
    author: OrcidAuthor
    rp_profile_dir: str | None = None
    warnings: list[str] = Field(default_factory=list)
    report: dict[str, int] = Field(default_factory=dict)
    #: "From Profile" imports only: what the profile's expertise /
    #: not_interests did to the concept list (see rp_profile_import)
    rp: dict | None = None


class OrcidWork(_Model):
    """One fetched work of an ORCID / profile draft, as the seed picker shows it."""

    openalex_id: str
    title: str
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    first_author: str | None = None
    position: str | None = None
    is_corresponding: bool = False
    author_index: int | None = None
    total_authors: int | None = None
    work_type: str | None = None
    cited_by_count: int = 0
    #: the person's ORCID record lists it (None: registry unavailable)
    claimed: bool | None = None
    #: datasets cannot be seeds
    seed_eligible: bool = True
    #: openalex_id of the copy the default rule keeps when this is a duplicate version
    dup_of: str | None = None
    has_abstract: bool = False
    #: what the default rule pre-checks
    default_selected: bool = False
    #: the user's last confirmed choice (None before any confirmation)
    selected: bool | None = None
    is_seed: bool = False


class SeedSelectRequest(_Model):
    """Body of ``POST /api/profiles/draft/{slug}/seeds/select``."""

    openalex_ids: list[str]


class ResearcherSummary(_Model):
    """One row of the user's Researchers library (``GET /api/researchers``)."""

    id: int
    rid: str
    orcid: str | None = None
    name: str
    affiliation: str | None = None
    field: str | None = None
    level: str | None = None
    provenance: str | None = None
    date_modified: str | None = None
    source_kind: str = "paste"
    imported_at: str | None = None
    updated_at: str | None = None
    n_expertise: int = 0
    n_not_interests: int = 0
    n_papers: int | None = None


class ResearcherDetail(ResearcherSummary):
    """``GET /api/researchers/{id}``: the parsed view plus the raw document."""

    parsed: dict = Field(default_factory=dict)
    doc: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ResearcherImportRequest(_Model):
    """Body of ``POST /api/researchers``: the text of a profile.jsonld."""

    profile_json: str
    source_kind: str = "paste"


class ResearcherImportResult(_Model):
    researcher: ResearcherSummary
    created: bool
    warnings: list[str] = Field(default_factory=list)


class RpDraftCreateRequest(_Model):
    """Body of ``POST /api/profiles/draft/from-profile``.

    ``profile_json`` is the text of a Researcher Profile ``profile.jsonld``
    (pasted or read from a file client-side). ``name`` defaults to the
    profile's ``name``. ``mailto`` joins the OpenAlex polite pool.
    """

    profile_json: str
    name: str | None = None
    mailto: str | None = None


class RpDraftStart(_Model):
    """Kickoff response for a profile import.

    ``mode`` is ``"orcid"`` when the profile carries an ORCID and the
    OpenAlex import was dispatched (poll ``run_id``), or ``"pdf"`` when it
    does not: the draft exists with the profile's metadata attached and
    the user uploads PDFs as seeds (``run_id`` is null).
    """

    slug: str
    name: str
    mode: str
    run_id: int | None = None
    orcid: str | None = None
    warnings: list[str] = Field(default_factory=list)


class OrcidImportStatus(_Model):
    """Status-poll response for ``GET /api/profiles/draft/{slug}/import/{run_id}``."""

    run: GatherRun
    result: OrcidImportResult | None = None


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


# ---------------------------------------------------------------------------
# Phase B+ — reranker comparison
# ---------------------------------------------------------------------------


class TopicYield(_Model):
    """What one topic's gather quota has actually produced."""

    topic_id: str
    display_name: str
    on: bool
    n_candidates: int = 0
    n_shown: int = 0
    n_saved: int = 0
    n_dismissed: int = 0
    last_fetched_at: str | None = None


class TopicYieldResponse(_Model):
    """Output of ``GET /api/profiles/{key}/topic-yield``."""

    ok: bool = True
    key: str
    days: int
    topics: list[TopicYield] = Field(default_factory=list)


class RerankerCandidate(_Model):
    """One row in the reranker comparison bump chart."""

    openalex_id: str
    title: str
    score_selector: float
    score_blended: float
    rank_before: int
    rank_after: int


class RerankerComparisonResponse(_Model):
    """Output of ``GET /api/profiles/{key}/reranker-comparison``."""

    ok: bool = True
    key: str
    n: int
    candidates: list[RerankerCandidate] = Field(default_factory=list)
    avg_rank_change: float = 0.0
    max_rank_up: int = 0
    max_rank_down: int = 0
    queries_used: list[str] = Field(default_factory=list)
