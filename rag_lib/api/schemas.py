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


# Rank-percentile fallback cut-points for ``bucket`` when a candidate row
# has no raw similarity (rows written before ``score_raw`` existed): top
# tenth is "high", top third "medium". Rows with a raw similarity are
# bucketed against the profile's threshold and seed band instead — see
# ``rag_lib.calibration`` and ``mappers.candidate_row_to_card``.
BUCKET_HIGH = 0.90
BUCKET_MEDIUM = 0.67


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
    # Calibrated reading of ``coherence`` (see ``rag_lib.calibration``):
    # one of focused / broad / mixed / single / none, plus a 0–100
    # agreement figure for display.
    coherenceLabel: str = "none"
    agreement: int | None = None
    # Leave-one-out similarity of the seeds to their centroid: the band a
    # candidate has to reach to be "as close as your own papers".
    seedSimMin: float | None = None
    seedSimMax: float | None = None
    # The stored researcher this interest was built from, if any.
    researcherId: int | None = None


class Card(_Model):
    id: str
    title: str
    authors: list[str]
    venue: str
    date: str
    doi: str | None
    openalex: str
    profile: str
    # ``score`` is the rank percentile within the profile's pool (1.0 =
    # top). ``similarity`` is the raw cosine to the seed centroid — the
    # number the profile threshold is compared against. ``bucket`` is
    # derived from ``similarity`` vs the threshold and seed band when
    # available, else from the percentile.
    score: float
    similarity: float | None = None
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
    source: str | None = None


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


class SeedSimilarity(_Model):
    """Leave-one-out cosine of each seed to the centroid of the others."""

    min: float
    median: float
    max: float


class DryRunResponse(_Model):
    ok: bool = True
    key: str
    n: int
    # Per-candidate raw cosine scores from the profile's persisted
    # candidate set. The UI uses this list to render a slider-driven
    # histogram showing "how many of the N fetched would pass at θ".
    scores: list[float] = Field(default_factory=list)
    suggested_threshold: float | None = None
    seed_similarity: SeedSimilarity | None = None
    # [lo, hi] the slider should span: observed scores + seed band.
    score_range: list[float] | None = None


class GatherRun(_Model):
    """One row from ``gather_runs`` shaped for API consumers."""

    id: int
    # Null for a researcher import, which has no profile.
    profile_id: int | None = None
    researcher_id: int | None = None
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


class ProsopiaImportRequest(_Model):
    """Body of ``POST /api/import/prosopia``.

    ``ref`` is a bare slug, a full profile URL — what a user copies out
    of the address bar, and rejecting it would be a gratuitous papercut —
    or an ORCID (bare or as an ``orcid.org`` URL), which is resolved to
    the slug of the profile that researcher published. ``base_url`` points at a non-default Prosopia
    instance. ``name`` overrides the draft name, which otherwise comes
    from the Prosopia profile's own metadata; ``embedding_model``
    overrides the configured default, exactly as on the wizard's
    create-draft route.
    """

    ref: str
    base_url: str | None = None
    name: str | None = None
    embedding_model: str | None = None
    # The ``id`` values from GET /api/import/prosopia/works to keep;
    # omitted = every paper on the profile.
    paper_ids: list[str] | None = None


class ProsopiaWork(_Model):
    """One paper on a Prosopia profile, slimmed for a pick list."""

    id: str
    title: str
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    openalex_id: str | None = None
    cited_by_count: int | None = None
    authors: list[str] = []
    n_authors: int | None = None


class ProsopiaWorksResponse(_Model):
    """``GET /api/import/prosopia/works`` — the profile's papers, to pick from."""

    slug: str
    name: str | None = None
    works: list[ProsopiaWork]


class OrcidWork(_Model):
    """One of an author's OpenAlex works, slimmed for a pick list."""

    openalex_id: str
    doi: str | None = None
    title: str
    year: int | None = None
    venue: str | None = None
    type: str | None = None
    cited_by_count: int | None = None
    authors: list[str] = []
    n_authors: int | None = None
    author_position: str | None = None


class OrcidWorksResponse(_Model):
    """``GET /api/import/orcid/{orcid}/works`` — what the author published.

    ``name`` is read off the authorships and is null when OpenAlex has
    no works for the ORCID (there is nothing to read it from).
    """

    orcid: str
    name: str | None = None
    works: list[OrcidWork]


class OrcidImportRequest(_Model):
    """Body of ``POST /api/import/orcid``.

    ``openalex_ids`` are the works the user kept from the listing. The
    draft is named after the author unless ``name`` says otherwise.
    """

    orcid: str
    openalex_ids: list[str]
    name: str | None = None
    embedding_model: str | None = None


class ProsopiaImportStart(_Model):
    """Kick-off ack for ``POST /api/import/prosopia``.

    The draft exists by the time this returns — the Prosopia read and
    the draft insert happen inline, because a bad slug should be a 404
    on the request rather than an error buried in a run row. Everything
    expensive (OpenAlex resolution, embedding every paper) runs in the
    background under ``run_id``.
    """

    # Null for a researcher import, which creates no draft.
    draft_slug: str | None = None
    run_id: int


class ProsopiaImportResult(_Model):
    """What the import actually managed to do.

    ``resolved_by`` is the point of this shape. Every paper lands as a
    seed either way, so the row count alone cannot distinguish a profile
    that imported cleanly from one reconstructed by fuzzy title search.
    The per-rung counts can: ``work_id`` and ``doi`` are the paper the
    profile named, ``title`` is our best guess at it, and ``none`` is a
    synthetic id embedded from the Prosopia summary.
    """

    slug: str
    # Null for an import that targeted a researcher rather than a draft.
    draft_slug: str | None = None
    # The stored researcher the papers were recorded under, when any.
    researcher_id: int | None = None
    name: str
    drafted: int
    resolved_by: dict[str, int] = Field(default_factory=dict)
    unresolved: list[str] = Field(default_factory=list)


class ProsopiaImportStatus(_Model):
    """Poll response for ``GET /api/import/prosopia/{run_id}``.

    Mirrors ``DraftDryRunStatus``: the audit row always, the payload
    only once the job finished cleanly. A failed job leaves ``result``
    null and the reason on ``run.error``.
    """

    run: GatherRun
    result: ProsopiaImportResult | None = None


# ---------------------------------------------------------------------------
# Researchers — stored people and the papers that came with them
# ---------------------------------------------------------------------------


class Researcher(_Model):
    """One stored person, as ``GET /api/researchers`` lists them."""

    id: int
    source: str                         # 'prosopia' | 'orcid'
    key: str                            # prosopia slug or bare ORCID
    name: str
    orcid: str | None = None
    affiliation: str | None = None
    url: str | None = None
    base_url: str | None = None
    n_papers: int = 0
    n_interests: int = 0
    imported_at: str | None = None
    last_run_id: int | None = None
    # True while the last import is still running; poll it through
    # ``GET /api/import/prosopia/{last_run_id}``.
    importing: bool = False
    last_error: str | None = None
    created_at: str | None = None


class ResearcherPaper(_Model):
    """One of a researcher's papers. ``id`` is the OpenAlex id (or the
    synthetic ``prosopia:`` id for a paper that resolved nowhere)."""

    id: str
    title: str
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    resolved_by: str | None = None
    summary: str | None = None
    pdf_url: str | None = None
    added_at: str | None = None


class ResearcherDetail(_Model):
    """``GET /api/researchers/{id}``: the person, what the source said
    about them, their papers, and the interests built from them."""

    researcher: Researcher
    expertise: str | None = None
    soul: str | None = None
    grants: list[dict] = Field(default_factory=list)
    papers: list[ResearcherPaper] = Field(default_factory=list)
    interests: list[Profile] = Field(default_factory=list)


class SuggestedInterest(_Model):
    """One group of a researcher's papers that could be an interest."""

    name: str
    # Papers to seed with, most typical first; ``loose_ids`` are papers
    # in the group that sit below the same-field floor — listed so the
    # user can tick them, unticked by default.
    paper_ids: list[str] = Field(default_factory=list)
    loose_ids: list[str] = Field(default_factory=list)
    topics: list[dict] = Field(default_factory=list)
    coherence_median: float | None = None
    agreement: int | None = None
    label: str = "none"
    seed_titles: list[str] = Field(default_factory=list)


class ResearcherSuggestions(_Model):
    """``GET /api/researchers/{id}/suggestions``: at most three groups.

    ``note`` says why there is only one (too few papers, or no split
    that was clearly better than the whole set).
    """

    researcher_id: int
    embedding_model: str
    n_papers: int
    n_embedded: int
    # Papers inside some suggestion; the rest did not belong clearly
    # enough to any group and are left out on purpose.
    n_grouped: int = 0
    suggestions: list[SuggestedInterest] = Field(default_factory=list)
    note: str | None = None


class ResearcherImportRequest(_Model):
    """Body of ``POST /api/researchers/import``.

    ``source`` says how to read ``ref``: ``prosopia`` takes a slug, a
    profile URL or an ORCID published there; ``orcid`` takes an ORCID
    and reads the works off OpenAlex. ``paper_ids`` keeps a selection
    (Prosopia paper ids, or OpenAlex work ids); omitted = everything.
    Re-importing an existing researcher refreshes it in place.
    """

    source: str
    ref: str
    base_url: str | None = None
    paper_ids: list[str] | None = None
    embedding_model: str | None = None


class ResearcherImportStart(_Model):
    """The researcher row exists when this returns; the papers arrive
    under ``run_id``, polled through ``GET /api/import/prosopia/{run_id}``."""

    researcher_id: int
    run_id: int


class ResearcherInterestRequest(_Model):
    """Body of ``POST /api/researchers/{id}/interests``: a draft named
    ``name`` seeded with ``openalex_ids`` (omitted = every paper)."""

    name: str
    openalex_ids: list[str] | None = None
    embedding_model: str | None = None


class ResearcherInterestResponse(_Model):
    draft: Draft
    n_seeds: int


class DraftSeedsRequest(_Model):
    """Body of ``POST /api/profiles/draft/{slug}/seeds``: papers the user
    already has (uploaded, or imported with a researcher) to attach as seeds."""

    openalex_ids: list[str]


class DraftSeedsResponse(_Model):
    attached: int
    # Ids that were not the user's to use; nothing was attached for these.
    rejected: list[str] = Field(default_factory=list)


class LeastSimilarPair(_Model):
    a_id: str
    a_title: str
    b_id: str
    b_title: str
    cosine: float


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
    # Calibrated reading for people: label, 0–100 agreement, a sentence,
    # the seed band, and which two seeds agree least.
    label: str = "none"
    agreement: int | None = None
    summary: str = ""
    seed_similarity: SeedSimilarity | None = None
    least_similar: LeastSimilarPair | None = None


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
    suggested_threshold: float | None = None
    seed_similarity: SeedSimilarity | None = None
    score_range: list[float] | None = None


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
