# rag_lib — Personal Research Radar backend

> **Quickstart with docker-compose:** if you just want to run the demo,
> see [`../README.md`](../README.md) at the repo root — one command
> brings up backend + frontend + (optional) Ollama. This README covers
> the Python library internals and dev workflow against a local
> checkout.

Track 1 Python library for the Personal Research Radar and RAG System
(build spec v1.1). This README documents the Phase 1A integration
contract that both tracks build against. The protocols and the feedback
log schema are **frozen at Phase 1A exit**; changes after that require
explicit cross-track re-agreement.

## Current phase

**Phase 1B — Selectors, Ingestion, Radar.** Shipped on top of 1A:

- `rag_lib/embed.py` — `build_embedding_input(paper, max_tokens=512)` with
  labeled sections and drop-priority truncation (body → substances →
  keywords → mesh; title + abstract always preserved).
- `rag_lib/embedders.py` — `specter2_embed` added (lazy SPECTER2 +
  proximity-adapter loader, requires `phase1b` extras). `EMBEDDERS`
  registry + `get_embedder(name)` lookup so selectors resolve at score
  time by `profile.embedding_model`.
- `rag_lib/coherence.py` — pairwise-cosine median / IQR / bimodal flag
  over a seed corpus.
- `rag_lib/selectors/centroid.py` — real fit/select. Reads seed vectors
  from the profile, scores candidates by cosine to the centroid (mapped
  to `[0, 1]`), reuses stored candidate embeddings when present and
  otherwise computes via the registry. Diagnostics include coherence.
- `rag_lib/openalex_client.py` — `search_works` real (cursor-paginated,
  OR-joined topic / subfield / field / domain / primary_topic filters,
  AND-joined with `from_created_date` and extras, polite-pool mailto on
  every request).
- `rag_lib/gatherers/openalex.py` — `fetch` real; calls `search_works`
  then `paper_from_work`, records api_calls + wall_seconds.
- `rag_lib/vault.py` — `ingest_pdf(path)` (pdfplumber, lazy import) →
  `PdfIngestRecord(title, body_text, doi, n_pages)`.
- `Profile.from_pdfs` — real; ingests a PDF directory, OpenAlex-enriches,
  embeds, aggregates topic_filters.
- `rag_lib/radar.py` — `dry_run(selector, profile, gatherer, thresholds,
  days)` and `centroid_drift(v1, v2)`.
- `cli/ingest.py` — end-to-end prototype CLI (`--csv` or `--pdf-dir` →
  Profile, fit, dry-run, live).
- `radar/pdfs/paper-corpus.csv` — sample manifest derived from the
  existing 41-paper corpus, ready to feed `Profile.from_csv`.

**Phase 1A — Protocol and Scaffolding.** Shipped:

- **Paper / Topic / TopicNode** data classes (JSON round-trip, OpenAlex
  four-level topic hierarchy, embeddings keyed by model name, local PDF
  path).
- **Profile** data class — name, seed papers with embeddings + topics,
  aggregated topic_filters, selector/gatherer configs, threshold; JSON
  file I/O; stands on its own without the source PDFs.
- **OpenAlexClient** — `lookup_by_doi` / `lookup_by_title` / `slim_work`
  / `paper_from_work` lifted from `radar/openalex_lookup.py` into a
  reusable class. `search_works` (cursor-paginated candidate fetch)
  stays stubbed for Phase 1B.
- **Profile build methods** — `Profile.from_csv` and `Profile.from_pdfs`
  classmethods (both real as of 1B). `Profile.aggregate_topic_filters`
  staticmethod.
- **embedders** — `rag_lib.embedders.placeholder_embed` (deterministic
  hash-seeded unit-norm vector). Phase 1B adds `specter2_embed`
  alongside it; both slot into `Profile.from_csv` via the `embedder=`
  keyword.
- **Selector** protocol, parametrized compliance test, FixtureSelector
  reference, typed stubs for CentroidSelector and LLMRetrainSelector.
- **Gatherer** protocol, parametrized compliance test, FixtureGatherer
  reference, typed stub for OpenAlexGatherer.

Phase 1B will land: SPECTER2 embedder, real CentroidSelector body,
`OpenAlexClient.search_works`, `OpenAlexGatherer.fetch`, PDF ingestion
in `build_from_pdfs`, and the end-to-end CLI. Track 2 owns the
`LLMRetrainSelector` body on its own schedule.

## Quick start

```bash
cd radar/radar-backend
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest -v
```

Expected result:
- Paper/Profile/embedder/build tests pass (`test_paper.py`,
  `test_profile.py`, `test_embedders.py`, `test_profile_build.py`).
- FixtureSelector + FixtureGatherer pass their respective full suites.
- CentroidSelector, LLMRetrainSelector, and OpenAlexGatherer stubs pass
  the non-invoking subset and the `TestStubsRaise` sentinel.

## Running the API

The FastAPI service lives at `rag_lib.api.app:app`. Boot it with:

```bash
cd radar/radar-backend
python -m cli.serve                         # 127.0.0.1:8000 by default
python -m cli.serve --host 0.0.0.0 --port 8000 --reload
curl http://localhost:8000/api/health       # {"status":"ok","version":...}
open http://localhost:8000/docs              # interactive route list
```

Configuration is loaded from environment variables and an optional
`.env` file in the working directory; see `.env.example` for every
key. Settings of note:

| key                       | default                  | meaning                                  |
|---------------------------|--------------------------|------------------------------------------|
| `RADAR_DB_PATH`           | `data/radar.db`          | SQLite file the API reads/writes.        |
| `RADAR_VAULT_DIR`         | `data/vault`             | Per-user PDF storage root.               |
| `RADAR_CHROMA_DIR`        | `data/chroma`            | Vector index for Phase 9 RAG chat.       |
| `RADAR_OLLAMA_URL`        | `http://localhost:11434` | Ollama service for chat (Phase 9).       |
| `RADAR_CORS_ORIGINS`      | `["http://localhost:5173"]` | Frontend dev origin.                  |
| `RADAR_LOG_JSON`          | `true`                   | JSON logs vs friendly console renderer.  |
| `RADAR_SCHEDULER_ENABLED` | `true`                   | Phase 7 toggle for in-process APScheduler. |

Phase 4 ships the skeleton: `/api/health` is live, every other route
returns HTTP 501 with a "not yet implemented" message identifying the
phase that owns the body.

## Database

The SQLite layer designed in `DB_DESIGN.md` is implemented under
`rag_lib/db/` with hand-rolled migrations, repos, and a `cli/db.py`
management entry point. The default file is `data/radar.db`. Schema
includes `users` (multi-user-capable; default user 1 is `demo@example.com`),
`profiles` (with `slug` URL key and `user_id` FK), `papers`,
`paper_embeddings`, `profile_seeds`, `profile_candidates`, and
`gather_runs`.

```bash
python -m cli.db init                # create the DB and apply migrations
python -m cli.db status              # list applied migrations
python -m cli.db upgrade             # apply any pending migrations
python -m cli.db profiles            # list profiles for the demo user
python -m cli.db candidates --profile <slug> --unshown
```

`cli/ingest.py` and `cli/gather.py` will accept a `--db` flag in Phase 2
so profile + seeds + embeddings + candidates persist into the same file
the FastAPI service will read in Phase 4+.

## Data model

### Paper

```python
@dataclass
class Paper:
    doi: str | None
    openalex_id: str | None
    title: str
    abstract: str = ""
    year: int | None = None
    venue: str | None = None
    mesh: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    substances: list[str] = field(default_factory=list)
    embeddings: dict[str, list[float]] = field(default_factory=dict)  # {model_name: vector}
    primary_topic: Topic | None = None
    topics: list[Topic] = field(default_factory=list)
    local_path: str | None = None     # path to PDF on disk
    source: str = "unknown"
    added: str | None = None          # ISO-8601 UTC
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Paper": ...
    def embedding_for(self, model: str) -> list[float] | None: ...
```

`embeddings` is keyed by model name so the same Paper can carry multiple
vectors side-by-side (placeholder + SPECTER2 + whatever 1B adds). The
Selector reads `profile.embedding_model` and looks up the matching key.

`local_path` is the on-disk path to the PDF when we have one. The
Profile JSON carries the string; the PDF itself stays on disk. Phase 4's
RAG chat will reopen these for deeper context; Phase 1 only uses them
as a pointer.

### Topic hierarchy (OpenAlex)

```
Domain    (4: Health Sciences, Life Sciences, Physical Sciences, Social Sciences)
  Field       (~26: Medicine, CS, Biochemistry, ...)
    Subfield    (~250: Pediatrics, Software, AI, ...)
      Topic       (~4,500: "Heart Rate Variability and Autonomic Control", ...)
```

Each `Topic` on a Paper carries its position at all four levels plus an
OpenAlex classifier score. The profile_builder aggregates these across
the seed into `profile.topic_filters` with per-level ID lists, so
Gatherers can filter at whatever specificity a profile needs.

### Profile

```python
@dataclass
class Profile:
    name: str
    papers: list[Paper]                  # seed corpus, with embeddings + topics
    topic_filters: dict                  # {"topics":[{id,display_name,count}], "subfields":[...], "fields":[...], "domains":[...]}
    embedding_model: str                 # which embedding key selectors read
    selector_config: dict
    gatherer_config: dict
    threshold: float | None
    created: str | None
    last_radar_date: str | None
    def to_json(self, path) / from_json(path): ...
    def seed_embeddings(self, model=None) -> list[list[float]]: ...
```

Profile JSON is the portable artifact. Once built (PDFs + OpenAlex
lookup + embeddings), it stands on its own: the Gatherer runs against
it, the Selector fits and scores against it, and no step needs the
original PDFs except RAG chat (which uses `paper.local_path`).

## Profile build

```python
from rag_lib import Profile
from rag_lib.openalex_client import OpenAlexClient

client = OpenAlexClient(mailto="you@example.com")
profile = Profile.from_csv("manifest.csv", name="neonatal_vitals",
                           openalex_client=client)
profile.to_json("profiles/neonatal_vitals.json")
```

CSV columns (any subset; at least one resolution path required):

| column   | purpose                                                        |
|----------|----------------------------------------------------------------|
| `doi`    | tried first — `client.lookup_by_doi(doi)`                      |
| `title`  | OpenAlex search fallback when DOI misses                       |
| `year`   | narrows title-search filter                                    |
| `path`   | local PDF path, stored on `Paper.local_path`                   |
| `abstract` | used only if OpenAlex silent on this paper                  |

Papers OpenAlex can't resolve (paywalled, not indexed) still become
`Paper` records with `source="user_csv"` so the seed corpus stays
complete. Embeddings are generated for every row from title+abstract.

The default embedder (`rag_lib.embedders.placeholder_embed`) is a
deterministic hash-seeded unit-norm vector — not a real semantic
embedding, but enough to exercise fit/score/JSON plumbing end-to-end.
Phase 1B swaps in SPECTER2 by passing
`Profile.from_csv(..., embedder=specter2_embed, embedding_model="specter2")`.

Topic-filter aggregation is exposed separately as
`Profile.aggregate_topic_filters(papers, top_k_each=8)` (staticmethod)
so Phase 2+ re-aggregation workflows can invoke it without re-running
the full build.

## Selector protocol

```python
from typing import Protocol
from rag_lib.paper import Paper
from rag_lib.profile import Profile

class Selector(Protocol):
    name: str  # "centroid" | "llm_retrain" | "fixture"

    def fit(self, profile: Profile) -> None: ...
    def select(self, candidates: list[Paper], profile: Profile,
               threshold: float | None = None) -> list[tuple[float, Paper]]: ...
    def diagnostics(self) -> dict: ...
    def config(self) -> dict: ...
    @classmethod
    def from_config(cls, config: dict) -> "Selector": ...
    def cost(self) -> dict: ...
```

Both `fit` and `select` receive the Profile, so hybrid selectors can
reason about the target beyond their fitted internal state — matching
a candidate's tags against `profile.topic_filters`, prompting an LLM
with the profile's human-readable name, etc. Simple selectors that
encode the entire target into their fitted state (e.g.,
`CentroidSelector` — the centroid *is* the profile) may ignore it, but
every selector must accept the argument.

Score semantics are selector-specific but must be monotonic in relevance
and clipped to `[0, 1]` (tolerance `±1e-6` for near-antipodal cosine).
`select()` returns `(score, paper)` tuples sorted descending by score;
if `threshold` is provided the result is a suffix of the unfiltered
output.

## Gatherer protocol

A `Gatherer` is the source of candidate papers for a Profile. It reads
the profile's topic_filters + gatherer_config and returns new Papers
since a given date.

```python
class Gatherer(Protocol):
    name: str  # "openalex" | "fixture" | ...

    def fetch(self, profile: Profile, since: str,
              *, limit: int | None = None) -> list[Paper]: ...
    def diagnostics(self) -> dict: ...
    def config(self) -> dict: ...
    @classmethod
    def from_config(cls, config: dict) -> "Gatherer": ...
    def cost(self) -> dict: ...
```

**Deriving topic_filters from a seed corpus is NOT the gatherer's job.**
That lives in `rag_lib.profile_builder.aggregate_topic_filters`, which
runs at Profile-build time. The Gatherer only fetches candidates
against a fully-built Profile. This keeps sources swappable (OpenAlex
today, arXiv/bioRxiv/PubMed later) without entangling each one with
profile-construction semantics.

**One gatherer per profile, not per selector.** All selectors running
against a profile (including in benchmark mode) see the same gatherer
output. This is what makes the benchmark an apples-to-apples ranking
comparison: selectors are distinguished by how they rank a shared pool,
not by which papers they fetch. Enforced at the profile-schema layer in
Phase 2; don't route around it.

`Gatherer.cost()` includes `api_calls` alongside `wall_seconds`. This
matters for rate-limited sources like the OpenAlex polite pool.

## Compliance tests

Two parallel compliance files, same three-part structure in each:

- `tests/test_selector_protocol.py` — Selector compliance.
- `tests/test_gatherer_protocol.py` — Gatherer compliance.

| Parametrization   | Who passes                                           |
|-------------------|------------------------------------------------------|
| `TestNonInvoking` | Every implementation (stubs and full impls).         |
| `TestFull`        | Only full implementations. Fixture* today.           |
| `TestStubsRaise`  | The stubs. Removed from a stub's entry once shipped. |

**Benchmark enrollment gates on the full Selector suite.** A selector
cannot join a benchmark run until it passes `TestFull` in
`test_selector_protocol.py`. Gatherers are not benchmarked — a new
gatherer is added by swapping it at the profile-schema layer, not by
running it as a benchmark arm.

To add an implementation to the compliance run, append its class to
`FULL_SELECTORS` / `STUB_SELECTORS` (or the gatherer equivalents) at
the top of the relevant test file.

## Feedback log schema — frozen at 1A exit

One record per save or dismiss action, written append-only to
`~/rag_vault/state/feedback.jsonl`. Track 2 depends on this schema;
changes after 1A exit require cross-track re-agreement.

```jsonc
{
  "timestamp": "str",             // ISO-8601 UTC, required
  "profile": "str",               // profile name, required
  "selector": "str",              // "centroid" | "llm_retrain" | "fixture", required
  "selector_config_hash": "str",  // sha256 of selector.config() at score time
  "benchmark_run_id": "str | null", // set only when benchmark_active was true
  "doi": "str | null",            // DOI of the card, or null for DOI-less preprints
  "openalex_id": "str",           // OpenAlex work ID, always set
  "score": 0.0,                   // float in [0, 1], score at radar time
  "action": "str"                 // "save" | "dismiss", required
}
```

`selector_config_hash` is what lets a retraining run distinguish feedback
produced by an earlier model version from feedback produced by the
current model without losing earlier feedback. The log file itself is
written starting in Phase 2; the schema is locked here so Track 2 can
develop against it from day one.

## Track 2 quick-start

Track 2 owns the body of `LLMRetrainSelector`. Everything you need to
begin is in this repo at the Phase 1A tag.

1. **Clone and install.**
   ```bash
   cd radar/radar-backend
   python -m venv .venv && source .venv/bin/activate
   pip install -e '.[dev]'
   pytest -v     # confirm the suite is green before you change anything
   ```
2. **Replace the stub body** at `rag_lib/selectors/llm_retrain.py`. The
   class surface — `name`, `__init__`, `fit`, `select`, `diagnostics`,
   `config`, `from_config`, `cost` — is frozen. Keep every method
   signature identical; replace the two `raise NotImplementedError(...)`
   bodies with the real implementation. `fit(profile: Profile)` receives
   the seed corpus via `profile.papers` and the chosen embedding key via
   `profile.embedding_model`. `select(candidates, profile, threshold)`
   gets both the candidate Papers and the Profile so you can prompt with
   the profile's human-readable name and topic set.
3. **Promote your selector to the full suite.** In
   `tests/test_selector_protocol.py`, move `LLMRetrainSelector` from
   `STUB_SELECTORS` to `FULL_SELECTORS`. `TestFull` must pass before
   benchmark enrollment is allowed.
4. **Benchmark governance.** See "Selector Delivery Schedule" in the
   spec (`radar/radar_build_spec_v1_1.docx.docx`). Full-suite pass by
   Benchmark Start minus two weeks is the drop-dead gate.
5. **Candidate pool.** `LLMRetrainSelector` sees the same candidate pool
   as every other selector running against a profile. The pool is
   produced by the profile's `Gatherer` (typically `OpenAlexGatherer`).
   One gatherer per profile — you do not get to fetch your own
   candidates from a different source, and extracted search terms are
   not routed back to the gatherer. This keeps the benchmark a clean
   apples-to-apples ranking comparison.
6. **Determinism.** Record in `config()` at fit time: the Ollama
   base-model tag digest + model artifact hash, temperature (`0.0` for
   benchmark), Ollama seed, full prompts used in fit and select, and
   fit timestamp. `selector_config_hash` in the feedback log derives
   from `config()`; reproducibility depends on this field being stable.

## Repo layout

```
radar-backend/
├── pyproject.toml
├── README.md                       (this file)
├── rag_lib/
│   ├── __init__.py                 (exports Paper, Topic, TopicNode, Profile)
│   ├── paper.py                    (Paper, Topic, TopicNode dataclasses)
│   ├── profile.py                  (Profile dataclass + from_csv / from_pdfs / aggregate_topic_filters)
│   ├── embedders.py                (placeholder_embed; 1B adds specter2_embed)
│   ├── openalex_client.py          (lookup real; search_works 1B stub)
│   ├── selector.py                 (Selector Protocol — frozen at 1A exit)
│   ├── gatherer.py                 (Gatherer Protocol — frozen at 1A exit)
│   ├── selectors/
│   │   ├── centroid.py             (1A stub; 1B replaces body)
│   │   └── llm_retrain.py          (1A typed stub; Track 2 replaces body)
│   └── gatherers/
│       └── openalex.py             (1A stub; 1B replaces body)
└── tests/
    ├── fixture_selector.py         (deterministic random selector reference)
    ├── fixture_gatherer.py         (deterministic gatherer reference)
    ├── fake_openalex_client.py     (no-network OpenAlex for build tests)
    ├── test_paper.py
    ├── test_profile.py             (serialization, seed_embeddings, aggregate_topic_filters)
    ├── test_profile_build.py       (Profile.from_csv end-to-end)
    ├── test_embedders.py
    ├── test_selector_protocol.py   (parametrized selector compliance)
    └── test_gatherer_protocol.py   (parametrized gatherer compliance)
```

Phase 1B will add `rag_lib/embed.py` (SPECTER2), `rag_lib/coherence.py`,
`rag_lib/vault.py` (PDF ingestion + Chroma), `rag_lib/radar.py`, and
`cli/ingest.py`, and will replace the bodies of `centroid.py`,
`gatherers/openalex.py`, `OpenAlexClient.search_works`, and
`profile_builder.build_from_pdfs`.

## Phase 1A exit criteria

- [x] Paper / Topic / TopicNode dataclasses with JSON round-trip.
- [x] Profile dataclass with JSON file I/O.
- [x] `build_from_csv` produces a Profile end-to-end from a manifest.
- [x] Selector protocol defined, documented, frozen.
- [x] Gatherer protocol defined, documented, frozen.
- [x] FixtureSelector passes the full selector compliance suite.
- [x] FixtureGatherer passes the full gatherer compliance suite.
- [x] CentroidSelector, LLMRetrainSelector, and OpenAlexGatherer stubs
      import, instantiate, and pass their non-invoking subsets.
- [x] README documents both integration contracts with a Track 2
      quick-start.

See `../PHASE1_TRACK1_PROGRESS.md` for live status.
