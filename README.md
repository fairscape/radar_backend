# radar-backend

FastAPI service + Python library for the Personal Research Radar.
Tracks a researcher's interests from a seed corpus, fetches new
candidate papers from OpenAlex on a daily schedule, scores them against
the seed, and serves a per-user vault with optional RAG chat.

The deployed system runs via Docker Compose — see
[radar_deployment](https://github.com/fairscape/radar_deployment).
This README covers the standalone backend.

## What it does

- **Profiles.** A user uploads ~10 seed PDFs; the backend pulls
  metadata via OpenAlex, embeds each paper with SPECTER2, and
  aggregates topic filters.
- **Daily radar.** APScheduler runs a gatherer (OpenAlex) once a day,
  scores new candidates against the profile centroid, and stores the
  top-ranked cards.
- **Vault + chat.** PDFs are chunked into a per-user Chroma collection;
  `/api/chat` uses Ollama for RAG over the vault.
- **Feedback loop.** Save / dismiss actions append to
  `vault/<user>/state/feedback.jsonl` and feed selector retraining.

API surface lives at `rag_lib.api.app:app`; routers in
`rag_lib/api/routers/` (`health`, `users`, `profiles`, `radar`,
`vault`, `chat`). State persists in SQLite (`./data/radar.db`),
per-user PDFs/feedback under `./vault/`, vector indexes under `./chroma/`.

## Install

Requires Python 3.11+.

```sh
git clone https://github.com/fairscape/radar_backend.git
cd radar_backend
python -m venv .venv && source .venv/bin/activate
pip install -e '.[phase1b,dev]'   # phase1b = SPECTER2 + chroma + ollama + pdfplumber
cp .env.example .env
python -m cli.db init             # create SQLite + apply migrations
python -m cli.serve               # 127.0.0.1:8000
```

Verify: `curl http://localhost:8000/api/health`.

Settings load from env / `.env`; see `.env.example` for every key. The
ones most likely to need tuning: `RADAR_DEFAULT_MAILTO` (OpenAlex
polite-pool), `RADAR_OLLAMA_URL` (defaults to `http://ollama:11434` for
compose; set to `http://localhost:11434` for local dev), and
`RADAR_CORS_ORIGINS`.

## CLI

```sh
python -m cli.serve                          # run the API
python -m cli.db {init,status,upgrade}       # SQLite migrations
python -m cli.ingest --pdf-dir ./papers      # build a profile from PDFs
python -m cli.gather --profile <slug>        # run a gather pass now
python -m cli.reindex_chat --user <email>    # rebuild vault chat index
python -m cli.import_orcid <orcid> --user <email> [--commit]   # radar from a researcher's ORCID
```

## Profiles from an ORCID

Besides uploading seed PDFs, the wizard can seed a profile from a
researcher's own publications:

1. `POST /api/profiles/draft/from-orcid` with `{"orcid": "0000-0001-5643-4068",
   "name": optional, "mailto": optional}` resolves the author on OpenAlex
   (404 if unknown, 409 if an import of that ORCID is already running),
   creates the draft, and dispatches a scheduler job.
2. The job fetches every work carrying the ORCID, stores them as `papers`
   (`source="orcid"`), embeds the researcher's first/last/corresponding-
   author works (newest `RADAR_ORCID_MAX_SEEDS`, default 40; datasets and
   duplicate preprint/repository copies excluded) and attaches them as
   seeds, writes a topic list weighted over the whole corpus (lead-author
   works x2, top `RADAR_ORCID_TOPIC_TOP_K`), and drops an RP-format
   `profile.jsonld` + `sources/papers.jsonld` under
   `RADAR_RP_PROFILES_DIR/<user_id>/<slug>/`.
3. Poll `GET /api/profiles/draft/{slug}/import/{run_id}` (same shape as
   the dry-run poll); `GET /api/profiles/draft/{slug}/seeds` lists the
   seeds. Steps 2-4 of the wizard are unchanged. For ORCID-seeded drafts
   the topic aggregation keeps the import-time list and UMLS-mapped
   topics start switched off.

Since 2026-09-22 the import runs in two phases. Phase A (`tier_used =
orcid_import`) fetches and stores the works and writes the RP files but
seeds nothing; `GET /api/profiles/draft/{slug}/works` then lists every
kept work with its author position, ORCID-record flag, duplicate pointer
and the rule's pre-check (`default_selected`). The user confirms with
`POST /api/profiles/draft/{slug}/seeds/select {"openalex_ids": [...]}`
(phase B, `tier_used = orcid_seed`, same status endpoint): the chosen
works are embedded and attached, replacing any earlier choice, and the
concept list is aggregated **from the selected seeds only**. There is no
hard cap; the UI warns above `RADAR_ORCID_MAX_SEEDS`. Datasets cannot be
seeds. `python -m cli.import_orcid … --seeds default|lead|all|ids:<file>`
does both phases in one go.

Committing a draft (any kind) now also registers its cron gather on the
running scheduler; before 2026-09-10 a profile committed after boot was not
gathered until the next backend restart.

Service code: `rag_lib/api/services/orcid_import.py`; job:
`rag_lib.scheduler.jobs.import_orcid_for_draft`; tests:
`tests/test_orcid_import.py`, `tests/test_api_orcid.py`.

## Profiles from a Researcher Profile document

`POST /api/profiles/draft/from-profile` with `{"profile_json": "<text of a
databio profile.jsonld>", "name": optional, "mailto": optional}` seeds a
draft from a Researcher Profile (https://village.databio.org/researcher-profiles/rp-spec/).
Only the document is read — the papers it lists live in a separate file the
document merely names — so:

- With an ORCID (`rid` or `@id`), the ORCID import above runs unchanged
  (same job, status and seeds endpoints; the profile's OpenAlex author id
  is the fallback when OpenAlex does not know the ORCID). When it finishes,
  the profile's `expertise` and `not_interests` phrases are embedded with
  the UMLS mapper's model (`RADAR_UMLS_EMBEDDING_MODEL`, mxbai via ollama)
  and matched against the OpenAlex topic index: a `not_interests` phrase
  switches OFF its best-matching concept (cosine >= `RADAR_RP_NOT_INTEREST_MIN_SIM`,
  0.70), an `expertise` phrase switches ON up to `RADAR_RP_EXPERTISE_TOP_K`
  concepts already in the list (>= `RADAR_RP_EXPERTISE_MIN_SIM`, 0.65) and
  may ADD one concept the corpus never surfaced (best hit >= `RADAR_RP_EXPERTISE_ADD_MIN_SIM`,
  0.78; `source="rp_expertise"`). The higher cosine wins a conflict. The
  floors were set on Nathan Sheffield's profile: mxbai cosines are
  compressed (0.65-0.86 across the board) and the 2nd/3rd neighbours of a
  not-interest like "phylogenomics" are the researcher's core topics.
- Without an ORCID, the draft is created with the metadata attached (`mode:
  "pdf"`, no run); the user uploads PDFs and the signal is applied when Step 3
  aggregates the concepts.

Summary, affiliation, expertise, not_interests, collaborators, level and
provenance are stored on `profiles.rp_meta_json` (migration 0016) and shown
on the topic detail page. `python -m cli.import_profile <profile.jsonld>
--user <email> --dry-run-signals` prints the phrase → concept mapping without
writing anything; drop `--dry-run-signals` to run the import.

Service code: `rag_lib/api/services/rp_profile_import.py`; job:
`rag_lib.scheduler.jobs.import_profile_for_draft`; tests:
`tests/test_rp_profile_import.py`, `tests/test_api_rp_profile.py`.

## Researchers library

`/api/researchers` keeps imported Researcher Profile documents whole, one
row per `(user, rid)` (`GET` list, `POST {"profile_json", "source_kind"}`
import / overwrite, `GET /{id}` detail with the parsed view and the raw
document, `DELETE /{id}`). The parsed view carries every section of the
document — identity, summary, expertise / not_interests / collaborators,
training, career, career_stage, paper_stats, capability flags and a
summary of the file manifest (counts by role and visibility; the files
themselves are never fetched). It is independent of topics: importing here
creates no topic and feeds no gathering or scoring; the wizard's FROM
PROFILE path does not write here. Migration `0018_researchers`; service
`rag_lib/api/services/researchers.py`; tests `tests/test_researchers.py`.

## Tests

```sh
pytest -v
```

Compliance suites for the `Selector` and `Gatherer` protocols live in
`tests/test_selector_protocol.py` and `tests/test_gatherer_protocol.py`.
A new selector is added by appending its class to `FULL_SELECTORS` in
that file; passing `TestFull` is the gate to enroll in the benchmark.
