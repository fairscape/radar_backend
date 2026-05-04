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
```

## Tests

```sh
pytest -v
```

Compliance suites for the `Selector` and `Gatherer` protocols live in
`tests/test_selector_protocol.py` and `tests/test_gatherer_protocol.py`.
A new selector is added by appending its class to `FULL_SELECTORS` in
that file; passing `TestFull` is the gate to enroll in the benchmark.
