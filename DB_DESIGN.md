# radar-backend persistence layer

Today the system is JSON-on-disk: one Profile JSON per corpus, one
"candidates" JSON per gather run. That worked for a one-shot validation
but doesn't survive a real radar use case:

- **No dedup across gather runs.** A 365-day lookback rerun next week
  re-fetches and re-shows the same 69 neonatal papers.
- **No "shown / dismissed / saved" state.** Once a paper is in the JSON
  there's nowhere to record that the user already triaged it.
- **No way for a webapp** (the planned `radar/radar-website`) to query
  across profiles ("what's new for me today across all radars?").
- **No audit** of which gather runs happened, which tier was used, how
  many calls hit OpenAlex.

This document specifies the local SQLite layer that solves those problems.

## Why SQLite

- Zero-ops, single file (`radar/radar-backend/data/radar.db`).
- Server-side queryable from a Node/Python webapp without a separate
  daemon. The `radar-website` can `sqlite3.open` the same file the CLI
  writes to.
- Embeddings stored as BLOBs (numpy `float32` bytes) — fine for the
  scale we're at (12 seed papers + a few hundred candidates per profile).
- If vector search becomes hot, [`sqlite-vec`](https://github.com/asg017/sqlite-vec)
  is a drop-in extension; or migrate the `paper_embeddings` table to
  Chroma/LanceDB without touching the rest of the schema.
- Cross-platform, ships with Python's stdlib, easy to back up (just copy
  the file).

Alternatives considered:
- **Per-profile JSON** (today). Doesn't compose; rewriting the whole
  file on every gather is wasteful and racy.
- **DuckDB.** Better for analytics queries but the local-first concurrent
  read+write story is weaker for a small webapp. Overkill at this size.
- **Postgres.** Operational overhead not justified for personal use.

## Schema

All tables go in `radar.db`. SQL lives in
`radar/radar-backend/rag_lib/db/migrations/<NNNN>_<topic>.sql`. A bare
`_migrations_applied` table tracks what's been run.

```sql
CREATE TABLE _migrations_applied (
  version    TEXT PRIMARY KEY,
  applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per Profile. The seed papers and candidates live in
-- separate tables so a paper that appears in several profiles is
-- stored once.
CREATE TABLE profiles (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  name                TEXT NOT NULL UNIQUE,
  embedding_model     TEXT NOT NULL,
  centroid            BLOB,                  -- numpy float32 bytes
  threshold           REAL,
  topic_filters_json  TEXT NOT NULL,         -- aggregated, top-K per level
  selector_config_json TEXT,
  gatherer_config_json TEXT,
  coherence_median    REAL,
  coherence_iqr       REAL,
  coherence_bimodal   INTEGER,               -- 0/1
  n_seed              INTEGER NOT NULL,
  created_at          TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at          TEXT
);

-- Content-addressed by openalex_id. A paper exists once even if it's
-- a seed in profile A and a candidate in profile B.
CREATE TABLE papers (
  openalex_id   TEXT PRIMARY KEY,
  doi           TEXT,
  title         TEXT NOT NULL,
  abstract      TEXT,
  year          INTEGER,
  venue         TEXT,
  publication_date TEXT,
  topics_json   TEXT,                        -- primary_topic + topics[] full hierarchy
  source        TEXT NOT NULL,               -- 'openalex_gatherer', 'user_pdf', 'user_csv'
  local_path    TEXT,                        -- if ingested from a PDF
  body_text     TEXT,                        -- if extracted; can be large
  first_seen_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_papers_doi  ON papers(doi);
CREATE INDEX idx_papers_year ON papers(year);

-- Many embeddings per paper (one per model the system has run).
CREATE TABLE paper_embeddings (
  openalex_id      TEXT NOT NULL REFERENCES papers(openalex_id) ON DELETE CASCADE,
  embedding_model  TEXT NOT NULL,
  vector           BLOB NOT NULL,            -- numpy float32 bytes
  computed_at      TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (openalex_id, embedding_model)
);

-- Seed corpus membership.
CREATE TABLE profile_seeds (
  profile_id   INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  openalex_id  TEXT NOT NULL REFERENCES papers(openalex_id),
  added_at     TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (profile_id, openalex_id)
);

-- The radar log: every (profile, candidate) pair that gather has ever
-- surfaced, with the user's triage state. PRIMARY KEY enforces dedup.
CREATE TABLE profile_candidates (
  profile_id      INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  openalex_id     TEXT NOT NULL REFERENCES papers(openalex_id),
  score           REAL NOT NULL,             -- cosine to centroid at fetch time
  tier_used       TEXT,                      -- which gather tier surfaced it
  gather_run_id   INTEGER REFERENCES gather_runs(id),
  fetched_at      TEXT NOT NULL DEFAULT (datetime('now')),
  shown_at        TEXT,                      -- NULL = never surfaced to user
  dismissed_at    TEXT,                      -- user said "not interested"
  saved_at        TEXT,                      -- user starred it
  snoozed_until   TEXT,                      -- resurface after this date
  notes           TEXT,
  PRIMARY KEY (profile_id, openalex_id)
);
CREATE INDEX idx_pc_unshown   ON profile_candidates(profile_id, shown_at)
  WHERE shown_at IS NULL;
CREATE INDEX idx_pc_score     ON profile_candidates(profile_id, score DESC);
CREATE INDEX idx_pc_dismissed ON profile_candidates(profile_id, dismissed_at);

-- One row per gather invocation; the candidate rows reference it.
CREATE TABLE gather_runs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_id    INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  started_at    TEXT NOT NULL DEFAULT (datetime('now')),
  finished_at   TEXT,
  since_date    TEXT,
  filter_string TEXT,
  tier_used     TEXT,
  n_fetched     INTEGER,
  n_new         INTEGER,                    -- not previously in profile_candidates
  n_redup       INTEGER,                    -- already-known papers we saw again
  api_calls     INTEGER,
  error         TEXT
);
```

## Dedup mechanics

The whole point of the new layer. When a gather run fetches K papers:

```sql
-- Insert each candidate; ON CONFLICT DO NOTHING preserves prior triage state.
INSERT INTO profile_candidates
  (profile_id, openalex_id, score, tier_used, gather_run_id)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (profile_id, openalex_id) DO NOTHING;
```

Then `n_new = changes()`, `n_redup = K - n_new`, and the webapp's
"what's new on my radar today" query is:

```sql
SELECT p.title, pc.score, pc.fetched_at
FROM profile_candidates pc
JOIN papers p USING (openalex_id)
WHERE pc.profile_id = ?
  AND pc.shown_at IS NULL
  AND pc.dismissed_at IS NULL
  AND (pc.snoozed_until IS NULL OR pc.snoozed_until < datetime('now'))
ORDER BY pc.score DESC
LIMIT 50;
```

Marking shown is `UPDATE profile_candidates SET shown_at = datetime('now')
WHERE ...`. Dismiss / save are the same shape with their respective
columns.

## Migration approach

Hand-rolled, no Alembic. The library is small enough that a custom
runner is less code than configuring a framework.

```
rag_lib/db/
  __init__.py
  schema.py          # connect(), apply_migrations(), helpers
  repo_profiles.py   # profile CRUD
  repo_papers.py     # paper + embedding upsert
  repo_candidates.py # candidate insert with dedup, triage updates
  migrations/
    0001_initial.sql
    0002_*.sql       # future
```

`apply_migrations(conn)` enumerates `migrations/*.sql` in lexicographic
order, looks each up in `_migrations_applied`, runs the missing ones in
a transaction, and records each one. Append-only — never edit a shipped
migration.

## CLI surface

New: `cli/db.py`
- `python -m cli.db init` — create the database file and apply all migrations
- `python -m cli.db status` — list applied migrations
- `python -m cli.db upgrade` — apply any pending migrations
- `python -m cli.db profiles` — list profiles in the DB
- `python -m cli.db candidates --profile <name> --unshown` — quick triage view

Modified:
- `cli/ingest.py` — keep `--out path.json` for compatibility, add `--db
  data/radar.db` to also (or instead) persist the Profile + seeds +
  embeddings to SQLite.
- `cli/gather.py` — add `--db data/radar.db`. When set, dedup against
  `profile_candidates` so already-known papers do not re-embed and re-rank.
  Write the `gather_runs` row before/after the call.

## Webapp consumption (radar-website)

The webapp is read-mostly. It opens the same SQLite file with
`better-sqlite3` (Node) or `sqlite3` (Python) and runs:

- list profiles → `SELECT id, name, n_seed, coherence_median FROM profiles`
- profile dashboard → joins `profile_candidates` × `papers` filtered by
  triage state
- triage actions → `UPDATE profile_candidates SET shown_at=... / dismissed_at=... / saved_at=...`
- recent activity → `SELECT * FROM gather_runs ORDER BY started_at DESC LIMIT 20`

No server-side migrations from the webapp; the CLI is the only writer
of schema. App-level writes (triage state) are restricted to columns
the schema explicitly carves out for that purpose.

## Open questions

- Do we want `body_text` in SQLite or in a sidecar (it can be 100s of KB
  per paper, blows up the DB size for large corpora)? Default plan:
  store it inline; revisit if `radar.db` grows past ~500 MB.
- The webapp may want full-text search over titles/abstracts; SQLite has
  FTS5 built in. Add `papers_fts` virtual table in a 0002 migration when
  the webapp needs it.
- Multi-user is out of scope; the schema assumes one human reading their
  own radar.
