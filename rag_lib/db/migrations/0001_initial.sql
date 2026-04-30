-- 0001_initial.sql — base schema from radar-backend/DB_DESIGN.md.
--
-- Tables:
--   profiles            one row per Profile
--   papers              content-addressed by openalex_id
--   paper_embeddings    many per paper, one per embedding model
--   profile_seeds       seed-corpus membership
--   gather_runs         audit trail of gather invocations
--   profile_candidates  the radar log; PK enforces dedup
--
-- The runner emits BEGIN/COMMIT around this script, so do not include
-- transaction control here.

CREATE TABLE profiles (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  name                  TEXT NOT NULL UNIQUE,
  embedding_model       TEXT NOT NULL,
  centroid              BLOB,
  threshold             REAL,
  topic_filters_json    TEXT NOT NULL,
  selector_config_json  TEXT,
  gatherer_config_json  TEXT,
  coherence_median      REAL,
  coherence_iqr         REAL,
  coherence_bimodal     INTEGER,
  n_seed                INTEGER NOT NULL,
  created_at            TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at            TEXT
);

CREATE TABLE papers (
  openalex_id      TEXT PRIMARY KEY,
  doi              TEXT,
  title            TEXT NOT NULL,
  abstract         TEXT,
  year             INTEGER,
  venue            TEXT,
  publication_date TEXT,
  topics_json      TEXT,
  source           TEXT NOT NULL,
  local_path       TEXT,
  body_text        TEXT,
  first_seen_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_papers_doi  ON papers(doi);
CREATE INDEX idx_papers_year ON papers(year);

CREATE TABLE paper_embeddings (
  openalex_id     TEXT NOT NULL REFERENCES papers(openalex_id) ON DELETE CASCADE,
  embedding_model TEXT NOT NULL,
  vector          BLOB NOT NULL,
  computed_at     TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (openalex_id, embedding_model)
);

CREATE TABLE profile_seeds (
  profile_id  INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  openalex_id TEXT NOT NULL REFERENCES papers(openalex_id),
  added_at    TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (profile_id, openalex_id)
);

-- gather_runs is created before profile_candidates so the FK target exists
-- when SQLite parses the referencing column.
CREATE TABLE gather_runs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_id    INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  started_at    TEXT NOT NULL DEFAULT (datetime('now')),
  finished_at   TEXT,
  since_date    TEXT,
  filter_string TEXT,
  tier_used     TEXT,
  n_fetched     INTEGER,
  n_new         INTEGER,
  n_redup       INTEGER,
  api_calls     INTEGER,
  error         TEXT
);

CREATE TABLE profile_candidates (
  profile_id    INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  openalex_id   TEXT NOT NULL REFERENCES papers(openalex_id),
  score         REAL NOT NULL,
  tier_used     TEXT,
  gather_run_id INTEGER REFERENCES gather_runs(id),
  fetched_at    TEXT NOT NULL DEFAULT (datetime('now')),
  shown_at      TEXT,
  dismissed_at  TEXT,
  saved_at      TEXT,
  snoozed_until TEXT,
  notes         TEXT,
  PRIMARY KEY (profile_id, openalex_id)
);
CREATE INDEX idx_pc_unshown   ON profile_candidates(profile_id, shown_at)
  WHERE shown_at IS NULL;
CREATE INDEX idx_pc_score     ON profile_candidates(profile_id, score DESC);
CREATE INDEX idx_pc_dismissed ON profile_candidates(profile_id, dismissed_at);
