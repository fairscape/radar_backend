-- Researchers: a stored person (a Prosopia profile or an ORCID) and the
-- papers that came with them. An interest can be built from any subset
-- of those papers, as often as wanted, without re-importing.
--
-- Papers stay content-addressed in ``papers`` and embedded once in
-- ``paper_embeddings``; ``researcher_papers`` is membership plus what
-- the source said about each paper (its own id, the resolution rung,
-- the summary if one was published).

CREATE TABLE researchers (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  source         TEXT    NOT NULL,            -- 'prosopia' | 'orcid'
  key            TEXT    NOT NULL,            -- prosopia slug, or bare ORCID
  base_url       TEXT,                        -- prosopia instance the profile was read from
  orcid          TEXT,
  name           TEXT    NOT NULL,
  affiliation    TEXT,
  url            TEXT,
  expertise      TEXT,
  soul           TEXT,
  grants_json    TEXT,
  document_json  TEXT,                        -- the profile's own metadata, as read
  n_papers       INTEGER NOT NULL DEFAULT 0,
  imported_at    TEXT,                        -- last successful import
  last_run_id    INTEGER,                     -- the gather_runs row of the last import
  created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
  updated_at     TEXT,
  UNIQUE (user_id, source, key)
);
CREATE INDEX idx_researchers_user ON researchers(user_id);

CREATE TABLE researcher_papers (
  researcher_id  INTEGER NOT NULL REFERENCES researchers(id) ON DELETE CASCADE,
  openalex_id    TEXT    NOT NULL REFERENCES papers(openalex_id),
  paper_id       TEXT,                        -- the source's own id for the paper
  resolved_by    TEXT,                        -- work_id | doi | pmcid | title | none
  summary        TEXT,                        -- the source's summary, when it had one
  added_at       TEXT    NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (researcher_id, openalex_id)
);
CREATE INDEX idx_researcher_papers_paper ON researcher_papers(openalex_id);

-- An interest remembers which researcher it was built from.
ALTER TABLE profiles ADD COLUMN researcher_id INTEGER REFERENCES researchers(id) ON DELETE SET NULL;

-- gather_runs audits researcher imports too, which have no profile. The
-- column was NOT NULL, and SQLite cannot relax that in place, so the
-- table is rebuilt. Foreign keys are on for every connection and cannot
-- be switched off inside a transaction, so the child rows in
-- profile_candidates are handled by deferring the checks: the DROP
-- counts one violation per candidate row, and re-inserting the runs
-- under their original ids counts each one back down before COMMIT.
PRAGMA defer_foreign_keys = ON;

CREATE TABLE _gather_runs_backup AS SELECT * FROM gather_runs;
-- The AUTOINCREMENT counter goes with the table; keep it so a run id
-- that was used and deleted is never handed out again.
CREATE TABLE _gather_runs_seq AS SELECT seq FROM sqlite_sequence WHERE name = 'gather_runs';
DROP TABLE gather_runs;

CREATE TABLE gather_runs (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_id          INTEGER REFERENCES profiles(id) ON DELETE CASCADE,
  researcher_id       INTEGER REFERENCES researchers(id) ON DELETE CASCADE,
  started_at          TEXT NOT NULL DEFAULT (datetime('now')),
  finished_at         TEXT,
  since_date          TEXT,
  filter_string       TEXT,
  tier_used           TEXT,
  n_fetched           INTEGER,
  n_new               INTEGER,
  n_redup             INTEGER,
  api_calls           INTEGER,
  error               TEXT,
  user_id             INTEGER REFERENCES users(id),
  current_step        TEXT,
  n_processed         INTEGER,
  n_total             INTEGER,
  last_message        TEXT,
  progress_updated_at TEXT,
  result_json         TEXT
);

INSERT INTO gather_runs (
  id, profile_id, started_at, finished_at, since_date, filter_string,
  tier_used, n_fetched, n_new, n_redup, api_calls, error, user_id,
  current_step, n_processed, n_total, last_message, progress_updated_at,
  result_json
)
SELECT
  id, profile_id, started_at, finished_at, since_date, filter_string,
  tier_used, n_fetched, n_new, n_redup, api_calls, error, user_id,
  current_step, n_processed, n_total, last_message, progress_updated_at,
  result_json
FROM _gather_runs_backup;

DROP TABLE _gather_runs_backup;

DELETE FROM sqlite_sequence WHERE name = 'gather_runs';
INSERT INTO sqlite_sequence (name, seq)
SELECT 'gather_runs', MAX(
  (SELECT COALESCE(MAX(id), 0) FROM gather_runs),
  (SELECT COALESCE(MAX(seq), 0) FROM _gather_runs_seq)
);
DROP TABLE _gather_runs_seq;

CREATE INDEX idx_gather_runs_user       ON gather_runs(user_id);
CREATE INDEX idx_gather_runs_researcher ON gather_runs(researcher_id);
