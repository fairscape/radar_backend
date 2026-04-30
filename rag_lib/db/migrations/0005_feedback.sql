-- 0005_feedback.sql — Phase 8 feedback log DB mirror.
--
-- Append-only audit of triage events (save / dismiss / snooze). The
-- canonical record is per-user JSONL at
-- ``{RADAR_VAULT_DIR}/{user_id}/state/feedback.jsonl``; this table
-- mirrors it for SQL queries (profile detail "recent feedback" panel,
-- the autotune sweep, the ``GET /api/profiles/{key}/feedback`` route).
--
-- ``shown`` is intentionally not logged here — that signal already
-- lives on ``profile_candidates.shown_at`` and would dominate volume.
--
-- ``score`` is ``profile_candidates.score_pct`` (Phase 3) at the time
-- of the event; selector_config_hash is sha256(json(selector.config()))
-- truncated to 16 chars so a future autotune can group events by the
-- exact selector configuration that produced them.
--
-- The migration runner emits BEGIN/COMMIT around this script, so do
-- not include transaction control here.

CREATE TABLE feedback_events (
  id                   INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id              INTEGER NOT NULL REFERENCES users(id),
  profile_id           INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  openalex_id          TEXT    NOT NULL REFERENCES papers(openalex_id),
  action               TEXT    NOT NULL,
  score                REAL,
  selector             TEXT,
  selector_config_hash TEXT,
  benchmark_run_id     INTEGER,
  ts                   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_feedback_profile_ts     ON feedback_events(profile_id, ts DESC);
CREATE INDEX idx_feedback_profile_action ON feedback_events(profile_id, action);
