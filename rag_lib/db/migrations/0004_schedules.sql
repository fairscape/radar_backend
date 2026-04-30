-- 0004_schedules.sql — per-profile gather schedules (Phase 7).
--
-- One row per profile, created lazily on first scheduler boot but
-- backfilled here so existing profiles are immediately scheduled at the
-- default daily-04:00-UTC cadence.
--
-- The runner emits BEGIN/COMMIT around this script, so do not include
-- transaction control here.

CREATE TABLE profile_schedules (
  profile_id INTEGER PRIMARY KEY REFERENCES profiles(id) ON DELETE CASCADE,
  cron       TEXT    NOT NULL DEFAULT '0 4 * * *',
  tz         TEXT    NOT NULL DEFAULT 'UTC',
  enabled    INTEGER NOT NULL DEFAULT 1,
  updated_at TEXT
);

INSERT INTO profile_schedules (profile_id)
SELECT id FROM profiles;
