-- 0018_researchers.sql — a per-user library of imported Researcher Profiles.
--
-- Independent of topics: a row here is a profile.jsonld the user chose to
-- keep (verbatim in ``doc_json`` plus a parsed view in ``parsed_json``),
-- one per (user, rid); re-importing the same researcher overwrites. Nothing
-- in this table feeds gathering or scoring. ``source_kind`` / ``source_url``
-- are reserved for a later base-URL / Prosopia import.

CREATE TABLE IF NOT EXISTS researchers (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  rid           TEXT    NOT NULL,
  orcid         TEXT,
  name          TEXT    NOT NULL,
  affiliation   TEXT,
  field         TEXT,
  level         TEXT,
  provenance    TEXT,
  date_modified TEXT,
  source_kind   TEXT    NOT NULL DEFAULT 'paste',
  source_url    TEXT,
  doc_json      TEXT    NOT NULL,
  parsed_json   TEXT    NOT NULL,
  imported_at   TEXT    NOT NULL DEFAULT (datetime('now')),
  updated_at    TEXT,
  UNIQUE (user_id, rid)
);

CREATE INDEX IF NOT EXISTS idx_researchers_user ON researchers(user_id, name);
