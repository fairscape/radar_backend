-- 0030_draft_owned.sql — what a draft interest brought into the database.
--
-- A draft is a sandbox: an uploaded file, a researcher imported for it and
-- the papers that import stored are new only for the draft. Discarding it
-- (or "start over", or an import that fails) removes them again, unless
-- something else has come to use them meanwhile; saving it clears the
-- list, and they stay (rag_lib.db.repos.draft_owned).
--
--   kind 'paper'       a papers row: an upload's version, its public record
--                      or a paper an import stored
--   kind 'researcher'  a researchers row created by the wizard's import

CREATE TABLE draft_owned (
  profile_id  INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  kind        TEXT    NOT NULL CHECK (kind IN ('paper', 'researcher')),
  item        TEXT    NOT NULL,
  added_at    TEXT    NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (profile_id, kind, item)
);
CREATE INDEX idx_draft_owned_item ON draft_owned (kind, item);

-- Where each seed came from, shown on the interest (FUNCTIONAL_SPEC D.2-3):
--   'upload'      a PDF uploaded into the interest
--   'researcher'  a researcher's papers (an import, or "from a profile"):
--                 source_researcher_id names whose
--   'vault'       picked from the vault
--   'feed'        added from the feed
-- NULL for seeds added before this was recorded. The first source stays
-- when the same paper is added again another way.
ALTER TABLE profile_seeds ADD COLUMN source TEXT;
ALTER TABLE profile_seeds ADD COLUMN source_researcher_id INTEGER REFERENCES researchers(id) ON DELETE SET NULL;
