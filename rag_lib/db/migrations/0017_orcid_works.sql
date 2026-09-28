-- 0017_orcid_works.sql — the works an ORCID / profile import fetched for a
-- draft, so the user can choose the seeds instead of a fixed rule.
--
-- One row per kept OpenAlex work of the researcher. ``default_selected``
-- is what the old rule would have seeded (lead-author, non-dataset, one
-- copy per paper, newest RADAR_ORCID_MAX_SEEDS); ``selected`` is the
-- user's answer (NULL until they confirm). ``dup_of`` points at the copy
-- the rule keeps when a preprint / repository version was collapsed.
-- Rows go with the profile (CASCADE); the papers themselves stay.

CREATE TABLE IF NOT EXISTS orcid_works (
  profile_id       INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  openalex_id      TEXT    NOT NULL REFERENCES papers(openalex_id) ON DELETE CASCADE,
  position         TEXT,
  is_corresponding INTEGER NOT NULL DEFAULT 0,
  author_index     INTEGER,
  total_authors    INTEGER,
  work_type        TEXT,
  cited_by_count   INTEGER NOT NULL DEFAULT 0,
  claimed          INTEGER,
  seed_eligible    INTEGER NOT NULL DEFAULT 1,
  dup_of           TEXT,
  default_selected INTEGER NOT NULL DEFAULT 0,
  selected         INTEGER,
  PRIMARY KEY (profile_id, openalex_id)
);

CREATE INDEX IF NOT EXISTS idx_orcid_works_profile ON orcid_works(profile_id);
