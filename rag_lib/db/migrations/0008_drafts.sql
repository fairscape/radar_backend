-- 0008_drafts.sql — Phase 11 wizard scaffolding.
--
-- Adds:
--   profiles.is_draft INTEGER NOT NULL DEFAULT 0
--
-- A draft is a profile row created by ``POST /api/profiles/draft`` while
-- the user is still walking through the wizard (uploading seeds,
-- inspecting coherence, picking topics, calibrating threshold).
-- ``profiles_repo.list_for_user`` filters drafts out by default so they
-- never appear in the sidebar / Radar / detail surfaces; the wizard's
-- own endpoints opt in via ``include_drafts=True``.
--
-- ``commit_draft`` flips the row to ``is_draft=0`` once the user hits
-- Save, at which point the existing read paths pick it up. There is no
-- separate "drafts" table — same row, just a status flag — so seeds
-- attached during the wizard survive the flip without copy.
--
-- The migration runner emits BEGIN/COMMIT around this script, so do
-- not include transaction control here.

ALTER TABLE profiles ADD COLUMN is_draft INTEGER NOT NULL DEFAULT 0;

UPDATE profiles SET is_draft = 0 WHERE is_draft IS NULL;
