-- 0015_orcid.sql — profiles created from a researcher's ORCID.
--
-- A profile built by the "From ORCID" wizard path is seeded from the
-- researcher's own OpenAlex works instead of uploaded PDFs. The ORCID is
-- the join key back to that person (and to any RP-format profile written
-- alongside), and ``orcid IS NOT NULL`` is what the topic aggregation
-- reads to keep the import-time weighted topic list instead of
-- recomputing from seeds. ``researcher_name`` is OpenAlex's display name,
-- kept separately from ``name`` because the user may rename the radar.

ALTER TABLE profiles ADD COLUMN orcid TEXT;
ALTER TABLE profiles ADD COLUMN researcher_name TEXT;

CREATE INDEX IF NOT EXISTS idx_profiles_orcid ON profiles(orcid);
