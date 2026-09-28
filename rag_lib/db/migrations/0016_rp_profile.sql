-- 0016_rp_profile.sql — profiles seeded from a Researcher Profile document.
--
-- The "From Profile" wizard path accepts a databio ``profile.jsonld`` and
-- keeps what RADAR cannot derive from OpenAlex: the researcher's declared
-- expertise and not_interests (which shape the concept list), the
-- one-paragraph summary, affiliation, collaborators, level/provenance.
-- One JSON blob rather than a column per field: the spec is pre-1.0 and
-- these are display/signal data, never queried relationally.

ALTER TABLE profiles ADD COLUMN rp_meta_json TEXT;
