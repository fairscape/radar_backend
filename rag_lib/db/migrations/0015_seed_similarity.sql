-- 0015_seed_similarity.sql — leave-one-out seed similarity band.
--
-- The band is the range of cosines the profile's own seeds reach
-- against the centroid of the other seeds. It answers "how close does
-- a candidate have to be to count as being like my seeds?", which is
-- what the threshold suggestion, the card colouring and the
-- calibration slider are anchored on. Computed at coherence time and
-- at commit; NULL for profiles that predate this migration until they
-- are recomputed.

ALTER TABLE profiles ADD COLUMN seed_sim_min    REAL;
ALTER TABLE profiles ADD COLUMN seed_sim_median REAL;
ALTER TABLE profiles ADD COLUMN seed_sim_max    REAL;
