-- 0003_scores.sql — break the SPECTER2 score-compression issue
-- (todos/phase-03-score-tuning.md).
--
-- Adds three diagnostic columns to profile_candidates:
--   score_raw       the selector's primary unmapped signal (raw cosine
--                   for centroid; max-cosine for max_seed). Range
--                   [-1, 1] after dropping the legacy (c+1)/2 mapping.
--   score_max_seed  max cosine to any single seed paper. CentroidSelector
--                   computes this in addition to its primary; for
--                   MaxSeedSelector it equals score_raw.
--   score_pct       per-batch rank percentile in [0, 1]. The primary
--                   numeric the API + UI use for bucketing.
--
-- The legacy ``score`` column stays NOT NULL and is kept populated by
-- Phase 2's persistence helper (set to ``score_pct`` going forward).
-- Backfill of the new columns for rows inserted before this migration
-- is handled out-of-band by ``cli/db.py rescore`` (opt-in).
--
-- The runner emits BEGIN/COMMIT around this script, so do not include
-- transaction control here.

ALTER TABLE profile_candidates ADD COLUMN score_raw      REAL;
ALTER TABLE profile_candidates ADD COLUMN score_max_seed REAL;
ALTER TABLE profile_candidates ADD COLUMN score_pct      REAL;

-- Pre-existing rows: copy the legacy ``score`` into ``score_pct`` so the
-- API's percentile-keyed reads return *something* sensible until a real
-- backfill runs. The legacy values were already mapped into [0, 1] by
-- the (c+1)/2 step, so they are at least order-preserving.
UPDATE profile_candidates SET score_pct = score WHERE score_pct IS NULL;
