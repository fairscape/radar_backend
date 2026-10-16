-- Candidates written before 0003 have no score_raw; "below the bar" (NOT
-- PASSES) could never show them, while "all" could. score was the raw
-- cosine then, so it is the raw score.
UPDATE profile_candidates SET score_raw = score WHERE score_raw IS NULL AND score IS NOT NULL;
