-- Candidates written before 0003 have no score_raw; "below the bar" (NOT
-- PASSES) could never show them, while "all" could. Their score was the
-- cosine mapped by (c+1)/2 (see 0003), so the raw cosine is 2*score - 1.
UPDATE profile_candidates SET score_raw = 2 * score - 1 WHERE score_raw IS NULL AND score IS NOT NULL;
