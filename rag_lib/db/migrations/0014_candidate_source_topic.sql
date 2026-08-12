-- 0014_candidate_source_topic.sql — per-topic gather attribution.
--
-- Under the per-topic quota gatherer every candidate enters the pool
-- through exactly one topic's quota, even though the paper itself
-- usually carries several OpenAlex topics. Recording which one claimed
-- it is what makes the Step 3 toggles legible: the UI can show "this
-- topic brought in 47 papers over the last 30 days, you saved 3", so
-- the user prunes on evidence instead of guessing from the name.
--
-- NULL for rows written before this migration, and for candidates that
-- arrived through a fallback tier rather than a topic quota.

ALTER TABLE profile_candidates ADD COLUMN sourced_by_topic_id TEXT;

CREATE INDEX IF NOT EXISTS idx_pc_source_topic
  ON profile_candidates(profile_id, sourced_by_topic_id);
