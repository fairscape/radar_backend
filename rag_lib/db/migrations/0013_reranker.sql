-- 0013_reranker.sql — MedCPT cross-encoder reranker columns.
--
-- Per-candidate reranker breakdown (parallels 0003_scores).
-- Profile-level reranker config (parallels selector_config_json).

ALTER TABLE profile_candidates ADD COLUMN score_reranker_raw  REAL;
ALTER TABLE profile_candidates ADD COLUMN score_reranker_norm REAL;
ALTER TABLE profile_candidates ADD COLUMN score_blended       REAL;

ALTER TABLE profiles ADD COLUMN reranker_config_json TEXT;
