-- 0019_reranker_query_key.sql — which query set a stored reranker logit is relative to.
--
-- score_reranker_raw is a cross-encoder logit of (query, paper), aggregated
-- over the interest's queries (topic names, research topics or seeds). It
-- was never recomputed when those changed, so the pool normalised logits
-- from different query sets together, and papers that started passing the
-- threshold after a scan were never reranked at all. The key (a hash of
-- model, mode, aggregation and queries; MedCPTReranker.query_key) marks
-- stale rows for jobs.refresh_reranker. NULL = unknown (older rows).

ALTER TABLE profile_candidates ADD COLUMN reranker_query_key TEXT;
