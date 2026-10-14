-- 0017_interest_terms.sql — research topics for an interest (demo).
--
-- Topics generated the way Prosopia generates them (rag_lib/interest_terms.py):
-- the researcher's own Prosopia fields, or the researcher_profiles SDK's
-- topic extraction over the interest's seeds. JSON:
--   {"source": "prosopia"|"rp-sdk-llm"|"rp-sdk-cluster", "model", "generated_at",
--    "seed_hash", "terms": [{"label","weight","paper_ids"}], "not_interests",
--    "edited"}
-- Used as reranker queries in query_mode="terms". NULL until generated.
-- Never holds the API key used to generate it.

ALTER TABLE profiles ADD COLUMN interest_terms_json TEXT;
