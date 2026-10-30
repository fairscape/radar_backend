-- 0032_scan_topic_ids.sql — the topics and seeds a scan searched with.
--
-- JSON lists of the OpenAlex topic ids switched on
-- (openalex_tiers.enabled_topic_ids) and of the seed ids the interest had
-- when a scan that succeeded ran. The edit page's threshold step compares
-- the interest's topics and seeds now with the last scan's, to say its
-- recommendations come from other ones (FUNCTIONAL_SPEC T.2-4, E.2-5).
-- NULL for trial scans, imports, failed scans and every run before these
-- columns.
ALTER TABLE gather_runs ADD COLUMN topic_ids_json TEXT;
ALTER TABLE gather_runs ADD COLUMN seed_ids_json TEXT;
