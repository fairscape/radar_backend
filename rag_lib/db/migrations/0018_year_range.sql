-- 0018_year_range.sql — the publication years an interest recommends from.
--
-- RADAR only ever gathered papers published since its last scan, so an
-- interest's pool was all recent. year_from / year_to bound the feed (and
-- drive a one-off backfill of those years, gatherers/openalex.fetch_years).
-- NULL = no bound; year_to NULL means "to now", so the daily scans' new
-- papers are never filtered out.

ALTER TABLE profiles ADD COLUMN year_from INTEGER;
ALTER TABLE profiles ADD COLUMN year_to   INTEGER;
