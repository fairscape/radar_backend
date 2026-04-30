-- 0009_oa.sql — capture OpenAlex open-access fields on papers.
--
-- Adds:
--   papers.pdf_url    direct PDF link from best_oa_location.pdf_url
--                     (falls back to primary_location.pdf_url at extract time).
--                     NULL when OpenAlex has no direct link, even if the
--                     paper is otherwise OA — those are candidates for a
--                     Unpaywall / preprint-server fallback later.
--   papers.oa_status  open_access.oa_status: gold / hybrid / bronze /
--                     green / closed / diamond. NULL when OpenAlex
--                     omits it.
--
-- Backfilling is intentionally skipped — existing rows stay NULL and
-- get populated on the next gather/ingest pass for that paper.
--
-- The migration runner emits BEGIN/COMMIT around this script, so do
-- not include transaction control here.

ALTER TABLE papers ADD COLUMN pdf_url   TEXT;
ALTER TABLE papers ADD COLUMN oa_status TEXT;
