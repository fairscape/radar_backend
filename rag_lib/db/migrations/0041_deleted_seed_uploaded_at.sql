-- FUNCTIONAL_SPEC CR.4 (user 11-03, option A): a seed marked deleted shows
-- when that file was uploaded, beside when it was deleted, so a PDF deleted
-- and uploaded again reads as two files: the old one (uploaded, deleted)
-- and the new one (uploaded). Taken from paper_uploads when the delete
-- writes the mark; NULL for marks written before this.
ALTER TABLE deleted_seeds ADD COLUMN uploaded_at TEXT;
