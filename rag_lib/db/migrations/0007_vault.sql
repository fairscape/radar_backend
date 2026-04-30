-- 0007_vault.sql — Phase 6 columns for the user-facing PDF vault.
--
-- Adds:
--   papers.file_hash             sha256 of the uploaded bytes; nullable
--                                because papers.gathered-from-OpenAlex
--                                rows have no on-disk file behind them.
--   papers.uploaded_by_user_id   FK to users(id); set when a paper
--                                arrives via POST /api/vault/upload.
--                                NULL on gather-sourced rows.
--   papers.n_pages               for VaultDoc.pages (UI). Set at upload
--                                from pdfplumber's page count; NULL
--                                otherwise.
--   papers.authors_json          OpenAlex enrichment can carry author
--                                names; we serialize the list here so
--                                VaultDoc.authors has somewhere to
--                                source from. NULL when unknown.
--
-- Index supports the upload dedup check
-- (uploaded_by_user_id, file_hash) — return the existing VaultDoc when
-- a user re-uploads the same file rather than creating a duplicate row.
--
-- The migration runner emits BEGIN/COMMIT around this script, so do not
-- include transaction control here.

ALTER TABLE papers ADD COLUMN file_hash           TEXT;
ALTER TABLE papers ADD COLUMN uploaded_by_user_id INTEGER REFERENCES users(id);
ALTER TABLE papers ADD COLUMN n_pages             INTEGER;
ALTER TABLE papers ADD COLUMN authors_json        TEXT;

CREATE INDEX idx_papers_user_hash
  ON papers(uploaded_by_user_id, file_hash);
