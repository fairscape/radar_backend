-- 0020_paper_uploads.sql — who uploaded a paper, one row per user.
--
-- papers.uploaded_by_user_id held one uploader per paper, and an upload
-- overwrote it: two users uploading the same paper moved it out of the
-- first user's vault into the second's. Each upload is now its own row,
-- with that user's file. papers.uploaded_by_user_id is kept as the first
-- uploader and no longer read for ownership.
--
-- Researcher imports used to stamp uploaded_by_user_id too (no file_hash),
-- so a forgotten researcher's papers stayed in the vault as "uploads".
-- Those stamps are cleared; such papers stay in the vault only through a
-- seed, a saved card or a stored researcher, like any other import.

CREATE TABLE paper_uploads (
  user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  openalex_id  TEXT NOT NULL REFERENCES papers(openalex_id) ON DELETE CASCADE,
  file_hash    TEXT NOT NULL,
  local_path   TEXT,
  n_pages      INTEGER,
  uploaded_at  TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (user_id, openalex_id)
);
CREATE INDEX idx_paper_uploads_hash ON paper_uploads (user_id, file_hash);

INSERT OR IGNORE INTO paper_uploads (user_id, openalex_id, file_hash, local_path, n_pages, uploaded_at)
SELECT uploaded_by_user_id, openalex_id, file_hash, local_path, n_pages,
       COALESCE(first_seen_at, datetime('now'))
FROM papers
WHERE uploaded_by_user_id IS NOT NULL AND file_hash IS NOT NULL;

UPDATE papers SET uploaded_by_user_id = NULL
WHERE uploaded_by_user_id IS NOT NULL AND file_hash IS NULL;
