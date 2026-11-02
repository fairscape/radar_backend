-- 0039_vault_kept.sql — papers that stay in a user's vault after the
-- interest, researcher or seed that brought them in is gone.
--
-- The vault is worked out from what the user has (uploads, researchers'
-- papers, seeds, saved papers). Deleting an interest or a researcher, or
-- removing a seed in Edit interest, took those papers out of it with them;
-- only the Vault's own delete may (FUNCTIONAL_SPEC 7, B2.2). Those three
-- record here, first, what they would take out.

CREATE TABLE vault_kept (
  user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  openalex_id  TEXT    NOT NULL REFERENCES papers(openalex_id) ON DELETE CASCADE,
  kept_at      TEXT    NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (user_id, openalex_id)
);

-- A seed whose PDF the user deleted from the Vault (F1). The seed row goes
-- with the paper's text and vectors; this keeps it on the interest's seed
-- list, last and marked deleted, until a scan has scored without it.
CREATE TABLE deleted_seeds (
  profile_id   INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  openalex_id  TEXT    NOT NULL,
  title        TEXT,
  deleted_at   TEXT    NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (profile_id, openalex_id)
);
