-- 0002_users.sql — multi-user extension.
--
-- Adds:
--   users                       table — one row per email, default user 1 = demo
--   profiles.user_id            FK to users(id), backfilled to 1
--   profiles.slug               URL key (kebab-case), backfilled from name
--   gather_runs.user_id         FK to users(id), backfilled to 1
--
-- SQLite ALTER TABLE supports inline REFERENCES on ADD COLUMN but does
-- not support inline UNIQUE; the slug uniqueness is enforced via a
-- separate unique index after backfill.

CREATE TABLE users (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  email      TEXT NOT NULL UNIQUE,
  mailto     TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO users (id, email, mailto)
VALUES (1, 'demo@example.com', 'demo@example.com');

ALTER TABLE profiles ADD COLUMN user_id INTEGER REFERENCES users(id);
UPDATE profiles SET user_id = 1 WHERE user_id IS NULL;

ALTER TABLE profiles ADD COLUMN slug TEXT;
-- Backfill: lowercase, replace spaces and underscores with hyphens.
-- Profiles created post-migration set their own slug at insert time
-- (see rag_lib.db.repos.profiles.slugify).
UPDATE profiles
SET slug = lower(replace(replace(name, ' ', '-'), '_', '-'))
WHERE slug IS NULL;

CREATE UNIQUE INDEX idx_profiles_slug ON profiles(slug);
CREATE INDEX        idx_profiles_user ON profiles(user_id);

ALTER TABLE gather_runs ADD COLUMN user_id INTEGER REFERENCES users(id);
UPDATE gather_runs SET user_id = 1 WHERE user_id IS NULL;
CREATE INDEX idx_gather_runs_user ON gather_runs(user_id);
