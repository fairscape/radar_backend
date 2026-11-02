-- A saved paper stays in the Vault until it is deleted there
-- (FUNCTIONAL_SPEC 5.5): un-saving or dismissing it no longer takes it
-- out. ``vaulted_at``: when it was first saved; only the Vault's own
-- delete clears it. ``unsaved_at``: when it was un-saved, so the feed
-- puts it last; saving it again clears it.
ALTER TABLE profile_candidates ADD COLUMN vaulted_at TEXT;
ALTER TABLE profile_candidates ADD COLUMN unsaved_at TEXT;
UPDATE profile_candidates SET vaulted_at = saved_at WHERE saved_at IS NOT NULL;
