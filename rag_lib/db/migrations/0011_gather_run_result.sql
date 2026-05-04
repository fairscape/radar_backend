-- 0011_gather_run_result.sql — carry async dry-run results on the audit row.
--
-- The wizard's "calibrate threshold" step (Phase 11) used to be a
-- synchronous endpoint that returned a sweep + preview + raw scores in
-- the response body. With per-step progress reporting (0010) the dry-
-- run now runs as a background job: the API kicks it off and the
-- frontend polls gather_runs for stage updates until finished_at is
-- set. This column is where the job stashes its serialized
-- DraftDryRun so the status-poll endpoint can hand it back without
-- re-running the work.
--
-- Regular gathers leave this NULL; only tier_used='dry_run' rows
-- populate it.
--
-- The migration runner wraps this script in BEGIN/COMMIT, so do not
-- include transaction control here.

ALTER TABLE gather_runs ADD COLUMN result_json TEXT;
