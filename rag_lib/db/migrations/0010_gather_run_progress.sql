-- 0010_gather_run_progress.sql — incremental progress reporting on gather runs.
--
-- A gather run can take several minutes (SPECTER2 embedding of new
-- candidates dominates wall time). The frontend already polls
-- gather_runs to learn when finished_at is set; these columns let it
-- also render a live "embedding 47 / 230" status so the user knows the
-- run is still moving and doesn't reflexively refresh.
--
-- Columns:
--   current_step         coarse phase: 'loading_profile' | 'fetching' |
--                        'embedding' | 'persisting' | 'done'.
--                        NULL on rows from before this migration; the
--                        UI treats NULL as "no progress info, fall back
--                        to a plain spinner."
--   n_processed          counter within current_step (e.g., embeddings
--                        completed). Reset to 0 on each step transition.
--   n_total              denominator for current_step when known
--                        upfront (e.g., total candidates to embed).
--   last_message         short human-readable status string for the UI.
--   progress_updated_at  wall-clock of the most recent update — kept
--                        for future stuck-job detection (not surfaced
--                        in the UI yet).
--
-- The migration runner wraps this script in BEGIN/COMMIT, so do not
-- include transaction control here.

ALTER TABLE gather_runs ADD COLUMN current_step        TEXT;
ALTER TABLE gather_runs ADD COLUMN n_processed         INTEGER;
ALTER TABLE gather_runs ADD COLUMN n_total             INTEGER;
ALTER TABLE gather_runs ADD COLUMN last_message        TEXT;
ALTER TABLE gather_runs ADD COLUMN progress_updated_at TEXT;
