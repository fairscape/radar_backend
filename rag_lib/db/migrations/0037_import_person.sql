-- Who a wizard import is importing, on the run itself: since a wizard
-- import saves no researcher profile unless asked (FUNCTIONAL_SPEC M.2),
-- the researcher row can no longer say it, and a second request for the
-- same person (a double click) must still join the running import rather
-- than open a second draft and job. ``opened_draft``: the run's draft was
-- opened by this import (not one it added papers to).
ALTER TABLE gather_runs ADD COLUMN import_orcid TEXT;
ALTER TABLE gather_runs ADD COLUMN import_key TEXT;
ALTER TABLE gather_runs ADD COLUMN opened_draft INTEGER NOT NULL DEFAULT 0;
