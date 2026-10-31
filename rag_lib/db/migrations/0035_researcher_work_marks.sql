-- What an import of a researcher did not take, by the id the source lists
-- the work under (the Prosopia paper id; the OpenAlex work id for ORCID).
-- FUNCTIONAL_SPEC O.2:
--   skipped    offered and left unticked; "check for new papers" lists it
--              apart, unticked (E)
--   duplicate  another record of a paper the researcher already has (the
--              same OpenAlex work); counted once, never offered as new (H)
CREATE TABLE researcher_work_marks (
  researcher_id INTEGER NOT NULL REFERENCES researchers(id) ON DELETE CASCADE,
  work_id       TEXT    NOT NULL,
  mark          TEXT    NOT NULL CHECK (mark IN ('skipped', 'duplicate')),
  marked_at     TEXT    NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (researcher_id, work_id)
);
