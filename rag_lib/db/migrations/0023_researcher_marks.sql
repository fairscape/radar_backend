-- A researcher can be marked as the user themself ('self') or as someone
-- they follow ('followed'); marked ones list first. ``position`` is the
-- order the user arranged the list in (lower first; NULL = unplaced, sorted
-- by name after the placed ones).
ALTER TABLE researchers ADD COLUMN mark TEXT CHECK (mark IN ('self', 'followed'));
ALTER TABLE researchers ADD COLUMN position INTEGER;
