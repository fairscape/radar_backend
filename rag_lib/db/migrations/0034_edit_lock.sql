-- 0034_edit_lock.sql — first come, first served on Edit interest.
--
-- An interest is edited in one tab at a time: a copy (``edit_of``) belongs
-- to the tab that opened it (``edit_tab``, an id the page keeps for the
-- life of its tab), which says it is still there every 30 s
-- (``edit_seen_at``, UTC). Another tab's Edit interest is refused while
-- the copy is heard from; one not heard from for 2 minutes (the tab
-- closed without leaving, the computer slept) is left behind and goes
-- (FUNCTIONAL_SPEC E.7 R5). NULL on copies from before: left behind.
ALTER TABLE profiles ADD COLUMN edit_tab TEXT;
ALTER TABLE profiles ADD COLUMN edit_seen_at TEXT;
