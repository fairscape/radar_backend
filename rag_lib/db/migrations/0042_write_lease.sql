-- FUNCTIONAL_SPEC W.A (user 2026-11-04): an account is changed from one tab
-- at a time. The tab that holds that right (``X-Radar-Tab``) and when it last
-- said it is still there; a row not heard from for 2 minutes holds nothing
-- (rag_lib/api/write_lease.py). One row per user at most.
CREATE TABLE IF NOT EXISTS write_leases (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    tab     TEXT NOT NULL,
    seen_at TEXT NOT NULL
);
