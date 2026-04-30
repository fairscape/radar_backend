-- 0006_chat.sql — Phase 9 chat history table.
--
-- One row per turn (user prompt or assistant response). The router
-- writes a 'user' row first, then an 'assistant' row after Ollama
-- returns; both rows share the same scope_json so a turn can be
-- reconstructed by pairing them on (user_id, ts).
--
--   role         'user' | 'assistant'
--   body         the prompt text or the assistant reply, stored verbatim
--   sources_json JSON-serialized list of {n, title, score} citations on
--                assistant rows; NULL on user rows
--   scope_json   JSON-serialized list of profile slugs the request was
--                scoped to; NULL for unscoped (search-all) chats
--
-- Index supports the history endpoint (most-recent-first per user).
-- The migration runner emits BEGIN/COMMIT around this script, so do not
-- include transaction control here.

CREATE TABLE chat_turns (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id      INTEGER NOT NULL REFERENCES users(id),
  role         TEXT NOT NULL,
  body         TEXT NOT NULL,
  sources_json TEXT,
  scope_json   TEXT,
  ts           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_chat_turns_user_ts ON chat_turns(user_id, ts DESC);
