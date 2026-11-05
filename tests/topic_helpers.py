"""A test draft whose seeds bring no topic (uploaded PDFs, bare ids) cannot
be saved since 2026-11-05 (review BT4, user: "没有主题时，不保存，要求用户
至少保留一个主题"). Tests that only need *a* saved interest give the draft
one topic, switched on, the way the wizard's step 3 would have it."""

from __future__ import annotations

import json

from rag_lib.api import settings as settings_module
from rag_lib.db import connect

TEST_TOPIC = "https://openalex.org/T10001"


def ensure_a_topic(slug: str, db_path=None) -> list[str]:
    """The ids switched on on draft ``slug``; when it has none, one test topic
    is added and switched on. Returns the ids to commit with."""
    db_path = db_path or settings_module.get_settings().RADAR_DB_PATH
    c = connect(db_path)
    try:
        row = c.execute("SELECT topic_filters_json FROM profiles WHERE slug = ? AND is_draft = 1",
                        (slug,)).fetchone()
        if row is None:
            return []
        tf = json.loads(row[0]) if row[0] else {}
        on = [t["id"] for t in tf.get("topics") or [] if t.get("id") and t.get("on", True)]
        if on:
            return on
        # On the seeds too, as an OpenAlex topic: every rebuild of the list
        # (a trial, an Edit-interest seed change) aggregates it again.
        c.execute(
            "UPDATE papers SET topics_json = ? WHERE (topics_json IS NULL OR topics_json IN ('', 'null', '{}'))"
            " AND openalex_id IN (SELECT ps.openalex_id FROM profile_seeds ps JOIN profiles p ON p.id = ps.profile_id"
            " WHERE p.slug = ? AND p.is_draft = 1)",
            (json.dumps({"primary_topic": {"id": TEST_TOPIC, "display_name": "Test topic"},
                         "topics": [{"id": TEST_TOPIC, "display_name": "Test topic"}]}), slug))
        topics = [*(tf.get("topics") or []),
                  {"id": TEST_TOPIC, "display_name": "Test topic", "count": 1, "on": True}]
        c.execute("UPDATE profiles SET topic_filters_json = ? WHERE slug = ? AND is_draft = 1",
                  (json.dumps({**tf, "topics": topics}), slug))
        c.commit()
        return [TEST_TOPIC]
    finally:
        c.close()
