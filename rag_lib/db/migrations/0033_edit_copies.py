"""0033_edit_copies -- editing a saved interest on a copy (FUNCTIONAL_SPEC E).

``profiles.edit_of``: a copy made when the user opens Edit interest
points at the interest it copies, one copy per interest. It is a draft (``is_draft = 1``): not
listed, scanned or scored; Save changes writes it back onto the interest
and deletes it, leaving without saving discards it (``draft_owned``).

``profiles.researcher_doc_json``: the researcher's profile document as it
was when the interest applied its topics (``{"researcher_id", "name",
"document"}``). The interest's topic list reads this copy, not the
researcher: re-importing or deleting the researcher no longer changes an
interest built from them (E.7). Interests that have the topics applied
now get the document they would have read today.
"""

from __future__ import annotations

import json
import sqlite3

FLAG = "researcher_topics"     # rag_lib.researcher_topics.FLAG


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE profiles ADD COLUMN edit_of INTEGER REFERENCES profiles(id) ON DELETE CASCADE")
    conn.execute("ALTER TABLE profiles ADD COLUMN researcher_doc_json TEXT")
    # One copy per interest: two Edit interest opens at once made two, and
    # only one of them was ever found again.
    conn.execute("CREATE UNIQUE INDEX idx_profiles_edit_of ON profiles(edit_of) WHERE edit_of IS NOT NULL")
    rows = conn.execute(
        "SELECT p.id, p.topic_filters_json, r.id, r.name, r.document_json"
        " FROM profiles p JOIN researchers r ON r.id = p.researcher_id"
    ).fetchall()
    for pid, raw_tf, rid, name, doc in rows:
        try:
            tf = json.loads(raw_tf) if raw_tf else {}
            document = json.loads(doc) if doc else None
        except (ValueError, TypeError):
            continue
        if not isinstance(tf, dict) or not tf.get(FLAG):
            continue
        conn.execute(
            "UPDATE profiles SET researcher_doc_json = ? WHERE id = ?",
            (json.dumps({"researcher_id": rid, "name": name, "document": document}), pid),
        )
