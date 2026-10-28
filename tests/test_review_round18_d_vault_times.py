"""Review round 18 (d): a second user's upload of a file someone else
uploaded first showed the first uploader's time ("added 5 h ago", the
stats' last upload), telling them someone had, and when."""

from __future__ import annotations

import io

from rag_lib.db import connect
from tests.test_review_round12_isolation import A, B, _request, world  # noqa: F401 -- fixture
from tests.test_researchers import env  # noqa: F401 -- fixture
from tests.upload_inbox import inbox


def test_second_uploader_sees_own_time(world):
    app, env = world["app"], world["env"]
    conn = connect(env)
    try:
        # A's upload, long ago.
        conn.execute("UPDATE papers SET first_seen_at = '2001-01-01 00:00:00' WHERE openalex_id = ?", (world["upload"],))
        conn.execute("UPDATE paper_uploads SET uploaded_at = '2001-01-01 00:00:00' WHERE openalex_id = ?", (world["upload"],))
        conn.commit()
    finally:
        conn.close()
    files = {"file": ("same.pdf", io.BytesIO(b"%PDF-1.4\nalice-bytes"), "application/pdf")}
    r = _request(app, "POST", "/api/vault/upload", files=files, headers=B,
                 data={"profile_slug": inbox(_request, app, B)})
    assert r.status_code == 200, r.text
    # B's copy is B's own paper now (0029), so it never had A's time.
    b_upload = r.json()["id"]
    assert b_upload != world["upload"]
    assert not r.json()["added"].startswith("2001"), r.json()
    docs = _request(app, "GET", "/api/vault/docs", headers=B).json()
    docs = docs.get("docs", docs) if isinstance(docs, dict) else docs
    mine = [d for d in docs if d["id"] == b_upload]
    assert mine and not mine[0]["added"].startswith("2001"), mine
    stats = _request(app, "GET", "/api/vault/stats", headers=B).json()
    assert not str(stats.get("lastIngest")).startswith("2001"), stats
    # A still sees their own time.
    docs_a = _request(app, "GET", "/api/vault/docs", headers=A).json()
    docs_a = docs_a.get("docs", docs_a) if isinstance(docs_a, dict) else docs_a
    assert [d for d in docs_a if d["id"] == world["upload"]][0]["added"].startswith("2001")
