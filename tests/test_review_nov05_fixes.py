"""Fixes from the 2026-11-05 review (review_records/2026-11-05-code-review).

BT3  the lite RP profile (seed text + vectors) written to rp_profiles/<id>
     for research topics is scratch: gone once the topics are out.
BT4  an interest is never saved with no topic on, even when its topic list
     is empty (user 2026-11-05: "没有主题时，不保存，要求用户至少保留一个主题").
BT1  the OpenAlex key is scrubbed in its URL-encoded forms too.
BT2  a configured key refused (401/403) is told as that, not "without a key".
BL-3 a request from an idle tab (X-Radar-Idle) does not renew the sign-in.
FC-2/BL-2  a tab without the right can still drop its own edit copy, never another's.
"""

from __future__ import annotations

import os
import time

import pytest
import requests

from rag_lib import openalex_status
from rag_lib.api import settings as settings_module
from rag_lib.db import connect
from rag_lib.db.repos import papers as papers_repo, profiles as profiles_repo
from tests.test_interest_terms import _request, app  # noqa: F401  (fixture)

MARK = "UNPUBLISHED-MANUSCRIPT-ZQX"


def _draft_with_seeds(app, tmp_path, name="Secret draft", *, text=False, vectors=False):
    slug = _request(app, "POST", "/api/profiles/draft", json={"name": name}).json()["slug"]
    conn = connect(tmp_path / "radar.db")
    pid = conn.execute("SELECT id FROM profiles WHERE slug = ?", (slug,)).fetchone()[0]
    model = settings_module.get_settings().RADAR_DEFAULT_EMBEDDING_MODEL
    from rag_lib.db.repos import embeddings as embeddings_repo
    from rag_lib.embedders import placeholder_embed
    for i in range(4):
        if text:
            oa = f"local:upload:1:{i:064d}:t{i}"
            papers_repo.upsert(conn, {"openalex_id": oa, "title": f"{MARK} draft paper {i}", "abstract": None,
                                      "body_text": f"{MARK} body text of manuscript {i}. " * 20, "source": "user_pdf"})
        else:
            oa = f"https://openalex.org/W90{i}"
        profiles_repo.attach_seed(conn, pid, oa)
        if vectors:
            embeddings_repo.upsert(conn, oa, model, placeholder_embed(oa))
    conn.commit()
    conn.close()
    return slug, pid


def test_bt3_research_topics_leave_no_manuscript_text_on_disk(app, tmp_path):
    base = tmp_path / "rp_profiles"
    # One left behind by an older build, untouched for 2 hours: swept.
    old = base / "999"
    (old / "sources").mkdir(parents=True)
    (old / "sources" / "papers.jsonld").write_text(f"{MARK} old")
    two_h = time.time() - 7200
    os.utime(old, (two_h, two_h))

    slug, pid = _draft_with_seeds(app, tmp_path, text=True)
    r = _request(app, "POST", f"/api/profiles/{slug}/terms/generate", json={})
    assert r.status_code == 200, r.text
    assert r.json()["terms"], r.text                  # it still generated topics
    assert not [d for d in base.iterdir() if d.name.startswith(f"{pid}-")] if base.exists() else True
    assert not old.exists()
    left = [p for p in base.rglob("*") if p.is_file() and MARK.encode() in p.read_bytes()] if base.exists() else []
    assert left == []
    # Regenerate rebuilds it from scratch and tidies again.
    r = _request(app, "POST", f"/api/profiles/{slug}/terms/generate", json={})
    assert r.status_code == 200
    assert not base.exists() or not any(base.iterdir())


def test_r3_each_generation_builds_in_a_folder_of_its_own(app, tmp_path, monkeypatch):
    """Review R3: two generations for one draft (two tabs, or Back then
    Regenerate) used one folder and removed each other's."""
    from rag_lib import interest_terms
    real = interest_terms.build_lite_profile
    roots = []

    def spy(root, *a, **k):
        roots.append(root)
        assert not root.exists()                       # nobody else's
        return real(root, *a, **k)
    monkeypatch.setattr(interest_terms, "build_lite_profile", spy)
    slug, pid = _draft_with_seeds(app, tmp_path, text=True)
    for _ in range(2):
        assert _request(app, "POST", f"/api/profiles/{slug}/terms/generate", json={}).status_code == 200
    assert len(roots) == 2 and roots[0] != roots[1]
    assert all(r.parent == roots[0].parent and r.name.startswith(f"{pid}-") for r in roots)
    assert not any(r.exists() for r in roots)


def _commit(app, slug, ids):
    return _request(app, "POST", "/api/profiles", json={
        "slug": slug, "threshold": 0.5, "selected_topic_ids": ids, "cron": "0 4 * * *", "tz": "UTC", "years": None})


def test_bt4_a_draft_with_no_topic_at_all_is_not_saved(app, tmp_path):
    slug, _ = _draft_with_seeds(app, tmp_path, name="No topics", vectors=True)
    assert _request(app, "GET", f"/api/profiles/draft/{slug}/topics").json() == []
    r = _commit(app, slug, [])
    assert r.status_code == 400 and "no topic to search by" in r.text, r.text
    conn = connect(tmp_path / "radar.db")
    assert conn.execute("SELECT is_draft FROM profiles WHERE slug = ?", (slug,)).fetchone()[0] == 1
    conn.close()


def test_bt4_a_saved_interest_with_an_empty_list_keeps_refusing(app, tmp_path):
    """Saved with no topic (as before this rule): an edit of it is not saved
    until a topic is on. (A saved interest's topics change only in Edit
    interest; PUT topics refuses a saved interest.)"""
    conn = connect(tmp_path / "radar.db")
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE slug = 'chromatin'", ('{"topics": []}',))
    conn.commit()
    conn.close()
    r = _request(app, "POST", "/api/profiles/chromatin/edit")
    assert r.status_code == 200, r.text
    copy = r.json()["slug"]
    r = _request(app, "POST", f"/api/profiles/draft/{copy}/save-edit")
    assert r.status_code in (400, 409) and "no topic to search by" in r.text, r.text


class _Resp(requests.Response):
    def __init__(self, code, url, text=""):
        super().__init__()
        self.status_code = code
        self.url = url
        self.reason = "Bad"
        self._content = text.encode()


class _Session:
    """Answers every GET with ``answer(prepared_url)``; no network."""
    def __init__(self, answer):
        self.answer = answer
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        full = requests.Request("GET", url, params=params).prepare().url
        return self.answer(full)


@pytest.mark.parametrize("key", ["plainKEY123abc", "abc+def/ghi=jk", "key with space"])
def test_bt1_the_key_is_scrubbed_in_every_form(key):
    from urllib.parse import quote, quote_plus

    from rag_lib.openalex_client import OpenAlexClient
    forms = {key, quote(key, safe=""), quote_plus(key)}

    def leaked(text):
        return [f for f in forms if f in text]

    openalex_status.reset()
    c = OpenAlexClient("a@example.com", rate_limit_sleep=0, api_key=key,
                       session=_Session(lambda u: _Resp(400, u, "bad request")))
    with pytest.raises(requests.HTTPError) as e:
        c._get("/works/W400")
    assert leaked(str(e.value)) == [], str(e.value)

    def refuse(u):
        raise requests.ConnectionError(f"HTTPConnectionPool: Max retries exceeded with url: /works/W1?api_key={quote_plus(key)}&mailto=x")
    c = OpenAlexClient("a@example.com", rate_limit_sleep=0, api_key=key, session=_Session(refuse))
    with pytest.raises(requests.HTTPError) as e:
        c._get("/works/W1")
    assert leaked(str(e.value)) == [], str(e.value)

    # A 429 body that quotes the key across the 200-character cut.
    body = "x" * 190 + key + "y" * 50
    c = OpenAlexClient("a@example.com", rate_limit_sleep=0, api_key=key,
                       session=_Session(lambda u: _Resp(429, u, body)))
    with pytest.raises(requests.HTTPError) as e:
        c._get("/works/W429")
    assert not any(f[:6] in str(e.value) for f in forms), str(e.value)
    openalex_status.reset()


def test_bt2_a_refused_key_is_told_as_a_refused_key():
    from rag_lib.openalex_client import OpenAlexClient
    from rag_lib.openalex_status import OpenAlexUnavailable

    for key, kind, says in (("realKEY123", "key", "rejected the configured API key"),
                            (None, "auth", "without an API key")):
        openalex_status.reset()
        c = OpenAlexClient("a@example.com", rate_limit_sleep=0, api_key=key or "",
                           session=_Session(lambda u: _Resp(401, u, "Invalid api_key")))
        c.api_key = key
        with pytest.raises(OpenAlexUnavailable) as e:
            c._get("/works/W1")
        assert e.value.kind == kind and says in e.value.user_message, e.value.user_message
        assert "401" in e.value.detail
        assert openalex_status.status()["kind"] == kind
    openalex_status.reset()
