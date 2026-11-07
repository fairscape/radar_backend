"""Text that looks like an API key is never stored (user 2026-11-06).

A Claude API key typed into "Add a topic" was saved as a research topic and
kept, struck out, after it was removed. ``rag_lib.secret_text`` is the rule;
every request field a person types into and the server keeps refuses it,
the refusal does not echo the text back, and the browser's copy of the rule
(src/lib/secretText.ts) is the same pattern. No real key appears here: the
examples are made up.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_lib.api import schemas as S
from rag_lib.secret_text import KNOWN_PREFIX, MESSAGE, RANDOM_TOKEN, looks_like_secret
from tests.test_api_wizard import _request, app  # noqa: F401 -- fixture

FAKE_ANTHROPIC = "sk-ant-api03-" + "Xy7_Qw3-" * 12          # made up, the shape of one
SECRETS = [
    FAKE_ANTHROPIC,
    f"my key {FAKE_ANTHROPIC} please",
    "sk-proj-" + "a1B2c3D4e5" * 3,
    "sk-" + "abcdefghij1klmnopqrstuvwx",
    "ghp_" + "A1b2C3d4E5f6G7h8I9j0",
    "github_pat_" + "11ABCDEFG0" * 3,
    "xoxb-" + "1234567890-abcdefghij",
    "hf_" + "AbCdEfGhIjKlMn0pQrSt",
    "AKIA" + "ABCDEFGHIJKLMNOP",
    "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6",
    "Bearer " + "eyJhbGciOiJIUzI1NiJ9.e30.abc",
    "Zq8" + "rT5mW2xK9pL4vB7nC1hJ6gF3dS0aE" + "Yu",   # a random token, no known prefix
]
ORDINARY = [
    "Clinical outcome prediction",
    "Natural language processing for healthcare",
    "risk-stratification-for-sepsis-in-neonatal-intensive-care",
    "sk-learn baselines",
    "jane-doe-university-of-virginia-2024",
    "https://prosopia.org/profiles/jane-doe-university-of-virginia-2024",
    "10.1101/2024.01.01.123456",
    "0000-0002-1825-0097",
    "lei@example.org",
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # a lower-case hash
    "Bearer of bad news",
    # False positives the review found (2026-11-06)
    "Standard-bearer organizations/networks",
    "Risk-bearer reinsurance/securitization markets",
    "scikit-learn sk-learn-model-selection-tools",
    "COVID-19_Long-Term_Outcomes_2020-2024_UVA",
    "",
]


@pytest.mark.parametrize("text", SECRETS)
def test_secrets_are_recognised(text):
    assert looks_like_secret(text)


@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_text_is_not(text):
    assert not looks_like_secret(text)


def test_the_browser_applies_the_same_pattern():
    ts = (Path(__file__).resolve().parents[2] / "radar_frontend/src/lib/secretText.ts")
    if not ts.exists():
        pytest.skip("frontend tree not next to the backend")
    src = ts.read_text()
    raw = dict(re.findall(r"export const (KNOWN_PREFIX|RANDOM_TOKEN) = String\.raw`([^`]*)`;", src))
    assert raw == {"KNOWN_PREFIX": KNOWN_PREFIX, "RANDOM_TOKEN": RANDOM_TOKEN}
    assert repr(MESSAGE) in src or f"'{MESSAGE}'" in src


# Every request field a person types into and the server keeps.
TYPED = [
    (S.DraftCreateRequest, {"name": "{}"}),
    (S.ResearcherInterestRequest, {"name": "{}"}),
    (S.ProsopiaImportRequest, {"ref": "jane-doe", "name": "{}"}),
    (S.ProsopiaImportRequest, {"ref": "{}"}),
    (S.OrcidImportRequest, {"orcid": "0000-0002-1825-0097", "openalex_ids": [], "name": "{}"}),
    (S.OrcidImportRequest, {"orcid": "{}", "openalex_ids": []}),
    (S.ResearcherImportRequest, {"source": "prosopia", "ref": "{}"}),
    (S.UserUpdateRequest, {"mailto": "{}"}),
    (S.ProsopiaImportRequest, {"ref": "jane-doe", "base_url": "{}"}),
    (S.ResearcherImportRequest, {"source": "prosopia", "ref": "jane-doe", "base_url": "{}"}),
    (S.InterestTermsUpdate, {"model": "{}"}),
    (S.InterestTermsUpdate, {"seed_hash": "{}"}),
    (S.InterestTermsUpdate, {"generated_at": "{}"}),
    # Research topic labels and not-interests: checked in the route (only
    # new ones), see test_a_refused_topic... and test_a_key_already_stored...
]


def _fill(body, text):
    if isinstance(body, dict):
        return {k: _fill(v, text) for k, v in body.items()}
    if isinstance(body, list):
        return [_fill(v, text) for v in body]
    return text if body == "{}" else body


@pytest.mark.parametrize(("model", "body"), TYPED, ids=lambda x: getattr(x, "__name__", ""))
def test_every_typed_field_refuses_a_secret_and_takes_ordinary_text(model, body):
    with pytest.raises(ValidationError, match="looks like an API key"):
        model(**_fill(body, FAKE_ANTHROPIC))
    ok = "lei@example.org" if model is S.UserUpdateRequest else "https://prosopia.org" if "base_url" in body else (
        "0000-0002-1825-0097" if body.get("orcid") == "{}" else "Clinical outcome prediction")
    model(**_fill(body, ok))


def test_a_refused_name_is_not_stored_nor_echoed(app, tmp_path):
    r = _request(app, "POST", "/api/profiles/draft", json={"name": FAKE_ANTHROPIC})
    assert r.status_code == 422
    assert "looks like an API key" in r.text
    assert FAKE_ANTHROPIC not in r.text and "Xy7_Qw3" not in r.text
    import sqlite3
    conn = sqlite3.connect(tmp_path / "radar.db")
    assert conn.execute("SELECT COUNT(*) FROM profiles WHERE name LIKE '%sk-ant%'").fetchone()[0] == 0


def test_a_refused_topic_leaves_the_stored_topics_as_they_were(app, tmp_path):
    slug = _request(app, "POST", "/api/profiles/draft", json={"name": "Topics"}).json()["slug"]
    ok = _request(app, "PUT", f"/api/profiles/{slug}/terms", json={"terms": [{"label": "Clinical NLP"}]})
    assert ok.status_code == 200, ok.text
    r = _request(app, "PUT", f"/api/profiles/{slug}/terms",
                 json={"terms": [{"label": "Clinical NLP"}], "removed": [{"label": FAKE_ANTHROPIC}]})
    assert r.status_code == 422
    assert FAKE_ANTHROPIC not in r.text and "Xy7_Qw3" not in r.text
    got = _request(app, "GET", f"/api/profiles/{slug}/terms").json()
    assert [t["label"] for t in got["terms"]] == ["Clinical NLP"] and got.get("removed", []) == []


def test_a_struck_topic_deleted_for_good_is_gone(app):
    """RD-1: struck out it is kept (to put back); deleted for good, the
    stored record no longer has it."""
    slug = _request(app, "POST", "/api/profiles/draft", json={"name": "Purge"}).json()["slug"]
    put = lambda body: _request(app, "PUT", f"/api/profiles/{slug}/terms", json=body)  # noqa: E731
    assert put({"terms": [{"label": "Kept"}], "removed": [{"label": "Struck"}]}).status_code == 200
    got = _request(app, "GET", f"/api/profiles/{slug}/terms").json()
    assert [t["label"] for t in got["removed"]] == ["Struck"]
    assert put({"terms": [{"label": "Kept"}], "removed": []}).status_code == 200
    got = _request(app, "GET", f"/api/profiles/{slug}/terms").json()
    assert got["removed"] == [] and [t["label"] for t in got["terms"]] == ["Kept"]


def test_a_key_already_stored_passes_so_it_can_be_deleted(app, tmp_path):
    """Review M2: a record holding a key from before the rule must not block
    every later save; it passes until deleted for good. A new one does not."""
    import json
    import sqlite3

    slug = _request(app, "POST", "/api/profiles/draft", json={"name": "Old key"}).json()["slug"]
    conn = sqlite3.connect(tmp_path / "radar.db")
    conn.execute("UPDATE profiles SET interest_terms_json = ? WHERE slug = ?", (json.dumps(
        {"source": "manual", "terms": [{"label": "Kept"}], "removed": [{"label": FAKE_ANTHROPIC}],
         "not_interests": []}), slug))
    conn.commit()
    put = lambda body: _request(app, "PUT", f"/api/profiles/{slug}/terms", json=body)  # noqa: E731
    r = put({"terms": [{"label": "Kept"}, {"label": "Another"}], "removed": [{"label": FAKE_ANTHROPIC}]})
    assert r.status_code == 200, r.text                                  # the old one rides along
    other = "sk-ant-api03-" + "Pp4_Rr8-" * 12
    r = put({"terms": [{"label": "Kept"}, {"label": other}], "removed": [{"label": FAKE_ANTHROPIC}]})
    assert r.status_code == 422 and other[:20] not in r.text              # a new one does not
    assert put({"terms": [{"label": "Kept"}], "removed": []}).status_code == 200   # deleted for good
    stored = conn.execute("SELECT interest_terms_json FROM profiles WHERE slug = ?", (slug,)).fetchone()[0]
    assert "sk-ant" not in stored
    conn.close()


def test_no_refusal_echoes_a_key_even_when_another_check_fails_first(app):
    """Review L5: a label over the length limit is refused before the key
    check; the 422 still must not carry the key."""
    slug = _request(app, "POST", "/api/profiles/draft", json={"name": "Echo"}).json()["slug"]
    r = _request(app, "PUT", f"/api/profiles/{slug}/terms",
                 json={"terms": [{"label": "x" * 400}, {"label": FAKE_ANTHROPIC}]})
    assert r.status_code == 422 and "Xy7_Qw3" not in r.text
    r = _request(app, "PUT", f"/api/profiles/{slug}/terms",
                 json={"terms": [{"label": "t"}], "removed": [{"label": FAKE_ANTHROPIC}] * 500})
    assert r.status_code == 422 and "Xy7_Qw3" not in r.text


def test_the_lookups_do_not_take_or_echo_a_key(app):
    r = _request(app, "GET", "/api/import/prosopia/works", params={"ref": FAKE_ANTHROPIC})
    assert r.status_code == 422 and "Xy7_Qw3" not in r.text and "looks like an API key" in r.text
    r = _request(app, "GET", f"/api/import/orcid/{FAKE_ANTHROPIC}/works")
    assert r.status_code == 400 and "Xy7_Qw3" not in r.text
