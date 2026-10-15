"""The contact email from Settings goes with every OpenAlex request.

Only the scheduled scans used it; trial scans, imports, uploads and
researcher refreshes sent the deployment default.
"""

from __future__ import annotations

from types import SimpleNamespace

from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import profiles as profiles_repo, users as users_repo

def test_the_rule_is_settings_email_then_login_email_then_default(tmp_path):
    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    set_ = int(users_repo.upsert(conn, "a@example.com", mailto="lab@example.org")["id"])
    plain = int(users_repo.upsert(conn, "b@example.com")["id"])
    conn.execute("UPDATE users SET mailto = NULL WHERE id = ?", (plain,))
    conn.commit()
    assert users_repo.contact_mailto(conn, set_, "default@x") == "lab@example.org"
    assert users_repo.contact_mailto(conn, plain, "default@x") == "b@example.com"
    assert users_repo.contact_mailto(conn, None, "default@x") == "default@x"
    assert users_repo.contact_mailto(conn, 999, "default@x") == "default@x"

def test_the_trial_scan_asks_openalex_with_the_users_contact_email(tmp_path, monkeypatch):
    from rag_lib.api.services import wizard

    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    uid = int(users_repo.upsert(conn, "a@example.com", mailto="lab@example.org")["id"])
    pid = profiles_repo.upsert(conn, user_id=uid, name="X", embedding_model="placeholder-v1",
                               n_seed=1, topic_filters={"topics": []}, threshold=0.8)
    row = profiles_repo.get(conn, pid)

    seen = []

    class Gatherer:
        def __init__(self, mailto):
            seen.append(mailto)

        def fetch(self, profile, *, since, limit):
            return []

        def cost(self):
            return {}

    monkeypatch.setattr(wizard, "OpenAlexGatherer", Gatherer)
    # A seed so the trial gets as far as the fetch.
    from rag_lib.paper import Paper
    monkeypatch.setattr(wizard, "_load_profile", lambda c, r: SimpleNamespace(
        papers=[Paper(doi=None, openalex_id="W1", title="t")], topic_filters={"topics": []}))
    monkeypatch.setattr(wizard, "_new_selector_for_draft", lambda r, p: SimpleNamespace(
        fit=lambda p: None, select=lambda c, p, threshold=None: []))
    settings = SimpleNamespace(RADAR_DEFAULT_MAILTO="default@x", RADAR_GATHER_MAX_CANDIDATES=None)
    try:
        wizard.compute_dry_run_for_profile(conn, settings, profile_row=row, slug="x")
    except Exception:  # noqa: BLE001 -- only the gatherer's construction matters here
        pass
    assert seen == ["lab@example.org"]
