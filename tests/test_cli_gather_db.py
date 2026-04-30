"""cli.gather --db end-to-end: dedup across runs + gather_runs audit trail.

Network-free: patches `cli.gather._cascade_fetch` and `cli.gather.OpenAlexClient`
so no HTTP traffic is generated. Builds a small Profile JSON on disk via
Profile.from_csv, then runs gather twice and asserts:
  - First run: n_new = K, n_redup = 0
  - Second run: n_new = 0, n_redup = K (prior triage state preserved)
  - gather_runs has 2 rows
"""

from __future__ import annotations

import csv
import json
import sys

import pytest

import cli.gather as gather_cli
import cli.ingest as ingest_cli
from rag_lib.db import connect
from rag_lib.db.repos import (
    candidates as candidates_repo,
    profiles as profiles_repo,
    users as users_repo,
)
from rag_lib.paper import Paper
from tests.fake_openalex_client import FakeOpenAlexClient, canned_openalex_work


def _write_csv(tmp_path, rows):
    p = tmp_path / "manifest.csv"
    cols = sorted({k for r in rows for k in r.keys()})
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


@pytest.fixture()
def built_profile_json(tmp_path, monkeypatch):
    """Run cli.ingest once with a fake OpenAlex client to produce a Profile JSON."""
    fake_client = FakeOpenAlexClient(canned_works={
        "10.1/a": canned_openalex_work(
            doi="10.1/a", openalex_id="W1", title="Paper A", year=2024,
            abstract_words=["alpha"], primary_topic_id="T1",
        ),
        "10.1/b": canned_openalex_work(
            doi="10.1/b", openalex_id="W2", title="Paper B", year=2023,
            abstract_words=["beta"], primary_topic_id="T1",
        ),
    })
    csv_path = _write_csv(tmp_path, [{"doi": "10.1/a"}, {"doi": "10.1/b"}])
    out_json = tmp_path / "profile.json"

    monkeypatch.setattr(ingest_cli, "OpenAlexClient", lambda **kw: fake_client)
    monkeypatch.setattr(sys, "argv", [
        "ingest", "--csv", str(csv_path),
        "--name", "neonatal_vitals", "--email", "demo@example.com",
        "--out", str(out_json),
    ])
    assert ingest_cli.main() == 0

    # Sanity: the JSON has selector_config + papers populated.
    data = json.loads(out_json.read_text())
    assert data["name"] == "neonatal_vitals"
    assert data["selector_config"]["type"] == "centroid"
    return out_json


def _candidate_papers() -> list[Paper]:
    """Three synthetic candidates the patched _cascade_fetch returns."""
    return [
        Paper(
            doi=f"10.9/{oa.lower()}", openalex_id=oa,
            title=f"Cand {oa}", abstract="abs", year=2025,
            source="openalex_gatherer",
        )
        for oa in ("W100", "W101", "W102")
    ]


def _patch_gather_network(monkeypatch, candidates):
    """Replace OpenAlexClient and _cascade_fetch so gather runs offline."""

    class _NoopClient:
        api_calls = 0
        def __init__(self, **_): pass

    monkeypatch.setattr(gather_cli, "OpenAlexClient", _NoopClient)
    monkeypatch.setattr(
        gather_cli, "_cascade_fetch",
        lambda *a, **kw: (list(candidates), "must-have-AND",
                          "from_publication_date:2026-01-01", []),
    )


def test_gather_db_first_run_inserts_all_new(
    tmp_path, monkeypatch, built_profile_json
):
    db_path = tmp_path / "radar.db"
    out_json = tmp_path / "candidates.json"
    candidates = _candidate_papers()
    _patch_gather_network(monkeypatch, candidates)

    monkeypatch.setattr(sys, "argv", [
        "gather",
        "--profile", str(built_profile_json),
        "--email", "demo@example.com",
        "--days", "30",
        "--limit", "10",
        "--out", str(out_json),
        "--db", str(db_path),
        "--user", "demo@example.com",
    ])
    assert gather_cli.main() == 0

    conn = connect(db_path)
    user = users_repo.get_by_email(conn, "demo@example.com")
    profile_row = profiles_repo.get_by_name(conn, user["id"], "neonatal_vitals")

    runs = conn.execute(
        "SELECT n_fetched, n_new, n_redup, error FROM gather_runs ORDER BY id"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["n_fetched"] == 3
    assert runs[0]["n_new"] == 3
    assert runs[0]["n_redup"] == 0
    assert runs[0]["error"] is None
    assert candidates_repo.count_for_profile(conn, profile_row["id"]) == 3


def test_gather_db_second_run_dedups(
    tmp_path, monkeypatch, built_profile_json
):
    db_path = tmp_path / "radar.db"
    candidates = _candidate_papers()
    _patch_gather_network(monkeypatch, candidates)

    common_argv = [
        "gather",
        "--profile", str(built_profile_json),
        "--email", "demo@example.com",
        "--days", "30",
        "--limit", "10",
        "--db", str(db_path),
        "--user", "demo@example.com",
    ]

    # First run
    monkeypatch.setattr(sys, "argv", common_argv + ["--out", str(tmp_path / "r1.json")])
    assert gather_cli.main() == 0

    # User saves one of the candidates between runs.
    conn = connect(db_path)
    user = users_repo.get_by_email(conn, "demo@example.com")
    profile_row = profiles_repo.get_by_name(conn, user["id"], "neonatal_vitals")
    candidates_repo.mark_saved(conn, profile_row["id"], "W100")
    conn.close()

    # Second run with the same candidate set
    monkeypatch.setattr(sys, "argv", common_argv + ["--out", str(tmp_path / "r2.json")])
    assert gather_cli.main() == 0

    conn = connect(db_path)
    runs = conn.execute(
        "SELECT n_fetched, n_new, n_redup FROM gather_runs ORDER BY id"
    ).fetchall()
    assert len(runs) == 2
    assert runs[1]["n_fetched"] == 3
    assert runs[1]["n_new"] == 0
    assert runs[1]["n_redup"] == 3

    # Save state preserved across the resurface.
    saved = conn.execute(
        "SELECT saved_at FROM profile_candidates WHERE profile_id=? AND openalex_id=?",
        (profile_row["id"], "W100"),
    ).fetchone()
    assert saved["saved_at"] is not None


def test_gather_db_records_error_on_exception(
    tmp_path, monkeypatch, built_profile_json
):
    db_path = tmp_path / "radar.db"
    out_json = tmp_path / "candidates.json"

    class _NoopClient:
        api_calls = 7
        def __init__(self, **_): pass

    def _boom(*a, **kw):
        raise RuntimeError("HTTP 429 from upstream")

    monkeypatch.setattr(gather_cli, "OpenAlexClient", _NoopClient)
    monkeypatch.setattr(gather_cli, "_cascade_fetch", _boom)

    monkeypatch.setattr(sys, "argv", [
        "gather",
        "--profile", str(built_profile_json),
        "--email", "demo@example.com",
        "--out", str(out_json),
        "--db", str(db_path),
        "--user", "demo@example.com",
    ])
    with pytest.raises(RuntimeError):
        gather_cli.main()

    conn = connect(db_path)
    runs = conn.execute(
        "SELECT n_fetched, error, api_calls FROM gather_runs ORDER BY id"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["error"] is not None
    assert "429" in runs[0]["error"]
    assert runs[0]["api_calls"] == 7
