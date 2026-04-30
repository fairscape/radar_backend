"""cli.ingest --db end-to-end: build a Profile from CSV, persist to SQLite.

Network-free: patches `cli.ingest.OpenAlexClient` so the CSV's DOI
lookups go through a FakeOpenAlexClient.
"""

from __future__ import annotations

import csv
import sqlite3
import sys

import pytest

import cli.ingest as ingest_cli
from rag_lib.db import connect
from rag_lib.db.repos import (
    candidates as candidates_repo,
    embeddings as embeddings_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)
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
def fake_client():
    return FakeOpenAlexClient(canned_works={
        "10.1/a": canned_openalex_work(
            doi="10.1/a", openalex_id="W1", title="Paper A", year=2024,
            abstract_words=["alpha", "beta"], primary_topic_id="T1",
        ),
        "10.1/b": canned_openalex_work(
            doi="10.1/b", openalex_id="W2", title="Paper B", year=2023,
            abstract_words=["gamma", "delta"], primary_topic_id="T1",
        ),
    })


def _run_ingest_cli(monkeypatch, fake_client, argv):
    monkeypatch.setattr(
        ingest_cli, "OpenAlexClient", lambda **kw: fake_client,
    )
    monkeypatch.setattr(sys, "argv", argv)
    return ingest_cli.main()


def test_ingest_db_persists_profile_seeds_and_embeddings(
    tmp_path, monkeypatch, fake_client
):
    csv_path = _write_csv(tmp_path, [
        {"doi": "10.1/a"},
        {"doi": "10.1/b"},
    ])
    db_path = tmp_path / "radar.db"
    out_json = tmp_path / "profile.json"

    rc = _run_ingest_cli(monkeypatch, fake_client, [
        "ingest",
        "--csv", str(csv_path),
        "--name", "neonatal_vitals",
        "--email", "demo@example.com",
        "--out", str(out_json),
        "--db", str(db_path),
        "--user", "demo@example.com",
    ])
    assert rc == 0
    assert db_path.exists()
    assert out_json.exists()

    conn = connect(db_path)
    user = users_repo.get_by_email(conn, "demo@example.com")
    assert user is not None

    profile_row = profiles_repo.get_by_name(conn, user["id"], "neonatal_vitals")
    assert profile_row is not None
    assert profile_row["slug"] == "neonatal-vitals"
    assert profile_row["n_seed"] == 2
    assert profile_row["embedding_model"] == "placeholder-v1"

    seeds = profiles_repo.list_seed_openalex_ids(conn, profile_row["id"])
    assert sorted(seeds) == ["W1", "W2"]
    assert papers_repo.get_by_openalex_id(conn, "W1") is not None
    assert embeddings_repo.has(conn, "W1", "placeholder-v1")
    assert embeddings_repo.has(conn, "W2", "placeholder-v1")
    # No candidates yet — that's the gather CLI's job.
    assert candidates_repo.count_for_profile(conn, profile_row["id"]) == 0


def test_ingest_db_idempotent_on_rerun(tmp_path, monkeypatch, fake_client):
    csv_path = _write_csv(tmp_path, [{"doi": "10.1/a"}, {"doi": "10.1/b"}])
    db_path = tmp_path / "radar.db"
    out_json = tmp_path / "profile.json"

    argv = [
        "ingest",
        "--csv", str(csv_path),
        "--name", "neonatal_vitals",
        "--email", "demo@example.com",
        "--out", str(out_json),
        "--db", str(db_path),
        "--user", "demo@example.com",
    ]
    assert _run_ingest_cli(monkeypatch, fake_client, list(argv)) == 0
    assert _run_ingest_cli(monkeypatch, fake_client, list(argv)) == 0

    conn = connect(db_path)
    # Still one profile row, two papers, two embeddings.
    rows = conn.execute("SELECT count(*) AS n FROM profiles").fetchone()
    assert rows["n"] == 1
    rows = conn.execute("SELECT count(*) AS n FROM papers").fetchone()
    assert rows["n"] == 2
    rows = conn.execute("SELECT count(*) AS n FROM paper_embeddings").fetchone()
    assert rows["n"] == 2
    rows = conn.execute("SELECT count(*) AS n FROM profile_seeds").fetchone()
    assert rows["n"] == 2
