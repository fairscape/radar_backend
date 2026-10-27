"""Every setting the docs and code name is one ``Settings`` reads.

``Settings`` ignores unknown keys (``extra="ignore"``), so a key nothing
reads is dropped without a word. Six ORCID / researcher-profile
thresholds sat in the live configuration that way for a month after the
code that read them was rewritten, and the tracked ``.env`` had drifted
from the live one (reranker query mode ``topic`` against ``article``).
The repository now ships only templates; these tests keep them honest.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from rag_lib.api.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
FIELDS = set(Settings.model_fields)

DOCS = [
    ".env.example", "deploy/env.example", "DEPLOY.md", "README.md",
    "PLUGINS.md", "deploy/README.md", "deploy/start_backend.sh", "deploy/radar.sh",
]
CODE_DIRS = ["rag_lib", "cli", "scripts"]

# Read from the process environment, never from .env.
ENVIRON_ONLY = {"RADAR_SQLITE_JOURNAL_MODE"}
# start_backend.sh's own switch, not a backend setting.
SCRIPT_ONLY = {"RADAR_ALLOW_EMPTY_DB"}

ANY_KEY = re.compile(r"\bRADAR_[A-Z0-9_]+\b")
# A quoted name in Python: getattr(settings, "X"), _opt('X', ...), environ["X"].
QUOTED_KEY = re.compile(r"[\"'](RADAR_[A-Z0-9_]+)[\"']")
# "# RADAR_X=value" -- a commented-out line documents the default.
COMMENTED = re.compile(r"^#\s*(RADAR_[A-Z0-9_]+)=(.*)$", re.MULTILINE)


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", DOCS)
def test_documented_keys_are_settings(name):
    keys = set(ANY_KEY.findall(_text(name)))
    assert keys - FIELDS - ENVIRON_ONLY - SCRIPT_ONLY == set()


def test_code_reads_only_settings_that_exist():
    hits = {
        f"{p.relative_to(ROOT)}: {key}"
        for d in CODE_DIRS
        for p in sorted((ROOT / d).rglob("*.py"))
        for key in QUOTED_KEY.findall(p.read_text(encoding="utf-8"))
        if key not in FIELDS | ENVIRON_ONLY
    }
    assert hits == set()


def test_env_example_documents_every_setting():
    assert FIELDS - set(ANY_KEY.findall(_text(".env.example"))) == set()


def test_env_example_shows_the_code_defaults(monkeypatch):
    for key in FIELDS:
        monkeypatch.delenv(key, raising=False)
    wrong = []
    for key, value in COMMENTED.findall(_text(".env.example")):
        if key in ENVIRON_ONLY:
            continue
        default = Settings.model_fields[key].default
        if default is None:  # unset by default; the line shows an example value
            continue
        monkeypatch.setenv(key, value.strip())
        if getattr(Settings(_env_file=None), key) != default:
            wrong.append(f"{key}={value.strip()} (default {default!r})")
        monkeypatch.delenv(key)
    assert wrong == []


def test_deploy_env_example_loads_as_the_live_site(monkeypatch):
    path = ROOT / "deploy" / "env.example"
    assert path.is_file()
    for key in FIELDS:
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=path)
    # Values that differ from the code defaults, so a file that failed to
    # load would not pass.
    assert s.RADAR_RERANKER_ENABLED is True
    assert s.RADAR_RERANKER_QUERY_MODE == "article"
    assert s.RADAR_REQUIRE_AUTH is True
    assert s.RADAR_LOG_JSON is False
    assert s.RADAR_HOST == "127.0.0.1"
    assert s.RADAR_NIGHTLY_GATHER_ENABLED is False


def test_deploy_env_example_has_no_comment_after_a_value():
    # start_backend.sh reads RADAR_DB_PATH with grep | cut, which keeps
    # anything after the value.
    lines = [
        line for line in _text("deploy/env.example").splitlines()
        if line.startswith("RADAR_") and "#" in line
    ]
    assert lines == []


def test_env_is_not_tracked():
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", ".env"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    assert out.strip() == "", ".env is tracked; ship deploy/env.example instead"
