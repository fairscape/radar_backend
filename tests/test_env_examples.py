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

# "RADAR_X=" at the start of a line, commented out or not.
ENV_KEY = re.compile(r"^#?\s*(RADAR_[A-Z0-9_]+)=", re.MULTILINE)
# getattr(settings, "RADAR_X", ...) -- a name the type checker cannot see.
GETATTR_KEY = re.compile(r"getattr\(\s*[\w.]+\s*,\s*\"(RADAR_[A-Z0-9_]+)\"")
# Keys read straight from the process environment, not through Settings.
ENVIRON_ONLY = {"RADAR_SQLITE_JOURNAL_MODE", "RADAR_OLLAMA_URL"}


@pytest.mark.parametrize("name", [".env.example", "deploy/env.example", "DEPLOY.md"])
def test_documented_keys_are_settings_fields(name):
    keys = set(ENV_KEY.findall((ROOT / name).read_text(encoding="utf-8")))
    assert keys, f"no RADAR_ keys found in {name}"
    assert keys - FIELDS - ENVIRON_ONLY == set()


def test_settings_read_by_name_exist():
    hits = {
        f"{p.relative_to(ROOT)}: {key}"
        for p in sorted((ROOT / "rag_lib").rglob("*.py"))
        for key in GETATTR_KEY.findall(p.read_text(encoding="utf-8"))
        if key not in FIELDS
    }
    assert hits == set()


def test_deploy_env_example_loads():
    s = Settings(_env_file=ROOT / "deploy" / "env.example")
    assert s.RADAR_HOST == "127.0.0.1"
    assert s.RADAR_NIGHTLY_GATHER_ENABLED is False


def test_env_is_not_tracked():
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", ".env"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    assert out.strip() == "", ".env is tracked; ship deploy/env.example instead"
