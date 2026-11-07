"""Text that looks like an API key or another secret is never stored (user 2026-11-06).

A Claude API key typed into "Add a topic" was saved as a research topic and
kept, struck out, after it was removed. Every request field a person types
into and the server keeps (interest names, research topics, not-interests,
Prosopia / ORCID references, the contact address) refuses such text with the
reason, rather than storing it. The browser applies the same rule before
sending, and before remembering a lookup (src/lib/secretText.ts: the two
patterns are kept identical, tests/test_secret_text.py checks).

Two patterns:

* a known key prefix followed by a run of key characters (Anthropic, OpenAI,
  GitHub, GitLab, Slack, Hugging Face, AWS, Google), or ``Bearer <token>``;
* any run of 24+ letters and digits, with no separator, that mixes upper
  case, lower case and digits -- a random token. Lower-case slugs
  ("jane-doe-uva-2024"), names joined by - or _ ("COVID-19_Long-Term_2020"),
  DOIs, ORCIDs, URLs and ordinary words do not match. A known prefix needs
  a digit after it ("sk-learn-model-tools" and "Standard-bearer" are words).
"""

from __future__ import annotations

import re

KNOWN_PREFIX = (
    r"(?:^|[^A-Za-z0-9_-])(?:sk-ant-|sk-proj-|sk-|ghp_|gho_|ghs_|ghu_|github_pat_|glpat-"
    r"|xox[abprs]-|hf_)(?=[A-Za-z0-9_-]*[0-9])[A-Za-z0-9_-]{16,}"
    r"|(?:^|[^A-Za-z0-9])AKIA[0-9A-Z]{16}"
    r"|(?:^|[^A-Za-z0-9])AIza[0-9A-Za-z_-]{30,}"
    r"|(?:^|[^A-Za-z0-9_-])[Bb]earer\s+(?=[A-Za-z0-9._~+/-]*[0-9])[A-Za-z0-9._~+/-]{16,}"
)
RANDOM_TOKEN = (
    r"(?:^|[^A-Za-z0-9])(?=[A-Za-z0-9]*[0-9])(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])"
    r"[A-Za-z0-9]{24,}(?![A-Za-z0-9])"
)
_PATTERN = re.compile(f"{KNOWN_PREFIX}|{RANDOM_TOKEN}")

MESSAGE = ("This looks like an API key or another secret, so it was not saved. "
           "Paste an API key only into the field that asks for one.")


def looks_like_secret(text: str | None) -> bool:
    return bool(text) and _PATTERN.search(text) is not None


def holds_secret(value) -> bool:
    """Whether any string inside ``value`` (nested lists and dicts) looks like a secret."""
    if isinstance(value, str):
        return looks_like_secret(value)
    if isinstance(value, dict):
        return any(holds_secret(k) or holds_secret(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(holds_secret(v) for v in value)
    return False


def refuse_secret(text):
    """Pydantic ``AfterValidator``: the text, or ValueError(MESSAGE)."""
    if isinstance(text, str) and looks_like_secret(text):
        raise ValueError(MESSAGE)
    return text
