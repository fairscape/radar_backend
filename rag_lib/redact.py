"""Error text that is safe to show a user.

Upstream errors were passed through verbatim: an OpenAlex failure carried
the full request URL, whose query string has the contact ``mailto``, and
it was stored on the run and shown in the UI. This keeps the message and
drops what is not the reader's business.
"""

from __future__ import annotations

import re

_URL_QUERY = re.compile(r"(https?://[^\s?'\"<>]+)\?[^\s'\"<>]*")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
MAX_LEN = 500


def public_error(value: object) -> str:
    """``value`` (an exception or text) without URL query strings or emails."""
    text = str(value) if value is not None else ""
    text = _URL_QUERY.sub(r"\1", text)
    text = _EMAIL.sub("<email>", text)
    return text if len(text) <= MAX_LEN else text[: MAX_LEN - 1] + "…"
