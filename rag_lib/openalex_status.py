"""Whether OpenAlex can be used right now.

Every step that needs OpenAlex stops when it cannot answer: an upload is
not stored as "not on OpenAlex" because OpenAlex did not reply, an interest
is not saved without its first scan. ``OpenAlexUnavailable`` is the one
error that says so, raised by ``OpenAlexClient._get`` for

  quota    429: the day's requests are used up (or too many at once)
  auth     401/403: OpenAlex wants an API key
  down     a 5xx, a timeout, no connection

``quota`` and ``auth`` pause OpenAlex for this process until the time
OpenAlex gave (the next UTC midnight when it gave none): every request in
between would be refused the same way and spend nothing but time. While
paused, ``_get`` refuses without calling, and the frontend shows a banner
(``GET /api/openalex/status``).

The pause lives in memory. A restart forgets it, and the first request
after that is refused by OpenAlex and pauses again.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import requests


# A 429 that does not say the day's quota is spent is a burst limit: it
# clears in about a second, so it is not worth stopping everything for long.
BURST_PAUSE_SECONDS = 60

_MESSAGES = {
    "quota": "OpenAlex's daily request limit has been reached.",
    "auth": "OpenAlex is refusing requests without an API key.",
    "down": "OpenAlex could not be reached.",
}


class OpenAlexUnavailable(requests.HTTPError):
    """OpenAlex did not answer, so nothing can be concluded from it.

    A ``requests.HTTPError`` so code that already treats an OpenAlex HTTP
    failure as one (a 404 check, a 5xx retry) keeps doing so.
    """

    def __init__(self, kind: str, *, until: float | None = None,
                 detail: str = "", response: requests.Response | None = None):
        self.kind = kind
        self.until = until
        self.detail = detail
        # What the user is told; str() adds what OpenAlex said, for logs and
        # a scan's recorded error.
        self.user_message = message(kind, until)
        super().__init__(self.user_message, response=response)

    def __str__(self) -> str:
        return f"{self.user_message} ({self.detail})" if self.detail else self.user_message


def message(kind: str, until: float | None = None) -> str:
    text = _MESSAGES.get(kind, _MESSAGES["down"])
    if until is not None:
        at = datetime.fromtimestamp(until, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        text += f" Paused until {at}."
    else:
        text += " Try again in a few minutes."
    return text + " Nothing was changed."


_lock = threading.Lock()
_paused: dict | None = None


def _next_utc_midnight(now: float) -> float:
    day = datetime.fromtimestamp(now, tz=timezone.utc).date() + timedelta(days=1)
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()


def pause(kind: str, *, reset_seconds: float | None = None, now: float | None = None) -> float:
    """Pause until ``reset_seconds`` from now (next UTC midnight if None)."""
    global _paused
    now = time.time() if now is None else now
    until = now + reset_seconds if reset_seconds is not None else _next_utc_midnight(now)
    with _lock:
        if _paused is None or _paused["until"] < until:
            _paused = {"kind": kind, "until": until}
        return _paused["until"]


def current(now: float | None = None) -> dict | None:
    """``{"kind", "until"}`` while paused, else None."""
    global _paused
    now = time.time() if now is None else now
    with _lock:
        if _paused is not None and _paused["until"] <= now:
            _paused = None
        return dict(_paused) if _paused else None


def check() -> None:
    """Raise ``OpenAlexUnavailable`` while paused."""
    state = current()
    if state is not None:
        raise OpenAlexUnavailable(state["kind"], until=state["until"], detail="paused")


def status() -> dict:
    """The ``GET /api/openalex/status`` body."""
    state = current()
    if state is None:
        return {"available": True, "kind": None, "until": None, "message": None}
    until = datetime.fromtimestamp(state["until"], tz=timezone.utc).isoformat()
    return {"available": False, "kind": state["kind"], "until": until,
            "message": message(state["kind"], state["until"])}


def reset() -> None:
    """Forget the pause (tests)."""
    global _paused
    with _lock:
        _paused = None
