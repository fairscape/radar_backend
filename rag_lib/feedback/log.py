"""Feedback log writer + reader.

``log_event`` is the single entry point: appends one JSON line to the
per-user JSONL and inserts a mirror row into ``feedback_events``. The
JSONL is the canonical record; the DB mirror exists for SQL queries.

Schema (per spec):
    timestamp, profile, selector, selector_config_hash,
    benchmark_run_id, doi, openalex_id, score, action
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag_lib.db.repos import feedback as feedback_repo


# Actions logged to JSONL + DB. ``shown`` is intentionally absent —
# captured by ``profile_candidates.shown_at``.
ALLOWED_ACTIONS = frozenset({"saved", "dismissed", "snoozed"})


@dataclass(frozen=True)
class FeedbackEvent:
    """One triage event ready to be written."""

    profile_id: int
    profile_slug: str
    openalex_id: str
    action: str
    score: float | None = None
    selector: str | None = None
    selector_config_hash: str | None = None
    benchmark_run_id: int | None = None
    doi: str | None = None
    timestamp: str | None = None  # filled by log_event when None

    def to_jsonl_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "profile": self.profile_slug,
            "selector": self.selector,
            "selector_config_hash": self.selector_config_hash,
            "benchmark_run_id": self.benchmark_run_id,
            "doi": self.doi,
            "openalex_id": self.openalex_id,
            "score": self.score,
            "action": self.action,
        }


def selector_config_hash(config: dict) -> str:
    """Stable 16-char sha256 of a selector config dict.

    ``json.dumps(..., sort_keys=True)`` is canonical enough at our scale;
    floats round-trip identically across runs because we never mutate
    them after fit. Returns the hex prefix, lowercase.
    """
    blob = json.dumps(config, sort_keys=True, default=_jsonable).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _jsonable(obj):
    """Fallback for json.dumps when a value is not JSON-native.

    Numpy arrays are the common case here (selector configs embed the
    centroid + seed matrix as numpy until ``.tolist()`` is called).
    """
    tolist = getattr(obj, "tolist", None)
    if tolist is not None:
        return tolist()
    return repr(obj)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonl_path_for(vault_dir: Path | str, user_id: int) -> Path:
    return Path(vault_dir) / str(user_id) / "state" / "feedback.jsonl"


def log_event(
    vault_dir: Path | str,
    user_id: int,
    db: sqlite3.Connection,
    event: FeedbackEvent,
) -> FeedbackEvent:
    """Append ``event`` to JSONL and insert into ``feedback_events``.

    Returns a copy of the event with ``timestamp`` populated when the
    caller passed ``None``. Skips silently if the action is outside
    ``ALLOWED_ACTIONS`` (so callers can pass through any state coming
    from ``mark_saved`` without filtering it themselves).
    """
    if event.action not in ALLOWED_ACTIONS:
        return event

    ts = event.timestamp or _utc_now_iso()
    stamped = FeedbackEvent(
        profile_id=event.profile_id,
        profile_slug=event.profile_slug,
        openalex_id=event.openalex_id,
        action=event.action,
        score=event.score,
        selector=event.selector,
        selector_config_hash=event.selector_config_hash,
        benchmark_run_id=event.benchmark_run_id,
        doi=event.doi,
        timestamp=ts,
    )

    path = jsonl_path_for(vault_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(stamped.to_jsonl_dict()) + "\n")

    feedback_repo.insert(
        db,
        user_id=user_id,
        profile_id=event.profile_id,
        openalex_id=event.openalex_id,
        action=event.action,
        score=event.score,
        selector=event.selector,
        selector_config_hash=event.selector_config_hash,
        benchmark_run_id=event.benchmark_run_id,
        ts=ts,
    )
    return stamped


def tail_jsonl(path: Path | str, n: int = 20) -> list[str]:
    """Return the last ``n`` lines of a JSONL file (oldest → newest).

    Streams the file once; safe to call on a missing file (returns []).
    Lines are returned with no trailing newline.
    """
    p = Path(path)
    if not p.exists():
        return []
    buf: deque[str] = deque(maxlen=max(1, n))
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if stripped:
                buf.append(stripped)
    return list(buf)


def format_event_line(row: sqlite3.Row | dict) -> str:
    """Render a feedback_events row as the UI's plain-string log line.

    Mirrors the mock-api shape:
        ``2026-04-16T07:00:12Z  SAVE     centroid  10.x/...  score=0.912``

    Falls back to the openalex id when DOI is missing.
    """
    get = row.__getitem__ if isinstance(row, sqlite3.Row) else row.get
    ts = get("ts") or ""
    action = (get("action") or "").upper()
    selector = get("selector") or "?"
    doi = None
    try:
        doi = get("doi")
    except (KeyError, IndexError):
        doi = None
    ident = doi or get("openalex_id") or "?"
    score = get("score")
    score_str = f"score={score:.3f}" if isinstance(score, (int, float)) else "score=NA"
    return f"{ts}  {action:<8} {selector}  {ident}  {score_str}"
