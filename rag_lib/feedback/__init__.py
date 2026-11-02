"""Feedback log (Phase 8).

Append-only record of triage decisions (save / dismiss / snooze). The
canonical store is per-user JSONL on disk; the DB mirror in
``feedback_events`` exists for queries (profile detail panel,
``GET /api/profiles/{key}/feedback``). Nothing scores or fits from it
(FUNCTIONAL_SPEC 5.2).
"""

from .log import (
    FeedbackEvent,
    log_event,
    selector_config_hash,
    tail_jsonl,
    format_event_line,
)

__all__ = [
    "FeedbackEvent",
    "log_event",
    "selector_config_hash",
    "tail_jsonl",
    "format_event_line",
]
