"""Publication-year ranges for an interest (migration 0018).

One place for the rules, used by the trial scan, the commit, the PATCH
route and the backfill: years between ``MIN_YEAR`` and this year,
``year_from <= year_to``, and a backfill spans at most ``MAX_BACKFILL_YEARS``
(one OpenAlex query per year, ``BACKFILL_PER_YEAR`` papers each).
"""

from __future__ import annotations

from datetime import datetime, timezone

MIN_YEAR = 1900
MAX_BACKFILL_YEARS = 15
BACKFILL_PER_YEAR = 100
TRIAL_PER_YEAR = 60
TRIAL_TOTAL = 600


def this_year() -> int:
    return datetime.now(timezone.utc).year


def check_range(year_from: int | None, year_to: int | None) -> tuple[int | None, int | None]:
    """The range, validated. Raises ``ValueError`` with a message for the user."""
    now = this_year()
    for name, y in (("year_from", year_from), ("year_to", year_to)):
        if y is not None and not (MIN_YEAR <= int(y) <= now):
            raise ValueError(f"{name} must be between {MIN_YEAR} and {now}")
    if year_from is not None and year_to is not None and int(year_from) > int(year_to):
        raise ValueError("year_from must not be after year_to")
    return (int(year_from) if year_from is not None else None,
            int(year_to) if year_to is not None else None)


def fetch_span(year_from: int, year_to: int | None) -> tuple[int, int]:
    """The years a fetch covers: ``year_to`` None is this year; at most
    ``MAX_BACKFILL_YEARS``, keeping the most recent ones."""
    end = year_to if year_to is not None else this_year()
    start = max(int(year_from), end - MAX_BACKFILL_YEARS + 1)
    return start, end


def trial_per_year(n_years: int) -> int:
    return max(1, min(TRIAL_PER_YEAR, TRIAL_TOTAL // max(1, n_years)))
