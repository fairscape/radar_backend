"""Topics once read from the text of an uploaded PDF OpenAlex does not have.

Until 2026-11-04 such a paper (``local:<hash>``) had OpenAlex topics
guessed for it by embedding its title and abstract against the topic
index, stored in ``papers.topics_json`` with ``source: pdf_content`` and
shown as "PDF" in the topic lists. The user dropped them (11-04): "对于
unpublished，只保留umls concept和claude api的结果" -- an unpublished paper's
topics come only from its UMLS concepts and from Claude's research topics
(FUNCTIONAL_SPEC P.4).

Nothing writes them now. Rows written before stay in the database (no data
is rewritten for a new feature), so every reader of ``topics_json`` goes
through ``seed_topics_raw``, which reads such a row as having no topics.
"""

from __future__ import annotations

import json

SOURCE = "pdf_content"


def is_from_content(raw: str | None) -> bool:
    """Whether a ``topics_json`` value is one read from a PDF's text."""
    if not raw:
        return False
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        return False
    return isinstance(d, dict) and d.get("source") == SOURCE


def seed_topics_raw(raw: str | None) -> str | None:
    """``topics_json`` as it counts now: None for one read from a PDF's text."""
    return None if is_from_content(raw) else raw
