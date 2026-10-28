"""OpenAlex topics for an uploaded PDF that OpenAlex does not have.

A PDF the vault cannot match to a published work is stored as
``local:<hash>`` with no topics, so it added nothing to what its interest
searches for. Its UMLS concepts did reach the topic list, but each concept
is mapped on its own name: a passing mention ("Evaluation", "Proteins")
maps as confidently as the subject, and UMLS covers only biomedicine.

Here the paper's own text -- its title and the start of its body, where
the abstract is -- is embedded whole and compared with the OpenAlex topic
names (the index the UMLS mapper uses). Measured on 21 uploaded PDFs that
OpenAlex *does* have, against the topics OpenAlex assigned them: at least
one of the 3 nearest topics was OpenAlex's for 17 of 21, against 4 of 21
for the UMLS mapping of the same papers, which found no topic at all for 4.

The topics are stored in ``papers.topics_json`` in OpenAlex's own shape,
so ``Profile.aggregate_topic_filters`` counts them like any seed's topics
-- they do not compete with UMLS for its additions -- and ``source``
marks them for the topic list.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger("rag_lib.content_topics")

SOURCE = "pdf_content"
TOP_K = 3
# Below this a topic is not about the paper. The nearest topic of a real
# paper measured 0.59-0.81 (median 0.67).
MIN_SIMILARITY = 0.55
# Title plus about the abstract's length of body; more did not score better.
BODY_CHARS = 1500


def topics_from_text(title: str | None, body: str | None, *, settings: Any) -> list[dict]:
    """The nearest OpenAlex topics to ``title`` + the start of ``body``.

    Each is a ``Topic.to_dict`` shape with ``score`` the cosine. The index
    carries subfield/field/domain by name only, so they have no id and are
    not counted at those levels. Raises if the index or embedder fails.
    """
    start = " ".join((body or "").split())[:BODY_CHARS]
    text = f"{(title or '').strip()}. {start}".strip(". ")
    if len(text) < 20:
        return []
    from .embedders import get_embedder
    from .umls.cache import get_topic_index

    model = settings.RADAR_UMLS_EMBEDDING_MODEL
    index = get_topic_index(str(settings.RADAR_UMLS_CACHE_DIR))
    index.check_compatible(model)
    vec = np.array(get_embedder(model)(text), dtype=np.float32)
    index.check_compatible(model, dim=int(vec.shape[-1]))
    node = lambda name: {"id": None, "display_name": name} if name else None  # noqa: E731
    out: list[dict] = []
    names: set[str] = set()
    # OpenAlex has topics that share a name ("Marine and coastal ecosystems"
    # twice); one of each name, so the next topic gets the place.
    for t, sim in index.search(vec, top_k=TOP_K * 3, min_similarity=MIN_SIMILARITY):
        name = (t.get("display_name") or "").strip()
        if not name or name.lower() in names:
            continue
        names.add(name.lower())
        out.append({
            "id": t["id"],
            "display_name": name,
            "subfield": node(t.get("subfield")),
            "field": node(t.get("field")),
            "domain": node(t.get("domain")),
            "score": round(sim, 4),
        })
        if len(out) == TOP_K:
            break
    return out


def topics_json(topics: list[dict]) -> str:
    """``papers.topics_json`` for content-derived topics (an empty list too:
    it records that the text was read, so it is not tried again)."""
    return json.dumps({"primary_topic": None, "topics": topics, "source": SOURCE})


def ids_from_content(raw: str | None) -> set[str]:
    """Topic ids in a ``topics_json`` that came from the paper's text."""
    if not raw:
        return set()
    try:
        d = json.loads(raw)
    except ValueError:
        return set()
    if not isinstance(d, dict) or d.get("source") != SOURCE:
        return set()
    return {t["id"] for t in d.get("topics") or [] if isinstance(t, dict) and t.get("id")}


def try_topics_json(title: str | None, body: str | None, *, settings: Any) -> str | None:
    """``topics_json`` for an unmatched paper, or None when it cannot be read
    now (index missing, embedder down) so a later aggregation retries."""
    if not getattr(settings, "RADAR_CONTENT_TOPICS_ENABLED", True):
        return None
    try:
        return topics_json(topics_from_text(title, body, settings=settings))
    except Exception as exc:  # noqa: BLE001 -- best effort; the upload must not fail on it
        log.warning("content_topics.failed", reason=str(exc)[:200])
        return None
