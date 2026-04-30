"""Embedding input construction.

``build_embedding_input`` produces the single string fed to an embedder
for a given Paper. Per the build spec, the baseline SPECTER2 regime is
title + abstract only; the Phase 1B expansion includes MeSH, keywords,
substances, and body when available. Labeled section markers let the
transformer attend per-field; a drop-priority truncation keeps the
result under the 512-token ceiling.

Drop order (applied from the first field that fits):
    body -> substances -> keywords -> mesh
Title and abstract are always preserved.

Token counting: we use a cheap whitespace-split heuristic by default,
which is conservative versus SPECTER2's BPE. Callers that need a
faithful token count can pass ``token_count_fn=len_bpe`` or similar.
"""

from __future__ import annotations

from typing import Callable

from .paper import Paper


DEFAULT_MAX_TOKENS = 512


def _ws_token_count(s: str) -> int:
    return len(s.split())


def build_embedding_input(
    paper: Paper,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    *,
    token_count_fn: Callable[[str], int] = _ws_token_count,
) -> str:
    """Assemble a labeled, truncated text blob for an embedder.

    Format::

        TITLE: {title}

        ABSTRACT: {abstract}

        MESH: {mesh joined by '; '}

        KEYWORDS: {keywords joined}

        SUBSTANCES: {substances joined}

        BODY: {body_text}

    Sections with empty content are omitted. If the assembled blob
    exceeds ``max_tokens`` per ``token_count_fn``, sections are dropped
    in the order body -> substances -> keywords -> mesh, and if still
    over budget the abstract is truncated token-wise. Title is never
    truncated.
    """
    sections = _build_sections(paper)
    while _tokens(sections, token_count_fn) > max_tokens and _has_droppable(sections):
        for key in ("BODY", "SUBSTANCES", "KEYWORDS", "MESH"):
            if sections.get(key):
                sections[key] = ""
                break

    text = _render(sections)
    if token_count_fn(text) > max_tokens:
        # Last-resort: truncate the abstract by whitespace tokens.
        text = _truncate_abstract(sections, max_tokens, token_count_fn)
    return text


def _build_sections(paper: Paper) -> dict[str, str]:
    mesh = "; ".join(paper.mesh) if paper.mesh else ""
    keywords = "; ".join(paper.keywords) if paper.keywords else ""
    substances = "; ".join(paper.substances) if paper.substances else ""
    return {
        "TITLE": paper.title or "",
        "ABSTRACT": paper.abstract or "",
        "MESH": mesh,
        "KEYWORDS": keywords,
        "SUBSTANCES": substances,
        "BODY": paper.body_text or "",
    }


def _render(sections: dict[str, str]) -> str:
    parts = [f"{k}: {v}" for k, v in sections.items() if v]
    return "\n\n".join(parts)


def _tokens(sections: dict[str, str], fn: Callable[[str], int]) -> int:
    return fn(_render(sections))


def _has_droppable(sections: dict[str, str]) -> bool:
    return any(sections.get(k) for k in ("BODY", "SUBSTANCES", "KEYWORDS", "MESH"))


def _truncate_abstract(
    sections: dict[str, str],
    max_tokens: int,
    fn: Callable[[str], int],
) -> str:
    """Whitespace-truncate the abstract until the full blob fits."""
    title_section = f"TITLE: {sections['TITLE']}"
    abstract_words = sections["ABSTRACT"].split()
    # Binary search on word count for the longest abstract that still fits.
    lo, hi = 0, len(abstract_words)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_abs = " ".join(abstract_words[:mid])
        candidate = title_section + "\n\nABSTRACT: " + candidate_abs
        if fn(candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best or title_section
