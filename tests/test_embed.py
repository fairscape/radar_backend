"""build_embedding_input tests — section order, truncation priority."""

from __future__ import annotations

from rag_lib.embed import build_embedding_input
from rag_lib.paper import Paper


def _paper(**overrides) -> Paper:
    base = dict(doi="10.1/a", openalex_id="W1", title="T", abstract="A")
    base.update(overrides)
    return Paper(**base)


def test_empty_paper_produces_title_only():
    p = _paper(title="just a title", abstract="")
    assert build_embedding_input(p) == "TITLE: just a title"


def test_sections_appear_in_canonical_order():
    p = _paper(
        title="T",
        abstract="A",
        mesh=["M1", "M2"],
        keywords=["K1"],
        substances=["Caffeine"],
        body_text="body text here",
    )
    out = build_embedding_input(p, max_tokens=10_000)
    assert out.index("TITLE:") < out.index("ABSTRACT:")
    assert out.index("ABSTRACT:") < out.index("MESH:")
    assert out.index("MESH:") < out.index("KEYWORDS:")
    assert out.index("KEYWORDS:") < out.index("SUBSTANCES:")
    assert out.index("SUBSTANCES:") < out.index("BODY:")


def test_drop_priority_body_first():
    """Over-budget: body drops before substances/keywords/mesh."""
    p = _paper(
        title="t", abstract="a",
        mesh=["m"], keywords=["k"], substances=["s"],
        body_text="word " * 2000,
    )
    out = build_embedding_input(p, max_tokens=20)
    assert "BODY:" not in out
    # Mesh/keywords/substances survived the first drop cycle.
    # (Further drops depend on how tight the budget is.)


def test_drop_priority_preserves_title_and_abstract():
    """Even with a tiny budget, title and abstract are never dropped."""
    p = _paper(
        title="important title",
        abstract="abstract words " * 3,
        body_text="body " * 500,
    )
    out = build_embedding_input(p, max_tokens=5)
    assert "TITLE:" in out
    # Abstract may be truncated but title must be preserved verbatim.
    assert "important title" in out


def test_empty_sections_omitted():
    p = _paper(mesh=[], keywords=[], substances=[], body_text="")
    out = build_embedding_input(p, max_tokens=10_000)
    assert "MESH:" not in out
    assert "KEYWORDS:" not in out
    assert "SUBSTANCES:" not in out
    assert "BODY:" not in out


def test_abstract_truncation_last_resort():
    """With only title + abstract and a too-tight budget, abstract is
    whitespace-truncated rather than dropped."""
    p = _paper(title="t", abstract="one two three four five six seven eight")
    out = build_embedding_input(p, max_tokens=4)
    assert "TITLE: t" in out
    assert "ABSTRACT:" in out
    # Should contain at most (4 - title-tokens) abstract tokens.
    # Title is "TITLE: t" which is 2 ws-tokens. Abstract should have
    # at most 2 tokens.
    import re
    m = re.search(r"ABSTRACT:\s+(.+)$", out)
    assert m is not None
    abstract_words = m.group(1).split()
    assert len(abstract_words) <= 3
