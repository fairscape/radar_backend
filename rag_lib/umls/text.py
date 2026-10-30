"""The text UMLS extraction reads for a paper.

Kept apart from the extraction itself (no spaCy import) so that the papers
repo can decide, on every write, whether a paper's extraction input
changed -- a seed imported without an abstract and later uploaded as a PDF
gets its body text, and its concepts must be re-extracted from it.
"""

from __future__ import annotations

from ..pdf_text import paper_text

# Body text cap; see rag_lib/api/services/vault.py for why there is one.
UMLS_TEXT_LIMIT = 20_000


def umls_input_text(
    title: str | None,
    abstract: str | None,
    body_text: str | None,
) -> str:
    """Pick the text to run UMLS extraction over: title, then abstract.

    The title is not decoration here. Abstracts introduce an
    abbreviation once and use it throughout, and the SciSpacy linker can
    only expand one when the expansion appears in the same text — so a
    paper whose abstract says "T2DM" nine times yields no diabetes
    concept at all. Titles spell the subject out. Prepending it recovered
    ``C0011860 Diabetes Mellitus, Non-Insulin-Dependent`` at 0.981 on a
    paper that had extracted nothing but "RF", "Fasting" and "Score".

    That failure is invisible downstream: the concept is simply absent,
    so no ranking or threshold further along can bring it back.
    """
    head = (title or "").strip()
    # Without an abstract, the body from where its reading starts (past
    # the journal name, authors and affiliations of a PDF).
    body = paper_text(abstract, body_text, UMLS_TEXT_LIMIT)
    return f"{head}\n\n{body}" if head and body else (head or body)
