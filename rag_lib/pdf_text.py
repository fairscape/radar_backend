"""How an uploaded PDF's text is read (FUNCTIONAL_SPEC P.3).

``read_pages`` lays each page out with pdfminer, the library pdfplumber
is built on: text blocks in reading order, so a two-column page comes out
one column after the other instead of the two columns' lines interleaved
(pdfplumber's ``page.extract_text`` joins whatever sits at the same
height). ``body_from_pages`` then drops the lines repeated on most pages
(running headers and footers) and joins words broken at a line end.

What is stored (``as_paragraphs``) has each text block on one line, so a
phrase can be searched for whatever the PDF's line breaks were; each page
ends with a form feed. The abstract is read from the line layout,
before that (``find_abstract``); for a paper with none, ``mark_lead``
marks where the first paragraph starts, past the journal name, authors
and affiliations, and ``lead_text`` reads from there.

The word list (``english_words.txt``) is the alphabetic words of the
Debian ``wamerican`` list (SCOWL), lowercased; it decides whether a
hyphen at a line end is part of the word.
"""

from __future__ import annotations

import collections
import re
from functools import lru_cache
from pathlib import Path

# pdfminer keeps ligature glyphs ("eﬃcacy"); pdfplumber expands these same
# seven, and the UMLS model only knows the plain letters.
LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})

# A running header or footer: page numbers ignored, the same line of at
# most HEADER_LINE characters, with HEADER_LETTERS letters or more (so a
# table's number-only rows stay), on more than half the pages of a PDF of
# HEADER_MIN_PAGES pages or more. Where it sits is not used: pdfminer may
# put a footer anywhere in a page's reading order.
HEADER_LINE, HEADER_LETTERS, HEADER_MIN_PAGES = 80, 4, 3

ABSTRACT_START = re.compile(
    r"^\s*(?:a\s?b\s?s\s?t\s?r\s?a\s?c\s?t|summary)\b\s*[:.—–-]?\s*(.*)$", re.I)
# What follows an abstract: a keyword list, the first section, a figure
# caption, or the journal's small print. Not "Background": a structured
# abstract has one.
ABSTRACT_END = re.compile(
    r"^\s*(?:key[\s-]?words?|index\s+terms|(?:\d+|[IVX]+)?\s*[.|]?\s*introduction\b"
    r"|(?:\d+|[IVX]+)\s*[.|]?\s+[A-Z][A-Za-z ,&-]{2,60}$"
    r"|©|copyright\b|peer[\s-]review under|this is an open access|\(?citation:|received:?\s"
    r"|(?:fig\.?|figure)\s*\d|\W*correspond|e-?mail(?: address(?:es)?)?:|\S+@\S+\.\w+|contents\s*\d*\s*$"
    r"|\(J [A-Z][a-z]+ .*\d{4}|(?:\(cid:\d+\)\s*)?j \w+ \w+ \w+\.? \d{4};)", re.I)
# The heading of the keyword list that may come between the abstract's
# heading and the abstract (Elsevier's side column).
KEYWORDS = re.compile(r"^\s*(?:key[\s-]?words?|index\s+terms)", re.I)
ABSTRACT_MIN, ABSTRACT_MAX = 300, 4000
ABSTRACT_PAGES = 2
# A line that is only a number or a mark, inside an abstract, is a stray
# from the page around it (a list number in a side column); two of them
# and the reading order is not to be trusted there.
STRAY = re.compile(r"^[\s\d.,;:()*†‡∗-]*$")
# The keyword list Elsevier prints beside an abstract: its short lines are
# stepped over when it comes between the heading and the abstract.
KEYWORD_LINE = 60
# Some PDFs carry each line twice, in two overlaid text layers: more than
# this share of the abstract's 8-word runs coming twice means that.
REPEAT_SHARE, REPEAT_RUN = 0.3, 8

# The fallback when there is no abstract starts at the first paragraph:
# the first PROSE_RUN lines in a row of PROSE_LINE characters or more that
# are not an author list (a comma every few words) or an affiliation. It
# is found on the PDF's lines, before they are joined into paragraphs,
# and kept in the stored text as a LEAD line of its own.
PROSE_LINE, PROSE_RUN = 60, 3
AFFILIATION = re.compile(r"universit|department|faculty|hospital|institut|school of|@|\bdoi\b", re.I)
LEAD = "\v"
PAGE = "\f"


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    path = Path(__file__).with_name("english_words.txt")
    return frozenset(path.read_text(encoding="utf-8").split())


def clean_line(line: str) -> str:
    """Ligatures as plain letters; justified text has runs of spaces
    between words, and one is enough."""
    return re.sub(r"[ \t\u00a0]{2,}", " ", line.translate(LIGATURES)).strip()


def read_pages(path: str | Path) -> list[list[str]]:
    """Each page's lines in pdfminer's reading order; "" ends a text block."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTFigure, LTTextContainer

    def boxes(container):
        for o in container:
            if isinstance(o, LTTextContainer):
                yield o
            elif isinstance(o, LTFigure):
                yield from boxes(o)

    pages = []
    # all_texts: lay out the text inside figures (form objects) too. Some
    # PDFs put a whole page in one; without it that page reads as nothing.
    for layout in extract_pages(str(path), laparams=LAParams(all_texts=True)):
        page: list[str] = []
        for box in boxes(layout):
            page += [clean_line(line) for line in box.get_text().splitlines() if line.strip()]
            page.append("")
        pages.append(page)
    return pages


def _header_key(line: str) -> str:
    return " ".join(re.sub(r"\d+", " ", line).split()).lower()


def body_from_pages(pages: list[list[str]], log: list[str] | None = None) -> str:
    """The pages' lines with running headers and footers dropped and words
    broken at a line end joined; pages separated by a form-feed line.
    ``log`` collects one line per header dropped and per hyphen decided."""
    log = [] if log is None else log
    seen = collections.Counter(
        k for p in pages for k in {_header_key(l) for l in p if _header_key(l) and len(l) <= HEADER_LINE})
    headers = {k for k, n in seen.items()
               if len(pages) >= HEADER_MIN_PAGES and n > len(pages) / 2
               and len(re.findall(r"[a-z]", k)) >= HEADER_LETTERS}
    kept: list[str] = []
    for i, page in enumerate(pages, 1):
        for line in page:
            if _header_key(line) in headers:
                log.append(f"page {i}: dropped header/footer line {line.strip()!r}")
            else:
                kept.append(line)
        kept.append(PAGE)
    text = "\n".join(kept)

    # A word broken at the end of a line ("recruit-\nment"): joined, or the
    # hyphen kept when it is part of the word ("cross-\nsectional"). Only
    # when the letters on both sides are lowercase ("ATP-binding", "IL-34"
    # stay). Blank lines between are allowed: a word may run over a page
    # or a text block.
    flat = re.sub(r"\s+", " ", text).lower()
    words = _words()

    def join(m: re.Match) -> str:
        a, gap, b = m.group(1), m.group(2), m.group(3)
        hyphened, unbroken = f"{a}-{b}".lower(), (a + b).lower()
        if re.search(rf"\b{re.escape(hyphened)}\b", flat):
            why, out = "hyphenated elsewhere in the text", f"{a}-{b}"
        elif re.search(rf"\b{re.escape(unbroken)}\b", flat):
            why, out = "written unbroken elsewhere in the text", a + b
        elif unbroken in words:
            why, out = "a dictionary word", a + b
        elif a.lower() in words and b.lower() in words:
            why, out = "two dictionary words", f"{a}-{b}"
        else:
            why, out = "neither", a + b
        log.append(f"joined {m.group(0)!r} -> {out!r} ({why})")
        # A word run over a page: the page still ends after it.
        return out + (f"\n{PAGE}\n" if PAGE in gap else "")

    return re.sub(r"\b([A-Za-z]*[a-z])-([ \t]*\n\s*)([a-z]+)", join, text)


def find_abstract(text: str) -> str | None:
    """The abstract from the first pages of ``text`` (``body_from_pages``'
    output), or None.

    Starts at a line beginning Abstract / Summary; a keyword list right
    after the heading is stepped over. Ends before the first keyword list,
    introduction, numbered section heading, figure caption or journal
    small print. None -- rather than a wrong abstract -- when there is no
    such end within ABSTRACT_MAX characters, when it is shorter than
    ABSTRACT_MIN, when two of its lines are stray numbers, when it starts
    in the middle of a sentence, or when most of it comes twice.
    """
    lines = "\n".join(text.split(PAGE)[:ABSTRACT_PAGES]).splitlines()
    lines = [l for k, l in enumerate(lines) if not (k and l.strip() and l == lines[k - 1])]
    for i, line in enumerate(lines):
        m = ABSTRACT_START.match(line)
        if not m:
            continue
        out = [m.group(1)] if m.group(1).strip() else []
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if ABSTRACT_END.match(nxt):
                if sum(map(len, out)) >= ABSTRACT_MIN:
                    break
                if out and not KEYWORDS.match(nxt):
                    # An end too early: a short abstract, or a line from the
                    # page around it (a correspondence note) in the middle.
                    # What would follow is not the abstract.
                    out = []
                    break
                # a keyword list before the abstract: skip its short lines
                j += 1
                while j < len(lines) and len(lines[j].strip()) < KEYWORD_LINE:
                    j += 1
                out = []
                continue
            if nxt.strip():
                out.append(nxt)
            j += 1
            if sum(map(len, out)) > ABSTRACT_MAX:
                break
        else:
            continue
        while out and STRAY.match(out[-1]):                 # a footnote mark before the end
            out.pop()
        # a kept hyphen or a URL broken at the line end joins without a space
        got = re.sub(r"(?<=[-/])\n", "", "\n".join(l.strip() for l in out))
        got = re.sub(r"\s+", " ", got).strip()
        got = re.sub(r"(?:\s*\([a-z]\))+$", "", got)           # panel labels of a figure below
        got = re.sub(r"(?<=[a-z][.!?])\s?\d{1,2}$", "", got)    # a trailing footnote mark
        strays = sum(bool(STRAY.match(l)) for l in out)
        # A lowercase start means its first words were read somewhere else.
        starts_well = bool(re.match(r"[A-Z0-9\"'“(\[]", got))
        w = got.lower().split()
        runs = [" ".join(w[k:k + REPEAT_RUN]) for k in range(len(w) - REPEAT_RUN + 1)]
        repeats = len(runs) - len(set(runs)) > REPEAT_SHARE * len(runs)
        if ABSTRACT_MIN <= len(got) <= ABSTRACT_MAX and strays < 2 and starts_well and not repeats:
            return got
    return None


def _is_prose(line: str) -> bool:
    words = line.split()
    return (len(line.strip()) >= PROSE_LINE and line.count(",") <= len(words) / 4
            and not AFFILIATION.search(line))


def mark_lead(text: str) -> str:
    """``text`` (``body_from_pages``' output) with a LEAD line before its
    first paragraph; unchanged when it has none."""
    lines = text.split("\n")
    for i in range(len(lines) - PROSE_RUN + 1):
        if all(_is_prose(l) for l in lines[i:i + PROSE_RUN]):
            return "\n".join(lines[:i] + ["", LEAD, ""] + lines[i:])
    return text


def as_paragraphs(text: str) -> str:
    """``text`` (``body_from_pages``' output) as it is stored: each text
    block on one line, blocks separated by a blank line, each page ended
    by a form feed on a line of its own."""
    pages = []
    for page in text.split(PAGE):
        out = []
        for block in re.split(r"\n[ \t]*\n", page):
            block = block.strip("\n")
            if block == LEAD:
                out.append(LEAD)
                continue
            block = re.sub(r"(?<=[-/])\n", "", block)
            if block.strip():
                out.append(re.sub(r"[ \t]*\n[ \t]*", " ", block).strip())
        pages.append("\n\n".join(out))
    while pages and not pages[-1]:
        pages.pop()
    return "".join(f"{p}\n{PAGE}\n" for p in pages)


def lead_text(body: str | None) -> str:
    """Where a stored PDF text's reading starts when it has no abstract:
    after its LEAD line (``mark_lead``), past the journal name, authors and
    affiliations. A text without one -- read before this reading
    (pdfplumber's), not a PDF's (a Prosopia summary), or with no
    paragraph -- is returned as it is."""
    body = body or ""
    at = body.find(f"\n{LEAD}\n")
    if at >= 0:
        return body[at + len(LEAD) + 2:].lstrip("\n")
    return body[len(LEAD) + 1:] if body.startswith(LEAD + "\n") else body


def paper_text(abstract: str | None, body: str | None, limit: int) -> str:
    """The text that stands for what a paper is about: its abstract, or
    ``limit`` characters from where its body's reading starts."""
    if abstract and abstract.strip():
        return abstract
    return lead_text(body)[:limit]
