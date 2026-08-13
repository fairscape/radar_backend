"""PDF ingestion for Profile.from_pdfs.

Extracts title, body text, and a best-effort DOI from a local PDF,
producing a lightweight record (``PdfIngestRecord``) that
``Profile.from_pdfs`` then runs through OpenAlex enrichment + embedding,
the same path CSV-driven builds use.

pdfplumber is imported lazily so ``rag_lib`` remains importable without
the ``phase1b`` extras installed.

What this module does NOT do:
  - Chroma vector index writes. The build spec calls for a Chroma-backed
    vault for later RAG chat (Phase 4A); for Phase 1B the Profile JSON
    itself is the corpus, with ``Paper.embeddings`` carrying vectors and
    ``Paper.local_path`` carrying PDF references. Adding Chroma here
    without a RAG consumer would just be overhead.
  - OpenAlex enrichment, embedding, or topic aggregation. Those happen
    in ``Profile.from_pdfs`` after ``ingest_pdf`` returns.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# The leading guard is a negative lookbehind rather than ``\b``. PDF text
# extraction routinely drops the space in front of the identifier, giving
# "DigitalObjectIdentifier10.1109/ACCESS.2023.3244712" — and between "r"
# and "1" there is no word boundary, so ``\b`` refuses to match. The
# paper's own DOI would then be skipped and the first *cleanly spaced*
# one found instead, which lives in the reference list and belongs to a
# different paper. The lookbehind still refuses to start mid-number.
DOI_REGEX = re.compile(
    r"(?<![\d.])10\.\d{4,9}/[-._;()/:A-Z0-9]{1,80}", re.IGNORECASE
)

# Metadata keys that carry the DOI. Publishers rarely use the obvious
# one: the IEEE and Elsevier PDFs here bury it in ``Subject``, e.g.
# "IEEE Access;2023;11; ;10.1109/ACCESS.2023.3244712".
_META_DOI_KEYS = (
    "doi", "DOI", "/doi", "/DOI",
    "Subject", "/Subject",
    "WPS-ARTICLEDOI", "wps-articledoi",
)


@dataclass
class PdfIngestRecord:
    """What ``ingest_pdf`` returns — intentionally narrower than Paper.

    A Paper gets built in ``Profile.from_pdfs`` by merging this with
    OpenAlex enrichment. Keeping the two shapes separate means we don't
    accidentally leak PDF-specific metadata into the OpenAlex paper_from_work
    path or vice versa.
    """

    path: str
    title: str                          # best-effort from first non-empty line
    body_text: str                      # all extractable text, joined pages
    doi: str | None = None              # first DOI found in the text / metadata
    n_pages: int = 0


def ingest_pdf(path: str | Path) -> PdfIngestRecord:
    """Extract title, body, and DOI from a local PDF.

    Uses pdfplumber. Title: the largest-font text block on page 1, with
    publisher boilerplate rejected and wrapped lines rejoined — see
    :func:`_title_from_page`, reconciled against the embedded metadata
    title; falls back to the first non-boilerplate line, then to the
    filename. DOI: page 1, then the metadata, then the rest of the body —
    see :func:`_extract_doi` for why that order is load-bearing.
    """
    path = Path(path)
    try:
        import pdfplumber  # type: ignore
    except ImportError as e:
        raise ImportError(
            "ingest_pdf requires the phase1b extras. "
            "Install with: pip install -e '.[dev,phase1b]'"
        ) from e

    pages_text: list[str] = []
    meta: dict = {}
    layout_title = ""
    with pdfplumber.open(str(path)) as pdf:
        meta = dict(pdf.metadata or {})
        for i, page in enumerate(pdf.pages):
            pages_text.append(page.extract_text() or "")
            if i == 0:
                layout_title = _title_from_page(page)

    body = "\n\n".join(pages_text).strip()
    first_page = pages_text[0] if pages_text else ""

    title = (
        _reconcile_title(layout_title, _meta_title(meta), first_page)
        or _title_from_lines(first_page)
        or path.stem
    )

    doi = _extract_doi(first_page, body, meta)

    return PdfIngestRecord(
        path=str(path),
        title=title,
        body_text=body,
        doi=doi,
        n_pages=len(pages_text),
    )


def fetch_pdf_text(url: str, *, timeout: float = 30.0) -> str:
    """Download a PDF from ``url`` and return its extracted body text.

    Writes to a temp file, delegates to :func:`ingest_pdf` for parsing,
    then deletes the temp file. Raises on network or parse errors so the
    caller can decide whether to fall back. Nothing is persisted to the
    papers DB or vault — selectors that fetch on demand treat the
    returned text as transient.
    """
    import os
    import tempfile
    import urllib.request

    req = urllib.request.Request(
        url, headers={"User-Agent": "radar-fulltext/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()

    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        record = ingest_pdf(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return record.body_text


def compute_file_hash(data: bytes | str | Path) -> str:
    """sha256 of file bytes, hex-encoded.

    Accepts a ``bytes`` payload (the upload path's `UploadFile.read()`
    result) or a path-like the function reads. Centralized so the
    upload route and any post-hoc admin tooling agree on the hash
    convention used by ``papers.file_hash``.
    """
    h = hashlib.sha256()
    if isinstance(data, (bytes, bytearray)):
        h.update(data)
        return h.hexdigest()
    p = Path(data)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# Lines that sit above, below or beside the title on a published PDF and
# are never part of it. The old heuristic — "first non-empty line of page
# one" — took whichever of these the publisher happened to put first, and
# on the three IEEE / JMIR papers used to develop this it was wrong every
# time: a running head, a "Received … accepted …" stamp, a journal name.
# A wrong title is not cosmetic. When the DOI lookup misses,
# ``_enrich_via_openalex`` matches on the title instead, so garbage here
# resolves the seed to an unrelated paper — and in the reranker's article
# mode the title heads the query.
_BOILERPLATE_PATTERNS = (
    re.compile(r"\bvol\.?\s*\d", re.I),               # "VOL.25,NO.4,APRIL2021"
    re.compile(r"^\s*received\b.{0,60}\baccepted\b", re.I),
    re.compile(r"digital\s*object\s*identifier|^\s*doi\b", re.I),
    re.compile(r"^\s*downloaded\s+from\b", re.I),
    re.compile(r"\bet\s+al\.?\b", re.I),              # running head w/ author
    re.compile(r"\b(journal|proceedings|transactions|symposium)\b.*\b(of|on)\b", re.I),
    re.compile(
        r"^\s*(review|article|research\s+article|original\s+(research|article)"
        r"|editorial|commentary|letter|case\s+report|perspective"
        r"|brief\s+report|systematic\s+review)\s*$",
        re.I,
    ),
    re.compile(r"^\W*\d{1,4}\W*$"),                   # bare page number
    re.compile(r"^\s*(open\s+access|this\s+work\s+is\s+licensed"
               r"|creative\s+commons|©|copyright)", re.I),
)

# Author bylines end the title block. Degree suffixes and IEEE membership
# grades are the reliable markers; superscript affiliation digits survive
# extraction as bare digits glued to a surname.
_AUTHOR_PATTERNS = (
    re.compile(r",\s*(ph\.?d|m\.?d|m\.?sc|b\.?sc|m\.?s|dr)\b", re.I),
    re.compile(r"\b(student|senior|life|fellow)?\s*member,\s*ieee\b", re.I),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),           # contact email
    re.compile(r"^\s*and\s+[A-Z]"),                   # "and Pantelis Georgiou"
    re.compile(r"[A-Za-z]{3}\d{1,2}\s*[,;]"),         # "Woldaregay1, MSc;"
)

# pdfplumber reports character positions in points; lines within this
# many points of each other belong to the same visual row.
_LINE_TOLERANCE = 2.0
# Font sizes within this many points of the largest are "the same size" —
# superscripts and the odd italic run vary slightly inside one heading.
_SIZE_TOLERANCE = 0.6
# A title spilling past this many rendered lines is almost certainly a
# run of body text that happens to share the heading's font size.
_MAX_TITLE_LINES = 6
_MIN_TITLE_CHARS = 12
# A column gutter is far wider than a word space. Whichever of these two
# is larger separates "beside the title" from "part of the title".
_MIN_GUTTER = 12.0
# In multiples of the heading's own font size: a candidate row further
# below the last title row than this belongs to something else.
_MAX_TITLE_GAP = 3.0
# Fraction of page 1 the title can appear in, measured from the top.
_TITLE_ZONE = 0.5


# PDF extraction routinely loses the spaces inside a running head — the
# IEEE header this was found on arrives as one token,
# "IEEEJOURNALOFBIOMEDICALANDHEALTHINFORMATICS,VOL.25,...". Any rule
# above that depends on whitespace silently stops firing on those, so
# these run against the whitespace-stripped text as well. Each fragment
# has to be one no real title would contain: "etal" is deliberately
# absent, since "fetal" and "metal" contain it.
_BOILERPLATE_SQUASHED = (
    "journalof",
    "proceedingsof",
    "transactionson",
    "conferenceon",
    "digitalobjectidentifier",
    "downloadedfrom",
    "thisworkislicensed",
    "creativecommons",
    "allrightsreserved",
    # Elsevier's header block, which is set in larger type than the
    # article title on at least some of their journals.
    "contentslistsavailable",
    "sciencedirect",
    "journalhomepage",
    "wwwelsevier",
)


def _is_boilerplate(text: str) -> bool:
    if any(p.search(text) for p in _BOILERPLATE_PATTERNS):
        return True
    squashed = re.sub(r"\s+", "", text).lower()
    if any(frag in squashed for frag in _BOILERPLATE_SQUASHED):
        return True
    if "received" in squashed and "accepted" in squashed:
        return True
    # Volume/issue/page furniture ("42, 688–698") and rendering
    # artefacts ("]]") are not headings. Requiring some letters, and
    # mostly letters, rejects both without naming either.
    letters = sum(ch.isalpha() for ch in text)
    return letters < 4 or letters < 0.5 * len(text.replace(" ", ""))


def _looks_like_authors(text: str) -> bool:
    return any(p.search(text) for p in _AUTHOR_PATTERNS)


def _page_lines(page) -> list[tuple[float, str, float]]:
    """Group a page's characters into ``(top, text, max_font_size)`` rows.

    ``page.extract_text()`` throws the font sizes away, and the size is
    the one signal that reliably separates a title from the boilerplate
    around it on an academic PDF. Sorting each row by ``x0`` rebuilds
    reading order for two-column layouts, where the raw char list is not
    necessarily in it.
    """
    try:
        chars = list(page.chars or [])
    except Exception:  # noqa: BLE001 — a malformed page must not fail ingest
        return []

    rows: dict[int, list] = {}
    for ch in chars:
        try:
            key = int(round(float(ch["top"]) / _LINE_TOLERANCE))
            float(ch["size"]), float(ch["x0"]), float(ch["x1"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.setdefault(key, []).append(ch)

    out: list[tuple[float, str, float]] = []
    for key in sorted(rows):
        line = sorted(rows[key], key=lambda c: float(c["x0"]))
        for segment in _split_columns(line):
            text = _join_chars(segment)
            if not text:
                continue
            out.append((key * _LINE_TOLERANCE, text, _row_font_size(segment)))
    return out


def _split_columns(line: list) -> list[list]:
    """Break one row at its column gutters.

    Grouping characters by vertical position alone merges anything that
    shares a baseline, and the older journal layouts set the journal name
    in a left margin beside the title. The merged row then reads
    "Journal of Applied Does infectious disease influence…" and is
    rejected wholesale as boilerplate, taking half the title with it. A
    gutter is much wider than a word space, so splitting on the wide gaps
    recovers the margin and the title as separate pieces.
    """
    segments: list[list] = []
    current: list = []
    prev_x1: float | None = None
    for c in line:
        x0, size = float(c["x0"]), float(c["size"])
        if prev_x1 is not None and x0 - prev_x1 > max(_MIN_GUTTER, 1.5 * size):
            if current:
                segments.append(current)
            current = []
        current.append(c)
        prev_x1 = float(c["x1"]) if prev_x1 is None else max(prev_x1, float(c["x1"]))
    if current:
        segments.append(current)
    return segments


def _join_chars(line: list) -> str:
    """Concatenate a row's characters, restoring the word gaps.

    Most PDFs encode spaces as horizontal displacement rather than as
    space characters, so ``page.chars`` simply has no space glyphs to
    concatenate — the naive join yields
    ``ImpactofNutritionalFactorsinBloodGlucose``. ``extract_text()``
    reinserts them from the geometry and we have to do the same:
    a gap wider than a fraction of the current font size is a space.
    """
    parts: list[str] = []
    prev_x1: float | None = None
    prev_size: float = 0.0
    for c in line:
        text = str(c.get("text") or "")
        if not text:
            continue
        x0, size = float(c["x0"]), float(c["size"])
        if prev_x1 is not None and x0 - prev_x1 > max(1.0, 0.22 * (prev_size or size)):
            parts.append(" ")
        parts.append(text)
        prev_x1, prev_size = float(c["x1"]), size
    return " ".join("".join(parts).split())


def _row_font_size(line: list) -> float:
    """The row's *modal* character size, deliberately not its maximum.

    Journals open a section with a drop cap two or three lines tall. It
    lands in the same row as the first line of body text, and taking the
    row's max size would then rate that row as the largest type on the
    page — beating the actual title. One oversized glyph cannot outvote
    the forty body-sized ones around it.
    """
    counts = Counter(round(float(c["size"]) * 2) / 2 for c in line)
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _title_from_page(page) -> str:
    """The largest-font run of lines on page 1, boilerplate excluded.

    Publishers set the title in the biggest type on the page; everything
    competing with it for the top of the page — running heads, date
    stamps, the DOI line — is smaller. Taking the largest surviving font
    and then reading consecutive rows at that size also rejoins a title
    that wrapped across three lines, which the line-at-a-time heuristic
    could never do.
    """
    rows = _page_lines(page)
    if not rows:
        return ""

    # Only the upper part of the page is eligible. Footers carry
    # rendering artefacts that can be set larger than the title — the
    # JMIR papers stamp a 20pt "XSLFO" at the bottom, which outranked
    # their own 18pt heading. No amount of blocklisting keeps up with
    # that; a title is simply never in the footer.
    try:
        height = float(getattr(page, "height", 0) or 0)
    except (TypeError, ValueError):
        height = 0.0
    if height <= 0:
        height = max((top for top, _, _ in rows), default=0.0) or 1.0
    zone = height * _TITLE_ZONE

    lines = [
        (top, text, size)
        for top, text, size in rows
        if top <= zone and not _is_boilerplate(text)
    ]
    if not lines:
        return ""

    biggest = max(size for _, _, size in lines)
    cutoff = biggest - _SIZE_TOLERANCE

    parts: list[str] = []
    last_top = 0.0
    for top, text, size in lines:
        if size < cutoff:
            # Skipped, not stopped. A margin column, a publisher
            # watermark or a volume line can sit *between* two rows of a
            # wrapped title; breaking on the first small row would return
            # only the title's first half.
            continue
        if parts and top - last_top > _MAX_TITLE_GAP * size:
            break            # too far down to still be the same heading
        if _looks_like_authors(text):
            break
        if text in parts:
            # Some typesetters leave two overlapping copies of the text
            # layer, so each line of the heading is drawn twice a few
            # points apart and the naive join reads "A A B B".
            continue
        parts.append(text)
        last_top = top
        if len(parts) >= _MAX_TITLE_LINES:
            break

    title = " ".join(parts).strip()
    return title if len(title) >= _MIN_TITLE_CHARS else ""


_META_TITLE_JUNK = re.compile(
    r"^\s*(untitled|unknown|no\s+title|microsoft\s+word|document\s*\d*"
    r"|manuscript|pii\b|doi\b)"
    r"|\.(doc|docx|pdf|tex|rtf|qxd|indd)\s*$",
    re.I,
)
# Publishers embed light inline markup in the metadata title:
# "…Using Deep Reinforcement Learning: An <italic>In Silico</italic> Validation".
_INLINE_MARKUP = re.compile(r"</?[a-z][a-z0-9:-]*\s*/?>", re.I)
# How much of the metadata title has to reappear on page 1 before we
# trust it. Long enough to be specific, short enough to survive a line
# break falling inside it.
_META_PROBE_CHARS = 20
# When the metadata title is a subset of the rendered one, how much of
# the rendered text it must still account for. Below this the metadata is
# more likely truncated than the rendering is padded.
_META_SUBSET_MIN_RATIO = 0.6


def _fold(text: str) -> str:
    """Lowercase alphanumerics only, with ligatures decomposed.

    Page text arrives with typographic ligatures — "inﬂuence",
    "efﬁcacy" — that the metadata spells out, so a raw comparison
    between the two fails on exactly the papers whose metadata is worth
    having. NFKC turns the ligature back into its letters.
    """
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", text).lower())


def _meta_title(meta: dict) -> str:
    """The publisher's embedded title, cleaned, or "" if it is junk.

    Populated correctly this is the best source available — no layout
    inference, and right even on the margin-column and oversized-masthead
    layouts that defeat the font heuristic. But it is just as often
    "Untitled", a typesetting filename, or the journal name, so it is
    never used unchecked.
    """
    raw = meta.get("Title") or meta.get("/Title") or meta.get("title")
    if not isinstance(raw, str):
        return ""
    title = " ".join(_INLINE_MARKUP.sub(" ", raw).split())
    if len(title) < _MIN_TITLE_CHARS:
        return ""
    if _META_TITLE_JUNK.search(title) or _is_boilerplate(title):
        return ""
    return title


def _reconcile_title(layout: str, meta: str, page_text: str) -> str:
    """Combine the rendered title and the embedded one.

    Neither source wins outright, and the reason is visible in the two
    ways they fail. The layout pass reads the real typography, so its
    punctuation is right, but a watermark or a margin column printed
    across the heading can cost it the first half. The metadata is
    complete, but it is sometimes a sanitized filename — one paper here
    has "diseases_ A systematic review" where the page reads
    "diseases: A systematic review".

    So: prefer what was rendered, and defer to the metadata whenever one
    strictly contains the other — either direction is a defect the other
    source does not have. Layout inside metadata is the truncation case.
    Metadata inside layout is the opposite: the rendering picked up
    something adjacent to the heading and set in the same type, such as
    the "OPEN" access badge Nature prints level with the title's first
    line, giving "open Early detection of type 2 diabetes…".

    The length guard is for metadata that is itself truncated, which
    would otherwise look exactly like the badge case and quietly shorten
    a correct title. Used on its own the metadata still has to show up on
    page 1 before it is believed.
    """
    if layout and meta:
        folded_layout, folded_meta = _fold(layout), _fold(meta)
        if folded_layout and folded_layout != folded_meta:
            if folded_layout in folded_meta:
                return meta
            if (
                folded_meta in folded_layout
                and len(folded_meta) >= _META_SUBSET_MIN_RATIO * len(folded_layout)
            ):
                return meta
        return layout
    if layout:
        return layout
    if meta and _fold(meta)[:_META_PROBE_CHARS] in _fold(page_text):
        return meta
    return ""


def _title_from_lines(page_text: str) -> str:
    """Fallback for pages that expose no usable character metrics.

    Same idea without the font signal: skip the boilerplate, then keep
    joining rows until a byline or a sentence end says the heading is
    over.
    """
    lines = [ln.strip() for ln in page_text.splitlines()]
    parts: list[str] = []
    for line in lines:
        if not line:
            if parts:
                break
            continue
        if _is_boilerplate(line):
            if parts:
                break
            continue
        if _looks_like_authors(line):
            break
        parts.append(line)
        if len(parts) >= _MAX_TITLE_LINES or line.endswith("."):
            break

    title = " ".join(parts).strip()
    return title if len(title) >= _MIN_TITLE_CHARS else (parts[0] if parts else "")


def _extract_doi(first_page: str, body: str, meta: dict) -> str | None:
    """Find the paper's own DOI, in descending order of trustworthiness.

    Order is the whole point. Scanning the full body and taking the first
    hit — what this used to do — reads the reference list whenever the
    front matter yields nothing, and every entry there is a *different*
    paper's DOI. That is worse than finding none at all: enrichment does
    an exact lookup on a DOI, so the seed silently acquires another
    paper's title, abstract and topics, with nothing to suggest anything
    went wrong. One of the papers this was found on came back tagged with
    a citation from its own bibliography.

    So: page 1 first, where a paper prints its own identifier and no
    references have started; then the embedded metadata; and only then
    the rest of the body.
    """
    return (
        _find_doi(first_page)
        or _doi_from_metadata(meta)
        or _find_doi(body)
    )


def _doi_from_metadata(meta: dict) -> str | None:
    for key in _META_DOI_KEYS:
        value = meta.get(key)
        if isinstance(value, str):
            found = _find_doi(value)
            if found:
                return found
    return None


def _find_doi(text: str) -> str | None:
    if not text:
        return None
    m = DOI_REGEX.search(text)
    if not m:
        return None
    # Trim trailing punctuation that the greedy regex sometimes catches.
    return m.group(0).rstrip(".,;:)")
