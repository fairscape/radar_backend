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
from dataclasses import dataclass
from pathlib import Path


DOI_REGEX = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


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

    Uses pdfplumber. Title heuristic: first non-empty line on page 1
    (reasonable on biomedical preprint layouts; falls back to the
    filename if extraction is empty). DOI heuristic: first regex hit
    anywhere in the body, then the embedded metadata's ``/doi`` if
    absent from text.
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
    with pdfplumber.open(str(path)) as pdf:
        meta = dict(pdf.metadata or {})
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")

    body = "\n\n".join(pages_text).strip()

    # Title: first non-empty line of the first page, trimmed.
    title = ""
    if pages_text:
        for line in pages_text[0].splitlines():
            line = line.strip()
            if line:
                title = line
                break
    if not title:
        title = path.stem

    # DOI: body regex first, then metadata keys that sometimes carry it.
    doi = _find_doi(body)
    if doi is None:
        for k in ("doi", "DOI", "/doi"):
            v = meta.get(k)
            if isinstance(v, str):
                m = _find_doi(v)
                if m:
                    doi = m
                    break

    return PdfIngestRecord(
        path=str(path),
        title=title,
        body_text=body,
        doi=doi,
        n_pages=len(pages_text),
    )


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


def _find_doi(text: str) -> str | None:
    if not text:
        return None
    m = DOI_REGEX.search(text)
    if not m:
        return None
    # Trim trailing punctuation that the greedy regex sometimes catches.
    return m.group(0).rstrip(".,;:)")
