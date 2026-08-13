"""L1 — PDF ingestion: rag_lib.vault + the vault service's UMLS text pick.

``test_api_vault.py`` patches ``ingest_pdf`` away, so until now nothing
exercised the parsing itself. pdfplumber is a phase1b extra and is not
installed in the compliance environment, so these tests inject a stub
module into ``sys.modules`` — that keeps the title / DOI / page-joining
heuristics under test without a binary fixture or the dependency.

Several tests here pin behaviour that is *heuristic*, not obviously
correct (the title is whatever the first non-empty line on page 1 is).
They are written to fail loudly if that heuristic changes, because a
wrong title propagates all the way into the reranker query in article
mode — a mis-titled seed paper is what produced the July "obesity"
failure, where an astrophysics title on an obesity paper dragged the
whole ranking into cosmology.
"""

from __future__ import annotations

import hashlib
import sys
import types

import pytest

from rag_lib.vault import (
    DOI_REGEX,
    PdfIngestRecord,
    _find_doi,
    compute_file_hash,
    ingest_pdf,
)


# -----------------------------------------------------------------------------
# pdfplumber stub
# -----------------------------------------------------------------------------


PAGE_HEIGHT = 792.0  # US Letter, in points — what pdfplumber reports


class _FakePage:
    def __init__(self, text: str | None, chars: list[dict] | None = None):
        self._text = text
        # A page with no character metrics is the realistic default here;
        # ``ingest_pdf`` then falls back to its line-based heuristic.
        # Tests that exercise the font-size path pass ``chars`` in.
        self.chars = chars if chars is not None else []
        self.height = PAGE_HEIGHT

    def extract_text(self):
        return self._text


class _FakePdf:
    def __init__(
        self,
        pages: list[str | None],
        metadata: dict | None,
        chars: list[dict] | None = None,
    ):
        self.pages = [
            _FakePage(t, chars if i == 0 else None)
            for i, t in enumerate(pages)
        ]
        self.metadata = metadata

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_pdfplumber(monkeypatch):
    """Install a stub ``pdfplumber`` and return a setter for its content.

    Usage::

        set_pdf(pages=["page one text"], metadata={"doi": "10.1/x"})
    """
    state: dict = {"pages": [], "metadata": {}, "chars": None}

    module = types.ModuleType("pdfplumber")

    def _open(_path):
        return _FakePdf(state["pages"], state["metadata"], state["chars"])

    module.open = _open  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pdfplumber", module)

    def set_pdf(*, pages, metadata=None, chars=None):
        state["pages"] = pages
        state["metadata"] = metadata
        state["chars"] = chars

    return set_pdf


def layout(rows, *, char_width=5.0, gap_ratio=0.35, left=50.0):
    """Build a ``page.chars`` list from ``(text, font_size, top)`` rows.

    Mirrors how a real PDF stores a page: no space glyphs at all, just a
    horizontal gap where a space belongs, which is why ``ingest_pdf`` has
    to reconstruct word breaks from the geometry. The gap scales with the
    font size, as it does in a real document — a fixed gap would fall
    below the detection threshold at large sizes and silently produce
    glued words in the fixture rather than in the code under test.
    """
    out: list[dict] = []
    for text, size, top in rows:
        x = left
        for i, word in enumerate(text.split(" ")):
            if i:
                x += gap_ratio * size
            for ch in word:
                out.append({
                    "text": ch, "size": size, "top": top,
                    "x0": x, "x1": x + char_width,
                })
                x += char_width
    return out


# -----------------------------------------------------------------------------
# ingest_pdf — title heuristic
# -----------------------------------------------------------------------------


def test_title_is_first_non_empty_line_of_page_one(fake_pdfplumber, tmp_path):
    fake_pdfplumber(pages=["\n  \nDeep Learning for Glycemic Control\nAuthors et al."])
    rec = ingest_pdf(tmp_path / "paper.pdf")
    assert rec.title == "Deep Learning for Glycemic Control"


def test_title_falls_back_to_filename_when_page_one_is_blank(
    fake_pdfplumber, tmp_path
):
    """A scanned PDF with no text layer yields nothing to read a title from."""
    fake_pdfplumber(pages=["", "   ", None])
    rec = ingest_pdf(tmp_path / "scan-of-something.pdf")
    assert rec.title == "scan-of-something"


def test_title_falls_back_when_pdf_has_zero_pages(fake_pdfplumber, tmp_path):
    fake_pdfplumber(pages=[])
    rec = ingest_pdf(tmp_path / "empty.pdf")
    assert rec.title == "empty"
    assert rec.n_pages == 0
    assert rec.body_text == ""


@pytest.mark.parametrize("boilerplate", [
    "Downloaded from https://academic.oup.com on 12 August 2026",
    "IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 25, NO. 4, APRIL 2021 1223",
    "JOURNAL OF MEDICAL INTERNET RESEARCH Woldaregay et al",
    "Received 14 January 2023, accepted 5 February 2023, date of publication 13 February 2023.",
    "Digital Object Identifier 10.1109/ACCESS.2023.3244712",
    "Review",
    "1223",
    "© 2023 IEEE",
])
def test_title_skips_publisher_boilerplate(fake_pdfplumber, tmp_path, boilerplate):
    """Every string here is a real first line from a published PDF.

    The first three come from the papers this was developed against; the
    old "first non-empty line" heuristic returned each of them as the
    title, three times out of three.
    """
    fake_pdfplumber(pages=[f"{boilerplate}\nThe Real Title Of The Paper\n"])
    rec = ingest_pdf(tmp_path / "paper.pdf")
    assert rec.title == "The Real Title Of The Paper"


def test_title_stops_at_the_author_byline(fake_pdfplumber, tmp_path):
    fake_pdfplumber(pages=[
        "A Perfectly Reasonable Title\n"
        "Taiyu Zhu , Student Member, IEEE, Kezhi Li , Member, IEEE\n"
        "Abstract—This paper describes...\n"
    ])
    rec = ingest_pdf(tmp_path / "paper.pdf")
    assert rec.title == "A Perfectly Reasonable Title"


def test_title_joins_a_wrapped_heading(fake_pdfplumber, tmp_path):
    """Titles wrap across two or three rendered lines; taking one line
    truncates them mid-phrase."""
    fake_pdfplumber(pages=[
        "Basal Glucose Control in Type 1 Diabetes\n"
        "Using Deep Reinforcement Learning:\n"
        "An In Silico Validation\n"
        "Taiyu Zhu , Student Member, IEEE\n"
    ])
    rec = ingest_pdf(tmp_path / "paper.pdf")
    assert rec.title == (
        "Basal Glucose Control in Type 1 Diabetes "
        "Using Deep Reinforcement Learning: An In Silico Validation"
    )


# -----------------------------------------------------------------------------
# ingest_pdf — title via font size (the primary path when a page exposes
# character metrics)
# -----------------------------------------------------------------------------


def test_title_is_the_largest_type_on_the_page(fake_pdfplumber, tmp_path):
    """Font size is the signal, not position.

    The running head is physically first; the title is merely biggest.
    """
    fake_pdfplumber(
        pages=["ignored — the chars below are what matters"],
        chars=layout([
            ("IEEE TRANSACTIONS ON SOMETHING VOL. 25", 7.0, 40.0),
            ("A Genuinely Large Heading", 15.0, 100.0),
            ("Some body text that follows the heading", 9.0, 200.0),
        ]),
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == "A Genuinely Large Heading"


def test_title_is_not_stolen_by_a_drop_cap(fake_pdfplumber, tmp_path):
    """A section-opening drop cap is the biggest glyph on the page.

    It shares a row with the first line of body text, so scoring a row by
    its largest character rates that row above the title — which is
    exactly what happened on the IEEE paper this was found on, yielding
    the title "DIABETESisachronicdiseasewhichaffectsmillionsof".
    """
    chars = layout([
        ("Basal Glucose Control in Type 1 Diabetes", 14.0, 100.0),
        ("DIABETES is a chronic disease affecting millions", 9.0, 300.0),
    ])
    for c in chars:                       # the drop cap: one huge glyph
        if c["top"] == 300.0:
            c["size"] = 30.0
            break

    fake_pdfplumber(pages=["ignored"], chars=chars)
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == "Basal Glucose Control in Type 1 Diabetes"


def test_title_restores_word_gaps(fake_pdfplumber, tmp_path):
    """PDFs encode spaces as displacement, so ``page.chars`` has none.

    Concatenating the glyphs verbatim produced
    ``ImpactofNutritionalFactorsinBloodGlucose``.
    """
    fake_pdfplumber(
        pages=["ignored"],
        chars=layout([("Impact of Nutritional Factors", 16.0, 100.0)]),
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == "Impact of Nutritional Factors"


def test_font_path_still_rejects_boilerplate_set_in_large_type(
    fake_pdfplumber, tmp_path
):
    """Some publishers set the journal name larger than the title."""
    fake_pdfplumber(
        pages=["ignored"],
        chars=layout([
            ("JOURNAL OF MEDICAL INTERNET RESEARCH Woldaregay et al", 20.0, 40.0),
            ("The Actual Paper Title", 14.0, 100.0),
        ]),
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == "The Actual Paper Title"


def test_falls_back_to_lines_when_the_page_has_no_char_metrics(
    fake_pdfplumber, tmp_path
):
    fake_pdfplumber(pages=["Downloaded from somewhere\nThe Actual Title Here\n"])
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == "The Actual Title Here"


def test_footer_artefacts_cannot_outrank_the_title(fake_pdfplumber, tmp_path):
    """JMIR stamps a 20pt "XSLFO" in the page footer.

    It is larger than their own 18pt heading and is real letters, so no
    blocklist entry short of naming it would help. Only the upper part of
    the page is eligible.
    """
    fake_pdfplumber(
        pages=["ignored"],
        chars=layout([
            ("Data-Driven Blood Glucose Pattern Classification", 18.0, 98.0),
            ("XSLFO", 20.0, 804.0),
        ]),
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == "Data-Driven Blood Glucose Pattern Classification"


def test_margin_column_does_not_swallow_half_the_title(fake_pdfplumber, tmp_path):
    """Older journal layouts run the journal name down the left margin.

    It shares a baseline with the title, so grouping by vertical position
    alone produces "Journal of Applied Does infectious disease…", which
    is then rejected as boilerplate — silently costing the title's first
    half. Splitting the row at the gutter keeps them apart.
    """
    chars = layout([("Journal of Applied", 9.0, 28.0)], left=40.0)
    chars += layout([("Does infectious disease influence the efficacy", 18.0, 28.0)],
                    left=200.0)
    chars += layout([("Ecology 2005", 9.0, 40.0)], left=40.0)
    chars += layout([("of marine protected areas? A theoretical framework", 18.0, 50.0)],
                    left=200.0)

    fake_pdfplumber(pages=["ignored"], chars=chars)
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == (
        "Does infectious disease influence the efficacy "
        "of marine protected areas? A theoretical framework"
    )


# -----------------------------------------------------------------------------
# ingest_pdf — title from embedded metadata
# -----------------------------------------------------------------------------


def _one_line(text, size=16.0, top=100.0):
    return layout([(text, size, top)])


def test_metadata_completes_a_title_the_layout_truncated(fake_pdfplumber, tmp_path):
    """The metadata is used exactly where the rendering came up short."""
    full = "Does infectious disease influence the efficacy of marine protected areas"
    fake_pdfplumber(
        pages=["ignored"],
        chars=_one_line("of marine protected areas"),
        metadata={"Title": full},
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == full


def test_rendered_title_wins_over_a_sanitized_metadata_title(
    fake_pdfplumber, tmp_path
):
    """Same words, worse punctuation — the metadata looks filename-derived.

    Folding to alphanumerics makes the two equal, so neither strictly
    contains the other and the rendered text keeps its colon.
    """
    fake_pdfplumber(
        pages=["ignored"],
        chars=_one_line("Predicting glucose: a systematic review"),
        metadata={"Title": "Predicting glucose_ a systematic review"},
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == "Predicting glucose: a systematic review"


def test_metadata_strips_inline_markup(fake_pdfplumber, tmp_path):
    fake_pdfplumber(
        pages=["Deep Reinforcement Learning: An In Silico Validation"],
        metadata={"Title": "Deep Reinforcement Learning: An <italic>In Silico</italic> Validation"},
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == "Deep Reinforcement Learning: An In Silico Validation"


@pytest.mark.parametrize("junk", [
    "Untitled",
    "Microsoft Word - draft7.doc",
    "Journal of Applied Ecology",
    "manuscript_final.pdf",
    "",
])
def test_junk_metadata_titles_are_rejected(fake_pdfplumber, tmp_path, junk):
    fake_pdfplumber(
        pages=["ignored"],
        chars=_one_line("The Rendered Title Wins"),
        metadata={"Title": junk},
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == "The Rendered Title Wins"


def test_metadata_drops_a_badge_the_layout_absorbed(fake_pdfplumber, tmp_path):
    """Nature prints an "OPEN" access badge level with the title's first
    line and in the same type, so the font pass cannot tell them apart:
    "open Early detection of type 2 diabetes…". The metadata is the same
    string without the badge, so it is a strict subset of the rendering.
    """
    clean = "Early detection of type 2 diabetes mellitus using machine learning"
    fake_pdfplumber(
        pages=["ignored"],
        chars=_one_line(f"open {clean}", size=26.0, top=152.0),
        metadata={"Title": clean},
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == clean


def test_a_truncated_metadata_title_does_not_shorten_a_good_one(
    fake_pdfplumber, tmp_path
):
    """The badge case and a truncated metadata title look identical —
    metadata strictly inside layout — so length decides between them."""
    full = "Prediction of Type 2 Diabetes using Machine Learning Classification Methods"
    fake_pdfplumber(
        pages=["ignored"],
        chars=_one_line(full),
        metadata={"Title": "Prediction of Type 2"},   # truncated at 20 chars
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == full


def test_a_doubled_text_layer_is_not_read_twice(fake_pdfplumber, tmp_path):
    """Some typesetters leave two overlapping copies of the text, drawn a
    few points apart, so each heading line appears twice and the rows
    interleave as A A B B."""
    a = "Prediction of Type 2 Diabetes using Machine Learning"
    b = "Classification Methods"
    chars = (
        _one_line(a, size=17.0, top=150.0)
        + _one_line(a, size=17.0, top=162.0)
        + _one_line(b, size=17.0, top=170.0)
        + _one_line(b, size=17.0, top=182.0)
    )
    fake_pdfplumber(pages=["ignored"], chars=chars)
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == f"{a} {b}"


def test_metadata_alone_must_appear_on_page_one(fake_pdfplumber, tmp_path):
    """With no layout to corroborate it, an unrelated metadata title is
    not trusted — some PDFs carry the title of whatever document the
    template was cloned from."""
    fake_pdfplumber(
        pages=["A page about something else entirely\n"],
        metadata={"Title": "The Title Of A Completely Different Paper"},
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.title == "A page about something else entirely"


# -----------------------------------------------------------------------------
# ingest_pdf — body / pages
# -----------------------------------------------------------------------------


def test_body_joins_pages_with_blank_line_and_strips(fake_pdfplumber, tmp_path):
    fake_pdfplumber(pages=["  first  ", "second"])
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.body_text == "first  \n\nsecond"


def test_pages_with_no_text_layer_count_toward_n_pages(fake_pdfplumber, tmp_path):
    """``extract_text()`` returning None must not drop the page from the count.

    n_pages is surfaced to the user as the document's length, so an
    image-only page still counts.
    """
    fake_pdfplumber(pages=["text", None, "more"])
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.n_pages == 3
    assert rec.body_text == "text\n\n\n\nmore"


def test_record_carries_the_path_as_given(fake_pdfplumber, tmp_path):
    fake_pdfplumber(pages=["Title"])
    target = tmp_path / "sub" / "p.pdf"
    rec = ingest_pdf(target)
    assert rec.path == str(target)
    assert isinstance(rec, PdfIngestRecord)


# -----------------------------------------------------------------------------
# ingest_pdf — DOI
# -----------------------------------------------------------------------------


def test_doi_found_in_body(fake_pdfplumber, tmp_path):
    fake_pdfplumber(pages=["Title\nhttps://doi.org/10.1038/s41586-024-07123-4 rest"])
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.doi == "10.1038/s41586-024-07123-4"


def test_doi_is_found_when_extraction_glued_it_to_the_label(
    fake_pdfplumber, tmp_path
):
    """Verbatim from an IEEE Access paper: the space is simply gone.

    A ``\\b`` in front of ``10\\.`` cannot match here — "r" and "1" are
    both word characters, so there is no boundary between them — and the
    scan then continued into the reference list.
    """
    fake_pdfplumber(pages=[
        "Received14January2023,accepted5February2023\n"
        "DigitalObjectIdentifier10.1109/ACCESS.2023.3244712\n"
        "Impact of Nutritional Factors\n"
    ])
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.doi == "10.1109/ACCESS.2023.3244712"


def test_page_one_doi_beats_a_citation_in_the_bibliography(
    fake_pdfplumber, tmp_path
):
    """The failure this ordering exists to prevent.

    Enrichment looks a DOI up exactly, so a reference's DOI resolves the
    seed to that other paper — correct-looking metadata for the wrong
    article, with no signal that anything is off.
    """
    fake_pdfplumber(pages=[
        "DigitalObjectIdentifier10.1109/ACCESS.2023.3244712\nTitle Of This Paper\n",
        "REFERENCES\n"
        "[1] X. Y., ''A type 1 diabetes simulator,'' J.DiabetesSci.Technol., "
        "10.1109/JIOT.2022.3143375. vol.8,no.1\n",
    ])
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.doi == "10.1109/ACCESS.2023.3244712"


def test_doi_comes_from_metadata_subject_when_page_one_has_none(
    fake_pdfplumber, tmp_path
):
    """Publishers rarely use a key called "doi".

    Both IEEE papers here carry it inside ``Subject``, formatted as
    "IEEE Access;2023;11; ;10.1109/ACCESS.2023.3244712".
    """
    fake_pdfplumber(
        pages=["A Title With No Identifier On It\n", "10.9999/from-the-references\n"],
        metadata={"Subject": "IEEE Access;2023;11; ;10.1109/ACCESS.2023.3244712"},
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.doi == "10.1109/ACCESS.2023.3244712"


def test_body_is_still_searched_when_page_one_and_metadata_have_nothing(
    fake_pdfplumber, tmp_path
):
    """JMIR prints the DOI in a footer, not on the front page, and its
    metadata is empty — the body scan is the only thing that finds it."""
    fake_pdfplumber(pages=[
        "JOURNAL OF MEDICAL INTERNET RESEARCH\nData-Driven Blood Glucose\n",
        "doi:10.2196/11030\n",
    ])
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.doi == "10.2196/11030"


def test_doi_does_not_start_inside_a_longer_number(fake_pdfplumber, tmp_path):
    """The lookbehind replaced ``\\b``; it still has to refuse this."""
    fake_pdfplumber(pages=["accession 9999910.1234/notadoi here\n"])
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.doi is None


def test_doi_falls_back_to_metadata_when_body_has_none(fake_pdfplumber, tmp_path):
    fake_pdfplumber(pages=["Title with no identifier"], metadata={"doi": "10.1234/abc.def"})
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.doi == "10.1234/abc.def"


def test_doi_metadata_slash_prefixed_key_is_read(fake_pdfplumber, tmp_path):
    """pdfplumber surfaces some XMP keys with the raw ``/Name`` spelling."""
    fake_pdfplumber(pages=["Title"], metadata={"/doi": "10.1234/xyz"})
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.doi == "10.1234/xyz"


def test_doi_ignores_non_string_metadata_values(fake_pdfplumber, tmp_path):
    """XMP values arrive as bytes or PDF objects often enough to matter."""
    fake_pdfplumber(pages=["Title"], metadata={"doi": b"10.1234/abc"})
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.doi is None


def test_doi_survives_missing_metadata(fake_pdfplumber, tmp_path):
    fake_pdfplumber(pages=["Title"], metadata=None)
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.doi is None


def test_body_doi_wins_over_metadata_doi(fake_pdfplumber, tmp_path):
    fake_pdfplumber(
        pages=["Title 10.1111/in-body"],
        metadata={"doi": "10.2222/in-metadata"},
    )
    rec = ingest_pdf(tmp_path / "p.pdf")
    assert rec.doi == "10.1111/in-body"


# -----------------------------------------------------------------------------
# _find_doi / DOI_REGEX
# -----------------------------------------------------------------------------


def test_find_doi_on_empty_input():
    assert _find_doi("") is None
    assert _find_doi("no identifier here") is None


def test_find_doi_trims_sentence_punctuation():
    assert _find_doi("see 10.1038/nature12373.") == "10.1038/nature12373"
    assert _find_doi("(10.1038/nature12373);") == "10.1038/nature12373"


def test_find_doi_is_case_insensitive_on_the_suffix():
    assert _find_doi("10.1038/NATURE-12373") == "10.1038/NATURE-12373"


def test_find_doi_returns_the_first_match():
    assert _find_doi("10.1000/aaa and 10.2000/bbb") == "10.1000/aaa"


def test_doi_regex_rejects_a_too_short_registrant():
    """Registrant codes are 4-9 digits; ``10.1/x`` is malformed."""
    assert DOI_REGEX.search("10.1/x") is None


def test_find_doi_trailing_paren_of_a_real_doi_is_lost():
    """Pinned known imprecision.

    A DOI may legitimately end in ``)``. ``rstrip('.,;:)')`` cannot tell
    that from a closing bracket the regex over-captured, so it strips it.
    Documented rather than fixed: truncating is the safer of the two
    failure modes for a lookup key.
    """
    assert _find_doi("10.1234/foo(bar)") == "10.1234/foo(bar"


# -----------------------------------------------------------------------------
# compute_file_hash
# -----------------------------------------------------------------------------


def test_hash_of_bytes_matches_sha256():
    data = b"%PDF-1.4 fake"
    assert compute_file_hash(data) == hashlib.sha256(data).hexdigest()


def test_hash_of_path_matches_hash_of_its_bytes(tmp_path):
    """The upload route hashes bytes; admin tooling hashes paths.

    Dedup on ``papers.file_hash`` only works if the two agree.
    """
    data = b"x" * 200_000  # spans several of the 64 KiB read chunks
    p = tmp_path / "f.pdf"
    p.write_bytes(data)
    assert compute_file_hash(p) == compute_file_hash(data)


def test_hash_accepts_bytearray():
    assert compute_file_hash(bytearray(b"abc")) == compute_file_hash(b"abc")


def test_hash_of_empty_input_is_stable():
    assert compute_file_hash(b"") == hashlib.sha256(b"").hexdigest()


# -----------------------------------------------------------------------------
# missing dependency
# -----------------------------------------------------------------------------


def test_ingest_pdf_without_pdfplumber_names_the_extra(monkeypatch, tmp_path):
    """The 503 the upload route returns quotes this message verbatim."""
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    monkeypatch.setattr(
        "builtins.__import__",
        _raising_import("pdfplumber"),
    )
    with pytest.raises(ImportError, match=r"phase1b"):
        ingest_pdf(tmp_path / "p.pdf")


def _raising_import(blocked: str):
    real_import = __import__

    def _import(name, *args, **kwargs):
        if name == blocked:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    return _import


# -----------------------------------------------------------------------------
# services.vault._umls_input_text — which text the extractor sees
# -----------------------------------------------------------------------------


def test_umls_input_leads_with_the_title():
    """The title is what makes the paper's subject extractable.

    An abstract introduces an abbreviation once and then uses it
    throughout, and the linker can only expand one whose expansion is in
    the same text. A paper whose abstract says "T2DM" nine times yields
    no diabetes concept at all; its title says "type 2 diabetes
    mellitus" in full.
    """
    from rag_lib.api.services.vault import _umls_input_text

    out = _umls_input_text("Early detection of type 2 diabetes mellitus",
                           "T2DM screening with ML", None)
    assert out.startswith("Early detection of type 2 diabetes mellitus")
    assert "T2DM screening with ML" in out


def test_umls_input_prefers_the_abstract_over_the_body():
    from rag_lib.api.services.vault import _umls_input_text

    assert _umls_input_text(None, "the abstract", "the body") == "the abstract"


def test_umls_input_falls_back_to_body_when_abstract_is_blank():
    from rag_lib.api.services.vault import _umls_input_text

    for blank in (None, "", "   \n "):
        assert _umls_input_text(None, blank, "the body") == "the body"


def test_umls_input_truncates_the_body_to_the_documented_limit():
    from rag_lib.api.services.vault import _UMLS_TEXT_LIMIT, _umls_input_text

    out = _umls_input_text(None, None, "b" * (_UMLS_TEXT_LIMIT * 3))
    assert len(out) == _UMLS_TEXT_LIMIT


def test_umls_input_does_not_truncate_the_abstract():
    """Pinned asymmetry.

    body_text is capped because a full PDF trips spaCy's 1,000,000-char
    ``nlp.max_length`` guard (ValueError E088), which ``_try_extract_umls``
    swallows — the paper silently ends up with no UMLS data. The abstract
    is not capped. Normally that is fine, but OpenAlex abstracts are
    rebuilt from an inverted index and a malformed record can carry an
    entire body there, which puts the uncapped branch back on the same
    cliff.
    """
    from rag_lib.api.services.vault import _UMLS_TEXT_LIMIT, _umls_input_text

    long_abstract = "a" * (_UMLS_TEXT_LIMIT * 3)
    assert _umls_input_text(None, long_abstract, None) == long_abstract


def test_umls_input_with_nothing_available_is_empty():
    from rag_lib.api.services.vault import _umls_input_text

    assert _umls_input_text(None, None, None) == ""
    assert _umls_input_text("", "", "") == ""


def test_umls_input_is_title_only_when_that_is_all_there_is():
    from rag_lib.api.services.vault import _umls_input_text

    assert _umls_input_text("A Title", None, None) == "A Title"
