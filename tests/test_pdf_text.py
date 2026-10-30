"""How an uploaded PDF's text is read (rag_lib.pdf_text, FUNCTIONAL_SPEC P.3-P.4).

The layout cases (two columns, text inside a form object) are read from
real PDF files, written here byte by byte so the test needs no PDF
library beyond pdfminer itself; the line rules are tested on lines.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rag_lib import pdf_text


# --- a minimal PDF writer -----------------------------------------------------

def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text_ops(lines) -> str:
    """``lines``: ``(x, y, text)`` in points from the bottom left."""
    return "".join(f"BT /F1 10 Tf {x} {y} Td ({_esc(t)}) Tj ET\n" for x, y, t in lines)


def write_pdf(path: Path, pages, *, form_pages=()) -> Path:
    """One page per entry of ``pages`` (lists of ``(x, y, text)``); a page
    whose index is in ``form_pages`` draws its text from a form XObject."""
    objs: list[bytes] = []

    def add(body: str | bytes) -> int:
        objs.append(body.encode("latin-1") if isinstance(body, str) else body)
        return len(objs)

    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    pages_id = len(objs) + 1 + 0     # placeholder, fixed below
    add("")                          # the Pages object, filled in at the end
    kids = []
    for i, lines in enumerate(pages):
        ops = _text_ops(lines)
        res = f"<< /Font << /F1 {font} 0 R >> >>"
        if i in form_pages:
            stream = ops.encode("latin-1")
            form = add(b"<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources "
                       + res.encode() + b" /Length " + str(len(stream)).encode() + b" >>\nstream\n"
                       + stream + b"\nendstream")
            ops, res = "/Fm0 Do\n", f"<< /XObject << /Fm0 {form} 0 R >> >>"
        stream = ops.encode("latin-1")
        content = add(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        kids.append(add(f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
                        f"/Resources {res} /Contents {content} 0 R >>"))
    objs[pages_id - 1] = (f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] "
                          f"/Count {len(kids)} >>").encode()
    catalog = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def column(x: float, top: float, texts) -> list:
    return [(x, top - 12 * k, t) for k, t in enumerate(texts)]


LEFT = ["The left column starts here and", "continues with its second line of", "text before it ends here."]
RIGHT = ["The right column is another text", "that must come after the left", "column, never interleaved with it."]


# --- reading the page ---------------------------------------------------------

def test_two_columns_come_out_one_after_the_other(tmp_path):
    pdf = write_pdf(tmp_path / "two.pdf", [column(50, 700, LEFT) + column(330, 700, RIGHT)])
    text = " ".join(pdf_text.as_paragraphs(pdf_text.body_from_pages(pdf_text.read_pages(pdf))).split())
    assert " ".join(LEFT) in text and " ".join(RIGHT) in text
    assert text.index(LEFT[-1]) < text.index(RIGHT[0])


def test_text_inside_a_form_object_is_read(tmp_path):
    pdf = write_pdf(tmp_path / "form.pdf", [column(50, 700, LEFT)], form_pages={0})
    lines = pdf_text.read_pages(pdf)[0]
    assert LEFT[0] in lines and LEFT[2] in lines


def test_ligatures_become_plain_letters_and_space_runs_one_space():
    assert pdf_text.clean_line("  eﬃcacy  of ﬁsh  oil ") == "efficacy of fish oil"
    assert pdf_text.clean_line("ﬀ ﬂ ﬄ ﬅ ﬆ") == "ff fl ffl st st"


def test_other_characters_are_left_alone():
    assert pdf_text.clean_line("α-synuclein — 5 µm, naïve") == "α-synuclein — 5 µm, naïve"


# --- headers and footers ------------------------------------------------------

BODIES = ["Microglia respond first.", "T cells follow later.", "Antibodies are tested.", "Trials are reviewed."]


def _pages(n):
    return [[f"Immunity 59, March 10, 2026 {520 + i}", BODIES[i % 4], ""] for i in range(n)]


def test_a_running_header_on_most_pages_is_dropped(tmp_path):
    log: list[str] = []
    text = pdf_text.body_from_pages(_pages(4), log)
    assert "Immunity 59" not in text
    assert all(b in text for b in BODIES)
    assert sum("dropped header/footer" in l for l in log) == 4


def test_no_header_is_dropped_from_a_pdf_under_three_pages():
    assert "Immunity 59" in pdf_text.body_from_pages(_pages(2))


def test_a_number_only_line_repeated_on_every_page_stays():
    pages = [["1 2 3 4", f"Row {i} of the table.", ""] for i in range(4)]
    assert pdf_text.body_from_pages(pages).count("1 2 3 4") == 4


def test_a_long_repeated_line_stays():
    long = "This sentence is longer than eighty characters and appears on every single page here."
    pages = [[long, f"Other text {i}.", ""] for i in range(4)]
    assert pdf_text.body_from_pages(pages).count(long) == 4


def test_a_line_on_half_the_pages_or_fewer_stays():
    pages = [["Running title", "a", ""], ["Running title", "b", ""], ["c", ""], ["d", ""]]
    assert pdf_text.body_from_pages(pages).count("Running title") == 2


# --- words broken at a line end -----------------------------------------------

def _joined(*lines):
    return pdf_text.body_from_pages([list(lines) + [""]])


def test_a_dictionary_word_is_joined():
    assert "recruitment of cells" in _joined("the recruit-", "ment of cells")


def test_hyphenated_elsewhere_keeps_the_hyphen():
    out = _joined("a cross-", "sectional design and a", "cross-sectional study")
    assert out.startswith("a cross-sectional design and a")


def test_unbroken_elsewhere_is_joined():
    assert "microglia" in _joined("the micro-", "glia and the microglia")


def test_two_dictionary_words_keep_the_hyphen():
    assert "a fine-tuned model" in _joined("a fine-", "tuned model")


def test_neither_is_joined():
    assert "Rosenbaumzqx" in _joined("Rosenbaum-", "zqx said")


def test_an_uppercase_letter_on_either_side_is_not_touched():
    out = _joined("the ATP-", "binding site and IL-", "34 levels")
    assert "ATP-\nbinding" in out and "IL-\n34" in out


def test_a_word_broken_over_a_page_is_joined_and_the_page_still_ends():
    pages = [["the recruit-", ""], ["ment of cells", ""], ["page three", ""]]
    text = pdf_text.body_from_pages(pages)
    assert "recruitment" in text and text.count(pdf_text.PAGE) == 3
    assert text.split(pdf_text.PAGE)[2].strip() == "page three"


# --- what is stored -----------------------------------------------------------

def test_a_block_is_one_line_and_a_kept_hyphen_or_url_joins_without_a_space():
    stored = pdf_text.as_paragraphs("a fine-\ntuned model at https://example.org/\npath\nend\n\nnext block\n\f\n")
    assert stored == "a fine-tuned model at https://example.org/path end\n\nnext block\n\f\n"


# --- the abstract -------------------------------------------------------------

ABSTRACT = ("Alzheimer's disease is the most common cause of dementia. Here we review how the "
            "innate and the adaptive immune systems shape its course, from microglia to T cells, "
            "and what this means for treatment. We discuss the genetic risk factors that point to "
            "immunity, the cells that carry them, the signals they exchange in the brain and in "
            "the blood, and the trials that target them.")
SENT = textwrap.wrap(ABSTRACT, 70)


def _page_text(*lines):
    return pdf_text.body_from_pages([list(lines) + [""]])


def test_abstract_from_its_heading_to_the_keywords():
    text = _page_text("A Journal 12 (2026)", "Abstract", *SENT, "Keywords: immunity; dementia")
    assert pdf_text.find_abstract(text) == ABSTRACT


@pytest.mark.parametrize("heading", ["ABSTRACT", "A B S T R A C T", "Abstract—", "Abstract:", "Summary"])
def test_abstract_headings(heading):
    text = _page_text(heading, *SENT, "1. Introduction")
    assert pdf_text.find_abstract(text).startswith("Alzheimer's disease is the most")


def test_abstract_text_on_the_heading_line():
    text = _page_text("Abstract—" + SENT[0], *SENT[1:], "Index Terms—dementia")
    assert pdf_text.find_abstract(text).startswith("Alzheimer's disease is the most")


@pytest.mark.parametrize("end", [
    "Introduction", "1 Introduction", "I. INTRODUCTION", "2. Related Work", "Figure 1. Overview",
    "Fig. 2 The model", "© 2026 Elsevier Ltd.", "Copyright 2026", "This is an open access article",
    "Received: 3 March 2026", "*Correspondence: someone", "E-mail: a@b.org", "someone@example.org",
    "(J Med Internet Res 2019;21(5):e1)", "Peer review under responsibility of X",
])
def test_abstract_ends_before(end):
    text = _page_text("Abstract", *SENT, end, "More text that is not the abstract at all.")
    got = pdf_text.find_abstract(text)
    assert got is not None and "not the abstract" not in got


def test_background_does_not_end_a_structured_abstract():
    text = _page_text("Abstract", "Background: " + SENT[0], *SENT[1:], "Methods: we did things.", "Keywords: x")
    got = pdf_text.find_abstract(text)
    assert got.startswith("Background:") and "Methods: we did things." in got


def test_a_keyword_column_before_the_abstract_is_stepped_over():
    text = _page_text("A B S T R A C T", "Keywords:", "Dementia", "Microglia", *SENT, "1. Introduction")
    got = pdf_text.find_abstract(text)
    assert got.startswith("Alzheimer's disease") and "Microglia" not in got[:20]


def test_trailing_footnote_mark_and_panel_labels_are_removed():
    text = _page_text("Abstract", *SENT[:-1], SENT[-1] + "1", "(a) (b)", "Keywords: x")
    assert pdf_text.find_abstract(text) == ABSTRACT


@pytest.mark.parametrize("lines", [
    ["Abstract", "Too short to be an abstract.", "Keywords: x"],                     # under 300
    ["Abstract", *(SENT * 20)],                                                      # no end within 4,000
    ["Abstract", "and so " + SENT[0], *SENT[1:], "Keywords: x"],                     # lowercase start
    ["Abstract", SENT[0], "12", *SENT[1:3], "13", *SENT[3:], "Keywords: x"],         # two stray lines
])
def test_not_found_rather_than_wrong(lines):
    assert pdf_text.find_abstract(pdf_text.body_from_pages([lines + [""]])) is None


def test_a_line_read_twice_in_a_row_counts_once():
    text = _page_text("Abstract", *[s for s in SENT for _ in (0, 1)], "Keywords: x")
    assert pdf_text.find_abstract(text) == ABSTRACT


def test_interleaved_double_layer_is_not_found():
    doubled = [s for pair in zip(SENT, SENT) for s in pair]
    doubled = [doubled[k] for k in range(len(doubled)) if k % 2 == 0] + SENT     # A..Z then A..Z again
    text = _page_text("Abstract", *doubled, "Keywords: x")
    assert pdf_text.find_abstract(text) is None


def test_only_the_first_two_pages_are_looked_at():
    pages = [["Title", ""], ["Some text", ""], ["Abstract", *SENT, "Keywords: x", ""]]
    assert pdf_text.find_abstract(pdf_text.body_from_pages(pages)) is None


def test_no_heading_no_abstract():
    assert pdf_text.find_abstract(_page_text(*SENT, "1. Introduction")) is None


# --- the fallback: where the reading starts -----------------------------------

PROSE = ["This is the first paragraph of the paper and it runs over a whole line of text.",
         "It goes on for a second line, which is just as long as the first one was here.",
         "And a third line closes the run that tells the reader a paragraph started now."]


def test_lead_skips_journal_authors_and_affiliations():
    text = _page_text("Journal of Things 12 (2026) 1-10", "Ann Author, Bob Bauthor, Cy Cauthor, Di Dauthor",
                      "Department of Medicine, University of Somewhere, 22903 Charlottesville, USA",
                      *PROSE)
    stored = pdf_text.as_paragraphs(pdf_text.mark_lead(text))
    assert pdf_text.lead_text(stored).startswith("This is the first paragraph")
    assert pdf_text.paper_text(None, stored, 30) == "This is the first paragraph of"
    assert pdf_text.paper_text("An abstract.", stored, 30) == "An abstract."


def test_lead_marker_is_its_own_line_and_the_text_is_otherwise_unchanged():
    text = _page_text("Journal", *PROSE)
    stored = pdf_text.as_paragraphs(pdf_text.mark_lead(text))
    assert f"\n{pdf_text.LEAD}\n" in stored
    assert " ".join(stored.replace(pdf_text.LEAD, " ").split()) == " ".join(pdf_text.as_paragraphs(text).split())


def test_text_without_a_marker_is_returned_as_it_is():
    old = "pdfplumber text\nJournal\n\nUniversity of X\n" + " ".join(PROSE)
    assert pdf_text.lead_text(old) == old
    assert pdf_text.lead_text(None) == ""
    assert pdf_text.lead_text("A Prosopia summary.") == "A Prosopia summary."


def test_no_paragraph_no_marker():
    text = _page_text("Short", "lines", "only")
    assert pdf_text.mark_lead(text) == text


def test_the_word_list_is_bundled():
    words = pdf_text._words()
    assert "recruitment" in words and "fine" in words and len(words) > 70_000


# --- what reads the fallback --------------------------------------------------

def _stored_with_front_matter():
    text = _page_text("Journal of Things 12 (2026) 1-10",
                      "Department of Medicine, University of Somewhere, 22903 Charlottesville, USA", *PROSE)
    return pdf_text.as_paragraphs(pdf_text.mark_lead(text))


def test_umls_reads_the_abstract_or_where_the_reading_starts():
    from rag_lib.umls.text import umls_input_text

    body = _stored_with_front_matter()
    assert umls_input_text("A title", None, body).startswith("A title\n\nThis is the first paragraph")
    assert umls_input_text("A title", "An abstract.", body) == "A title\n\nAn abstract."


def test_embedding_without_an_abstract_starts_at_the_first_paragraph():
    from rag_lib.embed import build_embedding_input
    from rag_lib.paper import Paper

    text = build_embedding_input(Paper(doi=None, openalex_id="local:x", title="A title",
                                       body_text=_stored_with_front_matter()))
    assert "This is the first paragraph" in text and "University of Somewhere" not in text


def test_content_topics_read_the_abstract_first(monkeypatch):
    from types import SimpleNamespace

    from rag_lib import content_topics

    seen = []
    monkeypatch.setattr(content_topics, "topics_from_text", lambda t, b, settings: seen.append(b) or [])
    content_topics.try_topics_json("T", "An abstract.", _stored_with_front_matter(), settings=SimpleNamespace())
    content_topics.try_topics_json("T", None, _stored_with_front_matter(), settings=SimpleNamespace())
    assert seen[0] == "An abstract." and seen[1].startswith("This is the first paragraph")


def test_an_end_before_300_characters_is_not_found_rather_than_reading_on():
    intro = textwrap.wrap("We study things in detail here, " * 20, 70)
    text = _page_text("Abstract", *SENT[:2], "1. Introduction", *intro, "2. Methods")
    assert pdf_text.find_abstract(text) is None


def test_a_correspondence_line_inside_the_abstract_is_not_found():
    text = _page_text("Abstract", SENT[0], "*Correspondence: someone", *SENT[1:], "Keywords: x")
    assert pdf_text.find_abstract(text) is None
