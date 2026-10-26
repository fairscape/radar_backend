"""Temporary diagnostics do not stay in the code.

A ``[umls-probe]`` block marked "delete once confirmed" sat in
``extract_umls_concepts`` for two months after its vault.py and wizard.py
halves were removed: it printed a funnel to the backend's stdout on every
extraction and read ``ent.text``, which broke eight extractor tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import rag_lib
from rag_lib.umls.extractor import extract_umls_concepts
from tests.test_umls_extractor import linked  # noqa: F401 -- fixture

ROOT = Path(rag_lib.__file__).parent

# "[umls-probe]", "[probe]", "临时诊断" (temporary diagnostic), "DEBUG-ONLY".
MARKER = re.compile(r"\[[\w-]*probe\]|临时诊断|DEBUG[-_ ]ONLY", re.IGNORECASE)


def test_no_probe_markers_in_rag_lib():
    hits = [
        f"{p.relative_to(ROOT)}:{i}: {line.strip()}"
        for p in sorted(ROOT.rglob("*.py"))
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if MARKER.search(line)
    ]
    assert hits == []


def test_extraction_prints_nothing(linked, capsys):
    linked([[("C0011860", 0.98)], [("C0011860", 0.4)], []],
           {"C0011860": ("Diabetes Mellitus", ["T047"])})
    assert len(extract_umls_concepts("diabetes mellitus")) == 1
    assert capsys.readouterr().out == ""
