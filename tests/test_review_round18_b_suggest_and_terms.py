"""Review 18 (b): suggested threshold rounding, and Prosopia not_interests."""

from __future__ import annotations

import numpy as np

from rag_lib import calibration, interest_terms


def test_suggested_threshold_keeps_the_top_two_percent():
    # Seed floor above every candidate: the suggestion is clamped to the
    # 98th percentile so "at least the top 2% of candidates pass". round(., 3)
    # then lifts it over the papers it was clamped to keep.
    scores = [0.80 + 0.001 * i for i in range(48)] + [0.9126, 0.9127]
    hi = float(np.percentile(scores, calibration.THRESHOLD_MAX_PERCENTILE))
    assert sum(s >= hi for s in scores) >= 1
    t = calibration.suggest_threshold(0.99, scores)
    assert sum(s >= t for s in scores) >= 1, f"suggested {t}: no candidate passes (top {sorted(scores)[-2:]})"


def test_prosopia_not_interests_listed_once(monkeypatch):
    # The shape of the stored sheffield-nathan document (researchers.id 8).
    fields = {
        "not_interests": ["phylogenomics", "wet-lab-protocol-development"],
        "research_interests": [
            {"concept": {"label": "phylogenomics", "unmapped": True}, "weight": -0.5, "method": "inferred"},
            {"concept": {"@id": "https://openalex.org/T10222", "code": "T10222",
                         "display": "Genomics and Chromatin Dynamics"}, "weight": 0.5, "method": "declared"},
        ],
    }

    class _Resp:
        status_code = 200

        def json(self):
            return {"fields": fields}

    monkeypatch.setattr(interest_terms.requests, "get", lambda *a, **k: _Resp())
    out = interest_terms.prosopia_terms("http://prosopia.invalid", "x")
    norm = [" ".join(n.replace("-", " ").lower().split()) for n in out["not_interests"]]
    # ResearchTopics.tsx shows these joined: "Not interested in: phylogenomics ·
    # wet-lab-protocol-development · phylogenomics · wet lab protocol development".
    assert len(norm) == len(set(norm)), out["not_interests"]
