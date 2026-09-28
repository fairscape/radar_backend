"""The Researchers library: ``/api/researchers`` and ``services.researchers``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_lib.api.services import researchers as svc
from rag_lib.api.services.rp_profile_import import RpProfileError
from rag_lib.db import connect
from tests.test_api_orcid import HDR, HDR_B, _app, _request, env  # noqa: F401

FIXTURE = Path(__file__).parent / "fixtures" / "sheffield_profile.jsonld"


def _text(**overrides) -> str:
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    d.update(overrides)
    return json.dumps(d)


# --------------------------------------------------------------------------- service


def test_parse_for_library_keeps_every_section():
    profile, parsed, doc_json = svc.parse_for_library(FIXTURE.read_text(encoding="utf-8"))
    assert profile.name == "Nathan C. Sheffield"
    assert parsed["rid"] == "0000-0001-5643-4068" and parsed["orcid"] == "0000-0001-5643-4068"
    assert parsed["n_expertise"] == 9 and parsed["n_not_interests"] == 4 and parsed["n_papers"] == 83
    assert [t["institution"] for t in parsed["training"]] == ["Duke University", "CeMM Research Center for Molecular Medicine"]
    assert len(parsed["career"]) == 3 and parsed["career"][1]["start_year"] == 2022
    assert parsed["career_stage"]["current_rank"] == "associate_professor"
    assert parsed["career_stage"]["tenure_status"] == "unknown"
    assert parsed["license"].startswith("https://creativecommons.org") and parsed["visibility"] == "public"
    assert parsed["capabilities"] == {"hasCitationGraph": True, "hasEmbeddingIndex": True, "expertiseCitesPaperIds": True}
    m = parsed["manifest"]
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert m["n_entries"] == len(fixture["hasPart"]) + len(fixture["subjectOf"])
    assert m["by_role"]["paper_summary"] == 2 and m["by_role"]["paper_fulltext"] == 1 and m["by_role"]["works"] == 1
    assert m["by_visibility"]["restricted"] >= 1 and m["persona_documents"] == ["soul", "expertise"]
    assert "paper_fulltext" in m["restricted_roles"] and m["works_url"] == "sources/papers.jsonld"
    assert m["paper_ids"][:2] == ["bock2016multi", "campbell2025taming"]
    assert json.loads(doc_json)["name"] == "Nathan C. Sheffield"
    assert parsed["warnings"] == []


def test_parse_for_library_tolerates_missing_sections():
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for k in ("training", "career", "career_stage", "paper_stats", "hasPart", "subjectOf", "license"):
        d.pop(k, None)
    d["collaborators"] = "not a list"
    _p, parsed, _ = svc.parse_for_library(json.dumps(d))
    assert parsed["training"] == [] and parsed["career"] == [] and parsed["career_stage"] is None
    assert parsed["n_papers"] is None and parsed["manifest"]["n_entries"] == 0
    assert parsed.get("collaborators") in (None, [])


def test_parse_for_library_rejects_bad_documents():
    for bad in ("", "nope", json.dumps({"@type": "Person"}), json.dumps([1])):
        with pytest.raises(RpProfileError):
            svc.parse_for_library(bad)


# --------------------------------------------------------------------------- API


def test_import_list_detail_reimport_delete(env):
    app = _app(None)
    r = _request(app, "POST", "/api/researchers", json={"profile_json": _text(), "source_kind": "file"}, headers=HDR)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["created"] is True and res["warnings"] == []
    s = res["researcher"]
    assert s["name"] == "Nathan C. Sheffield" and s["orcid"] == "0000-0001-5643-4068" and s["rid"] == "0000-0001-5643-4068"
    assert s["affiliation"] == "University of Virginia" and s["level"] == "full" and s["provenance"] == "self_published"
    assert s["n_expertise"] == 9 and s["n_not_interests"] == 4 and s["n_papers"] == 83 and s["source_kind"] == "file"
    assert s["imported_at"] and s["updated_at"] is None
    rid = s["id"]

    lst = _request(app, "GET", "/api/researchers", headers=HDR).json()
    assert [x["id"] for x in lst] == [rid] and lst[0]["n_papers"] == 83
    assert _request(app, "GET", "/api/researchers", headers=HDR_B).json() == []

    d = _request(app, "GET", f"/api/researchers/{rid}", headers=HDR).json()
    assert d["parsed"]["career_stage"]["current_rank"] == "associate_professor"
    assert len(d["parsed"]["training"]) == 2 and d["parsed"]["manifest"]["by_role"]["paper_summary"] == 2
    assert d["doc"]["rid"] == "0000-0001-5643-4068" and "hasPart" in d["doc"]
    assert _request(app, "GET", f"/api/researchers/{rid}", headers=HDR_B).status_code == 404

    # Re-import the same researcher with a newer document: same row, overwritten.
    r2 = _request(app, "POST", "/api/researchers", json={"profile_json": _text(dateModified="2026-10-01T00:00:00+00:00", summary="Updated summary.")}, headers=HDR)
    assert r2.status_code == 200 and r2.json()["created"] is False and r2.json()["researcher"]["id"] == rid
    d2 = _request(app, "GET", f"/api/researchers/{rid}", headers=HDR).json()
    assert d2["date_modified"].startswith("2026-10-01") and d2["parsed"]["summary"] == "Updated summary."
    assert d2["updated_at"] is not None and d2["source_kind"] == "paste"
    assert len(_request(app, "GET", "/api/researchers", headers=HDR).json()) == 1

    assert _request(app, "DELETE", f"/api/researchers/{rid}", headers=HDR_B).status_code == 404
    assert _request(app, "DELETE", f"/api/researchers/{rid}", headers=HDR).json() == {"ok": True}
    assert _request(app, "GET", "/api/researchers", headers=HDR).json() == []
    assert _request(app, "GET", f"/api/researchers/{rid}", headers=HDR).status_code == 404


def test_import_without_orcid_keys_on_the_local_rid(env):
    app = _app(None)
    r = _request(app, "POST", "/api/researchers", json={"profile_json": _text(rid="local:someone-ab12cd", **{"@id": "#me"})}, headers=HDR)
    assert r.status_code == 200, r.text
    s = r.json()["researcher"]
    assert s["rid"] == "local:someone-ab12cd" and s["orcid"] is None
    assert any("local:" in w for w in r.json()["warnings"])


def test_bad_document_is_422_and_stores_nothing(env):
    app = _app(None)
    for body in ({"profile_json": "not json"}, {"profile_json": json.dumps({"@type": "Person"})}):
        assert _request(app, "POST", "/api/researchers", json=body, headers=HDR).status_code == 422
    conn = connect(env["db"])
    assert conn.execute("SELECT COUNT(*) FROM researchers").fetchone()[0] == 0
    conn.close()


def test_library_import_does_not_touch_topics(env):
    app = _app(None)
    _request(app, "POST", "/api/researchers", json={"profile_json": _text()}, headers=HDR)
    conn = connect(env["db"])
    assert conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM gather_runs").fetchone()[0] == 0
    conn.close()
    assert _request(app, "GET", "/api/profiles/drafts", headers=HDR).json() == []


def test_same_orcid_written_differently_is_one_row(env):
    """rid as a bare ORCID, rid missing with an @id IRI, and rid as an IRI all key on the ORCID."""
    app = _app(None)
    a = _request(app, "POST", "/api/researchers", json={"profile_json": _text()}, headers=HDR).json()
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    d.pop("rid")                                           # only @id = https://orcid.org/…
    b = _request(app, "POST", "/api/researchers", json={"profile_json": json.dumps(d)}, headers=HDR).json()
    c = _request(app, "POST", "/api/researchers", json={"profile_json": _text(rid="https://orcid.org/0000-0001-5643-4068")}, headers=HDR).json()
    assert a["created"] is True and b["created"] is False and c["created"] is False
    assert a["researcher"]["id"] == b["researcher"]["id"] == c["researcher"]["id"]
    assert c["researcher"]["rid"] == "0000-0001-5643-4068"
    assert len(_request(app, "GET", "/api/researchers", headers=HDR).json()) == 1
