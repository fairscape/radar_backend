"""Class sweep (review 19): two tabs MODIFY the same interest at once.

Request A runs; just before its n-th transaction (on any connection it
opens, scan jobs included) request B runs to completion on its own
connection. Then invariants are checked, plus each request's own effect
when it answered 2xx (a lost update shows as an effect that is gone).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading

import pytest

from rag_lib.db import connect as _real_connect
from rag_lib.db.repos.profiles import bar
from tests.edit_helpers import edit_saved
from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    _go_live, _ok, _q, _request, _scan, _upload, app, env,
)

MAX_TX = int(os.environ.get("SWEEP_MAX_TX", "8"))


def _copy(frm, to):
    src, dst = sqlite3.connect(frm), sqlite3.connect(to)
    src.backup(dst)
    src.close(), dst.close()


def _db():
    return os.environ["RADAR_DB_PATH"]


@pytest.fixture()
def world(app, env, tmp_path):
    live, rid = _go_live(app, env, "Live One")
    _scan(app, live)
    pid = _q(env, "SELECT id FROM profiles WHERE slug = ?", live)[0][0]
    seeds = [r[0] for r in _q(env, "SELECT openalex_id FROM profile_seeds WHERE profile_id = ? ORDER BY added_at", pid)]
    # a vault paper that is not a seed, for "add seed"
    doc = _ok(_upload(app, b"extra-paper", "extra.pdf"))
    extra = doc.get("openalex_id") or doc.get("id")
    cards = [r[0] for r in _q(env, "SELECT openalex_id FROM profile_candidates WHERE profile_id = ? LIMIT 2", pid)]
    topics = json.loads(_q(env, "SELECT topic_filters_json FROM profiles WHERE id = ?", pid)[0][0] or "{}").get("topics") or []
    tids = [t["id"] for t in topics if t.get("id")]
    snap = tmp_path / "snap.db"
    _copy(_db(), snap)
    return dict(live=live, pid=pid, rid=rid, seeds=seeds, extra=extra, cards=cards, tids=tids, snap=snap)


def _calls(w):
    L, s0, s1, c0, rid = w["live"], w["seeds"][0], w["seeds"][-1], w["cards"][0], w["rid"]
    t_one = w["tids"][:1]
    return {
        "thr": ("PATCH", f"/api/profiles/{L}", {"json": {"threshold": 0.2}}),
        "years": ("PATCH", f"/api/profiles/{L}", {"json": {"years": {"year_from": 2019, "year_to": None}}}),
        "refit": ("POST", f"/api/profiles/{L}/refit", {}),
        "autotune": ("POST", f"/api/profiles/{L}/refit?autotune=1", {}),
        "recompute": ("POST", f"/api/profiles/{L}/recompute-topics", {}),
        "topics1": ("PUT", f"/api/profiles/{L}/topics", {"json": {"selected_topic_ids": t_one}}),
        "rt_on": ("PUT", f"/api/profiles/{L}/researcher-topics", {"json": {"enabled": True}}),
        "rt_off": ("PUT", f"/api/profiles/{L}/researcher-topics", {"json": {"enabled": False}}),
        "add": ("POST", f"/api/profiles/{L}/seeds", {"json": {"openalex_ids": [w["extra"]]}}),
        "rm0": ("DELETE", f"/api/profiles/{L}/seeds", {"params": {"openalex_id": s0}}),
        "rm1": ("DELETE", f"/api/profiles/{L}/seeds", {"params": {"openalex_id": s1}}),
        "upload": ("UPLOAD", L, {}),
        "scan": ("POST", f"/api/profiles/{L}/gather-now?days=7", {}),
        "save": ("POST", "/api/radar/cards/save", {"json": {"card_id": c0, "profile": L, "on": True}}),
        "dismiss": ("POST", "/api/radar/cards/dismiss", {"json": {"card_id": c0, "profile": L, "on": True}}),
        "terms": ("PUT", f"/api/profiles/{L}/terms", {"json": {"terms": [{"label": "zeta"}]}}),
        "sched": ("PATCH", f"/api/profiles/{L}/schedule", {"json": {"enabled": False}}),
        "mark": ("PATCH", f"/api/researchers/{rid}", {"json": {"mark": "self"}}),
        "reimport": ("POST", "/api/researchers/import", {"json": {"source": "prosopia", "ref": "sheffield-nathan"}}),
    }


def _send(app, call, tag):
    method, path, kw = call
    if method == "UPLOAD":
        return _upload(app, f"sweep-{tag}".encode(), "s.pdf", path)
    return _request(app, method, path, **kw)


def _state(w):
    c = _real_connect(_db())
    try:
        p = c.execute("SELECT * FROM profiles WHERE id = ?", (w["pid"],)).fetchone()
        seeds = sorted(r[0] for r in c.execute("SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", (w["pid"],)))
        embedded = c.execute(
            "SELECT COUNT(*) FROM profile_seeds s JOIN paper_embeddings e ON e.openalex_id = s.openalex_id "
            "AND e.embedding_model = ? WHERE s.profile_id = ?", (p["embedding_model"], w["pid"])).fetchone()[0]
        open_runs = c.execute("SELECT COUNT(*) FROM gather_runs WHERE finished_at IS NULL").fetchone()[0]
        card = c.execute("SELECT saved_at, dismissed_at FROM profile_candidates WHERE profile_id = ? AND openalex_id = ?",
                         (w["pid"], w["cards"][0])).fetchone()
        sched = c.execute("SELECT * FROM profile_schedules WHERE profile_id = ?", (w["pid"],)).fetchone() \
            if c.execute("SELECT 1 FROM sqlite_master WHERE name='profile_schedules'").fetchone() else None
        rmark = c.execute("SELECT * FROM researchers WHERE id = ?", (w["rid"],)).fetchone()
        last = c.execute("SELECT action FROM feedback_events WHERE profile_id = ? AND openalex_id = ? ORDER BY id DESC LIMIT 1",
                         (w["pid"], w["cards"][0])).fetchone()
        return dict(p=p, seeds=seeds, embedded=embedded, open_runs=open_runs, card=card, sched=sched, r=rmark,
                    last_event=last[0] if last else None)
    finally:
        c.close()


def _fit_seeds(p):
    try:
        ids = json.loads(p["selector_config_json"] or "{}").get("fit_seed_ids")
    except ValueError:
        return None
    return sorted(ids) if isinstance(ids, list) else None


def _topics(p):
    return (json.loads(p["topic_filters_json"] or "{}").get("topics")) or []


def _invariants(st):
    bad = []
    p = st["p"]
    if p is None:
        return ["interest vanished"]
    if st["embedded"] < 1:
        bad.append("live interest has no embedded seed")
    if p["n_seed"] != len(st["seeds"]):
        bad.append(f"n_seed={p['n_seed']} but {len(st['seeds'])} seeds")
    fit = _fit_seeds(p)
    if fit is not None and fit != st["seeds"]:
        bad.append(f"fit made from {len(fit)} seeds, interest has {len(st['seeds'])} (diff +{sorted(set(st['seeds']) - set(fit))} -{sorted(set(fit) - set(st['seeds']))})")
    if st["open_runs"]:
        bad.append(f"{st['open_runs']} run(s) unfinished")
    t = _topics(p)
    if t and not any(x.get("on", True) for x in t):
        bad.append("no topic on")
    # The activity log's latest event for the card agrees with its state.
    card, last = st["card"], st.get("last_event")
    if card is not None and last is not None:
        state = "saved" if card["saved_at"] else "dismissed" if card["dismissed_at"] else None
        if state is not None and last != state:
            bad.append(f"card {state} but the log's latest event is {last}")
    return bad


def _effect(name, w, st):
    """What a 2xx of ``name`` must have left in place (None = nothing to check)."""
    p = st["p"]
    if p is None:
        return None
    if name == "thr" and abs(p["threshold"] - bar(0.2)) > 1e-9:
        return f"threshold {p['threshold']} != {bar(0.2)}"
    if name == "years" and p["year_from"] != 2019:
        return f"year_from {p['year_from']}"
    if name == "add" and w["extra"] not in st["seeds"]:
        return "added seed gone"
    if name == "rm0" and w["seeds"][0] in st["seeds"]:
        return "removed seed back"
    if name == "rm1" and w["seeds"][-1] in st["seeds"]:
        return "removed seed back"
    if name == "topics1":
        on = sorted(x["id"] for x in _topics(p) if x.get("on", True))
        if on != sorted(w["tids"][:1]):
            return f"topics on {on} != {w['tids'][:1]}"
    if name == "save" and (st["card"] is None or st["card"]["saved_at"] is None):
        return "card not saved"
    if name == "dismiss" and (st["card"] is None or st["card"]["dismissed_at"] is None):
        return "card not dismissed"
    if name == "terms":
        labels = [t["label"] for t in (json.loads(p["interest_terms_json"] or "{}").get("terms") or [])]
        if labels != ["zeta"]:
            return f"terms {labels}"
    if name == "mark" and st["r"] is not None and "mark" in st["r"].keys() and st["r"]["mark"] != "self":
        return f"mark {st['r']['mark']}"
    return None


# Pairs where B's effect is legitimately undone by A (A runs last and means to change it).
OVERRIDES = {
    ("autotune", "thr"),          # autotune sets the threshold after
    ("save", "dismiss"), ("dismiss", "save"),
    ("topics1", "rt_on"), ("rt_on", "topics1"), ("rt_off", "topics1"),
    ("rm0", "add"), ("rm1", "add"),
}


@pytest.fixture()
def traced(app, monkeypatch):
    """Patch every module's ``connect`` so each connection request A opens is traced."""
    from rag_lib.api import deps
    state = {"n": 0, "at": None, "hook": None, "fired": False, "armed": False}

    def connect(path):
        conn = _real_connect(path)

        def trace(sql):
            if not state["armed"] or state["fired"] or not sql.startswith("BEGIN"):
                return
            state["n"] += 1
            if state["n"] == state["at"]:
                state["fired"] = True
                state["armed"] = False
                # B in its own thread: A may be on the event loop thread.
                # If A already holds a write lock elsewhere, B waits for it, as
                # it would in production: let A go on and collect B after.
                t = threading.Thread(target=state["hook"], daemon=True)
                t.start()
                t.join(timeout=3)
                state["thread"] = t
        conn.set_trace_callback(trace)
        return conn

    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("rag_lib") and getattr(mod, "connect", None) is _real_connect:
            monkeypatch.setattr(mod, "connect", connect)

    def traced_get_db(settings=None):
        conn = connect(_db())
        try:
            yield conn
        finally:
            conn.set_trace_callback(None)
            conn.close()
    app.dependency_overrides[deps.get_db] = traced_get_db
    yield state
    app.dependency_overrides.pop(deps.get_db, None)


A_NAMES = os.environ.get("SWEEP_A", "").split(",") if os.environ.get("SWEEP_A") else None
B_NAMES = os.environ.get("SWEEP_B", "").split(",") if os.environ.get("SWEEP_B") else None


def test_two_tabs_modify_the_same_interest(app, env, world, traced):
    calls = _calls(world)
    bad = []
    for a in A_NAMES or calls:
        for b in B_NAMES or calls:
            for at in range(1, MAX_TX + 1):
                _copy(world["snap"], _db())
                out = {}

                def hook(b=b, at=at):
                    out["b"] = _send(app, calls[b], f"b{at}")
                traced.update(n=0, at=at, fired=False, armed=True, hook=hook)
                ra = _send(app, calls[a], f"a{at}")
                traced["armed"] = False
                if not traced["fired"]:
                    break
                traced["thread"].join()
                rb = out["b"]
                sys.__stderr__.write(f"{a} {b} {at} {ra.status_code} {rb.status_code}\n")
                st = _state(world)
                label = f"A={a} B={b} (B before A's tx {at}) A->{ra.status_code} B->{rb.status_code}"
                probs = []
                for nm, r in (("A", ra), ("B", rb)):
                    if r.status_code >= 500:
                        probs.append(f"{nm} {r.status_code} {r.text[:200]}")
                probs += _invariants(st)
                if 200 <= ra.status_code < 300:
                    e = _effect(a, world, st)
                    if e:
                        probs.append(f"A's effect lost: {e}")
                if 200 <= rb.status_code < 300 and (a, b) not in OVERRIDES and a != b:
                    e = _effect(b, world, st)
                    if e:
                        probs.append(f"B's effect lost: {e}")
                if probs:
                    bad.append(f"{label}: " + "; ".join(probs))
    print("\n".join(bad))
    assert not bad, f"{len(bad)} bad interleavings:\n" + "\n".join(bad)


def test_commit_vs_seed_change_on_the_draft(app, env, traced, tmp_path):
    """Commit (A) while another tab attaches a seed to the draft (B)."""
    draft = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Draft C"}))["slug"]
    _ok(_upload(app, b"c-one", "one.pdf", draft))
    _ok(_upload(app, b"c-two", "two.pdf", draft))
    extra = _ok(_upload(app, b"c-extra", "x.pdf"))["id"]
    pid = _q(env, "SELECT id FROM profiles WHERE slug = ?", draft)[0][0]
    snap = tmp_path / "c.db"
    _copy(_db(), snap)
    Bs = {
        "upload": lambda at: _upload(app, f"c-late-{at}".encode(), "late.pdf", draft),
        "add": lambda at: _request(app, "POST", f"/api/profiles/draft/{draft}/seeds", json={"openalex_ids": [extra]}),
    }
    bad = []
    for b, send in Bs.items():
        for at in range(1, MAX_TX + 1):
            _copy(snap, _db())
            out = {}
            traced.update(n=0, at=at, fired=False, armed=True, hook=lambda send=send, at=at: out.update(b=send(at)))
            ra = _request(app, "POST", "/api/profiles", json={"slug": draft, "threshold": 0.1, "selected_topic_ids": []})
            traced["armed"] = False
            if not traced["fired"]:
                break
            traced["thread"].join()
            rb = out["b"]
            c = _real_connect(_db())
            p = c.execute("SELECT * FROM profiles WHERE id = ?", (pid,)).fetchone()
            seeds = sorted(r[0] for r in c.execute("SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", (pid,)))
            c.close()
            probs = []
            if ra.status_code >= 500 or rb.status_code >= 500:
                probs.append(f"5xx {ra.status_code}/{rb.status_code}")
            if not p["is_draft"]:
                if p["n_seed"] != len(seeds):
                    probs.append(f"n_seed={p['n_seed']} but {len(seeds)} seeds")
                fit = _fit_seeds(p)
                if fit is not None and fit != seeds:
                    probs.append(f"live fit from {len(fit)} seeds, has {len(seeds)}")
            if probs:
                bad.append(f"commit vs {b} (B before tx {at}) A->{ra.status_code} B->{rb.status_code}: " + "; ".join(probs))
    print("\n".join(bad))
    assert not bad, "\n".join(bad)


def _scan_seeds(w):
    """The seeds the interest's last successful scan recorded it scored with."""
    from rag_lib.db.repos import gather_runs as gather_runs_repo

    c = _real_connect(_db())
    try:
        last = gather_runs_repo.last_scan(c, w["pid"])
    finally:
        c.close()
    return sorted(last[1]) if last and last[1] is not None else None


def _copy_open(w):
    """Whether the copy the fixture opened (by id: a reopened copy may get
    its slug) is still there."""
    c = _real_connect(_db())
    try:
        return c.execute("SELECT 1 FROM profiles WHERE id = ?", (w["copy_id"],)).fetchone() is not None
    finally:
        c.close()


def _n_copies(w):
    c = _real_connect(_db())
    try:
        return c.execute("SELECT COUNT(*) FROM profiles WHERE edit_of = ?", (w["pid"],)).fetchone()[0]
    finally:
        c.close()


@pytest.fixture()
def editing(app, env, world, tmp_path):
    """``world`` with Edit interest open on the live interest (spec E): the
    copy has seed 0 removed, ``extra`` added and threshold 0.2."""
    L = world["live"]
    copy = _ok(_request(app, "POST", f"/api/profiles/{L}/edit"))["slug"]
    _ok(_request(app, "DELETE", f"/api/profiles/{copy}/seeds", params={"openalex_id": world["seeds"][0]}))
    _ok(_request(app, "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [world["extra"]]}))
    _ok(_request(app, "PATCH", f"/api/profiles/{copy}", json={"threshold": 0.2}))
    copy_id = _q(env, "SELECT id FROM profiles WHERE slug = ?", copy)[0][0]
    snap = tmp_path / "editing.db"
    _copy(_db(), snap)
    return dict(world, copy=copy, copy_id=copy_id, esnap=snap)


def _edit_calls(w):
    L, C, s1, c0 = w["live"], w["copy"], w["seeds"][-1], w["cards"][0]
    return {
        "save_edit": ("POST", f"/api/profiles/draft/{C}/save-edit", {}),
        "discard": ("DELETE", f"/api/profiles/draft/{C}", {}),
        "reopen": ("POST", f"/api/profiles/{L}/edit", {}),
        "copy_rm1": ("DELETE", f"/api/profiles/{C}/seeds", {"params": {"openalex_id": s1}}),
        "copy_topics1": ("PUT", f"/api/profiles/{C}/topics", {"json": {"selected_topic_ids": w["tids"][:1]}}),
        "copy_upload": ("UPLOAD", C, {}),
        "scan": ("POST", f"/api/profiles/{L}/gather-now?days=7", {}),
        "backfill": ("POST", f"/api/profiles/{L}/backfill", {"json": {"year_from": 2019, "year_to": None}}),
        "save": ("POST", "/api/radar/cards/save", {"json": {"card_id": c0, "profile": L, "on": True}}),
        "dismiss": ("POST", "/api/radar/cards/dismiss", {"json": {"card_id": c0, "profile": L, "on": True}}),
    }


def _edit_invariants(w, st):
    """Spec E: saving does not refit, and no save lands under a scan -- so
    the fit is the snapshot's or the current seeds', and the last scan's
    recorded seeds are the ones it fitted on."""
    bad = [x for x in _invariants(st) if not x.startswith("fit made from")]
    fit = _fit_seeds(st["p"])
    if fit is not None and fit not in (sorted(w["seeds"]), st["seeds"]):
        bad.append(f"fit made from {fit}, neither the old seeds nor the interest's {st['seeds']}")
    scanned = _scan_seeds(w)
    if scanned is not None and fit is not None and scanned != fit:
        bad.append(f"the last scan records seeds {scanned} but the fit is from {fit}")
    if _n_copies(w) > 1:
        bad.append(f"{_n_copies(w)} copies open")
    return bad


def _edit_effect(name, w, st):
    p = st["p"]
    if name == "save_edit":
        if w["seeds"][0] in st["seeds"] or w["extra"] not in st["seeds"]:
            return f"saved seeds lost: {st['seeds']}"
        if abs(p["threshold"] - bar(0.2)) > 1e-9:
            return f"saved threshold lost: {p['threshold']}"
        if _copy_open(w):
            return "the copy is still there after saving"
    if name == "discard" and _copy_open(w):
        return "the copy is still there after discarding"
    if name == "save" and (st["card"] is None or st["card"]["saved_at"] is None):
        return "card not saved"
    if name == "dismiss" and (st["card"] is None or st["card"]["dismissed_at"] is None):
        return "card not dismissed"
    return None


# B legitimately undoes A, A running last: A discards or reopens (which
# discards) the copy B changed; save/dismiss of one card.
EDIT_OVERRIDES = {
    ("discard", "copy_rm1"), ("discard", "copy_topics1"), ("discard", "copy_upload"),
    ("reopen", "copy_rm1"), ("reopen", "copy_topics1"), ("reopen", "copy_upload"),
    ("save", "dismiss"), ("dismiss", "save"),
}


def test_two_tabs_while_one_edits_the_interest(app, env, editing, traced):
    """Spec E's race surface: a copy is open in Edit interest; one tab saves,
    discards, reopens or changes it while another scans, backfills, triages
    or changes the copy. No save lands under a scan (E.6 R11), no scan runs
    while the copy is open (E.7 R9), and saving does not refit."""
    w = editing
    calls = _edit_calls(w)
    bad = []
    for a in calls:
        for b in calls:
            for at in range(1, MAX_TX + 1):
                _copy(w["esnap"], _db())
                out = {}

                def hook(b=b, at=at):
                    out["b"] = _send(app, calls[b], f"b{at}")
                traced.update(n=0, at=at, fired=False, armed=True, hook=hook)
                ra = _send(app, calls[a], f"a{at}")
                traced["armed"] = False
                if not traced["fired"]:
                    break
                traced["thread"].join()
                rb = out["b"]
                st = _state(w)
                label = f"A={a} B={b} (B before A's tx {at}) A->{ra.status_code} B->{rb.status_code}"
                probs = [f"{nm} {r.status_code} {r.text[:200]}" for nm, r in (("A", ra), ("B", rb)) if r.status_code >= 500]
                probs += _edit_invariants(w, st)
                if 200 <= ra.status_code < 300:
                    e = _edit_effect(a, w, st)
                    if e:
                        probs.append(f"A's effect lost: {e}")
                if 200 <= rb.status_code < 300 and (a, b) not in EDIT_OVERRIDES and a != b:
                    e = _edit_effect(b, w, st)
                    if e:
                        probs.append(f"B's effect lost: {e}")
                # The copy open: the scan is refused, unless B saved or discarded it first.
                if a in ("scan", "backfill") and 200 <= ra.status_code < 300 and not (
                        b in ("save_edit", "discard") and 200 <= rb.status_code < 300):
                    probs.append(f"A {a} ran while the copy was open")
                if probs:
                    bad.append(f"{label}: " + "; ".join(probs))
    print("\n".join(bad))
    assert not bad, f"{len(bad)} bad interleavings:\n" + "\n".join(bad)


def test_a_copy_saved_while_a_scan_refits_is_refused(app, env, world, traced):
    """Was: tab 1 presses Refit; while it fits, tab 2 removes a seed
    (answered 200) and tab 1's fit, made from the old seeds, is written
    last. Spec E: a saved interest is refitted only by a scan and its seeds
    change only when a copy is saved, which is refused while a scan runs.
    So: tab 1 scans (refitting first, the fit being stale); before each of
    its transactions tab 2 saves a copy that removes a seed. Whichever
    lands, the fit is the interest's seeds and the scan records them."""
    L = world["live"]
    removed = world["seeds"][0]
    # Stale fit: ``extra`` saved as a seed, not yet fitted.
    with edit_saved(lambda m, p, **k: _request(app, m, p, **k), L) as copy:
        _ok(_request(app, "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [world["extra"]]}))
    snap = env.parent / "race.db"
    _copy(_db(), snap)
    outcomes = set()
    for at in range(1, MAX_TX + 1):
        _copy(snap, _db())
        out = {}

        def hook():
            # Tab 2: Edit interest (it opens under a running scan), remove a seed, save.
            copy = _ok(_request(app, "POST", f"/api/profiles/{L}/edit"))["slug"]
            _ok(_request(app, "DELETE", f"/api/profiles/{copy}/seeds", params={"openalex_id": removed}))
            out["b"] = _request(app, "POST", f"/api/profiles/draft/{copy}/save-edit")
        traced.update(n=0, at=at, fired=False, armed=True, hook=hook)
        ra = _request(app, "POST", f"/api/profiles/{L}/gather-now?days=7")
        traced["armed"] = False
        if not traced["fired"]:
            break
        traced["thread"].join()
        rb = out["b"]
        assert ra.status_code == 200, (at, ra.text)
        assert rb.status_code in (200, 409), (at, rb.status_code, rb.text)
        outcomes.add(rb.status_code)
        st = _state(world)
        if rb.status_code == 200:
            assert removed not in st["seeds"], at
        else:
            assert "scan" in rb.text and removed in st["seeds"], (at, rb.text)   # nothing written
        fit = _fit_seeds(st["p"])
        assert fit == st["seeds"], (at, rb.status_code, "the fit is not the interest's seeds")
        assert _scan_seeds(world) == fit, (at, rb.status_code, "the scan records other seeds than it fitted")
        assert st["p"]["n_seed"] == len(st["seeds"])
        assert not st["open_runs"]
    assert 409 in outcomes, "no save landed under the running scan: the race was not exercised"
