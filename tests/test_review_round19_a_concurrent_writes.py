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
        return dict(p=p, seeds=seeds, embedded=embedded, open_runs=open_runs, card=card, sched=sched, r=rmark)
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


def test_refit_in_another_tab_puts_a_removed_seed_back_into_the_fit(app, env, world, traced):
    """Tab 1 presses Refit; while it fits, tab 2 removes a seed (answered 200).
    Tab 1's fit, made from the old seeds, is written last: the removed
    paper is still in the centroid the feed is scored with, and n_seed
    counts it."""
    calls = _calls(world)
    out = {}
    traced.update(n=0, at=1, fired=False, armed=True,
                  hook=lambda: out.update(b=_send(app, calls["rm0"], "x")))
    ra = _send(app, calls["refit"], "a")
    traced["armed"] = False
    traced["thread"].join()
    assert ra.status_code == 200 and out["b"].status_code == 200
    st = _state(world)
    removed = world["seeds"][0]
    assert removed not in st["seeds"]
    assert removed not in (_fit_seeds(st["p"]) or []), "removed seed is still in the stored fit"
    assert st["p"]["n_seed"] == len(st["seeds"])
