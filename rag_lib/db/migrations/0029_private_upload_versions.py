"""0029_private_upload_versions -- each user's upload becomes their own paper.

Before this an uploaded PDF was stored on the paper's public row (its
OpenAlex id, or ``local:<hash>``): the first uploader's text, embedding
and UMLS concepts were what every later uploader's seeds were scored
with, and a revised file of the same paper replaced the old one.

Each ``paper_uploads`` row now gets a version ``local:upload:<user>:<hash>``
(``rag_lib.db.repos.versions``), and the user's seeds, topic lists, fit
record and interest terms are renamed to it. What is moved, not rebuilt:

- An upload whose file the public row's text provably came from takes
  the text, embeddings, concepts and topics with it: the paper's only
  upload, by the user and with the hash the row records. Its interests
  keep their fit, topic choices, terms and feed exactly as they were.
- Any other upload of the paper gets the metadata only and waits for its
  own file to be read (``parsed = 0``, ``versions.parse_pending``). With
  two uploads the row's text may be either file's -- the old upload
  wrote a later uploader's text when the first left none, and kept the
  first one's hash -- so neither keeps it.
- A PDF ingested by the CLI (a ``local_path`` and no upload row) is an
  upload of its ``uploaded_by_user_id``. Without one nothing says whose
  file it was -- seeding it does not -- so it stays a public paper.
- A user's file that is also their upload of another paper (the same
  bytes under two ids) is one version; the second paper's seeds stay on
  its public record.

The public row keeps only bibliographic fields; its body-derived
embeddings and concepts are deleted (rebuilt from the metadata when
needed: the rows still in a feed, a researcher's papers or an interest
are queued in ``papers_to_embed`` for the periodic sweep,
``versions.embed_queued``). Any interest still seeded with a public row that lost its text
is marked stale, and so is any whose upload is waiting, so their next
scan refits them; their feed stays visible meanwhile. A ``local:<hash>``
row nothing refers to any more is removed. No file is read, moved or
deleted.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

PREFIX = "local:upload:"
FIT_SEEDS = "fit_seed_ids"     # rag_lib.api.services.wizard.FIT_SEEDS
PRIVATE = ("body_text", "local_path", "file_hash", "uploaded_by_user_id", "n_pages",
           "umls_concepts_json", "umls_mapped_topics_json")


def _seed_hash(ids) -> str:    # rag_lib.interest_terms.seed_hash
    return hashlib.sha1("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def _rename(value, renames):
    if isinstance(value, str):
        return renames.get(value, value)
    if isinstance(value, list):
        out = [_rename(v, renames) for v in value]
        if any(isinstance(v, str) and v in renames for v in value):
            seen: set = set()
            out = [v for v in out if not isinstance(v, str) or not (v in seen or seen.add(v))]
        return out
    if isinstance(value, dict):
        return {k: _rename(v, renames) for k, v in value.items()}
    return value


def _content_topics(raw) -> bool:
    try:
        d = json.loads(raw) if raw else None
    except ValueError:
        return False
    return isinstance(d, dict) and d.get("source") == "pdf_content"


def _schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE paper_versions (
          version_id TEXT PRIMARY KEY REFERENCES papers(openalex_id) ON DELETE CASCADE,
          user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          canonical_openalex_id TEXT REFERENCES papers(openalex_id),
          file_hash TEXT NOT NULL,
          -- 1: holds this file's text; 0: waiting for it; -1: file unreadable
          parsed INTEGER NOT NULL DEFAULT 1,
          -- 1: the PDF had no title, the title is its file's name
          title_from_file INTEGER NOT NULL DEFAULT 0,
          UNIQUE (user_id, file_hash)
        )
        """
    )
    conn.execute("CREATE INDEX idx_paper_versions_canonical ON paper_versions(canonical_openalex_id)")
    # Public papers whose vectors (made from an upload's text) were dropped
    # and are to be embedded again from their metadata.
    conn.execute(
        "CREATE TABLE papers_to_embed (openalex_id TEXT PRIMARY KEY REFERENCES papers(openalex_id) ON DELETE CASCADE,"
        " attempted_at TEXT)"
    )
    # Another user's version is never theirs to seed, score, list or claim.
    owner_of = {
        "profile_seeds": "(SELECT user_id FROM profiles WHERE id = NEW.profile_id)",
        "profile_candidates": "(SELECT user_id FROM profiles WHERE id = NEW.profile_id)",
        "researcher_papers": "(SELECT user_id FROM researchers WHERE id = NEW.researcher_id)",
        "paper_uploads": "NEW.user_id",
    }
    for table, owner in owner_of.items():
        for action in ("INSERT", "UPDATE"):
            conn.execute(
                f"""
                CREATE TRIGGER {table}_private_version_{action.lower()} BEFORE {action} ON {table}
                WHEN EXISTS (SELECT 1 FROM paper_versions v
                             WHERE v.version_id = NEW.openalex_id AND v.user_id IS NOT {owner})
                BEGIN SELECT RAISE(ABORT, 'that paper is another user''s upload'); END
                """
            )


def _loads(raw):
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def _stale(cfg, ids) -> bool:
    """Make a fit record no longer match seeds ``ids`` (by the seed ids,
    or by count for an older fit). Returns whether it changed."""
    if not ids or not isinstance(cfg, dict):
        return False
    if isinstance(cfg.get(FIT_SEEDS), list):
        kept = [i for i in cfg[FIT_SEEDS] if i not in ids]
        changed = len(kept) != len(cfg[FIT_SEEDS])
        cfg[FIT_SEEDS] = kept
        return changed
    return cfg.pop("seed_matrix", None) is not None


def migrate(conn: sqlite3.Connection) -> None:
    factory, conn.row_factory = conn.row_factory, sqlite3.Row
    try:
        _schema(conn)
        _move_uploads(conn)
    finally:
        conn.row_factory = factory


def _move_uploads(conn: sqlite3.Connection) -> None:
    columns = [r[1] for r in conn.execute("PRAGMA table_info(papers)")]

    uploads = [dict(r) for r in conn.execute(
        "SELECT * FROM paper_uploads ORDER BY openalex_id, uploaded_at, user_id")]
    n_uploads: dict[str, int] = {}
    for u in uploads:
        n_uploads[u["openalex_id"]] = n_uploads.get(u["openalex_id"], 0) + 1
    # A CLI-ingested PDF has its file on the paper and no upload row. It is
    # the upload of the user the row names, if any.
    originals: set[str] = set()
    for r in conn.execute(
        """SELECT openalex_id, uploaded_by_user_id, file_hash, local_path, n_pages, body_text
           FROM papers p WHERE local_path IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM paper_uploads u WHERE u.openalex_id = p.openalex_id)""",
    ).fetchall():
        owner = r["uploaded_by_user_id"]
        if owner is None:
            originals.add(r["openalex_id"])    # scrubbed below, nobody's upload
            continue
        digest = r["file_hash"] or hashlib.sha256((r["body_text"] or r["openalex_id"]).encode()).hexdigest()
        conn.execute(
            "INSERT INTO paper_uploads (user_id, openalex_id, file_hash, local_path, n_pages) VALUES (?, ?, ?, ?, ?)",
            (owner, r["openalex_id"], digest, r["local_path"], r["n_pages"]),
        )
        uploads.append({"user_id": owner, "openalex_id": r["openalex_id"], "file_hash": digest,
                        "local_path": r["local_path"], "n_pages": r["n_pages"], "owns_text": True})
        n_uploads[r["openalex_id"]] = 1

    # Each profile's terms record: current before, current after.
    terms_current = set()
    for pid, raw in conn.execute("SELECT id, interest_terms_json FROM profiles").fetchall():
        rec = _loads(raw)
        if isinstance(rec, dict) and rec.get("seed_hash"):
            ids = [x[0] for x in conn.execute("SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", (pid,))]
            if rec["seed_hash"] == _seed_hash(ids):
                terms_current.add(pid)

    renames: dict[int, dict[str, str]] = {}      # profile -> {old id: version}
    waiting: list[str] = []                       # versions with no text yet
    version_of: dict[str, str] = {}               # version -> the paper it is an upload of
    for u in uploads:
        original, uid, digest = u["openalex_id"], int(u["user_id"]), u["file_hash"]
        old = conn.execute("SELECT * FROM papers WHERE openalex_id = ?", (original,)).fetchone()
        if old is None:
            continue
        originals.add(original)
        vid = f"{PREFIX}{uid}:{digest}"
        canonical = None if original.startswith("local:") else original
        if version_of.setdefault(vid, original) != original:
            # The same bytes are this user's upload of another paper too:
            # one version, of that one. This paper's seeds stay public.
            conn.execute("DELETE FROM paper_uploads WHERE user_id = ? AND openalex_id = ?", (uid, original))
            continue
        owns_text = u.get("owns_text") or (
            n_uploads.get(original) == 1
            and old["uploaded_by_user_id"] == uid and old["file_hash"] == digest)
        if conn.execute("SELECT 1 FROM papers WHERE openalex_id = ?", (vid,)).fetchone() is None:
            data = dict(old)
            data.update(openalex_id=vid, source="user_pdf_version", local_path=u["local_path"],
                        file_hash=digest, uploaded_by_user_id=uid,
                        n_pages=u["n_pages"] if u["n_pages"] is not None else old["n_pages"])
            if not owns_text:
                data.update(body_text=None, umls_concepts_json=None, umls_mapped_topics_json=None)
                if _content_topics(data["topics_json"]):
                    data["topics_json"] = None
            conn.execute(
                f"INSERT INTO papers ({','.join(columns)}) VALUES ({','.join('?' * len(columns))})",
                [data.get(c) for c in columns],
            )
            conn.execute(
                "INSERT INTO paper_versions (version_id, user_id, canonical_openalex_id, file_hash, parsed) "
                "VALUES (?, ?, ?, ?, ?)",
                (vid, uid, canonical, digest, 1 if owns_text else 0),
            )
            if owns_text:
                conn.execute(
                    """INSERT INTO paper_embeddings (openalex_id, embedding_model, vector, computed_at)
                       SELECT ?, embedding_model, vector, computed_at FROM paper_embeddings WHERE openalex_id = ?""",
                    (vid, original),
                )
            else:
                waiting.append(vid)
        # The upload row, then this user's seeds, move to the version.
        conn.execute("DELETE FROM paper_uploads WHERE user_id = ? AND openalex_id = ?", (uid, original))
        conn.execute(
            "INSERT OR IGNORE INTO paper_uploads (user_id, openalex_id, file_hash, local_path, n_pages, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?, COALESCE(?, datetime('now')))",
            (uid, vid, digest, u["local_path"], u["n_pages"], u.get("uploaded_at")),
        )
        for (pid,) in conn.execute(
            """SELECT s.profile_id FROM profile_seeds s JOIN profiles p ON p.id = s.profile_id
               WHERE s.openalex_id = ? AND p.user_id = ?""", (original, uid),
        ).fetchall():
            if conn.execute("SELECT 1 FROM profile_seeds WHERE profile_id = ? AND openalex_id = ?",
                            (pid, vid)).fetchone():
                conn.execute("DELETE FROM profile_seeds WHERE profile_id = ? AND openalex_id = ?", (pid, original))
            else:
                conn.execute("UPDATE profile_seeds SET openalex_id = ? WHERE profile_id = ? AND openalex_id = ?",
                             (vid, pid, original))
            renames.setdefault(pid, {})[original] = vid

    # Seeds, topic lists (``seeds`` of a topic), the fit record and the
    # interest terms name papers by id: the same rename, nothing else.
    waiting_set = set(waiting)
    for pid, ren in renames.items():
        row = conn.execute(
            "SELECT topic_filters_json, selector_config_json, interest_terms_json FROM profiles WHERE id = ?",
            (pid,),
        ).fetchone()
        out = []
        for raw in row:
            try:
                out.append(json.dumps(_rename(json.loads(raw), ren)) if raw else raw)
            except ValueError:
                out.append(raw)
        cfg = _loads(out[1])
        if _stale(cfg, waiting_set & set(ren.values())):
            # Fitted from another file's text: refit at the next scan.
            out[1] = json.dumps(cfg)
        rec = _loads(out[2])
        if pid in terms_current and isinstance(rec, dict) and not waiting_set & set(ren.values()):
            # Renamed, not changed: the terms still describe these seeds.
            ids = [x[0] for x in conn.execute("SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", (pid,))]
            rec["seed_hash"] = _seed_hash(ids)
            out[2] = json.dumps(rec)
        conn.execute(
            """UPDATE profiles SET topic_filters_json = ?, selector_config_json = ?, interest_terms_json = ?,
               n_seed = (SELECT COUNT(*) FROM profile_seeds WHERE profile_id = ?) WHERE id = ?""",
            (*out, pid, pid),
        )

    # The public rows keep only what OpenAlex (or the PDF's metadata) says.
    referenced = (
        "SELECT 1 FROM profile_seeds WHERE openalex_id = :id UNION ALL "
        "SELECT 1 FROM profile_candidates WHERE openalex_id = :id UNION ALL "
        "SELECT 1 FROM researcher_papers WHERE openalex_id = :id UNION ALL "
        "SELECT 1 FROM feedback_events WHERE openalex_id = :id UNION ALL "
        "SELECT 1 FROM paper_uploads WHERE openalex_id = :id"
    )
    lost_text = {o for o in originals if conn.execute(
        "SELECT body_text IS NOT NULL OR umls_concepts_json IS NOT NULL FROM papers WHERE openalex_id = ?",
        (o,)).fetchone()[0]}
    for original in originals:
        if conn.execute("SELECT 1 FROM paper_embeddings WHERE openalex_id = ?", (original,)).fetchone():
            conn.execute("INSERT OR IGNORE INTO papers_to_embed (openalex_id) VALUES (?)", (original,))
        conn.execute("DELETE FROM paper_embeddings WHERE openalex_id = ?", (original,))
        if original.startswith("local:") and conn.execute(
                f"SELECT EXISTS ({referenced})", {"id": original}).fetchone()[0] == 0:
            conn.execute("DELETE FROM papers WHERE openalex_id = ?", (original,))
            continue
        raw = conn.execute("SELECT topics_json FROM papers WHERE openalex_id = ?", (original,)).fetchone()[0]
        conn.execute(
            f"UPDATE papers SET {', '.join(f'{c} = NULL' for c in PRIVATE)}"
            f"{', topics_json = NULL' if _content_topics(raw) else ''} WHERE openalex_id = ?",
            (original,),
        )

    # An interest still seeded with a public row that lost its text was
    # fitted from that text: refit at its next scan.
    for pid, raw in conn.execute(
        f"""SELECT DISTINCT p.id, p.selector_config_json FROM profiles p
            JOIN profile_seeds s ON s.profile_id = p.id
            WHERE s.openalex_id IN ({','.join('?' * len(lost_text))})""",
        sorted(lost_text),
    ).fetchall() if lost_text else []:
        cfg = _loads(raw)
        ids = {x[0] for x in conn.execute("SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", (pid,))}
        if _stale(cfg, ids & lost_text):
            conn.execute("UPDATE profiles SET selector_config_json = ? WHERE id = ?", (json.dumps(cfg), pid))
