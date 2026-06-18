"""SQLite + sqlite-vec + FTS5 index schema for docrag.

One DB file per corpus: ``{index_dir}/{corpus}.db``.

  files        (path, sha256, size, indexed_at)
  chunks       (id, path, corpus, source_file, kind, page, start_line,
                end_line, text, context_summary, content_hash, indexed_at,
                + section_id, section_number, section_title, breadcrumb,
                jurisdiction, edition, parent_section_id, node_type)
               -- one row per *embedded leaf* (smallest section body or a
               -- table or a sliding-window fallback span)
  section_nodes(id, corpus, path, jurisdiction, edition, section_id,
                section_number, section_title, breadcrumb,
                parent_section_id, full_text)
               -- the parent-document payload: one row per section, full_text
               -- = own body + all descendant bodies. NOT embedded.
  section_refs (src_section_id, dst_raw, dst_kind, dst_section_id, corpus)
               -- citation graph: cross-references parsed from section text.
  vec_chunks   (chunk_id, embedding[D])      -- sqlite-vec virtual, leaves only
  fts_chunks   (text, meta, source_file, kind) -- FTS5, standalone, leaves only

The only module that talks to SQLite.

CLI:
  python -m rag.db --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import sqlite_vec

from . import settings

# Bump when the schema changes in a way that requires a full rebuild.
#   v3: add domain-agnostic `metadata` JSON column to files+chunks; add
#       extraction_cache (cached per-doc LLM extractions, e.g. youtube map step).
SCHEMA_VERSION = 3

# Corpus names map directly to filesystem paths ({index_dir}/{corpus}.db).
_CORPUS_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def embedding_dims() -> int:
    try:
        return int(settings.embedding_dims())
    except Exception:  # noqa: BLE001
        return 3072


def _index_dir() -> Path:
    return Path(settings.index_dir())


def valid_corpus(corpus: str) -> bool:
    return bool(_CORPUS_RE.match((corpus or "").strip().lower()))


def open_db(corpus: str, db_path: str | os.PathLike | None = None) -> sqlite3.Connection:
    """Open or create the corpus DB. Loads sqlite-vec + ensures schema."""
    corpus = (corpus or "").strip().lower()
    if not _CORPUS_RE.match(corpus):
        raise ValueError("invalid corpus name: %r" % corpus)
    target = Path(db_path) if db_path is not None else _index_dir() / ("%s.db" % corpus)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


def schema_outdated(conn: sqlite3.Connection) -> bool:
    """True if this DB predates the current SCHEMA_VERSION (needs --full rebuild)."""
    row = conn.execute("PRAGMA user_version").fetchone()
    return (row[0] if row else 0) < SCHEMA_VERSION


def _ensure_schema(conn: sqlite3.Connection) -> None:
    dims = embedding_dims()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS files (
      path TEXT PRIMARY KEY,
      sha256 TEXT NOT NULL,
      size INTEGER NOT NULL,
      indexed_at INTEGER NOT NULL,
      metadata TEXT            -- JSON: per-doc domain metadata (youtube: channel,
                              -- title, url, upload_date; codes: {} / unused)
    );

    CREATE TABLE IF NOT EXISTS chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      path TEXT NOT NULL,
      corpus TEXT NOT NULL,
      source_file TEXT NOT NULL,
      kind TEXT NOT NULL DEFAULT 'document',
      page INTEGER,
      start_line INTEGER,
      end_line INTEGER,
      text TEXT NOT NULL,
      context_summary TEXT,
      content_hash TEXT NOT NULL,
      indexed_at INTEGER NOT NULL,
      section_id TEXT,
      section_number TEXT,
      section_title TEXT,
      breadcrumb TEXT,
      jurisdiction TEXT,
      edition TEXT,
      parent_section_id TEXT,
      node_type TEXT NOT NULL DEFAULT 'leaf',
      metadata TEXT,           -- JSON: domain-specific leaf metadata. codes use
                              -- the explicit columns above; youtube stores
                              -- {video_id, channel, title, url, start_time,
                              -- end_time, ...} here.
      FOREIGN KEY(path) REFERENCES files(path) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_path   ON chunks(path);
    CREATE INDEX IF NOT EXISTS idx_chunks_corpus ON chunks(corpus);

    CREATE TABLE IF NOT EXISTS section_nodes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      corpus TEXT NOT NULL,
      path TEXT NOT NULL,
      jurisdiction TEXT,
      edition TEXT,
      section_id TEXT NOT NULL,
      section_number TEXT,
      section_title TEXT,
      breadcrumb TEXT,
      parent_section_id TEXT,
      full_text TEXT NOT NULL,
      indexed_at INTEGER NOT NULL,
      UNIQUE(corpus, section_id)
    );
    CREATE INDEX IF NOT EXISTS idx_section_nodes_path ON section_nodes(path);

    CREATE TABLE IF NOT EXISTS section_refs (
      src_section_id TEXT NOT NULL,
      dst_raw TEXT NOT NULL,
      dst_kind TEXT,
      dst_section_id TEXT,
      corpus TEXT NOT NULL,
      path TEXT NOT NULL,
      PRIMARY KEY(src_section_id, dst_raw)
    );
    CREATE INDEX IF NOT EXISTS idx_section_refs_src  ON section_refs(src_section_id);
    CREATE INDEX IF NOT EXISTS idx_section_refs_path ON section_refs(path);

    CREATE TABLE IF NOT EXISTS extraction_cache (
      path TEXT NOT NULL,             -- doc id (codes: rel path; youtube: yt:<id>)
      corpus TEXT NOT NULL,
      content_hash TEXT NOT NULL,     -- invalidates when the source content changes
      extraction_kind TEXT NOT NULL,  -- e.g. "predictions" | "claims" | "summary"
      model TEXT NOT NULL,            -- invalidates if the extracting model changes
      result TEXT NOT NULL,           -- JSON: the cached per-doc extraction
      created_at INTEGER NOT NULL,
      PRIMARY KEY (path, extraction_kind, content_hash, model)
    );
    CREATE INDEX IF NOT EXISTS idx_extraction_cache_corpus ON extraction_cache(corpus);
    """)

    cur.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
        "chunk_id INTEGER PRIMARY KEY, embedding FLOAT[%d])" % dims
    )
    # Standalone FTS (not external-content): body in `text`, enrichment in
    # `meta`, so BM25 can weight them separately (see query.py bm25 weights).
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
      text,
      meta,
      source_file UNINDEXED,
      kind UNINDEXED
    );
    """)
    conn.execute("PRAGMA user_version=%d" % SCHEMA_VERSION)
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def file_row(conn: sqlite3.Connection, path: str) -> tuple[str, int] | None:
    cur = conn.execute("SELECT sha256, size FROM files WHERE path=?", (path,))
    row = cur.fetchone()
    return (row[0], row[1]) if row else None


def upsert_file(conn: sqlite3.Connection, path: str, sha256: str, size: int,
                metadata: dict | None = None) -> None:
    """Insert-or-replace a files row. Does NOT commit.

    ``metadata`` (optional) is a domain dict stored as JSON -- e.g. youtube's
    per-video {channel, title, url, upload_date}. Building-codes passes None.
    """
    conn.execute(
        "INSERT OR REPLACE INTO files(path, sha256, size, indexed_at, metadata) "
        "VALUES(?, ?, ?, ?, ?)",
        (path, sha256, int(size), int(time.time()),
         json.dumps(metadata) if metadata is not None else None),
    )


def purge_file(conn: sqlite3.Connection, path: str) -> None:
    """Delete chunks + vec + fts + section rows for this path, then files row."""
    cur = conn.cursor()
    cur.execute("SELECT id FROM chunks WHERE path=?", (path,))
    ids = [r[0] for r in cur.fetchall()]
    for i in range(0, len(ids), 500):
        batch = ids[i:i + 500]
        ph = ",".join("?" * len(batch))
        cur.execute("DELETE FROM vec_chunks WHERE chunk_id IN (%s)" % ph, batch)
        cur.execute("DELETE FROM fts_chunks WHERE rowid IN (%s)" % ph, batch)
    cur.execute("DELETE FROM chunks WHERE path=?", (path,))
    cur.execute("DELETE FROM section_nodes WHERE path=?", (path,))
    cur.execute("DELETE FROM section_refs WHERE path=?", (path,))
    cur.execute("DELETE FROM extraction_cache WHERE path=?", (path,))
    cur.execute("DELETE FROM files WHERE path=?", (path,))


def _pack_vec(vec: Iterable[float]) -> bytes:
    floats = list(vec)
    return struct.pack("%df" % len(floats), *floats)


def insert_chunks(
    conn: sqlite3.Connection,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> list[int]:
    """Insert leaf chunks + embeddings + fts rows. Returns ids. No commit."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            "chunks/embeddings length mismatch: %d vs %d"
            % (len(chunks), len(embeddings))
        )
    dims = embedding_dims()
    now = int(time.time())
    inserted: list[int] = []
    cur = conn.cursor()
    for ch, vec in zip(chunks, embeddings):
        if len(vec) != dims:
            raise ValueError(
                "embedding dim %d != configured %d for path=%s"
                % (len(vec), dims, ch.get("path"))
            )
        meta = ch.get("metadata")
        cur.execute(
            "INSERT INTO chunks(path, corpus, source_file, kind, page, "
            "start_line, end_line, text, context_summary, content_hash, "
            "indexed_at, section_id, section_number, section_title, "
            "breadcrumb, jurisdiction, edition, parent_section_id, node_type, "
            "metadata) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ch["path"], ch["corpus"], ch["source_file"],
                ch.get("kind", "document"), ch.get("page"),
                ch.get("start_line"), ch.get("end_line"),
                ch["text"], ch.get("context_summary"),
                ch["content_hash"], now,
                ch.get("section_id"), ch.get("section_number"),
                ch.get("section_title"), ch.get("breadcrumb"),
                ch.get("jurisdiction"), ch.get("edition"),
                ch.get("parent_section_id"), ch.get("node_type", "leaf"),
                json.dumps(meta) if meta is not None else None,
            ),
        )
        chunk_id = cur.lastrowid
        inserted.append(chunk_id)
        cur.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES(?, ?)",
            (chunk_id, _pack_vec(vec)),
        )
        cur.execute(
            "INSERT INTO fts_chunks(rowid, text, meta, source_file, kind) "
            "VALUES(?, ?, ?, ?, ?)",
            (chunk_id, ch["text"], ch.get("meta") or "",
             ch["source_file"], ch.get("kind", "document")),
        )
    return inserted


def insert_section_nodes(conn: sqlite3.Connection, nodes: list[dict]) -> None:
    """Insert-or-replace parent-document section rows. No commit."""
    now = int(time.time())
    cur = conn.cursor()
    for n in nodes:
        cur.execute(
            "INSERT OR REPLACE INTO section_nodes(corpus, path, jurisdiction, "
            "edition, section_id, section_number, section_title, breadcrumb, "
            "parent_section_id, full_text, indexed_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                n["corpus"], n["path"], n.get("jurisdiction"), n.get("edition"),
                n["section_id"], n.get("section_number"), n.get("section_title"),
                n.get("breadcrumb"), n.get("parent_section_id"),
                n.get("full_text") or "", now,
            ),
        )


def insert_section_refs(conn: sqlite3.Connection, refs: list[dict]) -> None:
    """Insert citation-graph edges. No commit."""
    cur = conn.cursor()
    for r in refs:
        cur.execute(
            "INSERT OR REPLACE INTO section_refs(src_section_id, dst_raw, "
            "dst_kind, dst_section_id, corpus, path) VALUES(?,?,?,?,?,?)",
            (r["src_section_id"], r["dst_raw"], r.get("dst_kind"),
             r.get("dst_section_id"), r["corpus"], r["path"]),
        )


def get_section(conn: sqlite3.Connection, corpus: str, section_id: str) -> dict | None:
    """Fetch one section_nodes row by (corpus, section_id) as a dict."""
    cur = conn.execute(
        "SELECT corpus, path, jurisdiction, edition, section_id, "
        "section_number, section_title, breadcrumb, parent_section_id, "
        "full_text FROM section_nodes WHERE corpus=? AND section_id=?",
        (corpus, section_id),
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def get_refs(conn: sqlite3.Connection, src_section_id: str) -> list[dict]:
    """Resolved outgoing citation edges for a section (dst_section_id NOT NULL)."""
    cur = conn.execute(
        "SELECT dst_raw, dst_kind, dst_section_id FROM section_refs "
        "WHERE src_section_id=? AND dst_section_id IS NOT NULL",
        (src_section_id,),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def get_extraction(conn: sqlite3.Connection, path: str, extraction_kind: str,
                   content_hash: str, model: str) -> dict | None:
    """Return a cached per-doc extraction (parsed JSON) or None on miss.

    Keyed on content_hash + model so a changed transcript or a model switch is a
    natural cache miss (no stale results)."""
    row = conn.execute(
        "SELECT result FROM extraction_cache WHERE path=? AND extraction_kind=? "
        "AND content_hash=? AND model=?",
        (path, extraction_kind, content_hash, model),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except (ValueError, TypeError):
        return None


def put_extraction(conn: sqlite3.Connection, path: str, corpus: str,
                   content_hash: str, extraction_kind: str, model: str,
                   result: object) -> None:
    """Cache a per-doc extraction (JSON-serializable). Does NOT commit."""
    conn.execute(
        "INSERT OR REPLACE INTO extraction_cache(path, corpus, content_hash, "
        "extraction_kind, model, result, created_at) VALUES(?,?,?,?,?,?,?)",
        (path, corpus, content_hash, extraction_kind, model,
         json.dumps(result), int(time.time())),
    )


def stats(conn: sqlite3.Connection, corpus: str) -> dict:
    file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    chunk_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE corpus=?", (corpus,)
    ).fetchone()[0]
    vec_count = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM fts_chunks").fetchone()[0]
    section_count = conn.execute(
        "SELECT COUNT(*) FROM section_nodes WHERE corpus=?", (corpus,)
    ).fetchone()[0]
    ref_count = conn.execute(
        "SELECT COUNT(*) FROM section_refs WHERE corpus=?", (corpus,)
    ).fetchone()[0]

    db_path = None
    for row in conn.execute("PRAGMA database_list"):
        if row[1] == "main":
            db_path = row[2]
            break
    db_size = os.path.getsize(db_path) if db_path and os.path.isfile(db_path) else 0

    return {
        "file_count": int(file_count),
        "chunk_count": int(chunk_count),
        "vec_count": int(vec_count),
        "fts_count": int(fts_count),
        "section_count": int(section_count),
        "ref_count": int(ref_count),
        "db_size_bytes": int(db_size),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> int:
    import random

    try:
        open_db("../etc")
    except ValueError:
        pass
    else:
        raise AssertionError("open_db('../etc') should have raised ValueError")

    tmp = Path(tempfile.gettempdir()) / "docrag_db_selftest.db"
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(tmp) + suffix)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    corpus = "selftest"
    dims = embedding_dims()
    conn = open_db(corpus, db_path=tmp)
    try:
        upsert_file(conn, "doc.html", "deadbeef" * 8, 12345)
        assert file_row(conn, "doc.html") == ("deadbeef" * 8, 12345)
        rng = random.Random(1337)
        chunk_dicts = [
            {"path": "doc.html", "corpus": corpus, "source_file": "NC 2024 Building Code",
             "kind": "document", "page": None, "text": "alpha bravo charlie",
             "meta": "R101.1 Title", "context_summary": "c1", "content_hash": "h1",
             "section_id": "Sec101.1", "section_number": "101.1",
             "section_title": "Title", "breadcrumb": "Ch1 > 101 > 101.1",
             "jurisdiction": "north-carolina", "edition": "NC 2024",
             "parent_section_id": "Sec101", "node_type": "leaf"},
            {"path": "doc.html", "corpus": corpus, "source_file": "NC 2024 Building Code",
             "kind": "document", "page": None, "text": "delta echo RX304 foxtrot",
             "meta": "101.2 Scope", "context_summary": "c2", "content_hash": "h2",
             "section_id": "Sec101.2", "section_number": "101.2",
             "section_title": "Scope", "breadcrumb": "Ch1 > 101 > 101.2",
             "jurisdiction": "north-carolina", "edition": "NC 2024",
             "parent_section_id": "Sec101", "node_type": "leaf",
             "metadata": {"start_time": 12.5, "channel": "selftest"}},
        ]
        embs = [[rng.uniform(-1, 1) for _ in range(dims)] for _ in chunk_dicts]
        ids = insert_chunks(conn, chunk_dicts, embs)
        assert len(ids) == 2

        insert_section_nodes(conn, [{
            "corpus": corpus, "path": "doc.html", "jurisdiction": "north-carolina",
            "edition": "NC 2024", "section_id": "Sec101", "section_number": "101",
            "section_title": "General", "breadcrumb": "Ch1 > 101",
            "parent_section_id": "Ch1",
            "full_text": "alpha bravo charlie delta echo RX304 foxtrot"}])
        insert_section_refs(conn, [{
            "src_section_id": "Sec101.2", "dst_raw": "160D-1110",
            "dst_kind": "ncgs", "dst_section_id": "SecNCGS", "corpus": corpus,
            "path": "doc.html"}])
        conn.commit()

        s = stats(conn, corpus)
        assert s["chunk_count"] == 2 and s["vec_count"] == 2 and s["fts_count"] == 2, s
        assert s["section_count"] == 1 and s["ref_count"] == 1, s

        parent = get_section(conn, corpus, "Sec101")
        assert parent and "RX304" in parent["full_text"], parent
        refs = get_refs(conn, "Sec101.2")
        assert refs and refs[0]["dst_raw"] == "160D-1110", refs

        # metadata JSON round-trips on the chunk row.
        meta_row = conn.execute(
            "SELECT metadata FROM chunks WHERE content_hash='h2'").fetchone()
        assert meta_row and json.loads(meta_row[0])["start_time"] == 12.5, meta_row

        # extraction_cache: miss -> put -> hit; model/hash mismatch -> miss.
        assert get_extraction(conn, "doc.html", "summary", "h2", "m1") is None
        put_extraction(conn, "doc.html", corpus, "h2", "summary", "m1",
                       {"points": ["a", "b"]})
        conn.commit()
        got = get_extraction(conn, "doc.html", "summary", "h2", "m1")
        assert got == {"points": ["a", "b"]}, got
        assert get_extraction(conn, "doc.html", "summary", "h2", "m2") is None
        assert get_extraction(conn, "doc.html", "summary", "OTHER", "m1") is None

        probe = _pack_vec(embs[0])
        vec_rows = conn.execute(
            "SELECT chunk_id FROM vec_chunks WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT 5", (probe,),
        ).fetchall()
        assert vec_rows and vec_rows[0][0] == ids[0], vec_rows

        fts_rows = conn.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ?", ("RX304",),
        ).fetchall()
        assert ids[1] in {r[0] for r in fts_rows}, fts_rows

        # meta column is searchable too (section number match).
        meta_rows = conn.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ?", ("meta:Scope",),
        ).fetchall()
        assert ids[1] in {r[0] for r in meta_rows}, meta_rows

        purge_file(conn, "doc.html")
        conn.commit()
        s2 = stats(conn, corpus)
        for k in ("file_count", "chunk_count", "vec_count", "fts_count",
                  "section_count", "ref_count"):
            assert s2[k] == 0, (k, s2[k])
        ec = conn.execute("SELECT COUNT(*) FROM extraction_cache").fetchone()[0]
        assert ec == 0, ("extraction_cache not purged", ec)
    finally:
        conn.close()
        for suffix in ("", "-wal", "-shm", "-journal"):
            p = Path(str(tmp) + suffix)
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    print("ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag.db")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.self_test:
        return _self_test()
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
