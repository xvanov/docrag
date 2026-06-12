"""SQLite + sqlite-vec + FTS5 index schema for docrag.

One DB file per corpus: ``{index_dir}/{corpus}.db``.

  files       (path, sha256, size, indexed_at)
  chunks      (id, path, corpus, source_file, kind, page,
               start_line, end_line, text, context_summary,
               content_hash, indexed_at)
  vec_chunks  (chunk_id, embedding[D])      -- sqlite-vec virtual
  fts_chunks  (text, source_file, kind)     -- FTS5 virtual, content='chunks'

The only module that talks to SQLite.

CLI:
  python -m docrag.db --self-test
"""

from __future__ import annotations

import argparse
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


def _ensure_schema(conn: sqlite3.Connection) -> None:
    dims = embedding_dims()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS files (
      path TEXT PRIMARY KEY,
      sha256 TEXT NOT NULL,
      size INTEGER NOT NULL,
      indexed_at INTEGER NOT NULL
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
      FOREIGN KEY(path) REFERENCES files(path) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_path   ON chunks(path);
    CREATE INDEX IF NOT EXISTS idx_chunks_corpus ON chunks(corpus);
    """)

    cur.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
        "chunk_id INTEGER PRIMARY KEY, embedding FLOAT[%d])" % dims
    )
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
      text,
      source_file UNINDEXED,
      kind UNINDEXED,
      content='chunks',
      content_rowid='id'
    );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def file_row(conn: sqlite3.Connection, path: str) -> tuple[str, int] | None:
    cur = conn.execute("SELECT sha256, size FROM files WHERE path=?", (path,))
    row = cur.fetchone()
    return (row[0], row[1]) if row else None


def upsert_file(conn: sqlite3.Connection, path: str, sha256: str, size: int) -> None:
    """Insert-or-replace a files row. Does NOT commit."""
    conn.execute(
        "INSERT OR REPLACE INTO files(path, sha256, size, indexed_at) "
        "VALUES(?, ?, ?, ?)",
        (path, sha256, int(size), int(time.time())),
    )


def purge_file(conn: sqlite3.Connection, path: str) -> None:
    """Delete chunks + vec + fts rows for this path, then the files row."""
    cur = conn.cursor()
    cur.execute("SELECT id FROM chunks WHERE path=?", (path,))
    ids = [r[0] for r in cur.fetchall()]
    for i in range(0, len(ids), 500):
        batch = ids[i:i + 500]
        ph = ",".join("?" * len(batch))
        cur.execute("DELETE FROM vec_chunks WHERE chunk_id IN (%s)" % ph, batch)
        cur.execute("DELETE FROM fts_chunks WHERE rowid IN (%s)" % ph, batch)
    cur.execute("DELETE FROM chunks WHERE path=?", (path,))
    cur.execute("DELETE FROM files WHERE path=?", (path,))


def _pack_vec(vec: Iterable[float]) -> bytes:
    floats = list(vec)
    return struct.pack("%df" % len(floats), *floats)


def insert_chunks(
    conn: sqlite3.Connection,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> list[int]:
    """Insert chunks + embeddings + fts rows. Returns ids. Does NOT commit."""
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
        cur.execute(
            "INSERT INTO chunks(path, corpus, source_file, kind, page, "
            "start_line, end_line, text, context_summary, content_hash, "
            "indexed_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ch["path"], ch["corpus"], ch["source_file"],
                ch.get("kind", "document"), ch.get("page"),
                ch.get("start_line"), ch.get("end_line"),
                ch["text"], ch.get("context_summary"),
                ch["content_hash"], now,
            ),
        )
        chunk_id = cur.lastrowid
        inserted.append(chunk_id)
        cur.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES(?, ?)",
            (chunk_id, _pack_vec(vec)),
        )
        cur.execute(
            "INSERT INTO fts_chunks(rowid, text, source_file, kind) "
            "VALUES(?, ?, ?, ?)",
            (chunk_id, ch["text"], ch["source_file"], ch.get("kind", "document")),
        )
    return inserted


def stats(conn: sqlite3.Connection, corpus: str) -> dict:
    file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    chunk_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE corpus=?", (corpus,)
    ).fetchone()[0]
    vec_count = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM fts_chunks").fetchone()[0]

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
        upsert_file(conn, "doc.pdf", "deadbeef" * 8, 12345)
        assert file_row(conn, "doc.pdf") == ("deadbeef" * 8, 12345)
        rng = random.Random(1337)
        chunk_dicts = [
            {"path": "doc.pdf", "corpus": corpus, "source_file": "doc.pdf",
             "kind": "document", "page": 1, "text": "alpha bravo charlie",
             "context_summary": "p1", "content_hash": "h1"},
            {"path": "doc.pdf", "corpus": corpus, "source_file": "doc.pdf",
             "kind": "document", "page": 2, "text": "delta echo RX304 foxtrot",
             "context_summary": "p2", "content_hash": "h2"},
        ]
        embs = [[rng.uniform(-1, 1) for _ in range(dims)] for _ in chunk_dicts]
        ids = insert_chunks(conn, chunk_dicts, embs)
        assert len(ids) == 2
        conn.commit()

        s = stats(conn, corpus)
        assert s == {**s, "file_count": 1, "chunk_count": 2,
                     "vec_count": 2, "fts_count": 2}, s
        assert s["db_size_bytes"] > 0

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

        purge_file(conn, "doc.pdf")
        conn.commit()
        s2 = stats(conn, corpus)
        for k in ("file_count", "chunk_count", "vec_count", "fts_count"):
            assert s2[k] == 0, (k, s2[k])
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
    p = argparse.ArgumentParser(prog="docrag.db")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.self_test:
        return _self_test()
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
