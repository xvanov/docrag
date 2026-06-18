"""index.py -- CLI to build / inspect / purge a corpus index.

Wires extract -> chunk -> embed -> upsert into the per-corpus SQLite-vec DB.
Incremental: files whose SHA256 matches the prior index are skipped.

CLI:
  python -m rag.index build  --corpus udo [--full] [--limit N]
                                 [--confirm] [--dry-run]
  python -m rag.index status --corpus udo
  python -m rag.index purge  --corpus udo --file <relpath>

Exit codes:
  0 success | 1 config/IO error | 2 bad usage
  3 Azure credentials missing | 4 cost guardrail tripped without --confirm
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback
from typing import Optional

from . import settings
from .chunk import chunk_extracted
from .db import (
    file_row, insert_chunks, insert_section_nodes, insert_section_refs,
    open_db, purge_file, schema_outdated, stats, upsert_file,
)
from .embed import EmbedError, embed_batch
from .extract import extract_file, iter_eligible_files, iter_skipped_files


EMBED_PRICE_PER_MTOKEN = 0.13          # text-embedding-3-large, Azure list price
COST_CONFIRM_THRESHOLD = 1.00          # USD per-corpus gate
EMBED_FLUSH_SIZE = 256
MAX_EMBED_INPUT_CHARS = 30000


def _sha256_size(abs_path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(abs_path, "rb") as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def _chunks_exist_for(conn, rel_path: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM chunks WHERE path=? LIMIT 1", (rel_path,)
    ).fetchone() is not None


def _check_azure_creds() -> None:
    if settings.azure_endpoint() and settings.azure_api_key():
        return
    sys.stderr.write(
        "ERROR: Azure OpenAI credentials missing.\n"
        "Add these to .env at the repo root (see .env.example):\n"
        "  AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/\n"
        "  AZURE_OPENAI_API_KEY=<key>\n"
        "  AZURE_OPENAI_API_VERSION=2024-10-21\n"
        "  AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large\n"
    )
    sys.exit(3)


def _human(num_bytes: int) -> str:
    if num_bytes < 1024:
        return "%d B" % num_bytes
    if num_bytes < 1024 * 1024:
        return "%d KB" % (num_bytes // 1024)
    return "%.1f MB" % (num_bytes / (1024 * 1024))


def _db_path_for(corpus: str) -> str:
    return os.path.join(settings.index_dir(), "%s.db" % corpus)


def _guard_embed_input(text: str, rel_path: str) -> str:
    if len(text) <= MAX_EMBED_INPUT_CHARS:
        return text
    sys.stderr.write(
        "[index] truncating oversize embed_input for %s (%d -> %d)\n"
        % (rel_path, len(text), MAX_EMBED_INPUT_CHARS)
    )
    return text[: MAX_EMBED_INPUT_CHARS - 3] + "..."


def _resolve_refs(conn, corpus: str) -> int:
    """Resolve section_refs.dst_section_id from dst_raw, after all sections
    exist. Internal refs match a section_number (same path preferred); NCGS
    refs match the section whose full_text most contains the statute number.
    Returns the count newly resolved."""
    rows = conn.execute(
        "SELECT section_id, section_number, path FROM section_nodes "
        "WHERE corpus=? AND section_number IS NOT NULL", (corpus,)
    ).fetchall()
    by_num: dict[str, list[tuple[str, str]]] = {}
    for sid, num, path in rows:
        by_num.setdefault(num, []).append((sid, path))
        # also index without a leading letter (R105.2 -> 105.2) for cross-code refs
        bare = num.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if bare != num:
            by_num.setdefault(bare, []).append((sid, path))

    pending = conn.execute(
        "SELECT rowid, src_section_id, dst_raw, dst_kind, path FROM section_refs "
        "WHERE corpus=? AND dst_section_id IS NULL", (corpus,)
    ).fetchall()
    n = 0
    cur = conn.cursor()
    for rowid, src, raw, kind, path in pending:
        target = None
        if kind == "internal_section":
            cands = by_num.get(raw) or by_num.get(raw.lstrip("R")) or []
            same = [sid for sid, p in cands if p == path and sid != src]
            other = [sid for sid, p in cands if sid != src]
            target = (same or other or [None])[0]
        elif kind == "ncgs":
            row = cur.execute(
                "SELECT section_id FROM section_nodes WHERE corpus=? "
                "AND section_id<>? AND full_text LIKE ? "
                "ORDER BY length(full_text) DESC LIMIT 1",
                (corpus, src, "%" + raw + "%"),
            ).fetchone()
            target = row[0] if row else None
        if target:
            cur.execute("UPDATE section_refs SET dst_section_id=? WHERE rowid=?",
                        (target, rowid))
            n += 1
    return n


def _plan_work(conn, corpus: str, full: bool, limit: int):
    todo: list[dict] = []
    reasons: dict = {"hash_unchanged": 0}
    for entry in iter_eligible_files(corpus):
        abs_path = entry["abs_path"]
        rel = entry["path"]
        try:
            sha, size = _sha256_size(abs_path)
        except OSError as e:
            sys.stderr.write("[index] cannot read %s: %s\n" % (abs_path, e))
            continue
        existing = file_row(conn, rel)
        if (not full) and existing and existing[0] == sha and _chunks_exist_for(conn, rel):
            reasons["hash_unchanged"] += 1
            continue
        if existing is not None:
            purge_file(conn, rel)
        e2 = dict(entry, sha256=sha, size=size)
        todo.append(e2)
        if limit and len(todo) >= limit:
            break
    return todo, reasons


def _do_build(args) -> int:
    corpus = args.corpus.strip().lower()
    docs_root = settings.docs_root()
    corpus_dir = os.path.join(docs_root, corpus)
    print("[paths] docs:  %s" % docs_root)
    print("[paths] index: %s" % settings.index_dir())
    print("[corpus] %s" % corpus)

    if not os.path.isdir(corpus_dir):
        sys.stderr.write("ERROR: corpus dir not found: %s\n" % corpus_dir)
        return 1

    if not args.dry_run:
        _check_azure_creds()

    try:
        conn = open_db(corpus)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("ERROR: cannot open DB for %s: %s\n" % (corpus, e))
        traceback.print_exc(file=sys.stderr)
        return 1

    # A schema bump invalidates old rows (chunk boundaries + columns changed).
    # Force a full reindex so every file is re-chunked under the new schema.
    if schema_outdated(conn) and not args.full:
        sys.stderr.write("[index] schema changed -> forcing --full reindex\n")
        args.full = True

    try:
        todo, reasons = _plan_work(conn, corpus, full=args.full, limit=args.limit)
        conn.commit()

        # Extract + chunk every todo file so we can price it.
        prepared: list[dict] = []
        total_chunks = 0
        total_chars = 0
        for entry in todo:
            extracted = extract_file(entry)
            if extracted.get("no_text"):
                sys.stderr.write("[index] no_text: %s\n" % entry["path"])
                upsert_file(conn, entry["path"], entry["sha256"], entry["size"])
                reasons["no_text_extracted"] = reasons.get("no_text_extracted", 0) + 1
                continue
            result = chunk_extracted(corpus, entry, extracted)
            leaves = result.get("chunks") or []
            if not leaves:
                sys.stderr.write("[index] no_chunks: %s\n" % entry["path"])
                upsert_file(conn, entry["path"], entry["sha256"], entry["size"])
                reasons["no_chunks"] = reasons.get("no_chunks", 0) + 1
                continue
            for c in leaves:
                c["embed_input"] = _guard_embed_input(c["embed_input"], entry["path"])
                total_chars += len(c["embed_input"])
            prepared.append({"entry": entry, "chunks": leaves,
                             "sections": result.get("sections") or [],
                             "refs": result.get("refs") or []})
            total_chunks += len(leaves)
        conn.commit()

        est_tokens = total_chars // 4 if total_chars else 0
        est_usd = (est_tokens / 1_000_000.0) * EMBED_PRICE_PER_MTOKEN
        print("[plan] %d files, %d chunks, ~%d tokens, ~$%.2f"
              % (len(prepared), total_chunks, est_tokens, est_usd))

        if est_usd > COST_CONFIRM_THRESHOLD and not args.confirm and not args.dry_run:
            sys.stderr.write(
                "ERROR: estimated cost ~$%.2f exceeds $%.2f. "
                "Re-run with --confirm or --dry-run.\n"
                % (est_usd, COST_CONFIRM_THRESHOLD)
            )
            return 4

        if args.dry_run:
            print("[dry-run] skipping embedding + insert")
            for item in prepared:
                sys.stderr.write("[dry-run] %s (%d chunks)\n"
                                 % (item["entry"]["path"], len(item["chunks"])))
            _print_summary(corpus, 0, 0, reasons, est_tokens, est_usd)
            return 0

        # Embed in batched flushes.
        file_embeddings: dict = {i: [] for i in range(len(prepared))}
        buffer: list[tuple[int, dict]] = []
        used_tokens = 0

        def flush(buf):
            nonlocal used_tokens
            if not buf:
                return
            texts = [c["embed_input"] for (_, c) in buf]
            vectors = embed_batch(texts)
            if len(vectors) != len(buf):
                raise RuntimeError(
                    "embed_batch returned %d vectors for %d texts"
                    % (len(vectors), len(buf))
                )
            for (idx, _c), vec in zip(buf, vectors):
                file_embeddings[idx].append(vec)
            used_tokens += sum(len(t) // 4 for t in texts)

        for idx, item in enumerate(prepared):
            for c in item["chunks"]:
                buffer.append((idx, c))
                if len(buffer) >= EMBED_FLUSH_SIZE:
                    flush(buffer)
                    buffer = []
        flush(buffer)

        indexed_files = 0
        indexed_chunks = 0
        for idx, item in enumerate(prepared):
            entry = item["entry"]
            chunks = item["chunks"]
            vecs = file_embeddings[idx]
            if len(vecs) != len(chunks):
                sys.stderr.write(
                    "[index] dim-mismatch for %s: %d chunks / %d vectors -- skip\n"
                    % (entry["path"], len(chunks), len(vecs))
                )
                continue
            upsert_file(conn, entry["path"], entry["sha256"], entry["size"])
            insert_section_nodes(conn, item.get("sections") or [])
            insert_chunks(conn, chunks, vecs)
            insert_section_refs(conn, item.get("refs") or [])
            conn.commit()
            indexed_files += 1
            indexed_chunks += len(chunks)
            sys.stderr.write("[index] %s/%s: %d chunks / %d sections\n"
                             % (corpus, entry["path"], len(chunks),
                                len(item.get("sections") or [])))

        # Resolve citation-graph edges now that every section exists.
        resolved = _resolve_refs(conn, corpus)
        conn.commit()
        if resolved:
            sys.stderr.write("[index] resolved %d citation refs\n" % resolved)

        used_usd = (used_tokens / 1_000_000.0) * EMBED_PRICE_PER_MTOKEN
        _print_summary(corpus, indexed_files, indexed_chunks, reasons,
                       used_tokens or est_tokens, used_usd if used_tokens else est_usd)
        return 0
    except EmbedError as e:
        sys.stderr.write("ERROR: embedding failed: %s\n" % e)
        return 1
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("ERROR: %s\n" % e)
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def _count_skipped(corpus: str, reasons: dict) -> dict:
    out = dict(reasons)
    for s in iter_skipped_files(corpus):
        r = s["reason"]
        out[r] = out.get(r, 0) + 1
    return out


def _print_summary(corpus, indexed_files, indexed_chunks, reasons,
                   tokens, usd) -> None:
    skip = _count_skipped(corpus, reasons)
    print("[done] %s" % corpus)
    print("  indexed:    %d files / %d chunks" % (indexed_files, indexed_chunks))
    print("  skipped:    %d files" % sum(skip.values()))
    for key in sorted(skip):
        if skip[key]:
            print("    - %-20s %d" % (key + ":", skip[key]))
    print("  tokens:     %d  (~$%.2f)" % (tokens, usd))
    print("  db:         %s" % _db_path_for(corpus))


def _do_status(args) -> int:
    corpus = args.corpus.strip().lower()
    db_path = _db_path_for(corpus)
    if not os.path.isfile(db_path):
        sys.stderr.write("ERROR: no index DB for %s at %s\n" % (corpus, db_path))
        return 1
    conn = open_db(corpus)
    try:
        s = stats(conn, corpus)
        eligible = sum(1 for _ in iter_eligible_files(corpus))
        print("[status] %s" % corpus)
        print("  db path:        %s  (%s)" % (db_path, _human(s["db_size_bytes"])))
        print("  files indexed:  %d" % s["file_count"])
        print("  chunks:         %d (vec=%d fts=%d)"
              % (s["chunk_count"], s["vec_count"], s["fts_count"]))
        print("  files on disk:  %d" % eligible)
        return 0
    finally:
        conn.close()


def _do_purge(args) -> int:
    corpus = args.corpus.strip().lower()
    rel = args.file.replace("\\", "/").strip()
    if not rel:
        sys.stderr.write("ERROR: --file must be non-empty.\n")
        return 2
    conn = open_db(corpus)
    try:
        if file_row(conn, rel) is None:
            sys.stderr.write("[purge] %s/%s: not indexed (no-op)\n" % (corpus, rel))
            return 0
        purge_file(conn, rel)
        conn.commit()
        print("[purge] %s/%s: removed" % (corpus, rel))
        return 0
    finally:
        conn.close()


def _parse_args(argv: Optional[list]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="rag.index",
                                description="Build / inspect / purge a corpus index.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Index a corpus folder.")
    pb.add_argument("--corpus", required=True)
    pb.add_argument("--full", action="store_true",
                    help="Reindex every file even if hash matches.")
    pb.add_argument("--limit", type=int, default=0,
                    help="Only process the first N (re)index candidates.")
    pb.add_argument("--confirm", action="store_true",
                    help="Confirm spend above the cost guardrail.")
    pb.add_argument("--dry-run", action="store_true",
                    help="Plan + estimate cost only; no embed or insert.")

    ps = sub.add_parser("status", help="Show index stats for a corpus.")
    ps.add_argument("--corpus", required=True)

    pp = sub.add_parser("purge", help="Remove all chunks for one file.")
    pp.add_argument("--corpus", required=True)
    pp.add_argument("--file", required=True, help="Relative path stored in chunks.path.")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    if args.cmd == "build":
        return _do_build(args)
    if args.cmd == "status":
        return _do_status(args)
    if args.cmd == "purge":
        return _do_purge(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
