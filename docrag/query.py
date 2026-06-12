"""query.py -- Hybrid (vector + BM25) retrieval over a corpus index.

Pipeline:
  1. Encode the query via embed.embed_one().
  2. Vector top-50 from vec_chunks (sqlite-vec MATCH).
  3. BM25 top-50 from fts_chunks (FTS5).
  4. Reciprocal Rank Fusion (k=60).
  5. Optional post-filters (kind whitelist, source_file glob).
  6. Word-boundary post-filter for short uppercase codes (anti-hallucination):
     if the query contains a token like ``^[A-Z0-9]{2,8}$`` with >=1 uppercase
     letter, drop chunks lacking a word-boundary match for at least one such
     token. Pure prose queries skip this filter.
  7. Trim to top_k.

Status values: "ok" | "no_results" | "low_confidence".

Public API:
    rag_query(corpus, query, top_k=12, filters=None) -> dict
"""

from __future__ import annotations

import fnmatch
import re
import struct

from .db import open_db
from .embed import embed_one


RRF_K = 60
VEC_TOP_K = 50
BM25_TOP_K = 50
LOW_CONFIDENCE_THRESHOLD = 0.01

_SHORT_CODE_RE = re.compile(r"^(?=[A-Z0-9]{2,8}$)[A-Z0-9]*[A-Z][A-Z0-9]*$")
# FTS5 syntax characters we strip to keep the query a plain token bag.
_FTS_SYNTAX_RE = re.compile(r'[\"\(\)\*\:\^]')


def _pack_vec(vec) -> bytes:
    floats = list(vec)
    return struct.pack("%df" % len(floats), *floats)


def _fts_query(raw: str) -> str:
    """Sanitize a user query into an FTS5 OR-of-tokens MATCH string."""
    cleaned = _FTS_SYNTAX_RE.sub(" ", raw or "")
    tokens = [t for t in re.split(r"\s+", cleaned) if t]
    if not tokens:
        return ""
    # Quote each token so FTS treats it literally; join with OR.
    return " OR ".join('"%s"' % t.replace('"', "") for t in tokens)


def _short_codes(query: str) -> list[str]:
    out = []
    for tok in re.split(r"[\s,;:()\[\]{}/]+", query or ""):
        tok = tok.strip(".-'\"")
        if tok and _SHORT_CODE_RE.match(tok):
            out.append(tok)
    return out


def _row_to_chunk(row: dict, score: float) -> dict:
    return {
        "chunk_id": row["id"],
        "path": row["path"],
        "source_file": row["source_file"],
        "kind": row["kind"],
        "page": row["page"],
        "start_line": row["start_line"],
        "end_line": row["end_line"],
        "text": row["text"],
        "score": score,
    }


def rag_query(corpus: str, query: str, top_k: int = 12,
              filters: dict | None = None) -> dict:
    """Hybrid retrieve top_k chunks for ``query`` from ``corpus``."""
    query = (query or "").strip()
    if not query:
        return {"status": "no_results", "results": []}

    conn = open_db(corpus)
    try:
        # --- Vector ranking -------------------------------------------------
        qvec = embed_one(query)
        vec_rows = conn.execute(
            "SELECT chunk_id FROM vec_chunks WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            (_pack_vec(qvec), VEC_TOP_K),
        ).fetchall()
        vec_rank = {r[0]: i + 1 for i, r in enumerate(vec_rows)}

        # --- BM25 ranking ---------------------------------------------------
        bm25_rank: dict[int, int] = {}
        fts_match = _fts_query(query)
        if fts_match:
            try:
                fts_rows = conn.execute(
                    "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (fts_match, BM25_TOP_K),
                ).fetchall()
                bm25_rank = {r[0]: i + 1 for i, r in enumerate(fts_rows)}
            except Exception:  # noqa: BLE001 -- malformed FTS query, vec-only
                bm25_rank = {}

        # --- Reciprocal Rank Fusion ----------------------------------------
        fused: dict[int, float] = {}
        for cid, rank in vec_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        for cid, rank in bm25_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        if not fused:
            return {"status": "no_results", "results": []}

        ordered_ids = sorted(fused, key=lambda c: fused[c], reverse=True)

        # --- Hydrate chunk rows --------------------------------------------
        ph = ",".join("?" * len(ordered_ids))
        cur = conn.execute(
            "SELECT id, path, source_file, kind, page, start_line, end_line, "
            "text FROM chunks WHERE id IN (%s)" % ph,
            ordered_ids,
        )
        cols = [d[0] for d in cur.description]
        by_id = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}

        results = []
        for cid in ordered_ids:
            row = by_id.get(cid)
            if row:
                results.append(_row_to_chunk(row, fused[cid]))

        # --- Post-filters ---------------------------------------------------
        filters = filters or {}
        types = filters.get("types")
        if types:
            allow = set(types)
            results = [r for r in results if r["kind"] in allow]
        file_glob = filters.get("file_glob")
        if file_glob:
            results = [r for r in results
                       if fnmatch.fnmatch(r["source_file"] or "", file_glob)]

        # --- Short-code word-boundary filter -------------------------------
        codes = _short_codes(query)
        if codes:
            patterns = [re.compile(r"\b%s\b" % re.escape(c)) for c in codes]
            kept = [r for r in results
                    if any(p.search(r["text"] or "") for p in patterns)]
            # Only apply if it doesn't wipe out everything (semantic hits may
            # legitimately paraphrase the code).
            if kept:
                results = kept

        if not results:
            return {"status": "no_results", "results": []}

        top_score = results[0]["score"]
        results = results[:top_k]
        status = "low_confidence" if top_score < LOW_CONFIDENCE_THRESHOLD else "ok"
        return {"status": status, "results": results}
    finally:
        conn.close()
