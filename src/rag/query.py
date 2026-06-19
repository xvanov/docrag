"""query.py -- hybrid retrieval + rerank + parent-document collapse.

Pipeline (2026 best-practice):
  1. Encode the query; vector top-50 (sqlite-vec) + BM25 top-50 (FTS5, with the
     section-metadata column down-weighted vs. body).
  2. Reciprocal Rank Fusion (k=60) over leaf chunks.
  3. Cross-encoder rerank of the fused leaf candidates (optional; no-op if the
     reranker isn't installed -- see rerank.py).
  4. Collapse leaves to their section's parent-document text (small-to-big),
     dedup, drop ancestors subsumed by a retained descendant.
  5. Adaptive jurisdiction balancing (skip when one jurisdiction dominates).
  6. One-hop citation-graph expansion: pull resolved cross-referenced sections
     ("see NCGS 160D-1110") as labeled supporting context.
  7. Trim to top_k.

Status: "ok" | "no_results" | "low_confidence".

Public API:
    rag_query(corpus, query, top_k=8, filters=None, balance=False) -> dict
"""

from __future__ import annotations

import fnmatch
import json
import re
import struct

from . import settings
from .db import get_refs, get_section, open_db
from .embed import embed_one
from .facets import facet_of, location_allows
from .rerank import rerank

RRF_K = 60
VEC_TOP_K = 80
BM25_TOP_K = 80
# When a location / version filter is active, over-fetch candidates so the
# filtered pool isn't starved by excluded jurisdictions / editions.
VEC_TOP_K_FILTERED = 250
BM25_TOP_K_FILTERED = 250
RERANK_POOL = 72          # leaves reranked before collapse (stratified pool)
LOW_CONFIDENCE_RRF = 0.01
LOW_CONFIDENCE_RERANK = 0.02
MAX_REF_EXPANSION = 3     # extra cross-referenced sections per query
_REF_SCORE = -1.0         # sentinel: expansion sections sort last, ignore in stats

_FTS_SYNTAX_RE = re.compile(r'[\"\(\)\*\:\^]')

# Building-codes-specific retrieval helpers (jurisdiction bucketing/balancing,
# the short-code section-number filter, lay->code query expansion) live in the
# building_codes domain (domains/building_codes/retrieval.py) and are
# lazy-imported only inside the branches below that the caller opts into
# (balance / short_code_filter / expand) -- so a plain domain like youtube never
# imports them and the core retrieval skeleton stays generic.


def _pack_vec(vec) -> bytes:
    floats = list(vec)
    return struct.pack("%df" % len(floats), *floats)


def _fts_query(raw: str) -> str:
    cleaned = _FTS_SYNTAX_RE.sub(" ", raw or "")
    tokens = [t for t in re.split(r"\s+", cleaned) if t]
    if not tokens:
        return ""
    return " OR ".join('"%s"' % t.replace('"', "") for t in tokens)


def _row_to_result(row: dict, score: float, text: str, referenced: bool = False) -> dict:
    meta = row.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            meta = None
    return {
        "chunk_id": row.get("id"),
        "path": row.get("path"),
        "source_file": row.get("source_file"),
        "kind": row.get("node_type") or row.get("kind") or "document",
        "page": row.get("page"),
        "start_line": row.get("start_line"),
        "end_line": row.get("end_line"),
        "section_id": row.get("section_id"),
        "section_number": row.get("section_number"),
        "section_title": row.get("section_title"),
        "breadcrumb": row.get("breadcrumb"),
        "jurisdiction": row.get("jurisdiction"),
        "edition": row.get("edition"),
        "metadata": meta,            # domain-specific (youtube: start_time/url/...)
        "text": text,
        "score": score,
        "referenced": referenced,
    }


def _expand_refs_enabled() -> bool:
    val = (settings.get("DOCRAG_EXPAND_REFS", "1") or "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def _fuse_one(conn, qtext: str, vlim: int, blim: int) -> dict[int, float]:
    """Vector + BM25 -> RRF leaf scores for a single query string."""
    fused: dict[int, float] = {}
    qvec = embed_one(qtext)
    vec_rows = conn.execute(
        "SELECT chunk_id FROM vec_chunks WHERE embedding MATCH ? "
        "ORDER BY distance LIMIT ?", (_pack_vec(qvec), vlim),
    ).fetchall()
    for i, r in enumerate(vec_rows):
        fused[r[0]] = fused.get(r[0], 0.0) + 1.0 / (RRF_K + i + 1)
    fts_match = _fts_query(qtext)
    if fts_match:
        try:
            fts_rows = conn.execute(
                "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ? "
                "ORDER BY bm25(fts_chunks, 1.0, 0.3) LIMIT ?", (fts_match, blim),
            ).fetchall()
            for i, r in enumerate(fts_rows):
                fused[r[0]] = fused.get(r[0], 0.0) + 1.0 / (RRF_K + i + 1)
        except Exception:  # noqa: BLE001 -- malformed FTS, vec-only
            pass
    return fused


def rag_query(corpus: str, query: str, top_k: int = 8,
              filters: dict | None = None, balance: bool = False,
              expand: bool = True, short_code_filter: bool = True) -> dict:
    """Hybrid retrieve + rerank + parent-collapse top_k sections for ``query``.

    ``expand`` controls the built-in LLM query expansion. The agentic layer
    (reason.py) does its own hypothesis-directed expansion, so it calls with
    ``expand=False`` to avoid a redundant paraphrase round.

    ``short_code_filter`` is the building-codes section-number hard-filter (a
    token like "R705" must appear in results). Domains without section numbers
    (e.g. youtube transcripts) pass ``False`` so tokens like "5G"/"Q4" don't
    wrongly empty the pool.
    """
    query = (query or "").strip()
    if not query:
        return {"status": "no_results", "results": [], "balanced": False}

    filters = filters or {}
    _loc = filters.get("location")
    _versions = filters.get("versions") or {}
    _filtered = bool(_loc or _versions)
    vlim = VEC_TOP_K_FILTERED if _filtered else VEC_TOP_K
    blim = BM25_TOP_K_FILTERED if _filtered else BM25_TOP_K

    conn = open_db(corpus)
    try:
        # --- Multi-query: original + LLM code-vocabulary paraphrases --------
        # Bridges the lay-vs-code vocabulary gap (e.g. "shed" -> "accessory
        # building") so scope/exemption rules are recalled regardless of how
        # the user phrases the question. Each query contributes vector+BM25 RRF;
        # scores are summed (RRF over all query x retriever rankings).
        if expand:
            from .domains.building_codes.retrieval import _expand_queries
            queries = [query] + _expand_queries(query)
        else:
            queries = [query]
        fused: dict[int, float] = {}
        for qtext in queries:
            for cid, score in _fuse_one(conn, qtext, vlim, blim).items():
                fused[cid] = fused.get(cid, 0.0) + score
        if not fused:
            return {"status": "no_results", "results": [], "balanced": False}

        ordered_ids = sorted(fused, key=lambda c: fused[c], reverse=True)

        # --- Hydrate leaf rows ---------------------------------------------
        ph = ",".join("?" * len(ordered_ids))
        cur = conn.execute(
            "SELECT id, path, corpus, source_file, kind, page, start_line, "
            "end_line, text, section_id, section_number, section_title, "
            "breadcrumb, jurisdiction, edition, node_type, metadata "
            "FROM chunks WHERE id IN (%s)" % ph, ordered_ids,
        )
        cols = [d[0] for d in cur.description]
        by_id = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
        leaves = [by_id[cid] for cid in ordered_ids if cid in by_id]
        for lf in leaves:
            lf["rrf"] = fused[lf["id"]]

        # --- Filters --------------------------------------------------------
        if filters.get("types"):
            allow = set(filters["types"])
            leaves = [r for r in leaves if (r.get("node_type") or "leaf") in allow]
        if filters.get("file_glob"):
            g = filters["file_glob"]
            leaves = [r for r in leaves if fnmatch.fnmatch(r.get("source_file") or "", g)]
        # Source include/omit (UI toggle): drop chunks whose path matches any
        # excluded prefix/glob. Lets the user mute whole source groups (e.g. the
        # durhamnc.gov agency-guidance pages) without reindexing.
        if filters.get("exclude_globs"):
            pats = filters["exclude_globs"]
            def _excluded(r):
                pp = (r.get("path") or "").replace("\\", "/")
                return any(pp.startswith(g) or fnmatch.fnmatch(pp, g) for g in pats)
            leaves = [r for r in leaves if not _excluded(r)]

        # Location (jurisdiction tier) + version (per-document edition) filters.
        if _filtered:
            kept = []
            for r in leaves:
                f = facet_of(r.get("path"), r.get("edition"))
                if _loc and not location_allows(_loc, f["jurisdiction"]):
                    continue
                want = _versions.get(f["doc_key"])
                if want and f["year"] and str(f["year"]) != str(want):
                    continue
                kept.append(r)
            leaves = kept

        if short_code_filter:
            from .domains.building_codes.retrieval import _short_codes
            codes = _short_codes(query)
        else:
            codes = []
        if codes:
            pats = [re.compile(r"\b%s\b" % re.escape(c)) for c in codes]
            kept = [r for r in leaves if any(p.search(r.get("text") or "") for p in pats)]
            if kept:
                leaves = kept
        if not leaves:
            return {"status": "no_results", "results": [], "balanced": False}

        # --- Cross-encoder rerank (optional) -------------------------------
        # When balancing, stratify the rerank input across jurisdictions so each
        # layer's top RRF leaves reach the cross-encoder even if one jurisdiction
        # floods RRF (e.g. "permit in Durham" must still surface the NC statute).
        # The reranker then decides the final order by true relevance -- no
        # fragile output-balancing/dominance toggle needed.
        stratified = bool(balance)
        if stratified:
            from .domains.building_codes.retrieval import _stratified_pool
            pool = _stratified_pool(leaves, RERANK_POOL)
        else:
            pool = leaves[:RERANK_POOL]
        rr = rerank(query, [lf.get("text") or "" for lf in pool])
        used_rerank = rr is not None
        if used_rerank:
            for lf, s in zip(pool, rr):
                lf["score"] = float(s)
            pool.sort(key=lambda r: r["score"], reverse=True)
        else:
            for lf in pool:
                lf["score"] = lf["rrf"]

        # --- Collapse leaves to parent-document sections -------------------
        seen: dict[str, dict] = {}
        order_secs: list[str] = []
        for lf in pool:
            sid = lf.get("section_id") or ("leaf:%s" % lf["id"])
            if sid not in seen:
                node = get_section(conn, corpus, lf.get("section_id") or "")
                text = (node or {}).get("full_text") or lf.get("text") or ""
                res = _row_to_result(lf, lf["score"], text)
                res["_parent_section_id"] = (node or {}).get("parent_section_id")
                seen[sid] = res
                order_secs.append(sid)
            else:
                if lf["score"] > seen[sid]["score"]:
                    seen[sid]["score"] = lf["score"]

        results = [seen[s] for s in order_secs]
        results.sort(key=lambda r: r["score"], reverse=True)
        results = _dedup_ancestors(results)

        top_score = results[0]["score"] if results else 0.0

        # Order is set by the reranker over a jurisdiction-stratified pool;
        # just take the top_k. (No output re-balancing -- the cross-encoder
        # already ranked across the layers that were guaranteed into the pool.)
        applied = stratified
        results = results[:top_k]

        # --- One-hop citation-graph expansion ------------------------------
        if _expand_refs_enabled():
            results = _expand_references(conn, corpus, results)
            # Expansion can pull cross-edition / cross-jurisdiction sections;
            # re-apply the active filters so version/location stay honored.
            if _filtered:
                kept = []
                for r in results:
                    f = facet_of(r.get("path"), r.get("edition"))
                    if _loc and not location_allows(_loc, f["jurisdiction"]):
                        continue
                    want = _versions.get(f["doc_key"])
                    if want and f["year"] and str(f["year"]) != str(want):
                        continue
                    kept.append(r)
                results = kept

        for r in results:
            r.pop("_parent_section_id", None)

        if not results:
            return {"status": "no_results", "results": [], "balanced": applied}
        thresh = LOW_CONFIDENCE_RERANK if used_rerank else LOW_CONFIDENCE_RRF
        status = "low_confidence" if top_score < thresh else "ok"
        return {"status": status, "results": results, "balanced": applied}
    finally:
        conn.close()


def _dedup_ancestors(results: list[dict]) -> list[dict]:
    """Drop a retained section when a retained descendant subsumes it (its
    full_text already contains the descendant). Keeps the more specific hit."""
    retained_ids = {r.get("section_id") for r in results if r.get("section_id")}
    descendant_parents = set()
    for r in results:
        p = r.get("_parent_section_id")
        # collect parent ids that are themselves retained -> they are ancestors
        if p and p in retained_ids:
            descendant_parents.add(p)
    if not descendant_parents:
        return results
    return [r for r in results if r.get("section_id") not in descendant_parents]


def _expand_references(conn, corpus: str, results: list[dict]) -> list[dict]:
    have = {r.get("section_id") for r in results}
    added = 0
    extra: list[dict] = []
    for r in results:
        if added >= MAX_REF_EXPANSION:
            break
        sid = r.get("section_id")
        if not sid:
            continue
        for ref in get_refs(conn, sid):
            if added >= MAX_REF_EXPANSION:
                break
            dst = ref.get("dst_section_id")
            if not dst or dst in have:
                continue
            node = get_section(conn, corpus, dst)
            if not node:
                continue
            have.add(dst)
            res = _row_to_result(node, _REF_SCORE, node.get("full_text") or "",
                                 referenced=True)
            res["referenced_by"] = r.get("section_number") or sid
            res["reference_raw"] = ref.get("dst_raw")
            extra.append(res)
            added += 1
    return results + extra
