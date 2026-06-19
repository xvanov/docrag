"""retrieval.py -- building-codes-specific retrieval helpers.

Relocated verbatim from the core query.py so the core hybrid-retrieval skeleton
carries no jurisdiction/code knowledge. core/query.py lazy-imports these ONLY
inside the building-code branches it already gated (balance / short_code_filter /
expand), so a plain domain (youtube) never imports this module.

Contents: jurisdiction bucketing + balancing/stratification, the short-code
section-number filter, and the lay->code LLM query expansion.
"""

from __future__ import annotations

import re

from ... import settings

_DOMINANCE_FRAC = 0.6

# Jurisdiction buckets for the building-codes corpus (model / state / local).
_JURISDICTIONS = (("north-carolina/", "north-carolina"), ("model/", "model"),
                  ("durham/", "durham"))

# A "short code" is an explicit provision/section/table identifier the user
# typed (e.g. R705, 160D, E3902) that we then REQUIRE in the results to suppress
# hallucination. It must mix a LETTER and a DIGIT: pure-alpha uppercase tokens
# ("NC", "IBC", "ADA") are abbreviations and pure-digit tokens ("20", "000" from
# "$20,000") are quantities -- neither is a section number, and hard-filtering
# on them wrongly empties the candidate pool (e.g. dropping R101.2.1 for a query
# that merely says "in Durham, NC" or "$20,000").
_SHORT_CODE_RE = re.compile(r"^(?=[A-Z0-9]{2,8}$)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+$")


def _short_codes(query: str) -> list[str]:
    out = []
    for tok in re.split(r"[\s,;:()\[\]{}/]+", query or ""):
        tok = tok.strip(".-'\"")
        if tok and _SHORT_CODE_RE.match(tok):
            out.append(tok)
    return out


def _jurisdiction_of(row: dict) -> str:
    j = row.get("jurisdiction")
    if j:
        return j
    p = (row.get("path") or "").replace("\\", "/").lstrip("/")
    for prefix, name in _JURISDICTIONS:
        if p.startswith(prefix):
            return name
    return p.split("/", 1)[0] if "/" in p else "_root"


def _balance_dominated(results: list[dict], window: int, frac: float = _DOMINANCE_FRAC) -> bool:
    top = results[:window]
    if not top:
        return True
    counts: dict[str, int] = {}
    for r in top:
        counts[_jurisdiction_of(r)] = counts.get(_jurisdiction_of(r), 0) + 1
    if len(counts) <= 1:
        return True
    return max(counts.values()) / len(top) >= frac


def _balance_by_jurisdiction(results: list[dict], top_k: int) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for r in results:
        buckets.setdefault(_jurisdiction_of(r), []).append(r)
    if len(buckets) <= 1:
        return results[:top_k]
    order = sorted(buckets, key=lambda k: results.index(buckets[k][0]))
    out: list[dict] = []
    i = 0
    while len(out) < top_k and any(buckets[k] for k in order):
        b = buckets[order[i % len(order)]]
        if b:
            out.append(b.pop(0))
        i += 1
    return out


def _stratified_pool(leaves: list[dict], cap: int) -> list[dict]:
    """Round-robin the RRF-ordered leaves across jurisdictions up to ``cap`` so
    every jurisdiction's best candidates reach the reranker (preventing one
    jurisdiction from flooding the pool). Order within a jurisdiction is the
    RRF order; the reranker re-scores the whole pool afterward."""
    buckets: dict[str, list[dict]] = {}
    for lf in leaves:
        buckets.setdefault(_jurisdiction_of(lf), []).append(lf)
    if len(buckets) <= 1:
        return leaves[:cap]
    order = sorted(buckets, key=lambda k: leaves.index(buckets[k][0]))
    out: list[dict] = []
    i = 0
    while len(out) < cap and any(buckets[k] for k in order):
        b = buckets[order[i % len(order)]]
        if b:
            out.append(b.pop(0))
        i += 1
    return out


def _expand_query_enabled() -> bool:
    val = (settings.get("DOCRAG_EXPAND_QUERY", "1") or "1").strip().lower()
    return val not in ("0", "false", "no", "off")


_EXPAND_N = 3
_EXPAND_SYS = (
    "You rewrite a user's question into alternative search queries that use the "
    "formal terminology found in building codes, statutes, and zoning "
    "ordinances. A code rarely uses lay words: a 'shed' is a 'detached "
    "accessory structure/building'; 'do I need a permit' maps to 'permit "
    "exemption / work not requiring a permit / when a permit is required'; "
    "'cost' maps to thresholds. Output exactly %d concise alternative queries, "
    "one per line, no numbering, no commentary -- each a different phrasing or "
    "the technical class names a code would use for the subject."
) % _EXPAND_N


def _expand_queries(query: str) -> list[str]:
    """LLM paraphrases of the question into code-vocabulary search queries.

    Bridges the lay-vs-code vocabulary gap so a rule phrased as a scope/exemption
    (e.g. R101.2.1 'Accessory buildings') is recalled even when the user says
    'shed'. Returns [] on any failure / when disabled -- retrieval then proceeds
    on the original query alone (graceful degradation). Generic: no per-question
    rules, the model generates the alternatives."""
    if not _expand_query_enabled() or not query:
        return []
    try:
        from openai import AzureOpenAI  # type: ignore
        endpoint = settings.azure_endpoint()
        api_key = settings.azure_api_key()
        if not endpoint or not api_key:
            return []
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key,
                             api_version=settings.azure_api_version())
        resp = client.chat.completions.create(
            model=settings.chat_deployment_synthesis(),
            messages=[{"role": "system", "content": _EXPAND_SYS},
                      {"role": "user", "content": query}],
            max_completion_tokens=160,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001 -- any LLM/transport failure -> no expansion
        return []
    out = []
    for ln in text.splitlines():
        ln = ln.strip().lstrip("0123456789.)-* \t").strip()
        if ln and ln.lower() != query.lower():
            out.append(ln)
    return out[:_EXPAND_N]
