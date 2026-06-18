"""answer.py -- Grounded LLM synthesis over retrieved chunks.

Pipeline:
  1. rag_query() to retrieve top_k chunks.
  2. If retrieval is empty / low-confidence -> refuse without an LLM call.
  3. Build a numbered chunks block (truncated per chunk).
  4. System prompt forbids out-of-context claims and mandates [N] citations.
  5. Azure chat call with exponential backoff on 429.
  6. Post-hoc: detect the canned refusal sentence; require >=1 in-range [N]
     citation, else flip to refused (anti-hallucination).

Public API:
    answer(corpus, query, history=None, top_k=12, filters=None) -> dict
"""

from __future__ import annotations

import re

from .core.chat import get_chat_provider
from .query import rag_query
from .domains.building_codes.prompts import (  # re-exported for reason.py/server.py
    CROSS_JURISDICTION_NOTE,
    MODEL_REFUSAL_SENTENCE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    _authorities,
    _authority_tier,
    _chunk_label,
    _designation,
)


CHUNK_CHAR_BUDGET = 4500          # per-section cap (parents are richer now)
TOTAL_CONTEXT_BUDGET = 48000      # overall cap; later chunks get tightened
CHAT_TEMPERATURE = 0.2
CHAT_MAX_TOKENS = 900


_CITE_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_LAST_PROMPT: dict | None = None


def _normalize_quotes(s: str) -> str:
    return (s or "").replace("’", "'").replace("‘", "'")


_QTERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-$]{1,}")
_STOPWORDS = frozenset(
    "a an the of to in for and or is are do does my i you it this that with "
    "be can will shall on at as by from need require required".split())


def _query_terms(query: str) -> list[str]:
    out = []
    for m in _QTERM_RE.finditer((query or "").lower()):
        t = m.group(0)
        if len(t) >= 3 and t not in _STOPWORDS:
            out.append(t)
    return out


def _truncate_chunk_text(text: str, budget: int = CHUNK_CHAR_BUDGET,
                         query: str | None = None) -> str:
    """Trim a chunk to ``budget`` chars.

    When the chunk is over budget, keep a window CENTERED on the query-relevant
    span (the earliest..latest position any query term appears) instead of the
    head -- so a long section never loses its operative clause to a blind head
    cut. Falls back to a head cut when no query term matches. Generic: uses only
    the user's own query terms, no document-specific rules.
    """
    text = " ".join((text or "").split())
    if len(text) <= budget:
        return text

    terms = _query_terms(query or "")
    low = text.lower()
    # All occurrences of every query term (not just the first) -- a long section
    # mentions a term in many places, and the operative clause is wherever they
    # CLUSTER, not the midpoint of the first..last span.
    hits = []
    for t in terms:
        start = 0
        while True:
            i = low.find(t, start)
            if i < 0:
                break
            hits.append(i)
            start = i + len(t)
    if not hits:
        return text[: budget - 3].rstrip() + "..."
    hits.sort()

    # Slide a budget-sized window; keep the one covering the most term hits
    # (densest relevant region). Anchor each candidate slightly before a hit so
    # the clause's lead-in is included.
    lead = budget // 5
    best_start, best_count = 0, -1
    for h in hits:
        s = max(0, min(h - lead, len(text) - budget))
        e = s + budget
        cnt = 0
        for x in hits:
            if x >= e:
                break
            if x >= s:
                cnt += 1
        if cnt > best_count:
            best_count, best_start = cnt, s
    start = best_start
    end = min(len(text), start + budget)
    out = text[start:end].strip()
    if start > 0:
        out = "..." + out
    if end < len(text):
        out = out + "..."
    return out


def _build_chunks_block(chunks: list[dict], query: str | None = None) -> str:
    """Numbered chunk block with a per-section and total context budget.

    Parent-document sections can be large, so cap each one (query-aware window,
    not a head cut) and tighten later chunks once the running total passes
    TOTAL_CONTEXT_BUDGET -- prevents prompt bloat / cost blowups while keeping
    the top hits' operative clauses intact."""
    lines = []
    total = 0
    for i, c in enumerate(chunks, start=1):
        budget = CHUNK_CHAR_BUDGET if total < TOTAL_CONTEXT_BUDGET else 600
        snippet = _truncate_chunk_text(c.get("text") or "", budget, query)
        total += len(snippet)
        lines.append('[%d] %s\n"%s"' % (i, _chunk_label(c), snippet))
    return "\n\n".join(lines)


def _parse_citations(answer_text: str, n_chunks: int) -> list[int]:
    if not answer_text or n_chunks <= 0:
        return []
    found: set[int] = set()
    for match in _CITE_RE.finditer(answer_text):
        for piece in match.group(1).split(","):
            piece = piece.strip()
            if piece.isdigit():
                n = int(piece)
                if 1 <= n <= n_chunks:
                    found.add(n)
    return sorted(found)


def _refused(chunks, reason, status):
    return {"answer": None, "citations": [], "chunks": chunks,
            "refused": True, "refusal_reason": reason, "status": status,
            "tokens": {"prompt": None, "completion": None}}


def answer(corpus: str, query: str, history: list[dict] | None = None,
           top_k: int = 12, filters: dict | None = None,
           balance: bool = False, location: str | None = None) -> dict:
    """Retrieve + synthesize a grounded, cited answer.

    ``balance=True`` enables jurisdiction-balanced retrieval + cross-code
    synthesis guidance (model IBC / NC state / local). ``location`` (a
    facets.LOCATIONS key) sets the "which governs in <place>" phrasing.

    Returns ``{answer, citations, chunks, refused, refusal_reason, status,
    tokens}``.
    """
    retrieval = rag_query(corpus, query, top_k=top_k, filters=filters,
                          balance=balance)
    chunks = retrieval.get("results") or []
    r_status = retrieval.get("status")
    # Balancing is adaptive: rag_query may decline to interleave when one
    # jurisdiction dominates. Key the cross-jurisdiction synthesis note off
    # what was actually applied, not what was requested.
    applied_balance = bool(retrieval.get("balanced"))

    if r_status == "no_results" or not chunks:
        return _refused(chunks, "no_supporting_documentation", "no_results")
    if r_status == "low_confidence":
        return _refused(chunks, "low_confidence", "refused")

    from .facets import answer_location
    cross_note = CROSS_JURISDICTION_NOTE % answer_location(location)
    system_prompt = SYSTEM_PROMPT + (cross_note if applied_balance else "")
    chunks_block = _build_chunks_block(chunks, query)
    user_prompt = USER_PROMPT_TEMPLATE.format(query=query, chunks_block=chunks_block)

    global _LAST_PROMPT
    _LAST_PROMPT = {"system": system_prompt, "user": user_prompt}

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})

    provider = get_chat_provider("azure")
    result = provider.complete(system=system_prompt, messages=messages[1:],
                               max_tokens=CHAT_MAX_TOKENS,
                               temperature=CHAT_TEMPERATURE)
    answer_text = result.text
    tokens = {"prompt": result.prompt_tokens, "completion": result.completion_tokens}

    if MODEL_REFUSAL_SENTENCE in _normalize_quotes(answer_text):
        out = _refused(chunks, "model_refused_insufficient_context", "refused")
        out["tokens"] = tokens
        return out

    citations = _parse_citations(answer_text, len(chunks))
    if not citations:
        out = _refused(chunks, "model_did_not_cite", "refused")
        out["tokens"] = tokens
        return out

    return {"answer": answer_text, "citations": citations, "chunks": chunks,
            "authorities": _authorities(chunks, citations),
            "refused": False, "refusal_reason": None, "status": "ok",
            "tokens": tokens, "balanced": applied_balance}
