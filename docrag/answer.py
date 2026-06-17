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
import sys
import time

from . import settings
from .query import rag_query


CHUNK_CHAR_BUDGET = 4500          # per-section cap (parents are richer now)
TOTAL_CONTEXT_BUDGET = 48000      # overall cap; later chunks get tightened
CHAT_TEMPERATURE = 0.2
CHAT_MAX_TOKENS = 900
MAX_ATTEMPTS = 6
INITIAL_DELAY_S = 1.0
MAX_DELAY_S = 60.0

MODEL_REFUSAL_SENTENCE = "I don't have documentation for that."

SYSTEM_PROMPT = (
    "You are a regulatory research assistant grounded in excerpts from a "
    "collection of building codes, statutes, and ordinances.\n\n"
    "RULES:\n"
    "1. Ground every claim in the provided chunks. Do not introduce facts, "
    "values, codes, or rules from outside the chunks.\n"
    "2. Every factual claim must be followed by a citation in square brackets "
    "like [1] or [2,3].\n"
    "3. NAME the specific provision you rely on, not just the document. Each "
    "chunk is labeled with its edition/jurisdiction and section designation "
    "(e.g. 'NC Residential Code R101.2.1 Accessory buildings', "
    "'NCGS 160D-1110', 'IBC Section 1010'). State that designation in prose "
    "alongside the bare [N], e.g. \"Under NC Residential Code R101.2.1 [3], ...\".\n"
    "4. REASON over the rules; do not merely quote them. Apply the cited rules "
    "to the specific facts in the question using ordinary deductive reasoning "
    "(compare a stated cost to a stated threshold; check whether the work or "
    "structure matches an enumerated trigger or exemption).\n"
    "5. Rules often define their own SCOPE -- the conditions under which a "
    "requirement applies. If a rule states WHEN something is regulated (a size, "
    "a cost threshold, an enumerated trigger) and the facts of the question "
    "fall OUTSIDE that scope, conclude that the requirement does not apply, and "
    "present this explicitly as an inference from the rule's scope, citing the "
    "provision. (Example shape: a rule says structures with any dimension over "
    "X must comply; a structure under X in every dimension therefore falls "
    "outside that requirement.) Distinguish 'the documents are silent on this' "
    "from 'the documents address this and the answer follows from them'.\n"
    "5a. Treat a scope threshold as DEFINITIONAL, not merely permissive. A "
    "general requirement (e.g. a statute saying construction needs a permit "
    "'as required by the Code') is qualified by the specific Code provisions "
    "that define what the Code actually regulates. So when the governing "
    "provision regulates a class only above a threshold and the facts fall "
    "below it, and no OTHER cited provision independently brings the item in, "
    "conclude the requirement does NOT apply -- do not default to 'required' "
    "merely because no clause uses the word 'exempt'. Absence of an explicit "
    "exemption is not the same as being regulated; reason from scope. State "
    "your assumption and note it follows from the cited provisions' scope.\n"
    "6. Do not fabricate. If a chunk doesn't state or imply it, don't state it. "
    "If the chunks genuinely do not let you reason to an answer, respond "
    "exactly: \"" + MODEL_REFUSAL_SENTENCE + "\"\n"
    "7. Be concise but show the reasoning steps that lead to the conclusion.\n"
    "8. Chunks may come from different source documents. When facts come from "
    "more than one, name the source/provision for each. If one source amends "
    "or is more specific than another (e.g. a local code or state statute vs. "
    "a model code), say so and make clear which governs.\n"
)

# Extra guidance when balanced retrieval spans the three building-code
# jurisdictions, so the answer synthesizes across all of them.
CROSS_JURISDICTION_NOTE = (
    "\nEach excerpt is tagged with an authority class: [STATE] (NC statutes / "
    "NC State Building Code), [LOCAL] (Durham UDO / county ordinance text), "
    "[LOCAL-GUIDANCE] (durhamnc.gov agency how-to pages), [MODEL] (IBC/IRC/... "
    "model codes), [FEDERAL]. Resolve which one governs in %s using NORTH "
    "CAROLINA's FIELD-PREEMPTION framework -- NOT a fixed rank:\n"
    "- State law is PRIMARY where it provides a complete, integrated regulatory "
    "scheme. The BUILDING-PERMIT REQUIREMENT (NCGS 160D-1110) and the NC STATE "
    "BUILDING CODE (construction / life-safety / accessibility standards) are "
    "such schemes: here [STATE] governs, and a [LOCAL] ordinance or "
    "[LOCAL-GUIDANCE] page CANNOT change the permit trigger or relax/tighten the "
    "building code. Base permit-required/exempt and code-standard conclusions on "
    "[STATE] law.\n"
    "- Where the state DELEGATES to local government -- zoning, land use, "
    "setbacks, placement, lot coverage, height-as-zoning, subdivision design, "
    "local stormwater -- the [LOCAL] Durham UDO governs and MAY be stricter than "
    "any state minimum. Base zoning/siting conclusions on the [LOCAL] UDO.\n"
    "- [LOCAL-GUIDANCE] pages (durhamnc.gov) are the permitting authority's own "
    "statements of local practice. When the [STATE] statute / [LOCAL] ordinance "
    "is SILENT or only general on the point, you MAY rely on the guidance as the "
    "best available answer (e.g. 'Durham does not require a permit for an "
    "ordinary fence') -- note it is agency guidance. But when a guidance page "
    "CONFLICTS with or OVER-GENERALIZES a more specific [STATE] statute or "
    "[LOCAL] ordinance (e.g. a flat 'a permit is always required' that ignores a "
    "statutory cost/scope exemption), defer to the primary instrument and FLAG "
    "the divergence. Do not refuse or hedge when the guidance gives a clear "
    "answer the primary law does not contradict.\n"
    "- [MODEL] provisions are background only unless NC adopted them; a model "
    "exemption absent from NC law does not apply in NC.\n"
    "- A SCOPE rule (which structures must meet the building code's construction "
    "standards) is a DIFFERENT question from the PERMIT requirement. Never cite "
    "a code-scope provision as evidence that no permit is required -- the permit "
    "trigger is NCGS 160D-1110, not the code's construction scope.\n"
    "Name the governing instrument and class for each conclusion; if the "
    "controlling layer is silent, say so rather than inferring.\n"
)

USER_PROMPT_TEMPLATE = (
    "Question / topic: {query}\n\n"
    "Chunks (numbered):\n{chunks_block}\n\n"
    "Instructions:\n"
    "- If the chunks describe the topic, summarize what they say and cite "
    "chunk numbers in brackets like [1] or [2,3].\n"
    "- A short topic is still a valid question -- treat it as \"what do these "
    "chunks say about <topic>?\".\n"
    "- Only respond with the exact refusal sentence if none of the chunks are "
    "about the topic at all.\n"
)

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


def _authority_tier(c: dict) -> str:
    """Authority class from the source path. NOT a flat rank -- it feeds a
    FIELD-PREEMPTION analysis (see CROSS_JURISDICTION_NOTE): which class governs
    depends on the subject. Classes:
      STATE          -- NC statutes (NCGS) + NC State Building Code (primary law)
      LOCAL          -- Durham UDO / county ordinance text (primary local law)
      LOCAL-GUIDANCE -- durhamnc.gov agency how-to pages (interpretive, NOT law)
      MODEL          -- IBC/IRC/... model codes (background unless adopted)
      FEDERAL        -- federal (ADA/FEMA/NFIP) primary law
    Derived structurally so the synthesis can be told which instrument controls.
    """
    p = (c.get("path") or "").replace("\\", "/").lower()
    head = p.split("/", 1)[0]
    if head.startswith("durham"):
        # The scraped UDO ordinance HTML is primary local law; the other
        # durhamnc.gov inspection pages are agency guidance (not law).
        if "/inspections/" in p and "/udo-" not in p:
            return "LOCAL-GUIDANCE"
        return "LOCAL"
    if head in ("north-carolina", "nc-state", "ncdot"):
        return "STATE"
    if head == "model":
        return "MODEL"
    if head == "federal":
        return "FEDERAL"
    return "OTHER"


def _chunk_label(c: dict) -> str:
    """Human-readable provenance for a retrieved section."""
    edition = c.get("edition") or c.get("jurisdiction")
    parts = ["[%s]" % _authority_tier(c)]
    if edition:
        parts.append(str(edition))
    parts.append(c.get("source_file") or "unknown")
    num = c.get("section_number")
    title = c.get("section_title")
    head = " ".join(x for x in (num, title) if x)
    if head:
        parts.append("§ " + head)
    page = c.get("page")
    if page is not None:
        parts.append("p.%s" % page)
    label = " | ".join(parts)
    if c.get("referenced"):
        label += " (cross-referenced by %s)" % (c.get("referenced_by") or "")
    return label


def _designation(c: dict) -> str:
    """Concise statute/section designation for the citation list.

    e.g. "NC Residential Code R101.2.1 Accessory buildings -- NCRC2024... p.-"
    Falls back to the source file + page when a section number is absent.
    """
    edition = c.get("edition") or c.get("jurisdiction") or ""
    num = c.get("section_number")
    title = (c.get("section_title") or "").rstrip(". ")
    head = " ".join(x for x in (num, title) if x)
    lead = " ".join(x for x in (str(edition), ("§ " + head) if head else "") if x)
    src = c.get("source_file") or "unknown"
    page = c.get("page")
    tail = src + (" p.%s" % page if page is not None else "")
    return (lead + " -- " + tail).strip(" -") if lead else tail


def _authorities(chunks: list[dict], citations: list[int]) -> list[dict]:
    """Map each cited [N] to a spelled-out authority designation."""
    out = []
    for n in citations:
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            out.append({
                "n": n,
                "designation": _designation(c),
                "section_number": c.get("section_number"),
                "section_title": c.get("section_title"),
                "edition": c.get("edition"),
                "jurisdiction": c.get("jurisdiction"),
                "source_file": c.get("source_file"),
                "page": c.get("page"),
            })
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


def _chat_client():
    from openai import AzureOpenAI  # type: ignore

    endpoint = settings.azure_endpoint()
    api_key = settings.azure_api_key()
    if not endpoint or not api_key:
        raise EnvironmentError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set."
        )
    # Bound every chat call: the SDK default 600s timeout (+ retries) means a
    # single stalled request can freeze a query / MCP tool call for 10-30 min.
    # Our own _chat_with_backoff handles transient retries, so cap hard here.
    # Override via DOCRAG_HTTP_TIMEOUT.
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=settings.azure_api_version(),
        timeout=float(settings.get("DOCRAG_HTTP_TIMEOUT", 60) or 60),
        max_retries=2,
    )


def _chat_with_backoff(client, deployment: str, messages: list[dict]):
    """chat.completions with backoff + token/temperature param probing."""
    use_max_tokens_legacy = False
    send_temperature = True
    delay = INITIAL_DELAY_S
    for attempt in range(MAX_ATTEMPTS):
        kwargs: dict = {"model": deployment, "messages": messages}
        if send_temperature:
            kwargs["temperature"] = CHAT_TEMPERATURE
        if use_max_tokens_legacy:
            kwargs["max_tokens"] = CHAT_MAX_TOKENS
        else:
            kwargs["max_completion_tokens"] = CHAT_MAX_TOKENS
        try:
            t0 = time.monotonic()
            resp = client.chat.completions.create(**kwargs)
            sys.stderr.write(
                "[answer] chat ok deployment=%s in %dms\n"
                % (deployment, int((time.monotonic() - t0) * 1000))
            )
            return resp
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            lower = msg.lower()
            transient = ("429" in msg or "rate" in lower or "throttle" in lower
                         or any(c in msg for c in (" 500", " 502", " 503", " 504"))
                         or "server_error" in lower or "overloaded" in lower)
            if transient:
                time.sleep(delay)
                delay = min(delay * 2, MAX_DELAY_S)
                continue
            if ("max_completion_tokens" in lower and "not supported" in lower
                    and not use_max_tokens_legacy):
                use_max_tokens_legacy = True
                continue
            if ("temperature" in lower
                    and ("not supported" in lower or "unsupported" in lower)
                    and send_temperature):
                send_temperature = False
                continue
            raise EnvironmentError("Azure OpenAI chat failed: %s" % msg) from e
    raise EnvironmentError(
        "Azure OpenAI chat failed after %d retries (rate-limited)." % MAX_ATTEMPTS
    )


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

    client = _chat_client()
    deployment = settings.chat_deployment_synthesis()
    resp = _chat_with_backoff(client, deployment, messages)

    try:
        answer_text = (resp.choices[0].message.content or "").strip()
    except (AttributeError, IndexError) as e:
        raise EnvironmentError("Azure chat returned unexpected shape: %s" % e) from e

    usage = getattr(resp, "usage", None)
    tokens = {
        "prompt": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion": getattr(usage, "completion_tokens", None) if usage else None,
    }

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
