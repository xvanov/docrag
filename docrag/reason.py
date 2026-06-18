"""reason.py -- agentic hypothesize-verify RAG over the grounded answer layer.

Correctness-first pipeline (multiple LLM calls by design):

  1. UNDERSTAND + HYPOTHESIZE (fast model, JSON): clean the question, extract
     the decision-relevant facts (structure type, dimensions, cost, scope,
     jurisdiction), state a common-sense hypothesis, and list the specific
     rules/topics and code-vocabulary search probes that would confirm or
     refute it.
  2. RETRIEVE (no LLM): hypothesis-directed retrieval -- run rag_query on the
     cleaned query plus every probe, fuse the results (RRF over the per-probe
     section rankings), keep the top sections.
  3. VERIFY + ANSWER (full model): given the hypothesis + retrieved provisions,
     confirm / refute / refine and produce the final grounded, cited answer. If
     the verifier names a rule it still needs, do one more targeted retrieve and
     re-answer (bounded loop).

The hypothesis is PRIVATE scaffolding: it directs search and tells the model
what to look for. The final answer stays grounded in the retrieved chunks and
cited -- the model's prior guides the search; the corpus decides the answer.

Public API mirrors answer.answer() so callers swap transparently:
    answer(corpus, query, history=None, top_k=15, filters=None,
           balance=False, location=None) -> dict
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import settings
from .answer import (
    CHAT_MAX_TOKENS,
    CHAT_TEMPERATURE,
    CROSS_JURISDICTION_NOTE,
    MODEL_REFUSAL_SENTENCE,
    _authorities,
    _build_chunks_block,
    _normalize_quotes,
    _parse_citations,
    _refused,
)
from .core.chat import get_chat_provider
from .query import rag_query

MAX_VERIFY_ROUNDS = 2          # extra hypothesis-directed retrieve+answer rounds
PER_PROBE_TOP_K = 8            # sections kept per search probe before fusion
MAX_PROBES = 8                 # cap concurrent retrieval probes per round
RRF_K = 60


def _enabled() -> bool:
    val = (settings.get("DOCRAG_AGENTIC", "1") or "1").strip().lower()
    return val not in ("0", "false", "no", "off")


# ---- Stage 1: understand + hypothesize --------------------------------------

_UNDERSTAND_SYS = (
    "You are the planning stage of a building-code question-answering system. "
    "Given a user's question (often lay, vague, or high-level), produce a JSON "
    "object that plans the retrieval. Use your general knowledge of how building "
    "codes, statutes, and zoning ordinances work to form a HYPOTHESIS, but the "
    "downstream step will verify it against the actual documents -- so your job "
    "is to make retrieval find the governing rule.\n\n"
    "Return ONLY a JSON object with these keys:\n"
    "  cleaned_query: a clear, specific restatement of the question.\n"
    "  facts: object of decision-relevant facts you can extract (e.g. "
    "structure_type, dimensions, cost, work_type, jurisdiction). Omit unknowns.\n"
    "  hypothesis: one sentence -- your best common-sense answer.\n"
    "  rules_to_check: array of the specific provisions/topics that would "
    "confirm or refute the hypothesis. ALWAYS cover BOTH sides and the "
    "governing law -- not just exemptions. For a permit question include: (a) "
    "the provision that REQUIRES a permit in the relevant jurisdiction (e.g. the "
    "state statute/local ordinance that mandates permits), (b) how the GOVERNING "
    "(state/local) code treats or scopes the subject, and (c) any exemption.\n"
    "  search_queries: array of 4-8 retrieval queries in the FORMAL terminology a "
    "code would use (translate lay terms: 'shed'->'detached accessory "
    "structure/building'). Cover the permit-REQUIREMENT angle AND the "
    "scope/exemption angle AND the specific jurisdiction (e.g. 'North Carolina "
    "statute requiring a building permit', 'when is a building permit required in "
    "North Carolina', 'NC residential code accessory building scope', 'Durham "
    "permit requirement accessory structure'). These drive the search.\n"
    "  When the subject has measurable attributes (size, dimension, cost, "
    "quantity, occupancy count, etc.), ALSO add at least one probe aimed at the "
    "governing provision's APPLICABILITY CRITERIA or THRESHOLD for that class of "
    "subject -- the rule that states the qualifying limit above/below which the "
    "scheme does or does not reach the subject -- not just the general "
    "requirement. That threshold provision is often what actually decides the "
    "case, and a generic requirement probe tends to miss it.\n"
    "AUTHORITY: NC uses field preemption. The building-PERMIT requirement and "
    "the NC State Building Code are state schemes (probe NCGS 160D-1110 and the "
    "NC code); zoning/setbacks/use are local (probe the Durham UDO). The "
    "durhamnc.gov agency pages are guidance, not law. Do NOT base the hypothesis "
    "on a model-code number (e.g. a square-foot exemption); it is only a hint to "
    "verify, and the controlling state statute / local ordinance decides.\n"
    "No commentary, no markdown fences -- just the JSON object."
)


def _understand(query: str, history: list[dict] | None) -> dict:
    """Stage 1: fast-model planning. Returns the parsed plan, or a minimal
    fallback plan (cleaned_query=query) on any failure."""
    fallback = {"cleaned_query": query, "facts": {}, "hypothesis": "",
                "rules_to_check": [], "search_queries": []}
    try:
        provider = get_chat_provider("azure")
    except Exception:  # noqa: BLE001 -- no Azure config
        return fallback
    messages = []
    if history:
        for turn in history[-4:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": query})
    try:
        # Use the full synthesis model for planning too -- the FAST deployment
        # is less reliable, and plan quality drives the whole pipeline.
        result = provider.complete(system=_UNDERSTAND_SYS, messages=messages,
                                   max_tokens=CHAT_MAX_TOKENS,
                                   temperature=CHAT_TEMPERATURE)
        raw = (result.text or "").strip()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("[reason] understand failed (%s); plain retrieval\n" % e)
        return fallback
    plan = _parse_json(raw)
    if not isinstance(plan, dict) or not plan.get("cleaned_query"):
        return fallback
    plan.setdefault("facts", {})
    plan.setdefault("hypothesis", "")
    plan.setdefault("rules_to_check", [])
    plan.setdefault("search_queries", [])
    return plan


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = re.sub(r"^(json|JSON)\s*", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        m = _JSON_RE.search(raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                return None
        return None


# ---- Stage 2: hypothesis-directed retrieval ---------------------------------

def _retrieve(corpus: str, plan: dict, top_k: int, filters: dict | None,
              balance: bool, extra_probes: list[str] | None = None) -> list[dict]:
    """Run rag_query on the cleaned query + every probe CONCURRENTLY, fuse the
    per-probe section rankings via RRF, return the top_k merged sections.

    The planner can emit ~16 probes (search_queries + rules_to_check); run
    serially that dominated latency. We cap the count and fan them out across a
    thread pool -- each rag_query opens its own SQLite connection and the embed
    call is HTTP, so they parallelize safely. Wall time ~= the slowest probe."""
    llm_probes = (list(plan.get("search_queries") or [])
                  + list(plan.get("rules_to_check") or []))
    extras = [(e or "").strip() for e in (extra_probes or []) if (e or "").strip()]
    probes: list[str] = []
    seen: set = set()
    for p in ([plan.get("cleaned_query")] + llm_probes + extras):
        p = (p or "").strip()
        if p and p not in seen:
            seen.add(p)
            probes.append(p)
    if not probes:
        probes = [plan.get("cleaned_query") or ""]
    # Cap probe count to bound cost, but always keep a targeted NEED-round probe.
    if len(probes) > MAX_PROBES:
        head = probes[:MAX_PROBES]
        for e in extras:
            if e not in head:
                head.append(e)
        probes = head
    # If the planner produced no code-vocabulary probes (e.g. it failed and we
    # fell back to the raw query), let rag_query do its own built-in expansion
    # so the lay->code vocabulary bridge still happens.
    base_expand = not llm_probes

    def _run(q):
        return rag_query(corpus, q, top_k=PER_PROBE_TOP_K, filters=filters,
                         balance=balance, expand=base_expand)

    # Preserve a deterministic fusion order (probe order) regardless of which
    # thread finishes first, so results are stable run-to-run.
    rqs: list = [None] * len(probes)
    with ThreadPoolExecutor(max_workers=min(MAX_PROBES, len(probes))) as ex:
        futs = {ex.submit(_run, q): i for i, q in enumerate(probes)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                rqs[i] = f.result()
            except Exception as e:  # noqa: BLE001 -- one bad probe shouldn't sink retrieval
                sys.stderr.write("[reason] probe %d failed: %s\n" % (i, e))

    fused: dict[str, float] = {}
    keep: dict[str, dict] = {}
    for rq in rqs:
        if not rq:
            continue
        for rank, r in enumerate(rq.get("results") or [], start=1):
            key = r.get("section_id") or "%s#%s" % (r.get("path"),
                                                     r.get("section_number"))
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
            # keep the richest copy (prefer a non-referenced, higher-scored one)
            prev = keep.get(key)
            if prev is None or (r.get("score") or 0) > (prev.get("score") or 0):
                keep[key] = r
    order = sorted(fused, key=lambda k: fused[k], reverse=True)
    return [keep[k] for k in order[:top_k]]


# ---- Stage 3: verify + answer -----------------------------------------------

_VERIFY_SYS = (
    "You are the verification + answering stage of a building-code assistant. "
    "You are given the user's question, a planning HYPOTHESIS from an earlier "
    "step, and numbered excerpts retrieved from the document collection.\n\n"
    "Your job: decide the answer by checking the hypothesis against the "
    "excerpts, then reason to a correct, grounded conclusion.\n\n"
    "RULES:\n"
    "1. The hypothesis is only a hint. Confirm, refute, or refine it using the "
    "excerpts. The final answer must be grounded in the excerpts, not the "
    "hypothesis or outside knowledge.\n"
    "2. Cite every factual claim with a BRACKETED chunk number [N] (e.g. [3] or "
    "[2,5]) AND name the provision in prose (e.g. 'NC Residential Code R101.2.1 "
    "[3]', 'NCGS 160D-1110 [2]'). The bracketed [N] is mandatory on every claim "
    "-- naming the provision without a [N] does not count. Your answer MUST "
    "contain at least one [N].\n"
    "3. REASON over the rules. Apply thresholds, enumerated lists, and scope "
    "conditions to the question's facts. Treat a scope threshold as "
    "DEFINITIONAL: if the governing provision regulates a class only above a "
    "threshold and the facts fall below it, and no other cited provision brings "
    "the item in, conclude the requirement does NOT apply -- absence of an "
    "explicit 'exempt' clause is not the same as being regulated.\n"
    "3a. AUTHORITY by FIELD PREEMPTION (NOT a fixed rank). Excerpts are tagged "
    "[STATE], [LOCAL], [LOCAL-GUIDANCE], [MODEL], [FEDERAL]. Which class governs "
    "depends on the SUBJECT: the BUILDING-PERMIT requirement (NCGS 160D-1110) "
    "and the NC STATE BUILDING CODE are complete state schemes -> [STATE] "
    "governs permit-required/exempt and construction-standard questions, and "
    "[LOCAL]/[LOCAL-GUIDANCE] cannot change them. ZONING/setbacks/placement/use/"
    "lot-coverage/subdivision are delegated -> the [LOCAL] UDO governs and may be "
    "stricter. [LOCAL-GUIDANCE] (durhamnc.gov pages) is interpretive/process, "
    "NOT law: use for practical context, defer to the [STATE] statute or [LOCAL] "
    "ordinance, and flag any over-generalization or conflict. [MODEL] is "
    "background unless NC adopted it.\n"
    "3b. APPLICABILITY-CRITERIA PRINCIPLE (general; applies to permits and to any "
    "other requirement). A statute or code provision imposes its requirement only "
    "when the facts satisfy the criteria that provision uses to define what it "
    "governs. If a governing provision defines the class it regulates by a "
    "qualifying threshold or condition, and the facts fall OUTSIDE those criteria, "
    "the provision does not apply and creates no requirement -- do not invent a "
    "requirement the text does not support. Apply this symmetrically: (i) a mere "
    "omission from an illustrative sub-list inside an otherwise-applicable "
    "provision does NOT place an item outside the governing scheme, so it is not a "
    "basis for 'not required'; but (ii) a genuine APPLICABILITY/scope condition of "
    "the governing statute or code -- the criterion that determines whether the "
    "scheme reaches the item at all -- IS decisive, and when the facts do not meet "
    "it the requirement does not attach. Decide a permit question the same way: "
    "from the governing permit statute and the governing code's applicability to "
    "the facts (including any explicit dollar/work or scope conditions). Name the "
    "specific criterion that is or is not met and conclude from it; do not default "
    "to 'required' or 'not required' by the item's label.\n"
    "4. If a SPECIFIC provision you need to decide is clearly missing from the "
    "excerpts, respond with a single line 'NEED: <short description of the rule "
    "or topic to search for>' and nothing else. Use this sparingly and only "
    "when you genuinely cannot decide without it.\n"
    "5. Otherwise give the final answer: a direct conclusion first, then the "
    "reasoning steps with citations. Be decisive even when the facts are "
    "incomplete -- state the assumption your conclusion depends on and answer "
    "anyway. Prefer a grounded best-effort answer over refusing: only respond "
    "with the exact refusal sentence \"" + MODEL_REFUSAL_SENTENCE + "\" if NONE "
    "of the excerpts are relevant to the subject at all. If relevant provisions "
    "are present but incomplete, reason from them and note the limitation.\n"
)

_NEED_RE = re.compile(r"^\s*NEED:\s*(.+)$", re.IGNORECASE)


def _verify_prompt(query: str, plan: dict, chunks_block: str) -> str:
    facts = json.dumps(plan.get("facts") or {}, ensure_ascii=False)
    return (
        "User question: %s\n\n"
        "Planning hypothesis (verify against the excerpts; do not trust "
        "blindly): %s\n"
        "Extracted facts: %s\n\n"
        "Excerpts (numbered):\n%s\n\n"
        "Decide the answer per the rules. Either output a single 'NEED: ...' "
        "line, or the final cited answer."
        % (query, plan.get("hypothesis") or "(none)", facts, chunks_block)
    )


def _consult_labels(chunks, cap: int = 6) -> list[str]:
    """Friendly, de-duplicated source names from retrieved chunks, for the
    live "Consulting <source>" progress line. Best-effort; never raises."""
    seen: set = set()
    out: list[str] = []
    for c in chunks or []:
        # Real document names only -- skip bare jurisdiction codes ("model",
        # "nc-state", ...) which read poorly as "Consulting model".
        lab = (c.get("source_label") or c.get("book")
               or c.get("source_file") or "")
        lab = str(lab).strip()
        if ("/" in lab) or ("\\" in lab):
            lab = lab.replace("\\", "/").split("/")[-1]
        for ext in (".pdf", ".html", ".htm", ".docx", ".doc", ".txt", ".md"):
            if lab.lower().endswith(ext):
                lab = lab[:-len(ext)]
                break
        lab = lab.strip()
        if lab and lab not in seen:
            seen.add(lab)
            out.append(lab)
        if len(out) >= cap:
            break
    return out


def _plan_brief(plan) -> dict:
    """The planner's reasoning, for the debug trace."""
    if not isinstance(plan, dict):
        return {}
    return {
        "cleaned_query": plan.get("cleaned_query"),
        "hypothesis": plan.get("hypothesis"),
        "facts": plan.get("facts"),
        "rules_to_check": plan.get("rules_to_check"),
        "search_queries": plan.get("search_queries"),
    }


def _sec_brief(chunks, cap: int = 20) -> list[dict]:
    """Compact view of retrieved sections (no full text) for the trace."""
    out = []
    for c in (chunks or [])[:cap]:
        out.append({
            "section_number": c.get("section_number"),
            "source": c.get("source_file") or c.get("source_label"),
            "jurisdiction": c.get("jurisdiction"),
        })
    return out


def answer(corpus: str, query: str, history: list[dict] | None = None,
           top_k: int = 15, filters: dict | None = None,
           balance: bool = False, location: str | None = None,
           progress=None) -> dict:
    """Agentic hypothesize-verify answer. Returns the same envelope shape as
    answer.answer(). Falls back to plain retrieval when planning is unavailable.

    ``progress`` (optional) is called with small dicts describing the current
    stage ({phase, message, sources?}) so a UI can show live status instead of a
    bare spinner. It is best-effort -- failures in the callback never affect the
    answer.
    """
    def emit(phase: str, message: str, **extra) -> None:
        if progress is None:
            return
        try:
            progress(dict(phase=phase, message=message, **extra))
        except Exception:  # noqa: BLE001 -- UI progress must never break answering
            pass

    # Structured debug trace: full chain (plan -> retrieval probes -> each
    # verify round's LLM output + timing -> final answer). The server persists
    # it so 2-min-vs-50s style discrepancies and "2x over the sources" can be
    # diagnosed after the fact.
    t_start = time.monotonic()
    rounds = 0
    trace = {"query": query, "stages": []}

    def _ms(t0):
        return int((time.monotonic() - t0) * 1000)

    def stage(name, **info):
        info["stage"] = name
        info["at_ms"] = _ms(t_start)
        trace["stages"].append(info)

    def attach(env):
        trace["total_ms"] = _ms(t_start)
        trace["rounds"] = rounds
        try:
            env["trace"] = trace
        except Exception:  # noqa: BLE001
            pass
        return env

    query = (query or "").strip()
    if not query:
        return attach(_refused([], "empty_query", "no_results"))

    emit("understand", "Understanding your question")
    _t = time.monotonic()
    plan = _understand(query, history)
    stage("understand", dur_ms=_ms(_t), plan=_plan_brief(plan))

    emit("search", "Searching across the sources")
    _t = time.monotonic()
    chunks = _retrieve(corpus, plan, top_k, filters, balance)
    _probes = (list(plan.get("search_queries") or []) + list(plan.get("rules_to_check") or []))
    stage("retrieve", dur_ms=_ms(_t), n_probes_planned=len(_probes) + 1,
          n_probes_run=min(len(_probes) + 1, MAX_PROBES), probe_cap=MAX_PROBES,
          probes=_probes, n_sections=len(chunks), sections=_sec_brief(chunks))
    if not chunks:
        return attach(_refused([], "no_supporting_documentation", "no_results"))
    emit("review", "Reviewing the relevant provisions",
         sources=_consult_labels(chunks))

    from .facets import answer_location
    applied_balance = bool(balance)
    cross_note = CROSS_JURISDICTION_NOTE % answer_location(location)
    system_prompt = _VERIFY_SYS + (cross_note if applied_balance else "")

    try:
        provider = get_chat_provider("azure")
    except Exception as e:  # noqa: BLE001
        return attach(_refused(chunks, "no_chat_client:%s" % e, "refused"))
    deployment = settings.chat_deployment_synthesis()

    while True:
        chunks_block = _build_chunks_block(chunks, query)
        messages = []
        if history:
            for turn in history:
                if turn.get("role") in ("user", "assistant") and turn.get("content"):
                    messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user",
                         "content": _verify_prompt(query, plan, chunks_block)})
        _t = time.monotonic()
        result = provider.complete(system=system_prompt, messages=messages,
                                   max_tokens=CHAT_MAX_TOKENS,
                                   temperature=CHAT_TEMPERATURE)
        _verify_ms = _ms(_t)
        text = (result.text or "").strip()
        tokens = {"prompt": result.prompt_tokens, "completion": result.completion_tokens}

        need = _NEED_RE.match(text)
        stage("verify", round=rounds, dur_ms=_verify_ms, deployment=deployment,
              tokens=tokens, prompt_chars=len(chunks_block), n_sections=len(chunks),
              is_need=bool(need),
              need_topic=(need.group(1).strip() if need else None),
              output=text[:6000])
        if need and rounds < MAX_VERIFY_ROUNDS:
            rounds += 1
            topic = need.group(1).strip()
            sys.stderr.write("[reason] verify round %d -> NEED: %s\n" % (rounds, topic))
            short = topic if len(topic) <= 60 else topic[:57] + "..."
            emit("refine", "Checking a specific rule: " + short)
            _t = time.monotonic()
            more = _retrieve(corpus, plan, top_k, filters, balance,
                             extra_probes=[topic])
            # union new sections into the working set (keep order, dedup by id)
            have = {c.get("section_id") for c in chunks}
            for c in more:
                if c.get("section_id") not in have:
                    chunks.append(c)
                    have.add(c.get("section_id"))
            stage("retrieve", round=rounds, reason="need", probe=topic,
                  dur_ms=_ms(_t), n_sections=len(chunks), sections=_sec_brief(chunks))
            emit("review", "Re-checking the provisions",
                 sources=_consult_labels(chunks))
            continue
        # NEED but out of rounds: strip the marker, answer from what we have.
        if need:
            text = "I could not find a more specific provision; answering from " \
                   "the available excerpts.\n\n" + text

        if MODEL_REFUSAL_SENTENCE in _normalize_quotes(text):
            out = _refused(chunks, "model_refused_insufficient_context", "refused")
            out["tokens"] = tokens
            out["plan"] = plan
            return attach(out)
        citations = _parse_citations(text, len(chunks))
        if not citations:
            out = _refused(chunks, "model_did_not_cite", "refused")
            out["tokens"] = tokens
            out["plan"] = plan
            return attach(out)
        return attach({"answer": text, "citations": citations, "chunks": chunks,
                       "authorities": _authorities(chunks, citations),
                       "refused": False, "refusal_reason": None, "status": "ok",
                       "tokens": tokens, "balanced": applied_balance, "plan": plan})
