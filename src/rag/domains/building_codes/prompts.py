"""prompts.py -- building-codes synthesis prompts + citation-label helpers.

Relocated verbatim from the core answer.py so the building-code regulatory
reasoning (field-preemption, jurisdiction layering, authority designation) lives
in the domain. answer.py / reason.py import these back, so behavior is
unchanged; only the definitions moved.
"""

from __future__ import annotations

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
