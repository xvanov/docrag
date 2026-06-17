"""mcp_server.py -- expose docrag as MCP tools so a Claude agent can decide to use it.

This is the "make docrag part of the system" surface: a stdio MCP server that
publishes the corpus as typed tools. Any MCP client (Claude Code, Claude Desktop)
that mounts it gets `docrag_ask` / `docrag_sources` / `docrag_corpora` and decides
on its own when to call them -- the same flow as querying docrag by hand in an
agent session, but first-class.

The agent (Claude) supplies the reasoning + web search; docrag supplies grounded,
cited retrieval over the local corpus. The two compose: the agent forms a plan,
calls docrag to ground building-code claims, web-searches for precedent, and
reconciles -- exactly the ask -> retrieve -> validate loop.

Run (stdio):
    python -m docrag.mcp_server

Register in Claude Code via .mcp.json at the repo root (already provided), or:
    claude mcp add docrag -- <python> -m docrag.mcp_server
"""

from __future__ import annotations

import glob
import os

# Keep tool calls responsive: never let a first query block on the cross-encoder
# reranker loading/DOWNLOADING its model mid-call (the README flags CPU rerank as
# minutes-slow, and on a CUDA box `auto` would fetch a ~2GB model on first use --
# the classic multi-minute "stuck" symptom inside an MCP tool). RRF-only is the
# documented default; set DOCRAG_RERANK=1 before launch to opt back in.
os.environ.setdefault("DOCRAG_RERANK", "0")

from mcp.server.fastmcp import FastMCP

from . import settings

mcp = FastMCP("docrag")

# Corpora answered with jurisdiction-balanced + agentic (hypothesize-verify)
# retrieval, matching the web UI / docrag.ask defaults.
_BALANCED = {"building-codes"}


def _list_corpora() -> list[str]:
    """Corpora that have a built index (one SQLite DB per corpus)."""
    out = []
    for p in sorted(glob.glob(os.path.join(settings.index_dir(), "*.db"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if not name.startswith("_"):
            out.append(name)
    return out


def _fmt_authorities(authorities: list[dict]) -> str:
    if not authorities:
        return ""
    lines = ["", "Authorities cited:"]
    for a in authorities:
        lines.append("  [%s] %s" % (a.get("n"), a.get("designation") or ""))
    return "\n".join(lines)


@mcp.tool()
def docrag_ask(question: str, corpus: str = "building-codes",
               balance: bool = True, top_k: int = 15) -> str:
    """Ask the docrag corpus a question and get a GROUNDED, CITED answer.

    Use this to ground or verify any claim about the building codes / statutes /
    ordinances in the corpus (IBC, NC State Building Code, NC statutes, Durham
    UDO). The answer is synthesized only from retrieved provisions and every
    claim carries a [N] citation mapped to a named authority. Treat it as the
    authoritative local-law layer; pair it with your own web search for
    real-world precedent, current process/fees, or anything the corpus lacks.

    Args:
        question: A natural-language question. Lay phrasing is fine -- retrieval
            translates it into code vocabulary.
        corpus: Corpus name (default "building-codes"). See docrag_corpora.
        balance: Balance retrieval across model/state/local layers (default
            True; only meaningful for the multi-source building-codes corpus).
        top_k: Sections to retrieve (3-20).
    """
    corpus = (corpus or "building-codes").strip().lower()
    question = (question or "").strip()
    if not question:
        return "ERROR: empty question."
    top_k = max(3, min(int(top_k or 15), 20))
    use_balance = bool(balance) and corpus in _BALANCED

    # Agentic hypothesize-verify for the balanced corpus; plain synthesis else.
    if use_balance:
        from .reason import answer as ans
    else:
        from .answer import answer as ans

    try:
        res = ans(corpus=corpus, query=question, top_k=top_k, balance=use_balance)
    except FileNotFoundError:
        return ("ERROR: corpus %r has no built index. Available: %s"
                % (corpus, ", ".join(_list_corpora()) or "(none)"))
    except Exception as e:  # noqa: BLE001
        return "ERROR: docrag query failed: %s" % e

    if res.get("refused"):
        return ("No grounded answer (%s). The corpus does not contain provisions "
                "that decide this; say so and rely on other sources."
                % res.get("refusal_reason"))
    return (res.get("answer") or "").strip() + _fmt_authorities(res.get("authorities") or [])


@mcp.tool()
def docrag_sources(question: str, corpus: str = "building-codes",
                   top_k: int = 12) -> str:
    """Retrieve raw corpus passages for a query WITHOUT the LLM synthesis layer.

    Use when you want to read the actual provision text yourself and reason over
    it, rather than getting a pre-synthesized answer. Returns the top sections
    with source file, section number/title, page, and a snippet.

    Args:
        question: What to search for.
        corpus: Corpus name (default "building-codes").
        top_k: Number of sections (3-20).
    """
    corpus = (corpus or "building-codes").strip().lower()
    question = (question or "").strip()
    if not question:
        return "ERROR: empty question."
    top_k = max(3, min(int(top_k or 12), 20))
    from .query import rag_query
    try:
        r = rag_query(corpus, question, top_k=top_k,
                      balance=corpus in _BALANCED)
    except FileNotFoundError:
        return ("ERROR: corpus %r has no built index. Available: %s"
                % (corpus, ", ".join(_list_corpora()) or "(none)"))
    except Exception as e:  # noqa: BLE001
        return "ERROR: retrieval failed: %s" % e

    results = r.get("results") or []
    if not results:
        return "No passages found (status: %s)." % r.get("status")
    out = []
    for i, c in enumerate(results, 1):
        loc = ("p.%s" % c["page"]) if c.get("page") else "%s-%s" % (
            c.get("start_line"), c.get("end_line"))
        head = " ".join(x for x in (c.get("section_number"),
                                    c.get("section_title")) if x)
        text = (c.get("text") or "").strip()
        if len(text) > 1200:
            text = text[:1200].rstrip() + "..."
        out.append("[%d] %s | %s%s\n%s" % (
            i, c.get("source_file") or "?", head or "(section)",
            "  (%s)" % loc if loc and loc != "None-None" else "", text))
    return "\n\n".join(out)


@mcp.tool()
def docrag_corpora() -> str:
    """List the docrag corpora that have a built index (valid `corpus` values)."""
    names = _list_corpora()
    return ("Available corpora: " + ", ".join(names)) if names else \
        "No corpora are indexed yet."


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
