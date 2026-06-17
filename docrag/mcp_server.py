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

import asyncio
import glob
import logging
import os
import sys
from concurrent.futures import TimeoutError as _FTimeout

# CRITICAL for stdio MCP: STDOUT is the JSON-RPC protocol channel. Any library
# that logs to stdout corrupts the stream and, once the OS pipe buffer fills,
# BLOCKS the writing thread mid-call -- which presents as a multi-minute "stuck"
# tool call (the same query runs in ~20s called directly, but stalls over
# stdio). The openai/httpx clients emit INFO "HTTP Request: ..." logs (often via
# a rich handler bound to stdout). Per-logger setLevel proved insufficient, so
# hard-disable everything at INFO and below GLOBALLY, force the remaining
# WARNING+ records to stderr, and point any rich/console output at stderr too.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)
logging.disable(logging.INFO)            # global: drop all INFO/DEBUG records
for _h in list(logging.getLogger().handlers):
    try:
        _h.setStream(sys.stderr)         # any root handler -> stderr, never stdout
    except Exception:  # noqa: BLE001
        pass

# Keep tool calls responsive: never let a first query block on the cross-encoder
# reranker loading/DOWNLOADING its model mid-call (the README flags CPU rerank as
# minutes-slow, and on a CUDA box `auto` would fetch a ~2GB model on first use --
# the classic multi-minute "stuck" symptom inside an MCP tool). RRF-only is the
# documented default; set DOCRAG_RERANK=1 before launch to opt back in.
os.environ.setdefault("DOCRAG_RERANK", "0")

from mcp.server.fastmcp import FastMCP

from . import settings

# Import the full retrieval/answer stack EAGERLY, in the MAIN thread, at module
# load -- NOT lazily inside a tool. These pull heavy native extensions (numpy via
# sqlite-vec; torch via the reranker import chain). Importing them for the first
# time inside a worker thread (run_in_executor) deadlocks on CPython's import
# lock against the FastMCP event-loop thread: the stdio server hangs on
# `import numpy` and the tool call never returns (confirmed by a thread-stack
# dump). Doing it here pays the ~15s cost once at startup; every tool call then
# only CALLS already-imported code, so the worker never imports.
from . import facets                       # noqa: E402
from .db import open_db                    # noqa: E402
from .query import rag_query               # noqa: E402
from .answer import answer as _answer_fast    # noqa: E402  single-pass synthesis
from .reason import answer as _answer_deep    # noqa: E402  agentic hypothesize->verify

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


def _location_filters(corpus: str, location: str) -> dict:
    """Build the retrieval ``filters`` dict for a location, mirroring the web
    server: the location (jurisdiction-tier) selector plus each document's
    latest edition. Returns {"location", "versions"}; versions is best-effort
    (empty if the index can't be opened)."""
    eff_versions: dict = {}
    try:
        conn = open_db(corpus)
        try:
            eff_versions = facets.resolve_versions(conn, {})
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 -- versions are optional sugar
        eff_versions = {}
    return {"location": facets.location(location)["key"], "versions": eff_versions}


def _tool_timeout() -> float:
    return float(settings.get("DOCRAG_TOOL_TIMEOUT", 150) or 150)


async def _run_bounded(fn):
    """Run a blocking call in a worker thread with a hard wall-clock cap, the
    RELIABLE way under FastMCP: the tool is async, so the asyncio event loop
    (which serves the stdio protocol) keeps control and can deliver the result
    or the timeout. ``asyncio.wait_for`` cancels the await at the cap; the
    abandoned executor thread dies on its own once the Azure client's own
    timeout fires. (A synchronous ThreadPoolExecutor.result(timeout) inside a
    sync tool did NOT reliably return under FastMCP -- hence the indefinite
    'stuck' even with a cap.) Raises _FTimeout on timeout."""
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(None, fn),
                                      timeout=_tool_timeout())
    except (asyncio.TimeoutError, TimeoutError) as e:
        raise _FTimeout from e


def _fmt_authorities(authorities: list[dict]) -> str:
    if not authorities:
        return ""
    lines = ["", "Authorities cited:"]
    for a in authorities:
        lines.append("  [%s] %s" % (a.get("n"), a.get("designation") or ""))
    return "\n".join(lines)


@mcp.tool()
async def docrag_ask(question: str, corpus: str = "building-codes",
                     balance: bool = True, top_k: int = 15,
                     location: str = "durham-nc", deep: bool = False) -> str:
    """Ask the docrag corpus a question and get a GROUNDED, CITED answer.

    Use this to ground or verify any claim about the building codes / statutes /
    ordinances in the corpus (IBC, NC State Building Code, NC statutes, and the
    local land-use ordinances). The answer is synthesized only from retrieved
    provisions and every claim carries a [N] citation mapped to a named
    authority. Treat it as the authoritative local-law layer; pair it with your
    own web search for real-world precedent, current process/fees, or anything
    the corpus lacks.

    Args:
        question: A natural-language question. Lay phrasing is fine -- retrieval
            translates it into code vocabulary.
        corpus: Corpus name (default "building-codes"). See docrag_corpora.
        balance: Balance retrieval across model/state/local layers (default
            True; only meaningful for the multi-source building-codes corpus).
        top_k: Sections to retrieve (3-20).
        location: Which jurisdiction's LOCAL layer to stack on the shared
            model/state/federal codes. The model+NC-state+federal layers are
            always included; this selects which local ordinance applies so a
            Durham question never sees Alamance ordinances and vice versa.
            Valid: "durham-nc" (default), "alamance-county-nc" (unincorporated
            Alamance), "burlington-nc", "graham-nc", "alamance-towns-nc",
            "north-carolina" (statewide, no local layer), "model" (I-Codes only).
    """
    corpus = (corpus or "building-codes").strip().lower()
    question = (question or "").strip()
    if not question:
        return "ERROR: empty question."
    top_k = max(3, min(int(top_k or 15), 20))
    use_balance = bool(balance) and corpus in _BALANCED
    location = (location or "durham-nc").strip().lower()

    def _work():
        # All blocking work (filters, the heavy reason/query import, retrieval +
        # LLM synthesis) runs in the worker thread so the event loop stays free.
        # Default = fast single-pass synthesis (~1 chat call): responsive and
        # robust to backend throttling. deep=True = the agentic hypothesize->
        # verify pipeline (~several chat calls; slower, higher-quality), matching
        # the web UI. Both are grounded + cited + jurisdiction-balanced.
        filters = _location_filters(corpus, location) if corpus in _BALANCED else None
        ans = _answer_deep if (deep and use_balance) else _answer_fast
        return ans(corpus=corpus, query=question, top_k=top_k,
                   balance=use_balance, filters=filters, location=location)

    try:
        res = await _run_bounded(_work)
    except _FTimeout:
        return ("ERROR: docrag timed out (>%ds) -- the LLM/embedding backend was "
                "slow or unreachable. Try again, or proceed from other sources."
                % int(_tool_timeout()))
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
async def docrag_sources(question: str, corpus: str = "building-codes",
                         top_k: int = 12, location: str = "durham-nc") -> str:
    """Retrieve raw corpus passages for a query WITHOUT the LLM synthesis layer.

    Use when you want to read the actual provision text yourself and reason over
    it, rather than getting a pre-synthesized answer. Returns the top sections
    with source file, section number/title, page, and a snippet.

    Args:
        question: What to search for.
        corpus: Corpus name (default "building-codes").
        top_k: Number of sections (3-20).
        location: Jurisdiction whose local layer to include (see docrag_ask).
            "durham-nc" (default), "alamance-county-nc", "burlington-nc",
            "graham-nc", "alamance-towns-nc", "north-carolina", "model".
    """
    corpus = (corpus or "building-codes").strip().lower()
    question = (question or "").strip()
    if not question:
        return "ERROR: empty question."
    top_k = max(3, min(int(top_k or 12), 20))
    location = (location or "durham-nc").strip().lower()

    def _work():
        filters = _location_filters(corpus, location) if corpus in _BALANCED else None
        return rag_query(corpus, question, top_k=top_k, filters=filters,
                         balance=corpus in _BALANCED)

    try:
        r = await _run_bounded(_work)
    except _FTimeout:
        return "ERROR: docrag retrieval timed out (>%ds)." % int(_tool_timeout())
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
    # Diagnostic (opt-in): dump all thread stacks to a file every N seconds so a
    # stall inside the stdio server can be located. DOCRAG_MCP_FAULTDUMP=<path>.
    _fd = os.environ.get("DOCRAG_MCP_FAULTDUMP")
    if _fd:
        import faulthandler
        _f = open(_fd, "w", encoding="utf-8")
        faulthandler.dump_traceback_later(20, repeat=True, file=_f)
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
