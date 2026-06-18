"""mcp_server.py -- expose the youtube domain as MCP tools.

Publishes youtube_ask / youtube_sources / youtube_exhaustive / youtube_corpora
over stdio so a Claude agent can query indexed channels. Mirrors the building-
codes server's hardening: stdout is the JSON-RPC channel (so all logging goes to
stderr), and the answer stack (which pulls the Anthropic SDK) is imported
EAGERLY in the main thread at module load -- never lazily inside a worker thread,
which would deadlock CPython's import lock against the event loop.

Run (stdio):  python -m docrag.domains.youtube.mcp_server
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import sys
from concurrent.futures import TimeoutError as _FTimeout

os.environ.setdefault("PYTHONUNBUFFERED", "1")
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)
logging.disable(logging.INFO)
for _h in list(logging.getLogger().handlers):
    try:
        _h.setStream(sys.stderr)
    except Exception:  # noqa: BLE001
        pass
os.environ.setdefault("DOCRAG_RERANK", "0")

from mcp.server.fastmcp import FastMCP

import anthropic  # noqa: E402,F401  EAGER main-thread load; never first-import in a worker

from ... import settings
# EAGER, main-thread imports (pull numpy via the answer/query stack).
from .answer import single_answer            # noqa: E402
from .mapreduce import mapreduce_answer       # noqa: E402
from ...query import rag_query                # noqa: E402
from .answer import excerpt_label             # noqa: E402

mcp = FastMCP("rag-youtube")


def _list_corpora() -> list[str]:
    out = []
    for p in sorted(glob.glob(os.path.join(settings.index_dir(), "*.db"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if not name.startswith("_"):
            out.append(name)
    return out


def _tool_timeout() -> float:
    return float(settings.get("DOCRAG_TOOL_TIMEOUT", 200) or 200)


async def _run_bounded(fn):
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(None, fn),
                                      timeout=_tool_timeout())
    except (asyncio.TimeoutError, TimeoutError) as e:
        raise _FTimeout from e


def _fmt_sources(authorities: list[dict]) -> str:
    if not authorities:
        return ""
    lines = ["", "Sources:"]
    for a in authorities:
        lines.append("  [%s] %s (%s) %s" % (a.get("n"), a.get("title") or "",
                                            a.get("timestamp") or "", a.get("url") or ""))
    return "\n".join(lines)


@mcp.tool()
async def youtube_ask(question: str, corpus: str, top_k: int = 12) -> str:
    """Ask a focused question over an indexed YouTube corpus; grounded + cited.

    Returns an answer synthesized only from retrieved transcript excerpts, each
    claim carrying a [N] citation mapped to a video + timestamp deep-link. Use
    for focused questions ("what does he say about X?"). For exhaustive "list ALL
    X across every video" questions, use youtube_exhaustive instead.

    Args:
        question: Natural-language question.
        corpus: Indexed corpus name (see youtube_corpora).
        top_k: Transcript excerpts to retrieve (3-20).
    """
    corpus = (corpus or "").strip().lower()
    question = (question or "").strip()
    if not question:
        return "ERROR: empty question."
    if not corpus:
        return "ERROR: corpus is required. Available: %s" % (", ".join(_list_corpora()) or "(none)")
    top_k = max(3, min(int(top_k or 12), 20))
    try:
        res = await _run_bounded(lambda: single_answer(corpus, question, top_k=top_k))
    except _FTimeout:
        return "ERROR: youtube_ask timed out (>%ds)." % int(_tool_timeout())
    except Exception as e:  # noqa: BLE001
        return "ERROR: youtube_ask failed: %s" % e
    if res.get("refused"):
        return "No grounded answer (%s)." % res.get("refusal_reason")
    return (res.get("answer") or "").strip() + _fmt_sources(res.get("authorities") or [])


@mcp.tool()
async def youtube_exhaustive(question: str, corpus: str, target: str = "") -> str:
    """Exhaustively extract across EVERY video, then synthesize (map-reduce).

    Use for "list ALL / every / how many X" questions where top-k retrieval would
    miss instances. Visits each video's full transcript once (cached), then
    dedupes/organizes. Slower and more expensive than youtube_ask.

    Args:
        question: The exhaustive question.
        corpus: Indexed corpus name.
        target: What to extract per video (defaults to the question).
    """
    corpus = (corpus or "").strip().lower()
    question = (question or "").strip()
    if not question or not corpus:
        return "ERROR: question and corpus are required."
    try:
        res = await _run_bounded(
            lambda: mapreduce_answer(corpus, question, target=(target or None)))
    except _FTimeout:
        return "ERROR: youtube_exhaustive timed out (>%ds)." % int(_tool_timeout())
    except Exception as e:  # noqa: BLE001
        return "ERROR: youtube_exhaustive failed: %s" % e
    tail = ("\n\n[map-reduce] %s" % res["stats"]) if res.get("stats") else ""
    return (res.get("answer") or "").strip() + tail


@mcp.tool()
async def youtube_sources(question: str, corpus: str, top_k: int = 12) -> str:
    """Retrieve raw transcript excerpts (no LLM synthesis) with timestamps.

    Args:
        question: What to search for.
        corpus: Indexed corpus name.
        top_k: Excerpts to return (3-20).
    """
    corpus = (corpus or "").strip().lower()
    question = (question or "").strip()
    if not question or not corpus:
        return "ERROR: question and corpus are required."
    top_k = max(3, min(int(top_k or 12), 20))

    def _work():
        return rag_query(corpus, question, top_k=top_k, balance=False,
                         expand=False, short_code_filter=False)
    try:
        r = await _run_bounded(_work)
    except _FTimeout:
        return "ERROR: retrieval timed out (>%ds)." % int(_tool_timeout())
    except Exception as e:  # noqa: BLE001
        return "ERROR: retrieval failed: %s" % e
    results = r.get("results") or []
    if not results:
        return "No excerpts found (status: %s)." % r.get("status")
    out = []
    for i, c in enumerate(results, 1):
        text = (c.get("text") or "").strip()
        if len(text) > 1000:
            text = text[:1000].rstrip() + "..."
        link = (c.get("metadata") or {}).get("deep_link") or ""
        out.append("[%d] %s\n%s\n%s" % (i, excerpt_label(c), link, text))
    return "\n\n".join(out)


@mcp.tool()
def youtube_corpora() -> str:
    """List indexed corpora (valid `corpus` values)."""
    names = _list_corpora()
    return ("Available corpora: " + ", ".join(names)) if names else \
        "No corpora are indexed yet."


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
