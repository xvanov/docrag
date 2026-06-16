"""ask.py -- terminal Q&A across a corpus, with the cross-jurisdiction mode.

For the multi-source `building-codes` corpus this defaults to jurisdiction-
balanced retrieval (model IBC + NC 2024 state code + Durham UDO) so one
question is answered from all three layers, with each claim attributed.

    python -m docrag.ask "min stair riser height for an exit stair?"
    python -m docrag.ask --corpus udo "what is a major site plan?"
    python -m docrag.ask --no-balance --top-k 12 "..."   # plain retrieval
    python -m docrag.ask --sources "fire separation distance"   # chunks only
"""
from __future__ import annotations

import argparse
import sys

from . import settings  # noqa: F401  (ensures .env is loaded via import chain)
from .answer import answer as rag_answer
from .query import rag_query

_BALANCED_CORPORA = {"building-codes"}


def _print_sources(chunks: list[dict]) -> None:
    print("\nSources:")
    for i, c in enumerate(chunks, 1):
        loc = ("p.%s" % c["page"]) if c.get("page") else (
            "%s-%s" % (c.get("start_line"), c.get("end_line")))
        print("  [%d] %s (%s)  %s"
              % (i, c.get("source_file") or "?", loc, c.get("path") or ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="docrag.ask")
    ap.add_argument("query", nargs="+", help="the question")
    ap.add_argument("--corpus", default="building-codes")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--no-balance", action="store_true",
                    help="disable jurisdiction-balanced retrieval")
    ap.add_argument("--sources", action="store_true",
                    help="print retrieved chunks only (no LLM)")
    ap.add_argument("--agentic", action="store_true",
                    help="use the hypothesize-verify pipeline (reason.py)")
    ap.add_argument("--plain", dest="agentic", action="store_false",
                    help="force the single-pass answer path")
    ap.set_defaults(agentic=None)
    args = ap.parse_args(argv)

    corpus = args.corpus.strip().lower()
    query = " ".join(args.query).strip()
    balance = (corpus in _BALANCED_CORPORA) and not args.no_balance

    if args.sources:
        r = rag_query(corpus, query, top_k=args.top_k, balance=balance)
        if not r.get("results"):
            print("(no results: %s)" % r.get("status"))
            return 0
        _print_sources(r["results"])
        return 0

    # Default: agentic for the balanced building-codes corpus; plain otherwise.
    use_agentic = args.agentic if args.agentic is not None else balance
    if use_agentic:
        from .reason import answer as rag_answer_fn
    else:
        rag_answer_fn = rag_answer
    res = rag_answer_fn(corpus=corpus, query=query, top_k=args.top_k,
                        balance=balance)
    if res.get("refused"):
        print("No grounded answer (%s)." % res.get("refusal_reason"))
        if res.get("chunks"):
            _print_sources(res["chunks"])
        return 0

    print(res["answer"])
    authorities = res.get("authorities") or []
    if authorities:
        print("\nAuthorities cited:")
        for a in authorities:
            print("  [%d] %s" % (a["n"], a["designation"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
