"""cli.py -- single multi-domain entry point: ``rag``.

  rag domains
  rag ask    --corpus zeihan [--domain youtube] [--strategy single|agentic|
             mapreduce|longctx] [--exhaustive] [--longctx] [--top-k N]
             [--location durham-nc] "question"
  rag index  --corpus zeihan [--domain youtube] [URL|id ...]
             [--full] [--dry-run] [--limit N] [--lang en,de]      (youtube)
  rag index  --corpus building-codes [--full] [--dry-run] [--confirm] (building-codes)
  rag status --corpus zeihan

``--domain`` defaults to the corpus's configured/known domain (building_codes
otherwise). The domain decides the answering strategy unless overridden.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import registry
from .core import orchestrate
from .domains.base import STRATEGY_LONGCTX, STRATEGY_MAPREDUCE


def _resolve_domain_name(args) -> str:
    return (getattr(args, "domain", None)
            or registry.domain_name_for_corpus(args.corpus))


def _cmd_domains(_args) -> int:
    print("available domains:", ", ".join(registry.available_domains()))
    return 0


def _cmd_ask(args) -> int:
    domain_name = _resolve_domain_name(args)
    strategy = args.strategy
    if args.exhaustive:
        strategy = STRATEGY_MAPREDUCE
    elif args.longctx:
        strategy = STRATEGY_LONGCTX
    kw = {}
    if args.location:
        kw["location"] = args.location
    if args.target:
        kw["target"] = args.target
    out = orchestrate.answer(args.corpus, args.query, domain=domain_name,
                             strategy=strategy, top_k=args.top_k, **kw)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        return 0 if not out.get("refused") else 1
    if out.get("refused"):
        sys.stderr.write("(refused: %s)\n" % out.get("refusal_reason"))
        return 1
    print(out.get("answer") or "")
    auths = out.get("authorities") or []
    if auths:
        print("\nSources:")
        for a in auths:
            if a.get("url"):  # youtube-style
                print("  [%s] %s (%s) %s" % (a.get("n"), a.get("title"),
                                             a.get("timestamp") or "", a.get("url")))
            else:             # building-codes-style
                print("  [%s] %s" % (a.get("n"), a.get("designation")))
    if "stats" in out:
        print("\n[map-reduce] %s" % out["stats"])
    return 0


def _cmd_index(args) -> int:
    domain_name = _resolve_domain_name(args)
    corpus = args.corpus.strip().lower()
    if domain_name == "youtube":
        from .domains.youtube.ingest import ingest
        if not args.targets:
            sys.stderr.write("ERROR: youtube index needs at least one video/"
                             "channel/playlist URL or id.\n")
            return 2
        langs = [s for s in (args.lang or "en").split(",") if s]
        return ingest(corpus, args.targets, languages=langs, limit=args.limit,
                      full=args.full, dry_run=args.dry_run)
    # building-codes (and any filesystem domain): delegate to the index CLI.
    from . import index
    flags = []
    if args.full:
        flags.append("--full")
    if args.dry_run:
        flags.append("--dry-run")
    if args.confirm:
        flags.append("--confirm")
    if args.limit:
        flags += ["--limit", str(args.limit)]
    return index.main(["build", "--corpus", corpus, *flags])


def _cmd_status(args) -> int:
    domain_name = _resolve_domain_name(args)
    from . import db
    corpus = args.corpus.strip().lower()
    conn = db.open_db(corpus)
    try:
        s = db.stats(conn, corpus)
    finally:
        conn.close()
    print("[status] %s (domain=%s)" % (corpus, domain_name))
    print("  files:  %d" % s["file_count"])
    print("  chunks: %d (vec=%d fts=%d)" % (s["chunk_count"], s["vec_count"], s["fts_count"]))
    if s["section_count"]:
        print("  sections: %d  refs: %d" % (s["section_count"], s["ref_count"]))
    return 0


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag",
                                description="Grounded hybrid-search RAG over pluggable domains.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("domains", help="List available domains.")

    pa = sub.add_parser("ask", help="Ask a grounded, cited question.")
    pa.add_argument("query")
    pa.add_argument("--corpus", required=True)
    pa.add_argument("--domain", help="Override the corpus's domain.")
    pa.add_argument("--strategy", choices=["single", "agentic", "mapreduce", "longctx"])
    pa.add_argument("--exhaustive", action="store_true",
                    help="Force map-reduce over every doc (exhaustive coverage).")
    pa.add_argument("--longctx", action="store_true",
                    help="Stuff the whole corpus into one (cached) prompt.")
    pa.add_argument("--target", help="What to extract (map-reduce); defaults to the query.")
    pa.add_argument("--top-k", type=int, default=12)
    pa.add_argument("--location", help="building-codes jurisdiction (e.g. durham-nc).")
    pa.add_argument("--json", action="store_true")

    pi = sub.add_parser("index", help="Build / update a corpus.")
    pi.add_argument("targets", nargs="*", help="youtube: video/channel/playlist URLs or ids.")
    pi.add_argument("--corpus", required=True)
    pi.add_argument("--domain", help="Override the corpus's domain.")
    pi.add_argument("--full", action="store_true")
    pi.add_argument("--dry-run", action="store_true")
    pi.add_argument("--confirm", action="store_true", help="building-codes cost gate.")
    pi.add_argument("--limit", type=int, default=0)
    pi.add_argument("--lang", help="youtube transcript languages, comma-sep (default en).")

    ps = sub.add_parser("status", help="Show corpus index stats.")
    ps.add_argument("--corpus", required=True)
    ps.add_argument("--domain")

    args = p.parse_args(argv)
    return {
        "domains": _cmd_domains, "ask": _cmd_ask,
        "index": _cmd_index, "status": _cmd_status,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
