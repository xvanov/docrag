"""eval.py -- section-grounded retrieval + answer evaluation.

Because every chunk carries a section number/breadcrumb, retrieval quality is
measured against expected tokens (section numbers / phrases) rather than fuzzy
cosine labels. Reports retrieval hit-rate@k, mean reciprocal rank of the first
hit, and (unless --no-answer) the non-refusal + citation rate from answer().

Golden set: ``evalset/<corpus>.jsonl``; each line:
    {"q": "...", "expect": ["160D-1110", "permit", ...], "note": "..."}
A case is a retrieval hit if any expected token appears in a retrieved
section's number / breadcrumb / source / text (case-insensitive).

CLI:
  python -m docrag.eval --corpus building-codes [--top-k 8]
                        [--no-answer] [--balance|--no-balance]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import settings
from .query import rag_query


def _evalset_path(corpus: str) -> str:
    return os.path.join(settings.repo_root(), "evalset", "%s.jsonl" % corpus)


def _load_cases(corpus: str) -> list[dict]:
    path = _evalset_path(corpus)
    if not os.path.isfile(path):
        sys.stderr.write("ERROR: no eval set at %s\n" % path)
        return []
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                cases.append(json.loads(ln))
    return cases


def _haystack(r: dict) -> str:
    return " ".join(str(r.get(k) or "") for k in
                    ("section_number", "section_title", "breadcrumb",
                     "source_file", "text")).lower()


def _first_hit_rank(results: list[dict], expect: list[str]) -> int:
    toks = [t.lower() for t in expect]
    for i, r in enumerate(results, start=1):
        hay = _haystack(r)
        if any(t in hay for t in toks):
            return i
    return 0


def run(corpus: str, top_k: int, do_answer: bool, balance: bool) -> int:
    cases = _load_cases(corpus)
    if not cases:
        return 1
    hits = 0
    rr_sum = 0.0
    answered = 0
    answerable = 0
    print("[eval] corpus=%s  cases=%d  top_k=%d  balance=%s  answer=%s\n"
          % (corpus, len(cases), top_k, balance, do_answer))
    for c in cases:
        q = c.get("q", "")
        expect = c.get("expect") or []
        retr = rag_query(corpus, q, top_k=top_k, balance=balance)
        results = retr.get("results") or []
        rank = _first_hit_rank(results, expect)
        hit = rank > 0
        hits += 1 if hit else 0
        rr_sum += (1.0 / rank) if rank else 0.0

        ans_mark = ""
        if do_answer:
            answerable += 1
            try:
                env = __import__("docrag.answer", fromlist=["answer"]).answer(
                    corpus, q, top_k=top_k, balance=balance)
                ok = not env.get("refused") and bool(env.get("citations"))
                answered += 1 if ok else 0
                ans_mark = "  ans=%s" % ("OK" if ok else ("REFUSED:" + str(env.get("refusal_reason"))))
            except Exception as e:  # noqa: BLE001
                ans_mark = "  ans=ERR:%s" % e
        print("  [%s] rank=%-2s  %s%s" % ("HIT" if hit else "MISS",
              rank or "-", q[:60], ans_mark))

    n = len(cases)
    print("\n[eval] retrieval hit-rate@%d: %d/%d (%.0f%%)  MRR=%.3f"
          % (top_k, hits, n, 100.0 * hits / n, rr_sum / n))
    if do_answer and answerable:
        print("[eval] answered (non-refused + cited): %d/%d (%.0f%%)"
              % (answered, answerable, 100.0 * answered / answerable))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="docrag.eval")
    p.add_argument("--corpus", required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--no-answer", action="store_true",
                   help="retrieval only; skip the LLM answer pass")
    bal = p.add_mutually_exclusive_group()
    bal.add_argument("--balance", dest="balance", action="store_true")
    bal.add_argument("--no-balance", dest="balance", action="store_false")
    p.set_defaults(balance=True)
    args = p.parse_args(argv)
    return run(args.corpus, args.top_k, not args.no_answer, args.balance)


if __name__ == "__main__":
    raise SystemExit(main())
