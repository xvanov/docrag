"""eval.py -- section-grounded retrieval + answer evaluation.

Because every chunk carries a section number/breadcrumb, retrieval quality is
measured against expected tokens (section numbers / phrases) rather than fuzzy
cosine labels. Reports retrieval hit-rate@k, mean reciprocal rank of the first
hit, and (unless --no-answer) the non-refusal + citation rate from answer().

Golden set: ``evalset/<corpus>.jsonl``; each line:
    {"q": "...", "expect": ["160D-1110", "permit", ...], "note": "...",
     "expect_cite": ["R101.2.1"],      # optional: designation the answer must cite
     "expect_answer": ["no permit"]}   # optional: phrases the conclusion must contain
A case is a retrieval hit if any ``expect`` token appears in a retrieved
section's number / breadcrumb / source / text (case-insensitive). When present,
``expect_cite`` checks the answer actually spelled out + cited the right
authority (any token), and ``expect_answer`` checks the conclusion (all tokens)
-- so the system must find, reason to, and cite the correct statute on its own.

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


def _answer_haystack(env: dict) -> str:
    """Answer text + spelled-out authority designations (lowercased)."""
    parts = [env.get("answer") or ""]
    for a in env.get("authorities") or []:
        parts.append(a.get("designation") or "")
        parts.append(str(a.get("section_number") or ""))
    return " ".join(parts).lower()


def _all_present(tokens: list[str], hay: str) -> bool:
    """Every token group must match. A token may be a '|'-separated set of
    acceptable phrasings (any-of) so a free-text conclusion isn't graded on one
    exact wording -- e.g. 'no permit|not require a permit|without a permit'."""
    if not tokens:
        return True
    for group in tokens:
        if not any(alt.strip().lower() in hay for alt in group.split("|")):
            return False
    return True


def _any_present(tokens: list[str], hay: str) -> bool:
    return any(t.lower() in hay for t in tokens) if tokens else True


def _answer_fn(agentic: bool):
    if agentic:
        from .reason import answer as fn
    else:
        from .answer import answer as fn
    return fn


def run(corpus: str, top_k: int, do_answer: bool, balance: bool,
        agentic: bool = False) -> int:
    cases = _load_cases(corpus)
    if not cases:
        return 1
    hits = 0
    rr_sum = 0.0
    answered = 0
    answerable = 0
    # New signals (only counted for cases that declare them).
    cite_total = cite_ok = 0
    concl_total = concl_ok = 0
    ans_fn = _answer_fn(agentic) if do_answer else None
    print("[eval] corpus=%s  cases=%d  top_k=%d  balance=%s  answer=%s  agentic=%s\n"
          % (corpus, len(cases), top_k, balance, do_answer, agentic))
    for c in cases:
        q = c.get("q", "")
        expect = c.get("expect") or []
        expect_cite = c.get("expect_cite") or []
        expect_answer = c.get("expect_answer") or []
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
                env = ans_fn(corpus, q, top_k=top_k, balance=balance)
                ok = not env.get("refused") and bool(env.get("citations"))
                answered += 1 if ok else 0
                ans_mark = "  ans=%s" % ("OK" if ok else ("REFUSED:" + str(env.get("refusal_reason"))))
                hay = _answer_haystack(env)
                if expect_cite:
                    cite_total += 1
                    cok = _any_present(expect_cite, hay)
                    cite_ok += 1 if cok else 0
                    ans_mark += "  cite=%s" % ("OK" if cok else "MISS")
                if expect_answer:
                    concl_total += 1
                    pok = _all_present(expect_answer, hay)
                    concl_ok += 1 if pok else 0
                    ans_mark += "  concl=%s" % ("OK" if pok else "MISS")
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
        if cite_total:
            print("[eval] cited-correct-authority: %d/%d (%.0f%%)"
                  % (cite_ok, cite_total, 100.0 * cite_ok / cite_total))
        if concl_total:
            print("[eval] correct-conclusion: %d/%d (%.0f%%)"
                  % (concl_ok, concl_total, 100.0 * concl_ok / concl_total))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="docrag.eval")
    p.add_argument("--corpus", required=True)
    p.add_argument("--top-k", type=int, default=15)  # match the balanced server path
    p.add_argument("--no-answer", action="store_true",
                   help="retrieval only; skip the LLM answer pass")
    bal = p.add_mutually_exclusive_group()
    bal.add_argument("--balance", dest="balance", action="store_true")
    bal.add_argument("--no-balance", dest="balance", action="store_false")
    p.set_defaults(balance=True)
    p.add_argument("--agentic", action="store_true",
                   help="use the hypothesize-verify pipeline (reason.py)")
    args = p.parse_args(argv)
    return run(args.corpus, args.top_k, not args.no_answer, args.balance,
               agentic=args.agentic)


if __name__ == "__main__":
    raise SystemExit(main())
