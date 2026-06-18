"""orchestrate.py -- domain-aware answer dispatch.

Resolves the domain for a corpus (or an explicit ``domain`` name) and delegates
to its ``answer`` method, which picks/honors a strategy (single / agentic /
mapreduce / longctx). This is the single entry point the CLI and MCP servers
call so they stay domain-blind.
"""

from __future__ import annotations

from ..registry import domain_for_corpus, get_domain


def answer(corpus: str, query: str, *, domain: str | None = None,
           strategy: str | None = None, history: list[dict] | None = None,
           top_k: int = 12, **kw) -> dict:
    dom = get_domain(domain) if domain else domain_for_corpus(corpus)
    return dom.answer(corpus, query, strategy=strategy, history=history,
                      top_k=top_k, **kw)
