"""youtube domain -- transcript Q&A over videos / channels.

Ingest: yt-dlp enumerate + youtube-transcript-api fetch (incremental).
Answer: Claude, with three strategies -- SINGLE (focused retrieval), MAPREDUCE
(exhaustive "list ALL X" across every video), LONGCTX (whole corpus in a cached
prompt). Retrieval runs plain hybrid (no jurisdiction balancing, no query
expansion, no section-number filter).
"""

from __future__ import annotations

import re
from typing import Iterable

from ..base import (Domain, STRATEGY_LONGCTX, STRATEGY_MAPREDUCE, STRATEGY_SINGLE)

# Heuristics for "exhaustive" questions that need map-reduce, not top-k.
_EXHAUSTIVE_RE = re.compile(
    r"\b(all|every|each|list|enumerate|complete list|how many|count|"
    r"everything|throughout|across (all|every)|over the years)\b", re.I)


class YoutubeDomain(Domain):
    name = "youtube"

    chat_provider = "claude"
    # Plain hybrid retrieval -- none of the building-code retrieval machinery.
    retrieval_balance = False
    retrieval_expand = False
    retrieval_short_code_filter = False

    supported_strategies = (STRATEGY_SINGLE, STRATEGY_MAPREDUCE, STRATEGY_LONGCTX)

    def default_strategy(self, query: str) -> str:
        if _EXHAUSTIVE_RE.search(query or ""):
            return STRATEGY_MAPREDUCE
        return STRATEGY_SINGLE

    def ingest(self, corpus: str, argv: Iterable[str] | None = None) -> int:
        # argv is the list of target URLs/ids (+ optional flags handled by CLI).
        from .ingest import ingest as _ingest
        targets = [a for a in (argv or []) if not a.startswith("-")]
        if not targets:
            import sys
            sys.stderr.write("ERROR: youtube ingest needs at least one video/"
                             "channel/playlist URL or id.\n")
            return 2
        return _ingest(corpus, targets)

    def citation_label(self, result: dict) -> str:
        from .answer import excerpt_label
        return excerpt_label(result)

    def answer(self, corpus: str, query: str, strategy: str | None = None,
               history: list[dict] | None = None, top_k: int = 12,
               target: str | None = None, **kw) -> dict:
        strat = strategy or self.default_strategy(query)
        model = self.chat_model
        if strat == STRATEGY_MAPREDUCE:
            from .mapreduce import mapreduce_answer
            return mapreduce_answer(corpus, query, target=target, model=model)
        if strat == STRATEGY_LONGCTX:
            from .answer import longctx_answer
            return longctx_answer(corpus, query, model=model)
        from .answer import single_answer
        return single_answer(corpus, query, model=model, history=history, top_k=top_k)


DOMAIN = YoutubeDomain()
