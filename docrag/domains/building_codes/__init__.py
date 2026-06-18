"""building_codes domain -- the original docrag use case.

A thin adapter over the existing building-codes pipeline (index / answer /
reason / facets). No logic is moved here yet; this wires the existing modules to
the Domain interface so the new domain-aware CLI/orchestrator can drive it the
same way it drives youtube. Heavy/vendor imports (openai via answer/index) are
deferred into methods so importing this module stays cheap and dependency-
isolated.
"""

from __future__ import annotations

from typing import Iterable

from ..base import Domain, STRATEGY_AGENTIC, STRATEGY_SINGLE


class BuildingCodesDomain(Domain):
    name = "building_codes"

    chat_provider = "azure"
    # Building-codes retrieval is jurisdiction-balanced, expands lay->code
    # vocabulary, and hard-filters on section numbers -- the original behavior.
    retrieval_balance = True
    retrieval_expand = True
    retrieval_short_code_filter = True

    supported_strategies = (STRATEGY_SINGLE, STRATEGY_AGENTIC)

    def default_strategy(self, query: str) -> str:
        # Building-codes answers go through the agentic hypothesize->verify path
        # by default (correctness over latency), matching the existing default.
        return STRATEGY_AGENTIC

    def ingest(self, corpus: str, argv: Iterable[str] | None = None) -> int:
        from ... import index  # lazy: pulls the Azure embedding stack
        return index.main(["build", "--corpus", corpus, *list(argv or [])])

    def citation_label(self, result: dict) -> str:
        from ... import answer  # lazy: keeps openai out of non-azure imports
        return answer._chunk_label(result)

    def answer(self, corpus: str, query: str, strategy: str | None = None,
               history: list[dict] | None = None, top_k: int = 12,
               location: str | None = None, **kw) -> dict:
        # Preserve the existing building-codes behavior exactly: balanced
        # retrieval, agentic by default, single-pass when asked.
        strat = strategy or self.default_strategy(query)
        if strat == STRATEGY_AGENTIC:
            from ... import reason
            return reason.answer(corpus, query, history=history, top_k=top_k,
                                 balance=True, location=location)
        from ... import answer as answer_mod
        return answer_mod.answer(corpus, query, history=history, top_k=top_k,
                                 balance=True, location=location)


DOMAIN = BuildingCodesDomain()
