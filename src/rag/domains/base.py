"""base.py -- the domain plugin interface.

A ``Domain`` is a thin object that adapts the generic core (retrieve / answer /
orchestrate) to one use case. The core provides defaults; a domain overrides
only what differs. Two live domains: ``building_codes`` and ``youtube``.

Design notes:
- Retrieval tuning is exposed as plain attributes (``retrieval_balance`` etc.)
  consumed by orchestrate/answer when calling ``query.rag_query``. Defaults
  reproduce the original building-codes behavior.
- ``ingest`` lets each domain own its own pipeline (building-codes walks a
  filesystem corpus via ``index``; youtube enumerates videos + fetches
  transcripts). It returns a process exit code (0 == ok).
- ``citation_label`` renders a retrieved result's provenance for prose/UI.
- ``default_strategy`` picks an answering strategy for a query
  ("single" | "agentic" | "mapreduce" | "longctx").
"""

from __future__ import annotations

from typing import Iterable


# Strategy names understood by core/orchestrate.py.
STRATEGY_SINGLE = "single"        # one-pass grounded synthesis (answer.py)
STRATEGY_AGENTIC = "agentic"      # hypothesize -> verify (reason.py)
STRATEGY_MAPREDUCE = "mapreduce"  # per-doc extract -> reduce (exhaustive queries)
STRATEGY_LONGCTX = "longctx"      # whole-corpus-in-prompt (fits the context window)


class Domain:
    """Base domain. Subclasses set ``name`` and override behavior as needed.

    The base defaults mirror a plain hybrid-RAG corpus on Azure with no
    building-code or youtube specifics, so a trivial new domain needs almost
    nothing.
    """

    name: str = "base"

    # --- chat backend ---
    chat_provider: str = "azure"      # "azure" | "claude"
    chat_model: str | None = None     # provider-specific model/deployment override

    # --- retrieval tuning (passed to query.rag_query) ---
    retrieval_balance: bool = False        # jurisdiction round-robin (building-codes)
    retrieval_expand: bool = True          # LLM query expansion
    retrieval_short_code_filter: bool = True  # section-number hard-filter

    # --- answering ---
    supported_strategies: tuple[str, ...] = (STRATEGY_SINGLE,)

    def rag_query_kwargs(self) -> dict:
        """Keyword args this domain wants passed to query.rag_query."""
        return {
            "balance": self.retrieval_balance,
            "expand": self.retrieval_expand,
            "short_code_filter": self.retrieval_short_code_filter,
        }

    def default_strategy(self, query: str) -> str:
        """Choose an answering strategy for ``query``. Override per domain."""
        return STRATEGY_SINGLE

    def ingest(self, corpus: str, argv: Iterable[str] | None = None) -> int:
        """Build / update the corpus index. Override per domain.

        ``argv`` carries domain-specific options (e.g. video URLs to add, or
        build flags). Returns an exit code (0 == ok)."""
        raise NotImplementedError("%s.ingest is not implemented" % self.name)

    def citation_label(self, result: dict) -> str:
        """Human-readable provenance for a retrieved result. Override per domain."""
        return result.get("source_file") or result.get("path") or "unknown"

    def answer(self, corpus: str, query: str, strategy: str | None = None,
               history: list[dict] | None = None, top_k: int = 12, **kw) -> dict:
        """Produce a grounded, cited answer dict. Override per domain.

        Returns at least {answer, citations, chunks, refused, refusal_reason,
        status, tokens}."""
        raise NotImplementedError("%s.answer is not implemented" % self.name)
