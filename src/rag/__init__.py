"""docrag -- a generalized document RAG: chat with any PDF/DOCX/TXT/MD corpus.

A standalone fork of the Microvellum knowledge-RAG pipeline, stripped of all
brand / code-tier / manifest machinery. The unit of partition is a *corpus*:
a named set of documents living under ``{docs_root}/{corpus}/`` and indexed
into ``{index_dir}/{corpus}.db``.

Pipeline:  extract -> chunk -> embed -> hybrid retrieve (vector + BM25 RRF)
           -> grounded synthesis with [N] citations + anti-hallucination gate.
"""

__version__ = "1.0.0"
