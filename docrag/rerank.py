"""rerank.py -- optional cross-encoder reranking for retrieval.

A cross-encoder reranker is the single highest-ROI retrieval upgrade (2025-2026
benchmarks: Recall@5 ~0.70 -> ~0.82). It is **optional**: if the model / its
dependency isn't installed, ``rerank`` returns ``None`` and the caller keeps the
RRF order unchanged -- so the system degrades gracefully and the stdlib server
still runs with zero ML dependencies.

Default model: BAAI/bge-reranker-v2-m3 (Apache-2.0, strong, multilingual),
loaded via FlagEmbedding. Override with DOCRAG_RERANK_MODEL; disable entirely
with DOCRAG_RERANK=0.

Public API:
    rerank(query, docs) -> list[float] | None   # one score per doc, or None
    available() -> bool
"""

from __future__ import annotations

import os
import sys

from . import settings

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# Sentinel: None = not yet loaded; False = unavailable (don't retry); else model.
_MODEL = None
_TRIED = False


def _enabled() -> bool:
    val = (settings.get("DOCRAG_RERANK", "1") or "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def _model_name() -> str:
    return settings.get("DOCRAG_RERANK_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL


def _load():
    """Lazily load the reranker. Returns the model or False if unavailable."""
    global _MODEL, _TRIED
    if _TRIED:
        return _MODEL
    _TRIED = True
    if not _enabled():
        _MODEL = False
        return _MODEL
    try:
        from FlagEmbedding import FlagReranker  # type: ignore
        _MODEL = FlagReranker(_model_name(), use_fp16=True)
        sys.stderr.write("[rerank] loaded %s\n" % _model_name())
    except Exception as exc:  # noqa: BLE001 -- dep absent / model download failed
        sys.stderr.write("[rerank] unavailable (%s); RRF order kept\n" % exc)
        _MODEL = False
    return _MODEL


def available() -> bool:
    return bool(_load())


def rerank(query: str, docs: list[str]) -> list[float] | None:
    """Score each doc against the query with the cross-encoder.

    Returns a list of floats (higher = more relevant), or ``None`` when the
    reranker is unavailable/disabled so the caller can keep the prior order.
    """
    if not query or not docs:
        return None
    model = _load()
    if not model:
        return None
    try:
        pairs = [[query, d or ""] for d in docs]
        scores = model.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        return [float(s) for s in scores]
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("[rerank] scoring failed (%s); RRF order kept\n" % exc)
        return None
