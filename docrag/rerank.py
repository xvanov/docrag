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
    """Reranker gate. Default 'auto': enable only when a CUDA GPU is present.

    The cross-encoder (bge-reranker-v2-m3, an XLM-RoBERTa-large) is minutes-slow
    per query on CPU -- impractical for interactive use -- so on CPU we default
    to off and let hybrid RRF stand. Force it on/off explicitly with
    DOCRAG_RERANK=1 / DOCRAG_RERANK=0 (honored even on CPU).
    """
    val = (settings.get("DOCRAG_RERANK", "auto") or "auto").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    if val in ("1", "true", "yes", "on"):
        return True
    try:  # auto
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 -- torch absent -> CPU -> off
        return False


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
        # fp16 only helps on CUDA; on CPU it is emulated and pathologically slow
        # (each rerank takes minutes). Use fp16 only when a GPU is present.
        use_fp16 = False
        try:
            import torch  # type: ignore
            use_fp16 = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 -- torch absent / probe failed -> CPU
            use_fp16 = False
        _MODEL = FlagReranker(_model_name(), use_fp16=use_fp16)
        sys.stderr.write("[rerank] loaded %s (fp16=%s)\n" % (_model_name(), use_fp16))
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
