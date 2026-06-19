"""answer.py -- youtube grounded synthesis (SINGLE + LONGCTX strategies).

SINGLE  : hybrid-retrieve top_k transcript excerpts, answer with Claude, cite
          [N] -> video + timestamp deep-link. For focused questions.
LONGCTX : stuff every video's full transcript into one cached prompt and answer.
          For when a corpus fits the context window and the question benefits
          from seeing everything (the system prompt block is prompt-cached so
          repeat questions over the same corpus are cheap).
"""

from __future__ import annotations

import json
import re

from ... import settings
from ...core.chat import get_chat_provider
from ...db import open_db
from ...query import rag_query
from . import prompts

_CITE_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
ANSWER_MAX_TOKENS = 1200
LONGCTX_CHAR_BUDGET = 2_400_000   # ~600k tokens of transcript; guard the prompt


def resolve_chat() -> tuple[str, str | None]:
    """Pick the chat backend for the youtube domain.

    Explicit override: RAG_YOUTUBE_CHAT_PROVIDER (azure|claude) +
    RAG_YOUTUBE_CHAT_MODEL. Otherwise: Claude if ANTHROPIC_API_KEY is set, else
    fall back to Azure if its credentials are present (so a youtube corpus can
    run entirely on Azure infra). Returns (provider_name, model_or_None).
    """
    name = (settings.get("RAG_YOUTUBE_CHAT_PROVIDER", "") or "").strip().lower()
    if not name:
        if settings.get("ANTHROPIC_API_KEY", ""):
            name = "claude"
        elif settings.azure_endpoint() and settings.azure_api_key():
            name = "azure"
        else:
            name = "claude"
    if name == "azure":
        # None -> AzureChat uses chat_deployment_synthesis().
        model = settings.get("RAG_YOUTUBE_CHAT_MODEL", "") or None
    else:
        model = (settings.get("RAG_YOUTUBE_CHAT_MODEL", "")
                 or settings.get("RAG_CLAUDE_MODEL", "claude-opus-4-8")
                 or "claude-opus-4-8")
    return name, model


def excerpt_label(c: dict) -> str:
    """'[channel] title (mm:ss)' from a retrieved transcript result."""
    m = c.get("metadata") or {}
    chan = m.get("channel") or ""
    title = m.get("title") or c.get("source_file") or "video"
    hms = m.get("hms")
    head = ("[%s] " % chan) if chan else ""
    return "%s%s%s" % (head, title, (" (%s)" % hms) if hms else "")


def _parse_citations(text: str, n: int) -> list[int]:
    found: set[int] = set()
    for mt in _CITE_RE.finditer(text or ""):
        for piece in mt.group(1).split(","):
            piece = piece.strip()
            if piece.isdigit() and 1 <= int(piece) <= n:
                found.add(int(piece))
    return sorted(found)


def _authorities(chunks: list[dict], citations: list[int]) -> list[dict]:
    out = []
    for nn in citations:
        if 1 <= nn <= len(chunks):
            m = chunks[nn - 1].get("metadata") or {}
            out.append({"n": nn, "title": m.get("title"), "channel": m.get("channel"),
                        "timestamp": m.get("hms"), "url": m.get("deep_link") or m.get("url")})
    return out


def _refused(chunks, reason, status):
    return {"answer": None, "citations": [], "chunks": chunks, "refused": True,
            "refusal_reason": reason, "status": status,
            "tokens": {"prompt": None, "completion": None}}


def single_answer(corpus: str, query: str, model: str | None = None,
                  history: list[dict] | None = None, top_k: int = 12) -> dict:
    retrieval = rag_query(corpus, query, top_k=top_k, balance=False,
                          expand=False, short_code_filter=False)
    chunks = retrieval.get("results") or []
    if retrieval.get("status") == "no_results" or not chunks:
        return _refused(chunks, "no_supporting_transcript", "no_results")
    if retrieval.get("status") == "low_confidence":
        return _refused(chunks, "low_confidence", "refused")

    block = prompts.build_chunks_block(chunks, excerpt_label)
    user = prompts.ANSWER_USER_TEMPLATE.format(query=query, chunks_block=block)
    messages: list[dict] = []
    for turn in history or []:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user})

    name, mdl = resolve_chat()
    provider = get_chat_provider(name, model=model or mdl)
    result = provider.complete(system=prompts.ANSWER_SYSTEM, messages=messages,
                               max_tokens=ANSWER_MAX_TOKENS)
    text = result.text
    tokens = {"prompt": result.prompt_tokens, "completion": result.completion_tokens}

    if prompts.REFUSAL_SENTENCE.lower() in (text or "").lower():
        out = _refused(chunks, "model_refused", "refused"); out["tokens"] = tokens
        return out
    citations = _parse_citations(text, len(chunks))
    if not citations:
        out = _refused(chunks, "model_did_not_cite", "refused"); out["tokens"] = tokens
        return out
    return {"answer": text, "citations": citations, "chunks": chunks,
            "authorities": _authorities(chunks, citations), "refused": False,
            "refusal_reason": None, "status": "ok", "tokens": tokens}


def _load_videos(conn, corpus: str) -> list[dict]:
    """Per-video {title, url, transcript_text} from the files rows' metadata."""
    rows = conn.execute(
        "SELECT path, metadata FROM files ORDER BY path").fetchall()
    out = []
    for path, meta_json in rows:
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except (ValueError, TypeError):
            meta = {}
        if not meta.get("transcript_text"):
            continue
        out.append({"path": path, "title": meta.get("title") or path,
                    "url": meta.get("url"), "channel": meta.get("channel") or "",
                    "transcript_text": meta["transcript_text"]})
    return out


def longctx_answer(corpus: str, query: str, model: str | None = None) -> dict:
    conn = open_db(corpus)
    try:
        videos = _load_videos(conn, corpus)
    finally:
        conn.close()
    if not videos:
        return _refused([], "no_transcripts", "no_results")

    parts, total = [], 0
    used = 0
    for i, v in enumerate(videos, start=1):
        seg = "[VIDEO %d] %s  (%s)\n%s" % (i, v["title"], v.get("url") or "",
                                           v["transcript_text"])
        if total + len(seg) > LONGCTX_CHAR_BUDGET:
            break
        parts.append(seg); total += len(seg); used += 1
    corpus_block = "\n\n".join(parts)
    system = (prompts.ANSWER_SYSTEM
              + "\n\nThe excerpts below are FULL transcripts, each headed "
                "[VIDEO n] with its title and url. Cite [VIDEO n] and an "
                "approximate timestamp/quote for each claim.\n\n"
              + corpus_block)
    name, mdl = resolve_chat()
    provider = get_chat_provider(name, model=model or mdl)
    # cache_system caches the (large, stable) corpus prefix on Claude; it's a
    # harmless no-op on Azure.
    result = provider.complete(system=system,
                               messages=[{"role": "user", "content": query}],
                               max_tokens=ANSWER_MAX_TOKENS, cache_system=True)
    return {"answer": result.text, "citations": [], "chunks": [],
            "refused": False, "refusal_reason": None, "status": "ok",
            "videos_used": used, "videos_total": len(videos),
            "tokens": {"prompt": result.prompt_tokens,
                       "completion": result.completion_tokens}}
