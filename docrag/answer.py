"""answer.py -- Grounded LLM synthesis over retrieved chunks.

Pipeline:
  1. rag_query() to retrieve top_k chunks.
  2. If retrieval is empty / low-confidence -> refuse without an LLM call.
  3. Build a numbered chunks block (truncated per chunk).
  4. System prompt forbids out-of-context claims and mandates [N] citations.
  5. Azure chat call with exponential backoff on 429.
  6. Post-hoc: detect the canned refusal sentence; require >=1 in-range [N]
     citation, else flip to refused (anti-hallucination).

Public API:
    answer(corpus, query, history=None, top_k=12, filters=None) -> dict
"""

from __future__ import annotations

import re
import sys
import time

from . import settings
from .query import rag_query


CHUNK_CHAR_BUDGET = 1500
CHAT_TEMPERATURE = 0.2
CHAT_MAX_TOKENS = 600
MAX_ATTEMPTS = 6
INITIAL_DELAY_S = 1.0
MAX_DELAY_S = 60.0

MODEL_REFUSAL_SENTENCE = "I don't have documentation for that."

SYSTEM_PROMPT = (
    "You are a research assistant grounded in excerpts from a document "
    "collection.\n\n"
    "RULES:\n"
    "1. Answer ONLY using the provided chunks below. Do not introduce facts "
    "from outside knowledge.\n"
    "2. Every factual claim must be followed by a citation in square brackets "
    "like [1] or [2,3].\n"
    "3. If the chunks do not contain enough information to answer, respond "
    "exactly: \"" + MODEL_REFUSAL_SENTENCE + "\"\n"
    "4. Do not fabricate values, names, codes, or behaviors. If a chunk "
    "doesn't state it, do not state it.\n"
    "5. Keep answers concise -- no more than 5 sentences unless the question "
    "clearly needs more.\n"
    "6. Quote short snippets when they are diagnostic, but always cite the "
    "source chunk index with bare [N].\n"
)

USER_PROMPT_TEMPLATE = (
    "Question / topic: {query}\n\n"
    "Chunks (numbered):\n{chunks_block}\n\n"
    "Instructions:\n"
    "- If the chunks describe the topic, summarize what they say and cite "
    "chunk numbers in brackets like [1] or [2,3].\n"
    "- A short topic is still a valid question -- treat it as \"what do these "
    "chunks say about <topic>?\".\n"
    "- Only respond with the exact refusal sentence if none of the chunks are "
    "about the topic at all.\n"
)

_CITE_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_LAST_PROMPT: dict | None = None


def _normalize_quotes(s: str) -> str:
    return (s or "").replace("’", "'").replace("‘", "'")


def _truncate_chunk_text(text: str, budget: int = CHUNK_CHAR_BUDGET) -> str:
    text = " ".join((text or "").split())
    if len(text) <= budget:
        return text
    return text[: budget - 3].rstrip() + "..."


def _build_chunks_block(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        snippet = _truncate_chunk_text(c.get("text") or "")
        page = c.get("page")
        page_str = str(page) if page is not None else "-"
        source_file = c.get("source_file") or "unknown"
        lines.append('[%d] %s (p.%s)\n"%s"' % (i, source_file, page_str, snippet))
    return "\n\n".join(lines)


def _parse_citations(answer_text: str, n_chunks: int) -> list[int]:
    if not answer_text or n_chunks <= 0:
        return []
    found: set[int] = set()
    for match in _CITE_RE.finditer(answer_text):
        for piece in match.group(1).split(","):
            piece = piece.strip()
            if piece.isdigit():
                n = int(piece)
                if 1 <= n <= n_chunks:
                    found.add(n)
    return sorted(found)


def _chat_client():
    from openai import AzureOpenAI  # type: ignore

    endpoint = settings.azure_endpoint()
    api_key = settings.azure_api_key()
    if not endpoint or not api_key:
        raise EnvironmentError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set."
        )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=settings.azure_api_version(),
    )


def _chat_with_backoff(client, deployment: str, messages: list[dict]):
    """chat.completions with backoff + token/temperature param probing."""
    use_max_tokens_legacy = False
    send_temperature = True
    delay = INITIAL_DELAY_S
    for attempt in range(MAX_ATTEMPTS):
        kwargs: dict = {"model": deployment, "messages": messages}
        if send_temperature:
            kwargs["temperature"] = CHAT_TEMPERATURE
        if use_max_tokens_legacy:
            kwargs["max_tokens"] = CHAT_MAX_TOKENS
        else:
            kwargs["max_completion_tokens"] = CHAT_MAX_TOKENS
        try:
            t0 = time.monotonic()
            resp = client.chat.completions.create(**kwargs)
            sys.stderr.write(
                "[answer] chat ok deployment=%s in %dms\n"
                % (deployment, int((time.monotonic() - t0) * 1000))
            )
            return resp
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            lower = msg.lower()
            if "429" in msg or "rate" in lower or "throttle" in lower:
                time.sleep(delay)
                delay = min(delay * 2, MAX_DELAY_S)
                continue
            if ("max_completion_tokens" in lower and "not supported" in lower
                    and not use_max_tokens_legacy):
                use_max_tokens_legacy = True
                continue
            if ("temperature" in lower
                    and ("not supported" in lower or "unsupported" in lower)
                    and send_temperature):
                send_temperature = False
                continue
            raise EnvironmentError("Azure OpenAI chat failed: %s" % msg) from e
    raise EnvironmentError(
        "Azure OpenAI chat failed after %d retries (rate-limited)." % MAX_ATTEMPTS
    )


def _refused(chunks, reason, status):
    return {"answer": None, "citations": [], "chunks": chunks,
            "refused": True, "refusal_reason": reason, "status": status,
            "tokens": {"prompt": None, "completion": None}}


def answer(corpus: str, query: str, history: list[dict] | None = None,
           top_k: int = 12, filters: dict | None = None) -> dict:
    """Retrieve + synthesize a grounded, cited answer.

    Returns ``{answer, citations, chunks, refused, refusal_reason, status,
    tokens}``.
    """
    retrieval = rag_query(corpus, query, top_k=top_k, filters=filters)
    chunks = retrieval.get("results") or []
    r_status = retrieval.get("status")

    if r_status == "no_results" or not chunks:
        return _refused(chunks, "no_supporting_documentation", "no_results")
    if r_status == "low_confidence":
        return _refused(chunks, "low_confidence", "refused")

    chunks_block = _build_chunks_block(chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(query=query, chunks_block=chunks_block)

    global _LAST_PROMPT
    _LAST_PROMPT = {"system": SYSTEM_PROMPT, "user": user_prompt}

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})

    client = _chat_client()
    deployment = settings.chat_deployment_fast()
    resp = _chat_with_backoff(client, deployment, messages)

    try:
        answer_text = (resp.choices[0].message.content or "").strip()
    except (AttributeError, IndexError) as e:
        raise EnvironmentError("Azure chat returned unexpected shape: %s" % e) from e

    usage = getattr(resp, "usage", None)
    tokens = {
        "prompt": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion": getattr(usage, "completion_tokens", None) if usage else None,
    }

    if MODEL_REFUSAL_SENTENCE in _normalize_quotes(answer_text):
        out = _refused(chunks, "model_refused_insufficient_context", "refused")
        out["tokens"] = tokens
        return out

    citations = _parse_citations(answer_text, len(chunks))
    if not citations:
        out = _refused(chunks, "model_did_not_cite", "refused")
        out["tokens"] = tokens
        return out

    return {"answer": answer_text, "citations": citations, "chunks": chunks,
            "refused": False, "refusal_reason": None, "status": "ok",
            "tokens": tokens}
