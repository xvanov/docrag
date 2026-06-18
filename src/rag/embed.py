"""Azure OpenAI embedding client for docrag.

Auth + deployment + dims come from ``settings`` (which reads .env). Never
logs the API key. Progress to stderr only.

Public API:
  embed_batch(texts) -> list[list[float]]   # batches of 256, exp backoff on 429
  embed_one(text)    -> list[float]
  EmbedError                                  # raised on terminal failure
"""

from __future__ import annotations

import sys
import time
from typing import Sequence

from . import settings


BATCH_SIZE = 256
# Azure caps a single embeddings request at 300000 tokens. Batch by a char
# budget as well as count so dense docs (e.g. code-book pages) don't trip it.
# Conservative: dense numeric/code text can run ~2.5 chars/token, so 500k
# chars stays well under 300k tokens. The adaptive split in
# ``_embed_batch_adaptive`` is the hard backstop if an estimate still overshoots.
MAX_BATCH_CHARS = 500000
MAX_ATTEMPTS = 6
INITIAL_DELAY_S = 1.0
MAX_DELAY_S = 60.0


class EmbedError(RuntimeError):
    """Terminal failure from the Azure OpenAI embeddings endpoint."""


def _client():
    """Construct an AzureOpenAI client. Lazy import so missing deps fail loud."""
    from openai import AzureOpenAI  # type: ignore

    endpoint = settings.azure_endpoint()
    api_key = settings.azure_api_key()
    if not endpoint or not api_key:
        raise EnvironmentError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set "
            "(see .env.example)."
        )
    # Bound every call: the SDK default is a 600s timeout with retries, so one
    # stalled request would freeze a query (and an MCP tool call) for many
    # minutes. Cap it. Override via DOCRAG_HTTP_TIMEOUT.
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=settings.azure_api_version(),
        timeout=float(settings.get("DOCRAG_HTTP_TIMEOUT", 60) or 60),
        max_retries=2,
    )


def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Embed a list of texts. Returns float vectors in input order."""
    if not texts:
        return []
    client = _client()
    deployment = settings.embedding_deployment()
    out: list[list[float]] = []
    for batch in _sub_batches(texts):
        out.extend(_embed_batch_adaptive(client, deployment, batch))
    return out


def _embed_batch_adaptive(client, deployment: str, batch: list[str]) -> list[list[float]]:
    """Embed one batch; on an over-size 400, split in half and retry.

    Guarantees termination: chunk inputs are length-capped upstream, so a
    single-element batch always fits under the per-request token limit.
    """
    try:
        resp = _embed_with_backoff(client, deployment, batch)
        return [d.embedding for d in resp.data]
    except EmbedError as e:
        lower = str(e).lower()
        if len(batch) > 1 and ("maximum request size" in lower
                               or "max" in lower and "token" in lower):
            mid = len(batch) // 2
            sys.stderr.write(
                "[embed] request too large; splitting %d -> %d + %d\n"
                % (len(batch), mid, len(batch) - mid)
            )
            return (_embed_batch_adaptive(client, deployment, batch[:mid])
                    + _embed_batch_adaptive(client, deployment, batch[mid:]))
        raise


def _sub_batches(texts: Sequence[str]) -> list[list[str]]:
    """Group texts into request batches bounded by BOTH count and total chars.

    A single text is never split (chunk inputs are pre-capped upstream); it
    just gets its own batch if it alone exceeds the char budget.
    """
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_chars = 0
    for t in texts:
        n = len(t)
        if cur and (len(cur) >= BATCH_SIZE or cur_chars + n > MAX_BATCH_CHARS):
            batches.append(cur)
            cur = []
            cur_chars = 0
        cur.append(t)
        cur_chars += n
    if cur:
        batches.append(cur)
    return batches


def embed_one(text: str) -> list[float]:
    return embed_batch([text])[0]


def _embed_with_backoff(client, deployment: str, batch: list[str]):
    delay = INITIAL_DELAY_S
    for attempt in range(MAX_ATTEMPTS):
        try:
            t0 = time.monotonic()
            resp = client.embeddings.create(model=deployment, input=batch)
            dt = int((time.monotonic() - t0) * 1000)
            sys.stderr.write("[embed] %d vectors in %dms\n" % (len(batch), dt))
            return resp
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            lower = msg.lower()
            if "429" in msg or "rate" in lower or "throttle" in lower:
                sys.stderr.write(
                    "[embed] rate-limited (attempt %d); sleeping %.1fs\n"
                    % (attempt + 1, delay)
                )
                time.sleep(delay)
                delay = min(delay * 2, MAX_DELAY_S)
                continue
            raise EmbedError("Azure OpenAI embed failed: %s" % msg) from e
    raise EmbedError(
        "Azure OpenAI embed failed after %d retries (rate-limited)."
        % MAX_ATTEMPTS
    )
