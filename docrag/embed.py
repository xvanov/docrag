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
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=settings.azure_api_version(),
    )


def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Embed a list of texts. Returns float vectors in input order."""
    if not texts:
        return []
    client = _client()
    deployment = settings.embedding_deployment()
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = list(texts[i:i + BATCH_SIZE])
        resp = _embed_with_backoff(client, deployment, batch)
        out.extend([d.embedding for d in resp.data])
    return out


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
