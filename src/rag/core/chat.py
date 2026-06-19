"""chat.py -- pluggable chat-completion providers.

A thin seam so the synthesis/agentic layers don't hardcode a vendor. Two
providers today:

  AzureChat  -- the original docrag path (Azure OpenAI chat.completions),
                lifted verbatim incl. backoff + max_tokens/temperature
                capability probing. Default for the building-codes domain.
  ClaudeChat -- Anthropic Messages API, with tool_loop() for agentic /
                map-reduce flows and optional prompt caching of the system
                block (whole-channel-in-prompt long-context). Used by youtube.

Vendor SDKs are imported lazily INSIDE each provider's __init__ -- never at
module top -- so a building-codes install (no `anthropic`) can import this
module, and a youtube install (no `openai`) likewise. Instantiate a provider
at *module load* of an entry point (CLI / MCP server, main thread), not inside
a worker thread, to avoid the CPython import-lock stall that bit the MCP server.

Factory:
    get_chat_provider("azure"|"claude", **cfg) -> ChatProvider
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .. import settings

# Backoff (shared shape across providers).
MAX_ATTEMPTS = 6
INITIAL_DELAY_S = 1.0
MAX_DELAY_S = 60.0


@dataclass
class ChatResult:
    """Uniform return shape across providers."""
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    stop_reason: str | None = None
    raw: object = None


class ChatProvider(Protocol):
    def complete(self, *, system: str, messages: list[dict], max_tokens: int,
                 temperature: float | None = None,
                 cache_system: bool = False) -> ChatResult:
        """One completion. ``messages`` are user/assistant turns only (no system)."""
        ...

    def tool_loop(self, *, system: str, messages: list[dict], tools: list[dict],
                  dispatch: Callable[[str, dict], str], max_tokens: int,
                  max_rounds: int = 8, cache_system: bool = False) -> ChatResult:
        """Run an agentic tool-use loop until the model stops requesting tools."""


def _is_transient(msg: str) -> bool:
    lower = msg.lower()
    return ("429" in msg or "rate" in lower or "throttle" in lower
            or any(c in msg for c in (" 500", " 502", " 503", " 504"))
            or "server_error" in lower or "overloaded" in lower)


# ---------------------------------------------------------------------------
# Azure OpenAI (building-codes default) -- behavior preserved from answer.py
# ---------------------------------------------------------------------------

class AzureChat:
    def __init__(self, deployment: str | None = None) -> None:
        from openai import AzureOpenAI  # lazy: keep openai out of non-azure installs

        endpoint = settings.azure_endpoint()
        api_key = settings.azure_api_key()
        if not endpoint or not api_key:
            raise EnvironmentError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set."
            )
        # Bound every chat call: the SDK default 600s timeout (+ retries) means a
        # single stalled request can freeze a query / MCP tool call for 10-30 min.
        # Our own backoff handles transient retries, so cap hard here.
        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=settings.azure_api_version(),
            timeout=float(settings.get("DOCRAG_HTTP_TIMEOUT", 60) or 60),
            max_retries=2,
        )
        self._deployment = deployment or settings.chat_deployment_synthesis()

    def complete(self, *, system: str, messages: list[dict], max_tokens: int,
                 temperature: float | None = None,
                 cache_system: bool = False) -> ChatResult:
        # cache_system is a no-op for Azure (the service caches automatically).
        full = [{"role": "system", "content": system}] + list(messages)
        resp = self._chat_with_backoff(full, max_tokens, temperature)
        try:
            text = (resp.choices[0].message.content or "").strip()
        except (AttributeError, IndexError) as e:
            raise EnvironmentError("Azure chat returned unexpected shape: %s" % e) from e
        usage = getattr(resp, "usage", None)
        return ChatResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            raw=resp,
        )

    def tool_loop(self, *, system, messages, tools, dispatch, max_tokens,
                  max_rounds=8, cache_system=False) -> ChatResult:
        raise NotImplementedError("AzureChat.tool_loop is not implemented; "
                                  "the building-codes agentic path uses reason.py.")

    def _chat_with_backoff(self, messages: list[dict], max_tokens: int,
                           temperature: float | None):
        """chat.completions with backoff + token/temperature param probing."""
        use_max_tokens_legacy = False
        send_temperature = temperature is not None
        delay = INITIAL_DELAY_S
        for _ in range(MAX_ATTEMPTS):
            kwargs: dict = {"model": self._deployment, "messages": messages}
            if send_temperature:
                kwargs["temperature"] = temperature
            if use_max_tokens_legacy:
                kwargs["max_tokens"] = max_tokens
            else:
                kwargs["max_completion_tokens"] = max_tokens
            try:
                t0 = time.monotonic()
                resp = self._client.chat.completions.create(**kwargs)
                sys.stderr.write(
                    "[chat] azure ok deployment=%s in %dms\n"
                    % (self._deployment, int((time.monotonic() - t0) * 1000))
                )
                return resp
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                lower = msg.lower()
                if _is_transient(msg):
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


# ---------------------------------------------------------------------------
# Anthropic / Claude (youtube default)
# ---------------------------------------------------------------------------

class ClaudeChat:
    def __init__(self, model: str | None = None) -> None:
        import anthropic  # lazy: keep anthropic out of building-codes installs

        api_key = settings.get("ANTHROPIC_API_KEY", "") or ""
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY must be set for ClaudeChat.")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=float(settings.get("DOCRAG_HTTP_TIMEOUT", 120) or 120),
            max_retries=2,
        )
        self._model = model or settings.get("RAG_CLAUDE_MODEL", "claude-opus-4-8") \
            or "claude-opus-4-8"

    def _system_param(self, system: str, cache_system: bool):
        """System as a plain string, or a cache-controlled block list.

        Caching the (large, stable) system block lets repeat questions over the
        same corpus hit the prompt cache (~90% input savings). Uses the default
        5-min ephemeral cache; for a 1h TTL set RAG_CACHE_TTL=1h (requires the
        extended-cache-ttl beta header, added here when requested).
        """
        if not cache_system:
            return system, {}
        block = {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        extra_headers = {}
        ttl = settings.get("RAG_CACHE_TTL", "") or ""
        if ttl == "1h":
            block["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
            extra_headers = {"anthropic-beta": "extended-cache-ttl-2025-04-11"}
        return [block], extra_headers

    def _create_with_backoff(self, **kwargs):
        delay = INITIAL_DELAY_S
        last = None
        for _ in range(MAX_ATTEMPTS):
            try:
                return self._client.messages.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                last = e
                if _is_transient(str(e)) or isinstance(
                        e, getattr(self._anthropic, "APIStatusError", Exception)):
                    if _is_transient(str(e)):
                        time.sleep(delay)
                        delay = min(delay * 2, MAX_DELAY_S)
                        continue
                raise EnvironmentError("Claude chat failed: %s" % e) from e
        raise EnvironmentError("Claude chat failed after %d retries: %s"
                               % (MAX_ATTEMPTS, last))

    @staticmethod
    def _text_of(resp) -> str:
        return "".join(b.text for b in resp.content
                       if getattr(b, "type", None) == "text").strip()

    def complete(self, *, system: str, messages: list[dict], max_tokens: int,
                 temperature: float | None = None,
                 cache_system: bool = False) -> ChatResult:
        system_param, extra_headers = self._system_param(system, cache_system)
        kwargs: dict = {"model": self._model, "max_tokens": max_tokens,
                        "system": system_param, "messages": list(messages)}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        resp = self._create_with_backoff(**kwargs)
        usage = getattr(resp, "usage", None)
        return ChatResult(
            text=self._text_of(resp),
            prompt_tokens=getattr(usage, "input_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "output_tokens", None) if usage else None,
            stop_reason=getattr(resp, "stop_reason", None),
            raw=resp,
        )

    def tool_loop(self, *, system: str, messages: list[dict], tools: list[dict],
                  dispatch: Callable[[str, dict], str], max_tokens: int,
                  max_rounds: int = 8, cache_system: bool = False) -> ChatResult:
        """Agentic loop: let the model call ``tools`` (dispatched by ``dispatch``)
        until it stops requesting them or ``max_rounds`` is hit. ``messages`` is
        copied; the running transcript is local."""
        system_param, extra_headers = self._system_param(system, cache_system)
        convo = list(messages)
        in_tok = out_tok = 0
        last = None
        for _ in range(max_rounds):
            kwargs: dict = {"model": self._model, "max_tokens": max_tokens,
                            "system": system_param, "messages": convo, "tools": tools}
            if extra_headers:
                kwargs["extra_headers"] = extra_headers
            resp = self._create_with_backoff(**kwargs)
            last = resp
            usage = getattr(resp, "usage", None)
            if usage:
                in_tok += getattr(usage, "input_tokens", 0) or 0
                out_tok += getattr(usage, "output_tokens", 0) or 0
            if getattr(resp, "stop_reason", None) != "tool_use":
                return ChatResult(text=self._text_of(resp), prompt_tokens=in_tok,
                                  completion_tokens=out_tok,
                                  stop_reason=resp.stop_reason, raw=resp)
            convo.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    try:
                        result = dispatch(block.name, dict(block.input or {}))
                    except Exception as e:  # noqa: BLE001
                        result = "tool error: %s" % e
                    tool_results.append({"type": "tool_result",
                                         "tool_use_id": block.id,
                                         "content": result})
            convo.append({"role": "user", "content": tool_results})
        # Ran out of rounds: return whatever text the last response carried.
        return ChatResult(text=self._text_of(last) if last else "",
                          prompt_tokens=in_tok, completion_tokens=out_tok,
                          stop_reason="max_rounds",
                          raw=last)


def get_chat_provider(name: str, **cfg) -> ChatProvider:
    name = (name or "azure").strip().lower()
    if name == "azure":
        return AzureChat(deployment=cfg.get("deployment"))
    if name == "claude":
        return ClaudeChat(model=cfg.get("model"))
    raise ValueError("unknown chat provider: %r" % name)
