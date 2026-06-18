"""mapreduce.py -- exhaustive queries over a whole channel.

For "list ALL X across every video" questions, top-k retrieval is structurally
wrong (it returns the k most similar chunks, not every instance). Instead:

  MAP    : visit each video's full transcript once, extract every instance of
           the target with Claude -> JSON. Cache the per-video result in
           extraction_cache, keyed on (path, kind, content_hash, model) so a
           re-run is cheap and a changed transcript / model invalidates it.
  REDUCE : feed all cached items (with provenance) to Claude to dedupe, organize,
           and answer with per-item source links.
"""

from __future__ import annotations

import json
import re

from ... import settings
from ...core.chat import get_chat_provider
from ...db import get_extraction, open_db, put_extraction
from . import prompts
from .answer import _load_videos, _refused

MAP_MAX_TOKENS = 2400
REDUCE_MAX_TOKENS = 2000
MAP_TRANSCRIPT_CHAR_CAP = 600_000   # ~150k tokens; guard a single huge video


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:60] or "x"


def _parse_items(text: str) -> list[dict]:
    """Parse the model's {"items":[...]} JSON, tolerating stray prose/fences."""
    if not text:
        return []
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    try:
        obj = json.loads(raw)
    except ValueError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            return []
    items = obj.get("items") if isinstance(obj, dict) else obj
    return [it for it in (items or []) if isinstance(it, dict)]


def mapreduce_answer(corpus: str, query: str, target: str | None = None,
                     model: str | None = None) -> dict:
    """Exhaustive extract-then-synthesize over every video in ``corpus``.

    ``target`` is what to extract per video (e.g. "predictions about geopolitics");
    defaults to the query itself."""
    target = (target or query).strip()
    from .answer import resolve_chat
    provider_name, resolved_model = resolve_chat()
    if provider_name == "azure":
        model = model or resolved_model or settings.chat_deployment_synthesis()
    else:
        model = model or resolved_model or "claude-opus-4-8"
    # ``model`` is both the provider model/deployment AND part of the
    # extraction_cache key (so switching backend/model is a clean cache miss).
    extraction_kind = "mapreduce:" + _slug(target)

    conn = open_db(corpus)
    try:
        videos = _load_videos(conn, corpus)
        if not videos:
            return _refused([], "no_transcripts", "no_results")

        provider = get_chat_provider(provider_name, model=model)
        map_system = prompts.MAP_SYSTEM_TEMPLATE.format(target=target)

        all_items: list[dict] = []
        cache_hits = mapped = failed = 0
        for v in videos:
            path = v["path"]
            chash = (conn.execute("SELECT sha256 FROM files WHERE path=?", (path,))
                     .fetchone() or [None])[0]
            cached = get_extraction(conn, path, extraction_kind, chash or "", model)
            if cached is not None:
                items = cached
                cache_hits += 1
            else:
                transcript = v["transcript_text"][:MAP_TRANSCRIPT_CHAR_CAP]
                user = prompts.MAP_USER_TEMPLATE.format(
                    title=v["title"], channel=v.get("channel") or "",
                    transcript=transcript)
                # One video must never sink the whole exhaustive run: a transient
                # error or an Azure content-filter rejection (Zeihan covers wars,
                # which trip the violence filter) is logged and skipped, not
                # cached -- so a later run can retry it.
                try:
                    res = provider.complete(
                        system=map_system,
                        messages=[{"role": "user", "content": user}],
                        max_tokens=MAP_MAX_TOKENS)
                    items = _parse_items(res.text)
                    put_extraction(conn, path, corpus, chash or "",
                                   extraction_kind, model, items)
                    conn.commit()
                    mapped += 1
                except Exception as e:  # noqa: BLE001
                    import sys
                    reason = "content_filter" if "content_filter" in str(e) else \
                        str(e).splitlines()[0][:120]
                    sys.stderr.write("[mapreduce] skip %s (%s)\n" % (path, reason))
                    failed += 1
                    continue
            for it in items:
                it = dict(it)
                it["_video"] = {"title": v["title"], "url": v.get("url"),
                                "channel": v.get("channel")}
                all_items.append(it)
    finally:
        conn.close()

    if not all_items:
        return {"answer": "No instances of '%s' were found across %d video(s)."
                % (target, len(videos)), "citations": [], "chunks": [],
                "refused": False, "refusal_reason": None, "status": "ok",
                "stats": {"videos": len(videos), "items": 0,
                          "cache_hits": cache_hits, "mapped": mapped,
                          "failed": failed}}

    stats = {"videos": len(videos), "items": len(all_items),
             "cache_hits": cache_hits, "mapped": mapped, "failed": failed}
    reduce_system = prompts.REDUCE_SYSTEM_TEMPLATE.format(query=query)
    reduce_user = ("Items (JSON):\n%s\n\nWrite the final answer now."
                   % json.dumps(all_items, ensure_ascii=False))
    provider = get_chat_provider(provider_name, model=model)
    try:
        res = provider.complete(system=reduce_system,
                                messages=[{"role": "user", "content": reduce_user}],
                                max_tokens=REDUCE_MAX_TOKENS)
    except Exception as e:  # noqa: BLE001 -- synthesis filtered/failed: return raw items
        lines = []
        for it in all_items:
            v = it.get("_video") or {}
            lines.append("- %s [%s](%s)" % (it.get("claim") or it.get("quote") or "",
                                            v.get("title") or "", v.get("url") or ""))
        note = "(synthesis step failed: %s; returning the raw extracted items)" % (
            "content_filter" if "content_filter" in str(e) else str(e).splitlines()[0][:120])
        return {"answer": note + "\n\n" + "\n".join(lines), "citations": [],
                "chunks": [], "refused": False, "refusal_reason": "reduce_failed",
                "status": "ok", "stats": stats}
    return {"answer": res.text, "citations": [], "chunks": [], "refused": False,
            "refusal_reason": None, "status": "ok", "stats": stats,
            "tokens": {"prompt": res.prompt_tokens, "completion": res.completion_tokens}}
