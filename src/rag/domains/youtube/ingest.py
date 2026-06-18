"""ingest.py -- pull YouTube transcripts into a corpus DB.

Pipeline per video: enumerate (yt-dlp) -> fetch timestamped transcript
(youtube-transcript-api) -> window into leaf chunks (chunker) -> embed (Azure,
shared with building-codes) -> insert into the per-corpus SQLite-vec DB.

Incremental: a video's content_hash is the sha256 of its transcript text; an
unchanged transcript is skipped (re-captioned videos re-index). The full
transcript text is stored on the files row's metadata so the map-reduce
exhaustive path can read each video whole without reconstructing from chunks.

Run locally on a residential IP: cloud/datacenter IPs are widely blocked by
YouTube for transcript and yt-dlp access.
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from typing import Iterable

from ... import settings
from ...db import (file_row, insert_chunks, open_db, purge_file, schema_outdated,
                   upsert_file)
from .chunker import chunk_transcript, transcript_fulltext

EMBED_FLUSH_SIZE = 256
MAX_EMBED_INPUT_CHARS = 30000
EMBED_PRICE_PER_MTOKEN = 0.13
COST_CONFIRM_THRESHOLD = 1.00

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_WATCH_RE = re.compile(r"[?&]v=([A-Za-z0-9_-]{11})")
_PATH_ID_RE = re.compile(r"/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})")
_SHORT_RE = re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})")


def parse_video_id(s: str) -> str | None:
    """Extract an 11-char video id from a URL or raw id; None if not a video."""
    s = (s or "").strip()
    if _ID_RE.match(s):
        return s
    for rx in (_WATCH_RE, _SHORT_RE, _PATH_ID_RE):
        m = rx.search(s)
        if m:
            return m.group(1)
    return None


def _looks_like_collection(target: str) -> bool:
    t = (target or "").lower()
    return any(k in t for k in ("/playlist", "list=", "/channel/", "/@", "/c/",
                                "/user/", "/videos", "/streams", "/shorts?"))


def _ydl(flat: bool):
    from yt_dlp import YoutubeDL
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "ignoreerrors": True}
    if flat:
        opts["extract_flat"] = "in_playlist"
    return YoutubeDL(opts)


def _meta_from_entry(e: dict) -> dict:
    vid = e.get("id")
    return {
        "video_id": vid,
        "title": e.get("title") or vid,
        "channel": e.get("channel") or e.get("uploader") or "",
        "url": e.get("webpage_url") or e.get("url") or ("https://youtu.be/%s" % vid),
        "upload_date": e.get("upload_date"),
    }


def enumerate_targets(targets: Iterable[str], limit: int = 0) -> list[dict]:
    """Resolve URLs/ids/channels to a flat list of video_meta dicts.

    A watch/youtu.be/shorts URL or bare id -> one video. A channel/playlist URL
    -> all its videos (flat, cheap; titles+ids without per-video extraction)."""
    out: list[dict] = []
    seen: set[str] = set()
    for target in targets:
        target = (target or "").strip()
        if not target:
            continue
        vid = parse_video_id(target)
        if vid and not _looks_like_collection(target):
            meta = {"video_id": vid, "title": vid, "channel": "",
                    "url": "https://www.youtube.com/watch?v=%s" % vid,
                    "upload_date": None}
            if vid not in seen:
                seen.add(vid); out.append(meta)
            continue
        # Collection (channel/playlist): flat-enumerate entries.
        with _ydl(flat=True) as ydl:
            info = ydl.extract_info(target, download=False)
        entries = (info or {}).get("entries") or ([info] if info else [])
        chan = (info or {}).get("channel") or (info or {}).get("uploader") or ""
        for e in entries:
            if not e or not e.get("id"):
                continue
            meta = _meta_from_entry(e)
            if not meta["channel"]:
                meta["channel"] = chan
            if meta["video_id"] in seen:
                continue
            seen.add(meta["video_id"]); out.append(meta)
            if limit and len(out) >= limit:
                return out
    return out


def fetch_transcript(video_id: str, languages: list[str]) -> tuple[list[dict], str, bool]:
    """Return ([{text,start,duration}], language_code, is_generated)."""
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=languages)
    snippets = [{"text": s.text, "start": float(s.start),
                 "duration": float(s.duration)} for s in fetched]
    return snippets, fetched.language_code, fetched.is_generated


def _guard_embed_input(text: str) -> str:
    if len(text) <= MAX_EMBED_INPUT_CHARS:
        return text
    return text[: MAX_EMBED_INPUT_CHARS - 3] + "..."


def _check_azure_creds() -> None:
    if settings.azure_endpoint() and settings.azure_api_key():
        return
    sys.stderr.write("ERROR: Azure OpenAI embedding credentials missing "
                     "(AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY).\n")
    raise SystemExit(3)


_BLOCK_RETRIES = 2            # per-video retries on an IP block
_BLOCK_BACKOFF_S = 30.0       # base backoff; doubles each retry
_MAX_CONSECUTIVE_BLOCKS = 4   # consecutive blocked videos -> abort the run


def _is_ip_block(e: Exception) -> bool:
    name = type(e).__name__.lower()
    if "ipblocked" in name or "requestblocked" in name or "toomanyrequests" in name:
        return True
    s = str(e).lower()
    return ("blocking requests from your ip" in s or "too many requests" in s
            or "your ip has been blocked" in s)


def ingest(corpus: str, targets: list[str], languages: list[str] | None = None,
           limit: int = 0, full: bool = False, confirm: bool = False,
           dry_run: bool = False, sleep: float | None = None) -> int:
    """Ingest ``targets`` (video/channel/playlist URLs or ids) into ``corpus``.

    ``sleep`` paces transcript fetches (seconds between videos) to avoid tripping
    YouTube's per-IP rate limiter; defaults to RAG_YT_SLEEP (1.0s). On a sustained
    IP block the run backs off, then stops early so it doesn't churn through
    thousands of failing requests -- re-run later and incremental resumes.
    """
    languages = languages or ["en"]
    sleep_s = float(settings.get("RAG_YT_SLEEP", 1.0) or 1.0) if sleep is None else sleep
    print("[corpus] %s  (index: %s)" % (corpus, settings.index_dir()))
    print("[enumerate] resolving %d target(s)..." % len(targets))
    videos = enumerate_targets(targets, limit=limit)
    print("[enumerate] %d video(s)" % len(videos))
    if not videos:
        sys.stderr.write("ERROR: no videos resolved from targets.\n")
        return 1

    if not dry_run:
        _check_azure_creds()

    from ...embed import embed_batch  # lazy: pulls the Azure embedding stack

    conn = open_db(corpus)
    if schema_outdated(conn) and not full:
        sys.stderr.write("[ingest] schema changed -> forcing --full\n")
        full = True

    indexed = skipped = no_transcript = blocked_skips = 0
    total_chunks = 0
    consecutive_blocks = 0
    aborted = False
    try:
        for vmeta in videos:
            vid = vmeta["video_id"]
            path = "yt:%s" % vid

            # Fetch with IP-block backoff/retry; non-block errors = no captions.
            snippets = lang = is_gen = None
            for attempt in range(_BLOCK_RETRIES + 1):
                try:
                    snippets, lang, is_gen = fetch_transcript(vid, languages)
                    consecutive_blocks = 0
                    break
                except Exception as e:  # noqa: BLE001
                    if _is_ip_block(e):
                        consecutive_blocks += 1
                        if consecutive_blocks >= _MAX_CONSECUTIVE_BLOCKS:
                            sys.stderr.write(
                                "[ingest] YouTube is rate-limiting this IP "
                                "(%d consecutive blocks). Stopping -- wait ~30-60 "
                                "min and re-run; incremental resumes.\n"
                                % consecutive_blocks)
                            aborted = True
                            snippets = None
                            break
                        if attempt < _BLOCK_RETRIES:
                            back = _BLOCK_BACKOFF_S * (2 ** attempt)
                            sys.stderr.write("[ingest] IP block on %s; backoff %ds "
                                             "(retry %d/%d)\n"
                                             % (vid, int(back), attempt + 1, _BLOCK_RETRIES))
                            time.sleep(back)
                            continue
                        sys.stderr.write("[ingest] still blocked on %s; skipping\n" % vid)
                        blocked_skips += 1
                    else:
                        sys.stderr.write("[ingest] no transcript for %s: %s\n"
                                         % (vid, str(e).splitlines()[0][:160]
                                            or type(e).__name__))
                        no_transcript += 1
                    snippets = None
                    break
            if aborted:
                break
            if snippets is None:
                continue
            # Pace successful fetches so we don't trip YouTube's per-IP limiter.
            if sleep_s > 0:
                time.sleep(sleep_s)
            fulltext = transcript_fulltext(snippets)
            chash = hashlib.sha256(fulltext.encode("utf-8")).hexdigest()

            existing = file_row(conn, path)
            if (not full) and existing and existing[0] == chash:
                skipped += 1
                continue
            if existing is not None:
                purge_file(conn, path)

            leaves = chunk_transcript(vmeta, snippets, corpus)
            if not leaves:
                no_transcript += 1
                continue
            for c in leaves:
                c["embed_input"] = _guard_embed_input(c["embed_input"])

            if dry_run:
                print("[dry-run] %s '%s' -> %d chunks (%d chars, %s%s)"
                      % (vid, vmeta.get("title"), len(leaves), len(fulltext),
                         lang, "/auto" if is_gen else ""))
                total_chunks += len(leaves)
                continue

            # Embed in flushes.
            vectors: list[list[float]] = []
            buf = [c["embed_input"] for c in leaves]
            for i in range(0, len(buf), EMBED_FLUSH_SIZE):
                vectors.extend(embed_batch(buf[i:i + EMBED_FLUSH_SIZE]))
            if len(vectors) != len(leaves):
                sys.stderr.write("[ingest] embed count mismatch for %s -- skip\n" % vid)
                continue

            file_meta = dict(vmeta)
            file_meta.update({"lang": lang, "is_generated": is_gen,
                              "n_snippets": len(snippets),
                              "transcript_text": fulltext})
            upsert_file(conn, path, chash, len(fulltext.encode("utf-8")),
                        metadata=file_meta)
            insert_chunks(conn, leaves, vectors)
            conn.commit()
            indexed += 1
            total_chunks += len(leaves)
            sys.stderr.write("[ingest] %s '%s': %d chunks\n"
                             % (vid, (vmeta.get("title") or "")[:60], len(leaves)))

        if dry_run:
            est_tokens = total_chunks * 450
            print("[dry-run] %d videos, %d chunks, ~$%.2f embed (no insert)"
                  % (len(videos), total_chunks,
                     est_tokens / 1_000_000.0 * EMBED_PRICE_PER_MTOKEN))
            return 0
        print("[done] %s: indexed=%d skipped=%d no_transcript=%d "
              "ip_blocked=%d chunks=%d%s"
              % (corpus, indexed, skipped, no_transcript, blocked_skips,
                 total_chunks, "  [ABORTED: IP rate-limited]" if aborted else ""))
        if aborted:
            sys.stderr.write("[ingest] Re-run the same command after a cooldown; "
                             "indexed videos are skipped automatically.\n")
        return 0
    finally:
        conn.close()
