"""chunker.py -- turn a video's timestamped transcript into leaf chunks.

A transcript is a flat list of timestamped snippets; there is no section
hierarchy (unlike building codes). We window consecutive snippets into
~target-sized leaves with small overlap (for retrieval recall), stamping each
leaf with its start/end time so a citation can deep-link to the moment:
``https://youtu.be/<id>?t=<seconds>``.

Each leaf dict is shaped for db.insert_chunks: path/corpus/source_file/text/
content_hash/node_type plus ``metadata`` (JSON: video_id, channel, title, url,
start_time, end_time, hms), ``embed_input`` (meta-prefixed body for embedding),
and ``meta`` (the FTS enrichment column).
"""

from __future__ import annotations

import hashlib

TARGET_CHARS = 1800       # ~450 tokens/leaf
OVERLAP_CHARS = 200
MIN_LEAF_CHARS = 40


def seconds_to_hms(seconds: float) -> str:
    s = int(seconds or 0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return "%d:%02d:%02d" % (h, m, sec) if h else "%d:%02d" % (m, sec)


def deep_link(url: str, start: float) -> str:
    base = (url or "").split("&t=")[0].split("?t=")[0]
    sep = "&" if "?" in base else "?"
    return "%s%st=%ds" % (base, sep, int(start or 0))


def _flush(group: list[dict], video_meta: dict, corpus: str) -> dict | None:
    if not group:
        return None
    text = " ".join(g["text"].strip() for g in group if g["text"].strip()).strip()
    if len(text) < MIN_LEAF_CHARS:
        return None
    start = float(group[0]["start"])
    last = group[-1]
    end = float(last["start"]) + float(last.get("duration") or 0)
    vid = video_meta["video_id"]
    title = video_meta.get("title") or vid
    channel = video_meta.get("channel") or ""
    url = video_meta.get("url") or ("https://youtu.be/%s" % vid)
    hms = seconds_to_hms(start)
    meta = " | ".join(x for x in (channel, title, hms) if x)
    metadata = {
        "video_id": vid, "channel": channel, "title": title, "url": url,
        "upload_date": video_meta.get("upload_date"),
        "start_time": start, "end_time": end, "hms": hms,
        "deep_link": deep_link(url, start),
    }
    return {
        "path": "yt:%s" % vid,
        "corpus": corpus,
        "source_file": ("%s -- %s" % (channel, title)) if channel else title,
        "kind": "transcript",
        "node_type": "leaf",
        "section_id": None,
        "text": text,
        "content_hash": hashlib.sha256(("%s|%.2f|%s" % (vid, start, text))
                                       .encode("utf-8")).hexdigest(),
        "metadata": metadata,
        "meta": meta,
        "embed_input": "%s\n%s" % (meta, text),
    }


def chunk_transcript(video_meta: dict, snippets: list[dict], corpus: str,
                     target_chars: int = TARGET_CHARS,
                     overlap_chars: int = OVERLAP_CHARS) -> list[dict]:
    """Window ``snippets`` ({text,start,duration}) into leaf chunks.

    Greedy fill to ``target_chars``; carry a tail of recent snippets covering
    ~``overlap_chars`` into the next window so a sentence split across a boundary
    is recoverable. Returns leaf dicts ready for db.insert_chunks."""
    leaves: list[dict] = []
    group: list[dict] = []
    group_chars = 0
    for sn in snippets:
        t = (sn.get("text") or "").strip()
        if not t:
            continue
        group.append(sn)
        group_chars += len(t) + 1
        if group_chars >= target_chars:
            leaf = _flush(group, video_meta, corpus)
            if leaf:
                leaves.append(leaf)
            # Build the overlap tail from the end of the just-emitted group.
            tail: list[dict] = []
            tail_chars = 0
            for s in reversed(group):
                tail.insert(0, s)
                tail_chars += len(s.get("text") or "") + 1
                if tail_chars >= overlap_chars:
                    break
            group = tail
            group_chars = sum(len(s.get("text") or "") + 1 for s in group)
    leaf = _flush(group, video_meta, corpus)
    # Avoid emitting a final leaf that is purely the overlap tail of the previous.
    if leaf and (not leaves or leaf["content_hash"] != leaves[-1]["content_hash"]):
        leaves.append(leaf)
    return leaves


def transcript_fulltext(snippets: list[dict]) -> str:
    """Plain concatenated transcript text (for map-reduce extraction)."""
    return " ".join((sn.get("text") or "").strip() for sn in snippets
                    if (sn.get("text") or "").strip()).strip()
