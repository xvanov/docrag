"""chunk.py -- turn extracted text into RAG chunks with metadata.

Strategy:
  * PDF: one chunk per page (1-indexed). Skip blank pages (<20 chars).
  * Everything else: sliding window ~800 tokens, 100 overlap, broken at the
    nearest paragraph / sentence boundary.

Each chunk carries a short ``context_summary`` (prepended to the text before
embedding via ``embed_input``) so isolated chunks still ground to a location.
Token sizing uses 1 token ~= 4 chars. No external tokenizer dependency.

Public API:
    chunk_extracted(corpus, file_entry, extracted) -> list[dict]
"""

from __future__ import annotations

import hashlib
from typing import Optional


_CHARS_PER_TOKEN = 4
_TARGET_CHARS = 800 * _CHARS_PER_TOKEN      # 3200
_OVERLAP_CHARS = 100 * _CHARS_PER_TOKEN     # 400
_MIN_PAGE_CHARS = 20


def _build_context_summary(
    corpus: str,
    file_entry: dict,
    page: Optional[int] = None,
    section: Optional[str] = None,
) -> str:
    basename = file_entry.get("basename") or "unknown"
    parts = ["Source: %s." % basename, "Corpus: %s." % corpus]
    if page is not None:
        parts.append("Page %d." % page)
    if section:
        parts.append("Section: %s." % section)
    return " ".join(parts)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _slide(text: str) -> list[tuple[int, int, str]]:
    """Split text into ~_TARGET_CHARS windows with _OVERLAP_CHARS overlap.

    Returns ``(start_offset, end_offset, chunk_text)`` tuples, broken at the
    nearest paragraph / sentence / whitespace boundary in the last 200 chars.
    """
    if not text:
        return []
    text_len = len(text)
    if text_len <= _TARGET_CHARS:
        return [(0, text_len, text)]

    windows: list[tuple[int, int, str]] = []
    start = 0
    while start < text_len:
        end = min(start + _TARGET_CHARS, text_len)
        if end < text_len:
            window_text = text[start:end]
            backoff = window_text[-200:]
            cut = -1
            para = backoff.rfind("\n\n")
            if para != -1:
                cut = para + 2
            else:
                for marker in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
                    pos = backoff.rfind(marker)
                    if pos > cut:
                        cut = pos + len(marker)
                if cut == -1:
                    space = backoff.rfind(" ")
                    if space != -1:
                        cut = space + 1
            if cut > 0:
                end = start + (len(window_text) - 200) + cut
        chunk = text[start:end].strip()
        if chunk:
            windows.append((start, end, chunk))
        if end >= text_len:
            break
        next_start = end - _OVERLAP_CHARS
        if next_start <= start:
            next_start = end  # avoid infinite loop on degenerate input
        start = next_start
    return windows


def _line_range_for_span(text: str, start: int, end: int) -> tuple[int, int]:
    if not text or start >= len(text):
        return (1, 1)
    start_line = text.count("\n", 0, start) + 1
    end_line = text.count("\n", 0, max(start, min(end, len(text)))) + 1
    return (start_line, end_line)


def chunk_extracted(corpus: str, file_entry: dict, extracted: dict) -> list[dict]:
    """Convert an extracted document into a list of chunk dicts.

    Each chunk: ``path, corpus, source_file, kind, page, start_line,
    end_line, text, context_summary, content_hash, embed_input``.
    ``embed_input`` (context_summary + text) is what gets embedded.
    """
    chunks: list[dict] = []
    basename = file_entry.get("basename") or "unknown"
    rel_path = file_entry.get("path") or basename
    kind = "document"

    if extracted.get("no_text"):
        return chunks

    if extracted.get("kind") == "pdf":
        for idx, page_text in enumerate(extracted.get("pages") or []):
            page_num = idx + 1
            if not page_text or len(page_text.strip()) < _MIN_PAGE_CHARS:
                continue
            ctx = _build_context_summary(corpus, file_entry, page=page_num)
            text = page_text.strip()
            embed_input = ctx + "\n\n" + text
            chunks.append({
                "path": rel_path, "corpus": corpus, "source_file": basename,
                "kind": kind, "page": page_num,
                "start_line": None, "end_line": None,
                "text": text, "context_summary": ctx,
                "content_hash": _hash_text(embed_input),
                "embed_input": embed_input,
            })
        return chunks

    text = extracted.get("text") or ""
    if not text.strip():
        return chunks

    for w_start, w_end, w_text in _slide(text):
        if not w_text.strip():
            continue
        start_line, end_line = _line_range_for_span(text, w_start, w_end)
        ctx = _build_context_summary(
            corpus, file_entry,
            section="lines %d-%d" % (start_line, end_line),
        )
        embed_input = ctx + "\n\n" + w_text
        chunks.append({
            "path": rel_path, "corpus": corpus, "source_file": basename,
            "kind": kind, "page": None,
            "start_line": start_line, "end_line": end_line,
            "text": w_text, "context_summary": ctx,
            "content_hash": _hash_text(embed_input),
            "embed_input": embed_input,
        })
    return chunks
