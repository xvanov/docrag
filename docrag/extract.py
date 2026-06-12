"""extract.py -- walk a corpus folder and extract text from each file.

The generalized replacement for the toolkit's manifest-gated rag_extract:
there is NO manifest. Every file under ``{docs_root}/{corpus}/`` whose
extension is extractable gets indexed. Binary / image extensions are skipped.

Public API:
    iter_eligible_files(corpus) -> Iterator[dict]
    iter_skipped_files(corpus)  -> Iterator[dict]
    extract_file(file_entry)    -> dict
"""

from __future__ import annotations

import os
import sys
from typing import Iterator

from . import settings
from .extractors import (
    extract_doc,
    extract_docx,
    extract_pdf_pages,
    sanitize_ascii,
)


# Extensions we know how to read.
_EXT_PDF = ".pdf"
_EXT_TEXT = frozenset({".txt", ".md", ".markdown", ".rst", ".csv", ".log",
                       ".json", ".xml", ".html", ".htm"})
_EXT_DOCX = ".docx"
_EXT_DOC = ".doc"

_EXT_ALL = _EXT_TEXT | {_EXT_PDF, _EXT_DOCX, _EXT_DOC}

# Hard limit per file (bytes). Larger files are skipped + logged.
_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


def _corpus_dir(corpus: str) -> str:
    return os.path.join(settings.docs_root(), corpus)


def _safe_path(p: str) -> str:
    r"""Windows long-path-safe (``\\?\`` prefix) when needed. No-op elsewhere."""
    if os.name != "nt":
        return p
    if p.startswith("\\\\?\\"):
        return p
    abs_p = os.path.abspath(p)
    if len(abs_p) >= 240:
        return "\\\\?\\" + abs_p.replace("/", "\\")
    return p


def _walk(corpus: str) -> Iterator[str]:
    """Yield absolute paths of every (non-hidden) file under the corpus dir."""
    root = _corpus_dir(corpus)
    if not os.path.isdir(root):
        return
    for dirpath, dirnames, filenames in os.walk(_safe_path(root)):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            yield os.path.join(dirpath, name)


def _entry_for(corpus: str, abs_path: str) -> dict:
    root = _corpus_dir(corpus)
    try:
        rel = os.path.relpath(abs_path, root).replace("\\", "/")
    except ValueError:
        rel = abs_path.replace("\\", "/")
    basename = os.path.basename(abs_path)
    return {
        "path": rel,
        "abs_path": abs_path,
        "basename": basename,
        "ext": os.path.splitext(basename)[1].lower(),
    }


def iter_eligible_files(corpus: str) -> Iterator[dict]:
    """Yield entry dicts for every extractable file in the corpus."""
    for abs_path in _walk(corpus):
        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in _EXT_ALL:
            continue
        try:
            if os.path.getsize(abs_path) > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield _entry_for(corpus, abs_path)


def iter_skipped_files(corpus: str) -> Iterator[dict]:
    """Yield ``{path, basename, reason}`` for every skipped file."""
    root = _corpus_dir(corpus)
    if not os.path.isdir(root):
        yield {"path": root, "basename": "(no corpus folder)",
               "reason": "no_corpus_folder", "ext": ""}
        return
    for abs_path in _walk(corpus):
        ext = os.path.splitext(abs_path)[1].lower()
        e = _entry_for(corpus, abs_path)
        if ext not in _EXT_ALL:
            yield {"path": e["path"], "basename": e["basename"],
                   "reason": "excluded_by_ext", "ext": ext}
            continue
        try:
            if os.path.getsize(abs_path) > _MAX_FILE_BYTES:
                yield {"path": e["path"], "basename": e["basename"],
                       "reason": "file_too_large", "ext": ext}
        except OSError:
            yield {"path": e["path"], "basename": e["basename"],
                   "reason": "read_error", "ext": ext}


# ---- Extraction -------------------------------------------------------------


def _read_text_file(abs_path: str) -> str:
    """Read a plain-text file with utf-8 then latin-1 fallback."""
    safe = _safe_path(abs_path)
    try:
        with open(safe, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(safe, "r", encoding="latin-1") as f:
                return f.read()
        except OSError:
            return ""
    except OSError:
        return ""


def extract_file(file_entry: dict) -> dict:
    """Extract text from one eligible file.

    Returns: ``{"kind": "pdf"|"text", "pages": list[str]|None,
                "text": str|None, "page_count": int, "no_text": bool}``.
    PDFs return per-page text; everything else a single text blob.
    """
    abs_path = file_entry["abs_path"]
    ext = file_entry["ext"]
    safe_abs = _safe_path(abs_path)

    result: dict = {"kind": "text", "pages": None, "text": None,
                    "page_count": 0, "no_text": False}

    if ext == _EXT_PDF:
        page_tuples = extract_pdf_pages(safe_abs) or []
        pages_text = [pt[1] for pt in page_tuples]
        result["kind"] = "pdf"
        result["pages"] = pages_text
        result["page_count"] = len(pages_text)
        result["no_text"] = sum(len(p.strip()) for p in pages_text) == 0
        return result

    if ext == _EXT_DOCX:
        try:
            text = sanitize_ascii(extract_docx(safe_abs))
        except RuntimeError as e:
            sys.stderr.write("[extract] %s: %s\n" % (file_entry["path"], e))
            text = ""
    elif ext == _EXT_DOC:
        try:
            text = sanitize_ascii(extract_doc(safe_abs))
        except RuntimeError as e:
            sys.stderr.write("[extract] %s: %s\n" % (file_entry["path"], e))
            text = ""
    else:
        text = sanitize_ascii(_read_text_file(abs_path))

    result["text"] = text
    result["page_count"] = 1 if text and text.strip() else 0
    result["no_text"] = not (text and text.strip())
    return result
