"""chunk.py -- turn extracted documents into RAG chunks + section tree.

Strategy (2026 best-practice, structure-aware):
  * HTML code books (kind="html_sections"): one *leaf* chunk per section's own
    body text; each section is also stored as a parent-document node whose
    full_text = own body + all descendant bodies (small-to-big retrieval).
  * PDF: detect numbered section headings (per-jurisdiction regex + a
    monotonicity check that rejects in-sentence cross-references). When a page
    span yields too few headings, fall back to one chunk per page.
  * Plain text / unstructured: sliding window (~800 tokens, 100 overlap).

Every leaf's ``embed_input`` is enriched with breadcrumb + section number +
title + jurisdiction + edition (the highest-ROI retrieval lever) and the same
string is stored as ``meta`` for the FTS metadata column. Cross-references
("Section 705.8", "NCGS 160D-1110") are parsed into raw ``refs`` for the
citation graph (resolved later in index.py).

Public API:
    chunk_extracted(corpus, file_entry, extracted) -> dict
        {"chunks": [...leaf dicts...], "sections": [...node dicts...],
         "refs": [...raw ref dicts...]}
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

_CHARS_PER_TOKEN = 4
_TARGET_CHARS = 800 * _CHARS_PER_TOKEN      # 3200
_OVERLAP_CHARS = 100 * _CHARS_PER_TOKEN     # 400
_MIN_LEAF_CHARS = 20
# A single embedding input must stay under the model's 8192-token limit.
# Dense legal text runs ~3 chars/token, so split section/page bodies larger
# than this into windowed sub-leaves (they keep the same section_id and so
# still collapse to one parent at retrieval time).
_LEAF_MAX_CHARS = 6000

# ---- Cross-reference patterns (citation graph) ------------------------------
# Internal code section: "Section 705.8", "§ 1004.5", "R105.2", "Table 601".
_REF_INTERNAL_RE = re.compile(
    r"(?:Section|Sec\.|§)\s*([A-Z]?\d+(?:\.\d+)+|\bR\d{3,4}(?:\.\d+)*)"
    r"|\b(R\d{3,4}(?:\.\d+)+)\b"
    r"|\bTable\s+([A-Z]?\d+(?:\.\d+)+)\b"
)
# NC General Statute: "N.C.G.S. 160D-1110", "160D-1110(b)" (often bare).
_REF_NCGS_RE = re.compile(
    r"(?:N\.?C\.?G\.?S\.?\s*)?\b(\d{2,3}[A-Z]?-\d+(?:\.\d+)?)(?:\([a-z0-9]+\))*\b"
)

# ---- PDF heading detection profiles -----------------------------------------
# A heading line is short, starts with the number token, and is followed by a
# Title-Case label. Monotonicity (number sequences forward) rejects the
# "...comply with Section 112.1..." in-sentence false positives.
_PDF_HEADING_RES = (
    re.compile(r"^\s*SECTION\s+(\d+)\b(.*)$"),                      # IBC SECTION 113
    re.compile(r"^\s*(\d{3,4}(?:\.\d+)*)\s+([A-Z][A-Za-z].{0,80})$"),  # 112.1 Title.
    re.compile(r"^\s*Sec\.\s+(\d+\.\d+)\s+(.{0,80})$"),            # UDO Sec. 2.1
    re.compile(r"^\s*(\d+\.\d+\.\d+)\.\s+([A-Z].{0,80})$"),        # UDO 2.1.1.
)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# ---- Provenance helpers -----------------------------------------------------


def _jurisdiction(rel_path: str) -> str:
    p = (rel_path or "").replace("\\", "/").lstrip("/")
    head = p.split("/", 1)[0] if "/" in p else ""
    if head in ("model", "north-carolina", "durham"):
        return head
    return head or ""


def _edition(source_label: Optional[str], basename: str) -> str:
    if source_label:
        return source_label
    b = basename or ""
    m = re.search(r"(\d{4})", b)
    year = m.group(1) if m else ""
    if "International_Building_Code" in b or "IBC" in b:
        return ("IBC " + year).strip()
    if "UDO" in b or "Unified_Development" in b:
        return "Durham UDO"
    return os.path.splitext(b)[0]


def _build_meta(edition, breadcrumb, number, title, jurisdiction) -> str:
    bits = []
    if edition:
        bits.append(edition)
    if breadcrumb:
        bits.append(breadcrumb)
    head = " ".join(x for x in (number, title) if x)
    if head:
        bits.append(head)
    if jurisdiction:
        bits.append(jurisdiction)
    return " | ".join(bits)


def _num_tuple(num: Optional[str]):
    """('R101.2.1') -> (101, 2, 1) for monotonic comparison; None if unparseable."""
    if not num:
        return None
    digits = re.sub(r"^[A-Za-z]+", "", num)
    parts = []
    for p in digits.split("."):
        if p.isdigit():
            parts.append(int(p))
        else:
            return None
    return tuple(parts) if parts else None


# ---- Reference extraction ---------------------------------------------------


def _refs_from_text(text: str, src_section_id: str, corpus: str,
                    rel_path: str) -> list[dict]:
    out = {}
    for m in _REF_INTERNAL_RE.finditer(text or ""):
        raw = next((g for g in m.groups() if g), None)
        if raw:
            out[("internal", raw)] = {
                "src_section_id": src_section_id, "dst_raw": raw,
                "dst_kind": "internal_section", "corpus": corpus, "path": rel_path}
    for m in _REF_NCGS_RE.finditer(text or ""):
        raw = m.group(1)
        # Require a hyphen+letter pattern typical of NCGS (e.g. 160D-1110) to
        # avoid matching plain section numbers already caught above.
        if raw and re.search(r"[A-Z]", raw):
            out[("ncgs", raw)] = {
                "src_section_id": src_section_id, "dst_raw": raw,
                "dst_kind": "ncgs", "corpus": corpus, "path": rel_path}
    return list(out.values())


# ---- Sliding window (fallback for unstructured text) ------------------------


def _slide(text: str) -> list[tuple[int, int, str]]:
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
            next_start = end
        start = next_start
    return windows


def _split_body(body: str) -> list[str]:
    """Keep a small body whole; window a large one so each embedding input
    stays under the model's token limit. Sub-leaves share the section_id, so
    they still collapse to one parent section at retrieval time."""
    if len(body) <= _LEAF_MAX_CHARS:
        return [body]
    return [w for _s, _e, w in _slide(body) if w.strip()]


# ---- Leaf builder (shared) --------------------------------------------------


def _make_leaf(corpus, rel_path, source_file, jurisdiction, edition, *,
               section_id, section_number, section_title, breadcrumb,
               body, page=None, node_type="leaf") -> dict:
    meta = _build_meta(edition, breadcrumb, section_number, section_title,
                       jurisdiction)
    embed_input = (meta + "\n\n" + body) if meta else body
    return {
        "path": rel_path, "corpus": corpus, "source_file": source_file,
        "kind": "document", "page": page, "start_line": None, "end_line": None,
        "text": body, "context_summary": meta, "meta": meta,
        "content_hash": _hash_text(embed_input), "embed_input": embed_input,
        "section_id": section_id, "section_number": section_number,
        "section_title": section_title, "breadcrumb": breadcrumb,
        "jurisdiction": jurisdiction, "edition": edition,
        "parent_section_id": section_id, "node_type": node_type,
    }


# ---- HTML sections ----------------------------------------------------------


def _breadcrumb(sid, by_id) -> str:
    """Root-first 'num title > num title' trail from ancestor links."""
    chain = []
    seen = set()
    cur = sid
    while cur and cur in by_id and cur not in seen:
        seen.add(cur)
        n = by_id[cur]
        label = " ".join(x for x in (n.get("section_number"),
                                     n.get("section_title")) if x)
        if label:
            chain.append(label)
        cur = n.get("parent_section_id")
    return " > ".join(reversed(chain))


def _chunks_from_html(corpus, file_entry, extracted) -> dict:
    rel_path = file_entry.get("path") or file_entry.get("basename") or "unknown"
    basename = file_entry.get("basename") or "unknown"
    source_label = extracted.get("source_label") or basename
    jurisdiction = _jurisdiction(rel_path)
    edition = _edition(source_label, basename)
    sections = extracted.get("sections") or []
    by_id = {s["section_id"]: s for s in sections}

    chunks, nodes, refs = [], [], []
    for s in sections:
        sid = s["section_id"]
        crumb = _breadcrumb(sid, by_id)
        full = (s.get("full_text") or "").strip()
        own = (s.get("own_text") or "").strip()
        # Parent-document node (retrieval payload).
        nodes.append({
            "corpus": corpus, "path": rel_path, "jurisdiction": jurisdiction,
            "edition": edition, "section_id": sid,
            "section_number": s.get("section_number"),
            "section_title": s.get("section_title"), "breadcrumb": crumb,
            "parent_section_id": s.get("parent_section_id"),
            "full_text": full or own,
        })
        # Leaf chunk for this section's own body (embedded + searched).
        if len(own) >= _MIN_LEAF_CHARS:
            nt = "table" if s.get("has_table") else "leaf"
            for body in _split_body(own):
                chunks.append(_make_leaf(
                    corpus, rel_path, source_label, jurisdiction, edition,
                    section_id=sid, section_number=s.get("section_number"),
                    section_title=s.get("section_title"), breadcrumb=crumb,
                    body=body, node_type=nt))
            refs.extend(_refs_from_text(own, sid, corpus, rel_path))
    return {"chunks": chunks, "sections": nodes, "refs": refs}


# ---- PDF (heading detection + fallback) -------------------------------------


def _detect_pdf_sections(pages: list[str]):
    """Return [(number, title, body, first_page)] or [] if structure is weak."""
    # Flatten pages, tracking page number per line.
    lines = []  # (text, page_no)
    for idx, ptext in enumerate(pages):
        for ln in (ptext or "").splitlines():
            lines.append((ln, idx + 1))

    headings = []  # (line_idx, number, title, page)
    last_tuple = None
    for i, (ln, pg) in enumerate(lines):
        for rx in _PDF_HEADING_RES:
            m = rx.match(ln)
            if not m:
                continue
            number = m.group(1)
            title = (m.group(2) if m.lastindex and m.lastindex >= 2 else "").strip(" .")
            nt = _num_tuple(number)
            # Monotonicity: accept only if sequencing forward, or a fresh top.
            if nt is not None and last_tuple is not None:
                if not (nt > last_tuple or len(nt) == 1):
                    break
            headings.append((i, number, title or None, pg))
            if nt is not None:
                last_tuple = nt
            break

    # Weak signal -> let caller fall back to page chunks.
    if len(headings) < 5:
        return []

    sections = []
    for h_idx, (li, number, title, pg) in enumerate(headings):
        end_li = headings[h_idx + 1][0] if h_idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[j][0] for j in range(li, end_li)).strip()
        if body:
            sections.append((number, title, body, pg))
    return sections


def _chunks_from_pdf(corpus, file_entry, extracted) -> dict:
    rel_path = file_entry.get("path") or file_entry.get("basename") or "unknown"
    basename = file_entry.get("basename") or "unknown"
    jurisdiction = _jurisdiction(rel_path)
    edition = _edition(None, basename)
    pages = extracted.get("pages") or []

    detected = _detect_pdf_sections(pages)
    chunks, nodes, refs = [], [], []

    if detected:
        # Parent links by numeric depth (101.2.1 -> 101.2 -> 101).
        num_to_sid = {}
        for number, _t, _b, pg in detected:
            num_to_sid[number] = "%s#%s" % (rel_path, number)
        for number, title, body, pg in detected:
            sid = num_to_sid[number]
            parent_sid = None
            if number and "." in number:
                parent_num = number.rsplit(".", 1)[0]
                parent_sid = num_to_sid.get(parent_num)
            crumb = " ".join(x for x in (number, title) if x)
            nodes.append({
                "corpus": corpus, "path": rel_path, "jurisdiction": jurisdiction,
                "edition": edition, "section_id": sid, "section_number": number,
                "section_title": title, "breadcrumb": crumb,
                "parent_section_id": parent_sid, "full_text": body,
            })
            for piece in _split_body(body):
                chunks.append(_make_leaf(
                    corpus, rel_path, basename, jurisdiction, edition,
                    section_id=sid, section_number=number, section_title=title,
                    breadcrumb=crumb, body=piece, page=pg))
            refs.extend(_refs_from_text(body, sid, corpus, rel_path))
        return {"chunks": chunks, "sections": nodes, "refs": refs}

    # Fallback: one chunk per page (legacy behavior), enriched with edition.
    for idx, page_text in enumerate(pages):
        page_num = idx + 1
        body = (page_text or "").strip()
        if len(body) < _MIN_LEAF_CHARS:
            continue
        crumb = "%s p.%d" % (edition or basename, page_num)
        for piece in _split_body(body):
            chunks.append(_make_leaf(
                corpus, rel_path, basename, jurisdiction, edition,
                section_id="%s#p%d" % (rel_path, page_num), section_number=None,
                section_title=None, breadcrumb=None, body=piece, page=page_num))
    return {"chunks": chunks, "sections": nodes, "refs": refs}


# ---- Plain text (sliding window) --------------------------------------------


def _chunks_from_text(corpus, file_entry, extracted) -> dict:
    rel_path = file_entry.get("path") or file_entry.get("basename") or "unknown"
    basename = file_entry.get("basename") or "unknown"
    source_label = extracted.get("source_label") or basename
    jurisdiction = _jurisdiction(rel_path)
    edition = _edition(source_label, basename)
    text = (extracted.get("text") or "")
    chunks = []
    if not text.strip():
        return {"chunks": [], "sections": [], "refs": []}
    for w_start, w_end, w_text in _slide(text):
        body = w_text.strip()
        if len(body) < _MIN_LEAF_CHARS:
            continue
        start_line = text.count("\n", 0, w_start) + 1
        sid = "%s#l%d" % (rel_path, start_line)
        leaf = _make_leaf(
            corpus, rel_path, source_label, jurisdiction, edition,
            section_id=sid, section_number=None, section_title=None,
            breadcrumb=None, body=body)
        leaf["start_line"] = start_line
        chunks.append(leaf)
    return {"chunks": chunks, "sections": [], "refs": []}


# ---- Public dispatch --------------------------------------------------------


def chunk_extracted(corpus: str, file_entry: dict, extracted: dict) -> dict:
    """Convert an extracted document into leaf chunks + section nodes + refs."""
    if extracted.get("no_text"):
        return {"chunks": [], "sections": [], "refs": []}
    kind = extracted.get("kind")
    if kind == "html_sections":
        return _chunks_from_html(corpus, file_entry, extracted)
    if kind == "pdf":
        return _chunks_from_pdf(corpus, file_entry, extracted)
    return _chunks_from_text(corpus, file_entry, extracted)
