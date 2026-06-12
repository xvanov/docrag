"""extractors.py -- text extraction for PDF / DOCX / DOC.

Vendored from the toolkit's doc_reader.py + pdf_evidence_locator.py so docrag
has no external-package dependency beyond PyPDF2 + python-docx. Plain-text
formats (.txt/.md) are read directly by extract.py.

Public API:
    sanitize_ascii(text) -> str
    extract_pdf_pages(path) -> list[(page_no, text)]   # 1-indexed
    extract_docx(path) -> str
    extract_doc(path) -> str        # legacy binary .doc, Latin-1 fallback
"""

from __future__ import annotations

import re
import sys


_REPLACEMENTS = {
    "‘": "'",   # left single curly quote
    "’": "'",   # right single curly quote
    "“": '"',   # left double curly quote
    "”": '"',   # right double curly quote
    "—": "--",  # em dash
    "–": "-",   # en dash
    "…": "...",  # ellipsis
    "•": "-",   # bullet
    " ": " ",   # non-breaking space
}


def sanitize_ascii(text: str) -> str:
    """Replace non-ASCII characters with safe equivalents.

    Single chokepoint for ASCII enforcement -- every extractor pipes output
    through this so downstream chunking / embedding sees consistent bytes.
    """
    if not text:
        return ""
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    # Strip control characters (keep \t \n \r).
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    out = []
    for ch in text:
        out.append(ch if ord(ch) <= 0x7E else "?")
    return "".join(out)


# -- PDF -----------------------------------------------------------------------


def extract_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text from every page, returning ``(page_no, text)`` tuples.

    page_no is 1-indexed. Pages with no extractable text are included as
    empty strings so page indexing stays correct. Output is ASCII-sanitized.
    Returns ``[]`` (with a stderr warning) when the PDF can't be opened.
    """
    try:
        import PyPDF2  # type: ignore
    except ImportError:
        sys.stderr.write("[extract] PyPDF2 not installed\n")
        return []

    try:
        reader = PyPDF2.PdfReader(pdf_path)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("[extract] cannot open PDF: %s\n" % exc)
        return []

    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty owner password; try that.
        try:
            reader.decrypt("")
        except Exception:  # noqa: BLE001
            sys.stderr.write("[extract] PDF is encrypted: %s\n" % pdf_path)
            return []

    pages: list[tuple[int, str]] = []
    for i in range(len(reader.pages)):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        pages.append((i + 1, sanitize_ascii(text)))
    return pages


# -- DOCX ----------------------------------------------------------------------


def extract_docx(filepath: str) -> str:
    """Extract text from a .docx file with light markdown structure.

    Returns raw (un-sanitized) text -- caller sanitizes. Raises RuntimeError
    if python-docx is missing or the file can't be opened, so the indexer can
    record the failure instead of crashing the process.
    """
    try:
        import docx  # type: ignore
    except ImportError as exc:
        raise RuntimeError("python-docx not installed") from exc

    try:
        document = docx.Document(filepath)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("cannot open DOCX: %s" % exc) from exc

    parts: list[str] = []

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name if para.style else ""
        if style_name.startswith("Heading"):
            m = re.search(r"(\d+)", style_name)
            level = int(m.group(1)) if m else 1
            parts.append("%s %s" % ("#" * level, text))
        elif "List" in style_name:
            parts.append("- %s" % text)
        else:
            parts.append(text)

    for table in document.tables:
        rows = []
        for row in table.rows:
            cells = []
            for cell in row.cells:
                cell_text = cell.text.strip() or " "
                cells.append(cell_text.replace("\n", " "))
            rows.append(cells)
        if rows:
            parts.append("")
            parts.append("| %s |" % " | ".join(rows[0]))
            parts.append("| %s |" % " | ".join("---" for _ in rows[0]))
            for row_cells in rows[1:]:
                while len(row_cells) < len(rows[0]):
                    row_cells.append(" ")
                parts.append("| %s |" % " | ".join(row_cells[:len(rows[0])]))
            parts.append("")

    return "\n".join(parts)


# -- DOC (legacy binary, Latin-1 fallback) -------------------------------------


def extract_doc(filepath: str) -> str:
    """Extract readable text from a binary .doc (OLE Word) file.

    Lossy Latin-1 decode -- no olefile dependency. Returns raw text.
    """
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("cannot read DOC: %s" % exc) from exc

    text = raw.decode("latin-1").replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    runs = re.findall(r"[\x20-\x7e\t\r\n]{20,}", text)

    clean_lines = []
    for run in runs:
        run = run.strip()
        if not run:
            continue
        alpha = sum(1 for ch in run if ch.isalpha())
        if alpha / len(run) < 0.3:
            continue
        clean_lines.append(re.sub(r"[ \t]+", " ", run))

    return ("(extracted via binary Latin-1 fallback -- formatting lost)\n\n"
            + "\n".join(clean_lines))
