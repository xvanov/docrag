"""scrape_ncosfm.py -- pull NC Office of State Fire Marshal (OSFM) code
interpretations, formal interpretations, appeals, and guidance papers into the
building-codes corpus.

These are North Carolina's OFFICIAL rulings on how the building code is actually
applied -- the reconciliation layer between formal code text and real-world
enforcement. Example: an OSFM interpretation states that a standalone outdoor
fire pit "is not listed in R101.2.2 Accessory Structures and is not required to
meet the provisions of the NC Residential Code." Pure code text can't yield that
conclusion; the interpretation can. Indexing this layer lets the agentic RAG
reach correct real-world answers (permit / no-permit, scope, exemptions).

Site shape (recon): every entry on ncosfm.gov is a metadata landing page that
links to a PDF at ``<detail-url>/open`` -- including the "Q&A" interpretations.
Plain urllib fetches fine (no bot wall). PDFs are born-digital text (PyMuPDF
extracts cleanly; PyPDF2 is the fallback). Scanned PDFs are rare and, since
Docling/OCR is unavailable in this env, are skipped + logged rather than indexed.

For each entry we: download the PDF (cached), extract + clean the text (strip
letterhead boilerplate), and write a cleaned ``.html`` wrapper carrying
``<meta name="book"/"chapter">`` provenance + a ``<section>`` whose ``tg-number``
is the INTERPRETED code section. The existing HTML extractor
(``docrag/extractors.extract_html``) then gives clean, jurisdiction-labeled
citations, and the citation graph links interpretation <-> the code section it
interprets.

Outputs:
  corpora/building-codes/nc-interpretations/<type>/<slug>.html   (indexed)
  corpora/building-codes/nc-interpretations/<type>/_coverage.json (not indexed)
  scraper/_ncosfm_cache/<type>/<slug>.pdf                         (not indexed)

Usage:
  python scraper/scrape_ncosfm.py --type all
  python scraper/scrape_ncosfm.py --type interpretations --limit 1   # validation
  python scraper/scrape_ncosfm.py --type appeals --force             # re-extract
Then:
  python -m docrag.index build --corpus building-codes --confirm

Authorized personal/local use of public records; do not redistribute the corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from docrag.extractors import extract_pdf_pages, sanitize_ascii  # noqa: E402

BASE = "https://www.ncosfm.gov"
OUT_ROOT = os.path.join(ROOT, "corpora", "building-codes", "nc-interpretations")
CACHE_ROOT = os.path.join(ROOT, "scraper", "_ncosfm_cache")

_UA = "Mozilla/5.0 (docrag corpus builder; authorized personal/local use)"
_SLEEP = 1.0          # polite delay between network hits (seconds)
_MIN_TEXT_CHARS = 120  # below this (after cleaning) -> treat as scanned/empty

# Per-type config. ``categories`` are the first URL path segment(s) that mark a
# real detail page for that type (the second segment always starts with a digit:
# a code number "0101.2..." or a case year "2026-...").
TYPES = {
    "interpretations": {
        "listing": BASE + "/interpretations",
        "categories": {
            "administrative", "building", "electrical", "energy-conservation",
            "existing-buildings", "fire-prevention", "fuel-gas", "mechanical",
            "plumbing", "residential",
        },
        "label": "Interpretation",
        # Interpretation slugs always begin with the code number (digit); the
        # category words double as ordinary content paths, so require a digit to
        # avoid pulling in menu pages.
        "require_digit": True,
    },
    "formal": {
        "listing": BASE + "/interpre-type/Formal-Interpretations",
        "categories": {"formal-interpretations"},
        "label": "Formal Interpretation",
        # Unique path -> every child is a ruling; some slugs start with letters
        # (e.g. "gs-160d-..."), so don't require a leading digit.
        "require_digit": False,
    },
    "appeals": {
        "listing": BASE + "/interpre-type/Appeals",
        "categories": {"appeals"},
        "label": "Appeal",
        "require_digit": False,
    },
    "guidance": {
        "listing": BASE + "/interpre-type/Guidance-Papers",
        "categories": {"guidance-papers"},
        "label": "Guidance Paper",
        "require_digit": False,
    },
}

_MAX_PAGES = 40  # safety cap on listing pagination

# Letterhead / footer lines to drop so chunks hold real content, not boilerplate.
_BOILER_RE = [
    re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.I),
    re.compile(r"north carolina office of the state fire marshal", re.I),
    re.compile(r"nc department of insurance", re.I),
    re.compile(r"department of insurance", re.I),
    re.compile(r"office of the state fire marshal", re.I),
    re.compile(r"engineering (and building codes )?division", re.I),
    re.compile(r"mail service center", re.I),
    re.compile(r"raleigh,?\s*nc\s*2769", re.I),
    re.compile(r"^\s*9\d{2}[-.\s]?\d{3}[-.\s]?\d{4}\s*$"),
    re.compile(r"^\s*[_\-=.\s]{6,}\s*$"),   # rule lines
    re.compile(r"^\s*www\.ncosfm\.gov\s*$", re.I),
]

# A code-section token: optional letter prefix + digit + dotted/dashed suffix,
# e.g. R101.2.2, 705.8, 160D-1110, E3601.1, Appendix M (kept loose).
_SECTION_RE = re.compile(r"\b([A-Z]{0,3}\d{2,4}(?:[.\-][\w]+)*)\b")


# ---- HTTP -------------------------------------------------------------------

def _get(url: str, *, binary: bool = False, tries: int = 3):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            data = urllib.request.urlopen(req, timeout=60).read()
            return data if binary else data.decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(_SLEEP * (attempt + 1))
    raise last  # type: ignore[misc]


# ---- listing enumeration ----------------------------------------------------

_HREF_RE = re.compile(r'href="(/[a-z][a-z0-9-]*/[^"#?]+)"')


def _detail_links(html: str, categories: set, require_digit: bool) -> list:
    out = []
    for href in _HREF_RE.findall(html):
        parts = href.strip("/").split("/")
        if len(parts) != 2:
            continue
        cat, slug = parts
        if cat not in categories or slug == "open":
            continue
        if require_digit and not slug[:1].isdigit():
            continue
        out.append(href)
    return out


def enumerate_details(listing: str, categories: set, require_digit: bool) -> list:
    """Page through a listing until no new detail links appear."""
    seen: list = []
    seen_set: set = set()
    for page in range(_MAX_PAGES):
        url = "%s?page=%d" % (listing, page)
        try:
            html = _get(url)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("[ncosfm] listing fetch failed %s: %s\n" % (url, e))
            break
        new = [h for h in _detail_links(html, categories, require_digit)
               if h not in seen_set]
        if not new:
            break
        for h in new:
            seen_set.add(h)
            seen.append(h)
        sys.stderr.write("[ncosfm]   page %d: +%d (total %d)\n"
                         % (page, len(new), len(seen)))
        time.sleep(_SLEEP)
    return seen


# ---- landing-page metadata --------------------------------------------------

def _field(html: str, name: str) -> str:
    m = re.search(
        r'field--name-%s.*?field--item[^>]*>\s*(.*?)\s*</div>' % re.escape(name),
        html, re.S)
    if not m:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(1))).strip()


def landing_meta(html: str) -> dict:
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S)
    title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", h1.group(1))).strip() if h1 else ""
    pub = _field(html, "document-entity-pub-date")
    pub = re.sub(r"^First Published\s*", "", pub).strip()
    edition = _field(html, "document-entity-terms").strip()
    return {"title": title, "pub_date": pub, "edition": edition}


# ---- PDF text extraction ----------------------------------------------------

def pdf_text(pdf_path: str) -> str:
    """Extract text. PyMuPDF (clean, fast) preferred; PyPDF2 fallback."""
    try:
        import fitz  # type: ignore  # PyMuPDF
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text() for page in doc)
    except Exception:  # noqa: BLE001 -- PyMuPDF absent or parse error
        pages = extract_pdf_pages(pdf_path) or []
        return "\n".join(t for _, t in pages)


def clean_text(text: str) -> str:
    out = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if any(rx.search(s) for rx in _BOILER_RE):
            continue
        out.append(re.sub(r"[ \t]+", " ", s))
    # Re-join into paragraphs: keep line breaks, cap blank runs.
    joined = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


# ---- field parsing (Q&A interpretations) ------------------------------------

def parse_qa_fields(text: str) -> dict:
    """Pull the structured Code:/Date:/Section: header of a Q&A interpretation.

    These appear near the top as 'Code: 2012 Residential Code  Date: ...
    Section: R101.2.2'. Returns {} when the doc isn't in that format
    (formal/appeals/guidance use letter/decision prose instead)."""
    out: dict = {}
    m = re.search(r"\bCode:\s*((?:19|20)\d{2})\s+([A-Za-z /]+?Code)\b", text)
    if m:
        out["edition"] = m.group(1)
        out["code_type"] = m.group(2).strip()
    m = re.search(r"\bSection:\s*([A-Z]{0,3}\d[\w.\-]*(?:\.\d+)*)", text)
    if m:
        out["section"] = m.group(1).strip()
    m = re.search(r"\bDate:\s*([A-Za-z]+ \d{1,2},? \d{4})", text)
    if m:
        out["date"] = m.group(1).strip()
    return out


def guess_section(title: str, text: str) -> str:
    """Best-effort interpreted code section from the title or body."""
    # Titles like "2018 NCRC R305.1 Minimum Height" or "0101.2 - Outdoor Fire Pits".
    for src in (title, text[:400]):
        m = re.search(r"\b([A-Z]\d{2,4}(?:\.\d+)+)\b", src)   # e.g. R305.1, E3601.1
        if m:
            return m.group(1)
    m = re.search(r"\b(160D-\d{3,4})\b", title + " " + text[:400])  # NCGS statute
    return m.group(1) if m else ""


# ---- HTML wrapper -----------------------------------------------------------

def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_html(*, slug: str, title: str, label: str, number: str, edition: str,
               code_type: str, section: str, date: str, url: str, body: str) -> str:
    # source label (book) -> drives citation attribution. Normalize code_type so
    # we don't emit "2012 NC NC Energy Conservation Code" when it already starts
    # with NC / North Carolina.
    ct = re.sub(r"^\s*(?:NC|North Carolina)\s+", "", code_type, flags=re.I).strip()
    if label == "Interpretation":
        edpart = (" (%s NC %s)" % (edition, ct)) if (edition and ct) \
            else ((" (%s NC Code)" % edition) if edition else "")
        book = "NC OSFM Interpretation%s" % edpart
    else:
        book = "NC OSFM %s%s" % (label, (" " + number) if number else "")
    # section_label (chapter). Strip a leading "<num> - " / "<num>: " label only
    # when the separator is space-padded (interpretation titles), so a case
    # number like "2026-0603" (dash, no spaces) is preserved.
    topic = re.sub(r"^\s*[\w.]+\s+[-:]\s+", "", title).strip() or title
    chapter = title
    if section and section not in title:
        chapter = "%s - SS%s" % (title, section)

    tg = ""
    if section:
        tg = '<span class="tg-number">%s</span> <span class="tg-title">%s</span>' \
             % (_esc(section), _esc(topic))
    else:
        tg = '<span class="tg-title">%s</span>' % _esc(topic)

    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    body_html = "\n".join("<p>%s</p>" % _esc(p.replace("\n", " ")) for p in paras)

    head = []
    head.append("Source: %s (NC Office of the State Fire Marshal, official site)." % url)
    if number:
        head.append("Document number: %s." % number)
    if edition:
        head.append("Code edition: %s." % edition)
    if section:
        head.append("Interpreted code section: %s." % section)
    if date:
        head.append("Date: %s." % date)
    head_html = "<p>%s</p>" % _esc(" ".join(head))

    return (
        "<html><head>"
        '<meta charset="utf-8">'
        '<meta name="book" content="%s">'
        '<meta name="chapter" content="%s">'
        "<title>%s</title></head><body>\n"
        '<section id="ncosfm-%s">\n%s\n%s\n%s\n</section>\n'
        "</body></html>\n"
        % (_esc(book), _esc(chapter), _esc(title or label),
           _esc(slug), tg, head_html, body_html)
    )


# ---- per-type driver --------------------------------------------------------

def scrape_type(tkey: str, *, limit: int = 0, force: bool = False) -> dict:
    cfg = TYPES[tkey]
    out_dir = os.path.join(OUT_ROOT, tkey)
    cache_dir = os.path.join(CACHE_ROOT, tkey)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    sys.stderr.write("[ncosfm] === %s ===\n" % tkey)
    details = enumerate_details(cfg["listing"], cfg["categories"],
                                cfg["require_digit"])
    if limit:
        details = details[:limit]
    sys.stderr.write("[ncosfm] %s: %d detail pages\n" % (tkey, len(details)))

    cov = {"type": tkey, "listing": cfg["listing"], "found": len(details),
           "saved": 0, "skipped_existing": 0, "dropped": []}

    for href in details:
        url = BASE + href
        slug = href.strip("/").split("/")[-1]
        out_path = os.path.join(out_dir, slug + ".html")
        if os.path.exists(out_path) and not force:
            cov["skipped_existing"] += 1
            continue
        try:
            landing = landing_meta(_get(url))
            time.sleep(_SLEEP)
            pdf_path = os.path.join(cache_dir, slug + ".pdf")
            if not os.path.exists(pdf_path):
                data = _get(url + "/open", binary=True)
                if not data[:5].startswith(b"%PDF"):
                    cov["dropped"].append({"slug": slug, "reason": "not_a_pdf"})
                    continue
                with open(pdf_path, "wb") as f:
                    f.write(data)
                time.sleep(_SLEEP)
            raw = pdf_text(pdf_path)
            body = sanitize_ascii(clean_text(raw))
            if len(re.sub(r"[^A-Za-z]", "", body)) < _MIN_TEXT_CHARS:
                cov["dropped"].append(
                    {"slug": slug, "reason": "no_text_layer (scanned; OCR unavailable)"})
                sys.stderr.write("[ncosfm]   DROP scanned/empty: %s\n" % slug)
                continue

            qa = parse_qa_fields(raw)
            title = landing.get("title") or slug
            number = ""
            mnum = re.match(r"((?:19|20)\d{2}-\d{3,4})", slug) or \
                re.match(r"((?:19|20)\d{2}-\d{3,4})", title)
            if mnum:
                number = mnum.group(1)
            elif tkey == "interpretations":
                mn = re.match(r"\s*([\d.]+)\b", title)
                number = mn.group(1) if mn else ""
            edition = qa.get("edition") or landing.get("edition") or ""
            section = qa.get("section") or guess_section(title, raw)
            date = qa.get("date") or landing.get("pub_date") or ""
            code_type = qa.get("code_type") or ""

            html = build_html(
                slug=slug, title=title, label=cfg["label"], number=number,
                edition=edition, code_type=code_type, section=section,
                date=date, url=url, body=body)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
            cov["saved"] += 1
            sys.stderr.write("[ncosfm]   saved %-55s SS=%s\n" % (slug, section or "-"))
        except Exception as e:  # noqa: BLE001
            cov["dropped"].append({"slug": slug, "reason": "error: %s" % e})
            sys.stderr.write("[ncosfm]   FAIL %s: %s\n" % (slug, e))

    with open(os.path.join(out_dir, "_coverage.json"), "w", encoding="utf-8") as f:
        json.dump(cov, f, indent=2)
    sys.stderr.write(
        "[ncosfm] %s done: found=%d saved=%d skipped_existing=%d dropped=%d\n"
        % (tkey, cov["found"], cov["saved"], cov["skipped_existing"], len(cov["dropped"])))
    return cov


def main():
    ap = argparse.ArgumentParser(description="Scrape NC OSFM code interpretations.")
    ap.add_argument("--type", choices=list(TYPES) + ["all"], default="all")
    ap.add_argument("--limit", type=int, default=0,
                    help="max detail pages per type (0 = all; use 1 to validate)")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch/re-extract even if output .html exists")
    args = ap.parse_args()

    keys = list(TYPES) if args.type == "all" else [args.type]
    summary = []
    for k in keys:
        summary.append(scrape_type(k, limit=args.limit, force=args.force))
    print("\n=== NC OSFM scrape summary ===")
    for c in summary:
        print("  %-16s found=%-4d saved=%-4d existing=%-4d dropped=%d"
              % (c["type"], c["found"], c["saved"], c["skipped_existing"],
                 len(c["dropped"])))


if __name__ == "__main__":
    main()
