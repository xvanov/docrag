"""
Scrape ICC Digital Codes into the docrag `building-codes` corpus.

Two collections (pick with --collection):
  nc-2024     -> the 9 "2024 North Carolina State Building Code" titles
                 saved under  building-codes/north-carolina/<book-slug>/
  icodes-2024 -> the 16 "2024 I-Codes" model titles (IBC, IRC, ...) saved
                 version-encoded under  building-codes/model/2024/<abbr>/

Strategy (reverse-engineered from the SPA's own JSON API):
  1. Enumerate titles via /api/search/search-results?category[]=<name>.
  2. For each book, GET /api/content/chapters/{doc} -> full TOC (every
     top-level entry, INCLUDING all appendices).
  3. For each top-level entry, GET /api/content/chapter-xml/{doc}/{contentId}
     -> the ENTIRE chapter HTML (every nested section + body) in one blob.
  4. Save one self-contained .html per chapter with a provenance header.

Resumable (skips chapters already saved), rate-limited, retries with backoff,
writes a per-book _manifest.json + a per-collection _coverage.json.

Usage:
    python scraper\\scrape.py --collection icodes-2024 --list
    python scraper\\scrape.py --collection icodes-2024
    python scraper\\scrape.py --collection nc-2024 --only 4302
"""
import argparse
import html
import json
import pathlib
import re
import sys
import time
import urllib.parse

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from icc_session import login, BASE  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS_ROOT = ROOT / "corpora" / "building-codes"
XHR = {"X-Requested-With": "XMLHttpRequest"}

SLEEP = 0.4          # polite delay between requests (s)
MAX_RETRY = 4

# --- Collections -------------------------------------------------------------
# Each collection: the ICC search category, and a function mapping a book's
# search record + info dict to its output directory (relative to CORPUS_ROOT).


def _nc_outdir(book, info):
    slug = info.get("document_slug") or slugify(book["display_title"])
    return pathlib.Path("north-carolina") / slug


def _abbr_from(book, info):
    """Clean code abbr (ibc / irc / ...) from the title's parenthetical, with a
    disambiguator for the two IECC variants (one bundles ASHRAE 90.1)."""
    title = book["display_title"]
    m = re.search(r"\(([A-Za-z0-9/]+)\)\s*$", title)
    abbr = (m.group(1) if m else "").strip().lower()
    if not abbr:
        for key in ("icc_abbreviation", "abbreviation", "book_short_code"):
            v = (info.get(key) or "").strip()
            if v:
                abbr = re.split(r"\d", v.lower(), 1)[0]
                break
    abbr = re.sub(r"[^a-z0-9]", "", abbr) or slugify(title)
    if abbr == "iecc" and "ashrae" in title.lower():
        abbr = "iecc-ashrae"
    return abbr


def _icodes_outdir(book, info):
    return pathlib.Path("model") / "2024" / _abbr_from(book, info)


COLLECTIONS = {
    "nc-2024": {
        "category": "North Carolina",
        "title_prefix": "2024 North Carolina State Building Code",
        "outdir": _nc_outdir,
    },
    "icodes-2024": {
        "category": "2024 I-Codes",
        "title_prefix": "",            # the category already scopes it
        "outdir": _icodes_outdir,
    },
}


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)[:80] or "section"


def get(sess: requests.Session, path: str):
    """GET with retry/backoff. Returns the requests.Response."""
    url = path if path.startswith("http") else BASE + path
    for attempt in range(MAX_RETRY):
        try:
            r = sess.get(url, headers=XHR, timeout=90)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            return r
        except Exception as e:
            wait = 2 ** attempt
            print(f"      ! {e} -> retry in {wait}s ({attempt+1}/{MAX_RETRY})")
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


def list_books(sess, category, title_prefix) -> list[dict]:
    base = "/api/search/search-results"
    items, page = [], 1
    while True:
        q = urllib.parse.urlencode(
            {"category[]": category, "page": page, "limit": 50, "search": ""}
        ).replace("category%5B%5D", "category[]")
        d = get(sess, f"{base}?{q}").json()
        items += d["data"]
        if len(items) >= d["pagination"]["totalResults"] or not d["data"]:
            break
        page += 1
    if title_prefix:
        items = [i for i in items if i["display_title"].startswith(title_prefix)]
    items.sort(key=lambda b: b["display_title"])
    return items


def book_info(sess, doc_id) -> dict:
    return get(sess, f"/api/content/info/{doc_id}").json()


def book_chapters(sess, doc_id) -> list[dict]:
    return get(sess, f"/api/content/chapters/{doc_id}").json()


def chapter_html(sess, doc_id, content_id) -> str:
    r = get(sess, f"/api/content/chapter-xml/{doc_id}/{content_id}")
    try:
        return r.json()
    except ValueError:
        return r.text


def count_sections(htmltext: str) -> int:
    return len(re.findall(r"<section\b", htmltext))


def wrap(book_title, chapter_title, doc_id, content_id, body) -> str:
    src = f"{BASE}/content/document/{doc_id}/{content_id}"
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{html.escape(book_title)} — {html.escape(chapter_title)}</title>\n"
        f"<meta name=\"source-url\" content=\"{html.escape(src)}\">\n"
        f"<meta name=\"book\" content=\"{html.escape(book_title)}\">\n"
        f"<meta name=\"chapter\" content=\"{html.escape(chapter_title)}\">\n"
        f"<meta name=\"icc-document-id\" content=\"{doc_id}\">\n"
        f"<meta name=\"icc-content-id\" content=\"{content_id}\">\n"
        "</head>\n<body>\n"
        f"<!-- Source: {src} -->\n{body}\n</body>\n</html>\n"
    )


def scrape_book(sess, book, outdir_fn, force=False) -> dict:
    doc_id = book["document_id"]
    title = book["display_title"]
    info = book_info(sess, doc_id)
    rel_dir = outdir_fn(book, info)
    out_dir = CORPUS_ROOT / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    chapters = book_chapters(sess, doc_id)
    (out_dir / "_toc.json").write_text(json.dumps(chapters, indent=1), encoding="utf-8")
    print(f"\n=== {title}  (doc {doc_id})  -> {rel_dir.as_posix()}  {len(chapters)} chapters")

    manifest = {"document_id": doc_id, "title": title,
                "rel_dir": rel_dir.as_posix(), "chapter_count": len(chapters),
                "chapters": []}
    for i, ch in enumerate(chapters, 1):
        cid = ch["content_id"]
        ctitle = (ch.get("title") or (ch.get("link") or {}).get("title")
                  or f"section-{cid}").strip()
        cslug = ch.get("chapter_slug") or slugify(ctitle)
        fname = f"{i:02d}-{cslug}.html"
        fpath = out_dir / fname
        if fpath.exists() and not force:
            print(f"  [{i:02d}/{len(chapters)}] skip (exists) {fname}")
            manifest["chapters"].append(
                {"n": i, "content_id": cid, "title": ctitle, "file": fname,
                 "bytes": fpath.stat().st_size, "skipped": True})
            continue
        body = chapter_html(sess, doc_id, cid)
        secs = count_sections(body)
        fpath.write_text(wrap(title, ctitle, doc_id, cid, body), encoding="utf-8")
        print(f"  [{i:02d}/{len(chapters)}] {fname}  {len(body):>8} chars  {secs:>4} sections")
        manifest["chapters"].append(
            {"n": i, "content_id": cid, "title": ctitle, "file": fname,
             "bytes": len(body), "sections": secs})
        time.sleep(SLEEP)

    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", choices=sorted(COLLECTIONS), default="nc-2024")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", help="single document_id")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    spec = COLLECTIONS[args.collection]
    sess = login(headless=args.headless)
    books = list_books(sess, spec["category"], spec["title_prefix"])
    print(f"\n[{args.collection}] {len(books)} titles:")
    for b in books:
        print(f"  doc={b['document_id']:>6}  {b['display_title']}")
    if args.list:
        return
    if args.only:
        books = [b for b in books if str(b["document_id"]) == str(args.only)]
        if not books:
            sys.exit(f"document_id {args.only} not in {args.collection}")

    report = []
    for b in books:
        report.append(scrape_book(sess, b, spec["outdir"], force=args.force))

    cov = CORPUS_ROOT / ("_coverage_%s.json" % args.collection)
    cov.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n========== COVERAGE (%s) ==========" % args.collection)
    total_ch = total_sec = 0
    for m in report:
        secs = sum(c.get("sections", 0) for c in m["chapters"])
        total_ch += m["chapter_count"]; total_sec += secs
        print(f"  {m['title'][:55]:55} chapters={m['chapter_count']:>3} sections={secs:>5}")
    print(f"  {'TOTAL':55} chapters={total_ch:>3} sections={total_sec:>5}")
    print("  wrote", cov)


if __name__ == "__main__":
    main()
