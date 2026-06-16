"""
Fetch legitimately-FREE construction docs that block plain curl (Cloudflare /
Akamai / CDN bot challenge -> 403) by driving a real Chromium page, exactly like
scraper/fetch_fema.py.

Three target groups, all confirmed free (govt / association-free / CC-BY OA):

  1. AISC free steel standards (aisc.org/standards publishes the standard + its
     commentary as free public PDFs; the AISC store sells printed/multi-user
     licensed copies, but the single-user PDF download is free). Bot-walled.
       - ANSI/AISC 360-22  Specification for Structural Steel Buildings
       - ANSI/AISC 341-22  Seismic Provisions for Structural Steel Buildings
       - ANSI/AISC 358-22  Prequalified Connections (moment frames)
       - ANSI/AISC 303-22  Code of Standard Practice for Buildings and Bridges
     -> corpora/methods/steel/
     We enumerate each "...-Download" landing page for the real PDF anchor (the
     filename hash on globalassets changes per edition, so scraping beats
     guessing). We also try a few known direct URLs as a fallback.

  2. MDPI open-access (CC-BY) earthen / natural-construction REVIEW papers.
     Cloudflare 403s curl. The PDF is the article URL + "/pdf".
     -> corpora/methods/earthen-reviews/

  3. Cal-Earth / SuperAdobe free structural test reports (calearth.org links them
     publicly from its "Resources for Builders" page).
     -> corpora/methods/earthbag/

Resumable: any target already valid on disk is skipped. PDFs must start with
%PDF and be >50 KB; HTML-only papers accepted at >20 KB. Anything that comes
back as an HTML error page (or fails validation) is deleted and reported.

Run:
    .venv\\Scripts\\python.exe scraper\\fetch_botwalled.py
"""
import pathlib
import re
import sys
import tempfile
import time
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPORA = ROOT / "corpora"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

PDF_MIN_BYTES = 50 * 1024   # 50 KB floor for PDFs
HTML_MIN_BYTES = 20 * 1024  # 20 KB floor for HTML-only papers

# ---------------------------------------------------------------------------
# Group 1: AISC free standards.
# ---------------------------------------------------------------------------
# Each entry: (download-landing-page, dest filename, [candidate direct PDF urls]).
# We first scrape the landing page for a .pdf anchor; if that fails we try the
# candidate direct URLs (filenames observed in the wild / search results).
AISC_HUB = "https://www.aisc.org/publications/steel-standards/"

# DROPPED: AISC 360/341/358/303-22 are NOT free -- copyrighted, purchase/DRM only
# (verified 2026-06-12). Only pirate copies exist; not touching those.
AISC_TARGETS = []
AISC_DIR = "methods/steel"

# ---------------------------------------------------------------------------
# Group 2: MDPI open-access (CC-BY) earthen-construction review papers.
# PDF url = article url + "/pdf". All open access, CC-BY 4.0.
# ---------------------------------------------------------------------------
# Verified real + open-access (CC-BY) via WebSearch 2026-06-12.
MDPI_TARGETS = [
    (
        "https://www.mdpi.com/2071-1050/16/2/670",
        "MDPI-Sustainability-2024-sustainable-earth-construction-materials-state-of-art-review.pdf",
    ),
    (
        "https://www.mdpi.com/2075-5309/16/8/1633",
        "MDPI-Buildings-compressed-stabilized-earth-blocks-prisma-systematic-review.pdf",
    ),
    (
        "https://www.mdpi.com/2075-5309/15/23/4362",
        "MDPI-Buildings-2025-carbon-conscious-compressed-stabilized-earth-blocks.pdf",
    ),
    (
        "https://www.mdpi.com/2075-5309/14/12/4034",
        "MDPI-Buildings-2024-rammed-earth-architecture-case-study.pdf",
    ),
]
MDPI_DIR = "methods/earthen-reviews"

# ---------------------------------------------------------------------------
# Group 3: Cal-Earth / SuperAdobe free structural test reports.
# Linked publicly from calearth.org "Resources for Builders".
# ---------------------------------------------------------------------------
# ICC-ES ESR-4126 = official code-compliance report for Cal-Earth SuperAdobe
# (bridges earthbag -> "allowed"). Scrape the real PDF link off the ICC-ES page,
# then try candidate direct CDN URLs as fallback.
CALEARTH_PAGES = [
    (
        "https://icc-es.org/report-listing/esr-4126/",
        "CalEarth-ICC-ES-ESR-4126-superadobe-evaluation-report.pdf",
        "4126",
        [
            "https://icc-es.org/wp-content/uploads/report-directory/ESR-4126.pdf",
            "https://cdn-v2.icc-es.org/wp-content/uploads/report-directory/ESR-4126.pdf",
        ],
    ),
]
CALEARTH_DIR = "methods/earthbag"

# Landing pages to warm up Cloudflare/Akamai clearance cookies per host.
WARMUP_URLS = [
    "https://www.mdpi.com/",
    "https://icc-es.org/",
]


def is_valid(path: pathlib.Path, pdf: bool) -> bool:
    """A file is valid if it's big enough and (for PDFs) starts with %PDF."""
    if not path.exists():
        return False
    floor = PDF_MIN_BYTES if pdf else HTML_MIN_BYTES
    if path.stat().st_size < floor:
        return False
    if pdf:
        with path.open("rb") as fh:
            return fh.read(5) == b"%PDF-"
    return True


def _looks_like_html(body: bytes) -> bool:
    head = body[:512].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") \
        or b"<head" in head or b"<title" in head


def _save_via_request(page, url: str, tmp: pathlib.Path) -> tuple[bool, str]:
    """Fetch bytes via the browser context's APIRequestContext. Reuses the
    Cloudflare clearance cookies banked during warm-up, and -- unlike page
    navigation -- returns the body of an inline-served PDF directly."""
    try:
        host = f"{urlparse(url).scheme}://{urlparse(url).netloc}/"
        resp = page.context.request.get(
            url, headers={"Referer": host, "Accept": "application/pdf,*/*"},
            timeout=90000,
        )
        if resp.status >= 400:
            return False, f"http {resp.status}"
        tmp.write_bytes(resp.body())
        return True, "request"
    except Exception as e:
        return False, f"request error: {repr(e)[:120]}"


def _save_via_download(page, url: str, tmp: pathlib.Path) -> tuple[bool, str]:
    """Drive Chromium so the PDF streams as a download (carries bot cookies)."""
    try:
        with page.expect_download(timeout=60000) as dl_info:
            try:
                page.goto(url, timeout=15000)
            except Exception:
                pass
        dl = dl_info.value
        dl.save_as(str(tmp))
        return True, "download"
    except Exception as e:
        return False, f"download error: {repr(e)[:120]}"


def _save_via_navigation(page, url: str, tmp: pathlib.Path) -> tuple[bool, str]:
    """Some hosts render the PDF inline instead of triggering a download. Grab
    the bytes from the navigation response."""
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=90000)
        if resp is None:
            return False, "no response"
        if resp.status >= 400:
            return False, f"http {resp.status}"
        body = resp.body()
        tmp.write_bytes(body)
        return True, "navigation"
    except Exception as e:
        return False, f"nav error: {repr(e)[:120]}"


def fetch(page, url: str, dest: pathlib.Path, pdf: bool) -> tuple[bool, str]:
    """Download one file by driving Chromium. Returns (ok, detail)."""
    if is_valid(dest, pdf):
        return True, f"skip (already on disk, {dest.stat().st_size // 1024} KB)"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.gettempdir()) / f"_bw_dl_{abs(hash(url))}.bin"

    def validate_and_store() -> tuple[bool, str]:
        if not tmp.exists():
            return False, "no bytes written"
        body = tmp.read_bytes()
        if pdf:
            if body[:5] != b"%PDF-":
                snippet = body[:24]
                tmp.unlink(missing_ok=True)
                if _looks_like_html(body):
                    return False, f"got HTML error page ({len(body)} bytes)"
                return False, f"not a PDF (starts {snippet!r}, {len(body)} bytes)"
            if len(body) < PDF_MIN_BYTES:
                tmp.unlink(missing_ok=True)
                return False, f"too small ({len(body)} bytes)"
        else:
            if len(body) < HTML_MIN_BYTES:
                tmp.unlink(missing_ok=True)
                return False, f"too small ({len(body)} bytes)"
        dest.write_bytes(body)
        tmp.unlink(missing_ok=True)
        return True, f"{len(body) // 1024} KB"

    def attempt() -> tuple[bool, str]:
        # 1) APIRequestContext (best for inline-served PDFs behind Cloudflare).
        ok0, how0 = _save_via_request(page, url, tmp)
        if ok0:
            good, detail = validate_and_store()
            if good:
                return True, detail
        # 2) Chromium download event (for attachment-served PDFs).
        ok, how = _save_via_download(page, url, tmp)
        if ok:
            good, detail = validate_and_store()
            if good:
                return True, detail
        # 3) Navigation response body.
        ok2, how2 = _save_via_navigation(page, url, tmp)
        if ok2:
            good, detail = validate_and_store()
            if good:
                return True, detail
            return False, detail
        return False, how0 if not ok0 else (how if not ok else how2)

    ok, detail = attempt()
    if not ok:
        # Re-warm cookies on the host landing page, then retry once.
        host = f"{urlparse(url).scheme}://{urlparse(url).netloc}/"
        try:
            page.goto(host, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(2500)
        except Exception:
            pass
        ok, detail = attempt()
    return ok, detail


def scrape_pdf_link(page, landing_url: str, must_match: str = "") -> str | None:
    """Open an AISC download landing page and return the first standard PDF
    anchor. must_match (e.g. '360') narrows to the right file when present."""
    try:
        page.goto(landing_url, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(2000)
    except Exception:
        return None
    try:
        anchors = page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => e.href)",
        )
    except Exception:
        return None
    pdfs = [h for h in anchors if h and h.lower().split("?")[0].endswith(".pdf")]
    # Prefer a globalassets standards PDF that matches the standard number.
    def score(h: str) -> int:
        s = 0
        low = h.lower()
        if "globalassets" in low:
            s += 2
        if must_match and must_match in low:
            s += 4
        if "standard" in low:
            s += 1
        return s
    pdfs.sort(key=score, reverse=True)
    return pdfs[0] if pdfs else None


def main():
    results: list[tuple[str, str, str]] = []  # (status, dest, detail)

    def record(ok: bool, dest: pathlib.Path, detail: str):
        status = "OK" if ok else "FAIL"
        results.append((status, str(dest), detail))
        print(f"[{status}] {dest.name}: {detail}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=UA, accept_downloads=True,
                                  viewport={"width": 1440, "height": 900})
        page = ctx.new_page()

        # Warm up: bank Cloudflare/Akamai clearance cookies per host.
        for w in WARMUP_URLS:
            try:
                page.goto(w, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(2000)
                print(f"[warmup] {w} -> ok")
            except Exception as e:
                print(f"[warmup] {w} -> {repr(e)[:80]}")

        # --- Group 1: AISC free standards ---
        print("\n=== AISC free standards ===")
        for landing, fname, candidates in AISC_TARGETS:
            dest = CORPORA / AISC_DIR / fname
            if is_valid(dest, pdf=True):
                record(True, dest, f"skip (already on disk, {dest.stat().st_size // 1024} KB)")
                continue
            num = re.search(r"(\d{3})-22", fname)
            must = num.group(1) if num else ""
            # Try scraping the real PDF anchor off the landing page first.
            pdf_url = scrape_pdf_link(page, landing, must_match=must)
            tried = []
            ok, detail = False, "no pdf link found on landing page"
            if pdf_url:
                tried.append(pdf_url)
                ok, detail = fetch(page, pdf_url, dest, pdf=True)
            # Fallback: known/candidate direct globalassets URLs.
            if not ok:
                for cand in candidates:
                    if cand in tried:
                        continue
                    tried.append(cand)
                    ok, detail = fetch(page, cand, dest, pdf=True)
                    if ok:
                        break
            record(ok, dest, detail if ok else f"{detail} (tried {len(tried)} url(s))")
            time.sleep(1)

        # --- Group 2: Cal-Earth SuperAdobe ICC-ES report (run first: high value) ---
        print("\n=== Cal-Earth / SuperAdobe ICC-ES report ===")
        for landing, fname, must, candidates in CALEARTH_PAGES:
            dest = CORPORA / CALEARTH_DIR / fname
            if is_valid(dest, pdf=True):
                record(True, dest, f"skip (already on disk, {dest.stat().st_size // 1024} KB)")
                continue
            pdf_url = scrape_pdf_link(page, landing, must_match=must)
            tried, ok, detail = [], False, "no pdf link found"
            if pdf_url:
                tried.append(pdf_url)
                ok, detail = fetch(page, pdf_url, dest, pdf=True)
            if not ok:
                for cand in candidates:
                    if cand in tried:
                        continue
                    tried.append(cand)
                    ok, detail = fetch(page, cand, dest, pdf=True)
                    if ok:
                        break
            record(ok, dest, detail if ok else f"{detail} (tried {len(tried)} url(s))")
            time.sleep(1)

        # --- Group 3: MDPI open-access review papers ---
        print("\n=== MDPI open-access earthen reviews ===")
        for article_url, fname in MDPI_TARGETS:
            dest = CORPORA / MDPI_DIR / fname
            pdf_url = article_url.rstrip("/") + "/pdf"
            ok, detail = fetch(page, pdf_url, dest, pdf=True)
            record(ok, dest, f"{detail}  <- {article_url}")
            time.sleep(1)

        browser.close()

    # Summary.
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for status, dest, detail in results:
        print(f"  [{status}] {dest}  ({detail})")
    n_ok = sum(1 for r in results if r[0] == "OK")
    print(f"\n{n_ok}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
