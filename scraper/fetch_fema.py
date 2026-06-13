"""
Fetch FEMA PDFs that block plain curl (Akamai bot-wall -> 403).

FEMA needs NO login. The Akamai challenge is passed by driving a real Chromium
page. The browser's APIRequestContext (`ctx.request.get`) still 403s here, so we
download through actual page navigation: Chromium streams the PDF as a download
(`page.expect_download`), which carries the page's bot-clearance cookies. We
warm up on FEMA landing pages first so Akamai sets those cookies.

Resumable: any target file already on disk (and valid) is skipped.

Run:
    .venv\\Scripts\\python.exe scraper\\fetch_fema.py
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

MIN_BYTES = 50 * 1024  # 50 KB sanity floor

# Static single-file targets: (url, dest relative to corpora/)
STATIC_TARGETS = [
    (
        "https://www.fema.gov/sites/default/files/documents/"
        "fema-480_floodplain-management-study-guide_local-officials.pdf",
        "building-codes/federal/FEMA-480-floodplain-management-study-guide.pdf",
    ),
    (
        "https://www.fema.gov/sites/default/files/2020-08/fema499_2010_edition.pdf",
        "methods/hazard/FEMA-P-499-coastal-construction-guide.pdf",
    ),
]

# Technical-bulletins index page to enumerate.
TB_INDEX = ("https://www.fema.gov/emergency-managers/risk-management/"
            "building-science/national-flood-insurance-technical-bulletins")
TB_DIR = "building-codes/federal/nfip-technical-bulletins"

# FEMA landing pages to visit first so the browser banks Akamai cookies.
WARMUP_URLS = [
    "https://www.fema.gov/",
    TB_INDEX,
]


def is_valid_pdf(path: pathlib.Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size < MIN_BYTES:
        return False
    with path.open("rb") as fh:
        return fh.read(5) == b"%PDF-"


def fetch_pdf(ctx, url: str, dest: pathlib.Path, page) -> tuple[bool, str]:
    """Download one PDF by driving Chromium. Returns (ok, detail).

    Chromium treats a direct PDF URL as a download, so we capture it via
    page.expect_download. The download inherits the page's Akamai cookies,
    which a bare ctx.request.get does not, hence this approach beats 403s.
    """
    if is_valid_pdf(dest):
        return True, f"skip (already on disk, {dest.stat().st_size // 1024} KB)"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.gettempdir()) / f"_fema_dl_{abs(hash(url))}.pdf"

    def attempt() -> tuple[bool, str]:
        try:
            with page.expect_download(timeout=120000) as dl_info:
                # Navigation that starts a download raises; that's expected.
                try:
                    page.goto(url, timeout=8000)
                except Exception:
                    pass
            dl = dl_info.value
            dl.save_as(str(tmp))
        except Exception as e:
            return False, f"download error: {repr(e)[:120]}"

        body = tmp.read_bytes()
        if body[:5] != b"%PDF-":
            tmp.unlink(missing_ok=True)
            return False, f"not a PDF (starts with {body[:16]!r}, {len(body)} bytes)"
        if len(body) < MIN_BYTES:
            tmp.unlink(missing_ok=True)
            return False, f"too small ({len(body)} bytes)"
        dest.write_bytes(body)
        tmp.unlink(missing_ok=True)
        return True, f"{len(body) // 1024} KB"

    ok, detail = attempt()
    if not ok:
        # Re-warm cookies on the FEMA host landing page, then retry once.
        host = f"{urlparse(url).scheme}://{urlparse(url).netloc}/"
        try:
            page.goto(host, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(2500)
        except Exception:
            pass
        ok, detail = attempt()
    return ok, detail


def enumerate_tb_pdfs(ctx, page) -> list[tuple[str, str]]:
    """Scrape the technical-bulletins index for TB PDF links.

    Returns list of (absolute_url, descriptive_filename).
    """
    # Use a real navigation so JS-rendered links (FEMA is Drupal) are present.
    page.goto(TB_INDEX, wait_until="networkidle", timeout=90000)
    page.wait_for_timeout(2000)
    anchors = page.eval_on_selector_all(
        "a[href]",
        "els => els.map(e => ({href: e.href, text: e.textContent.trim()}))",
    )

    found: dict[str, str] = {}
    for a in anchors:
        href = a["href"]
        text = a["text"]
        if not href.lower().endswith(".pdf"):
            continue
        low = (href + " " + text).lower()
        # Restrict to technical-bulletin PDFs.
        if "technical" not in low and "_tb_" not in low and "/tb" not in low \
                and not re.search(r"\btb[\s_-]?\d", low) \
                and "bulletin" not in low:
            continue
        abs_url = urljoin(TB_INDEX, href)
        name = descriptive_tb_name(abs_url, text)
        # Keep first occurrence per URL.
        found.setdefault(abs_url, name)
    return [(u, n) for u, n in found.items()]


# Map detectable TB numbers to a human-readable topic slug. The index link
# texts are mostly the generic "download document", so we lean on the URL
# filename, which carries the real identifiers.
TB_TOPIC = {
    "0": "users-guide-to-technical-bulletins",
    "1": "openings-in-foundation-walls-and-walls-of-enclosures",
    "2": "flood-damage-resistant-materials-requirements",
    "3": "non-residential-floodproofing-requirements-and-certification",
    "4": "elevator-installation",
    "5": "free-of-obstruction-requirements",
    "6": "below-grade-parking-requirements",
    "7": "wet-floodproofing-requirements",
    "8": "corrosion-protection-of-metal-connectors-in-coastal-areas",
    "9": "design-and-construction-guidance-for-breakaway-walls",
    "10": "ensuring-that-structures-built-on-fill-perform",
    "11": "crawlspace-construction-for-buildings-in-special-flood-hazard-areas",
}


# Topic keywords that identify a TB number when the URL has no explicit digit
# (e.g. fema_flood-openings-technical-bulletin_20210607.pdf -> TB-1).
TB_TOPIC_KEYWORDS = {
    "1": ["flood-opening", "openings"],
    "4": ["elevator"],
    "5": ["free-of-obstruction", "free_of_obstruction"],
    "7": ["wet-floodproofing", "wet_floodproofing"],
    "8": ["corrosion"],
    "11": ["crawspance", "crawlspace", "crawl-space"],
}


def detect_tb_number(url: str, text: str) -> str | None:
    """Pull the technical-bulletin number out of the URL filename / link text.

    Filenames carry trailing date stamps (e.g. ..._20210607.pdf), so we only
    accept a number that is immediately bound to a "bulletin"/"tb" token and is
    1-2 digits NOT followed by more digits (which would be a date). Topic
    keywords are tried before the loose patterns to avoid date collisions.
    """
    base = urlparse(url).path.rsplit("/", 1)[-1].lower()
    blob = base + " " + text.lower()
    # Tight patterns: number must be 1-2 digits and not part of a longer run.
    patterns = [
        r"technical[-_]bulletin[-_](\d{1,2})(?!\d)",
        r"(?:^|[-_ ])tb[-_ ]?(\d{1,2})(?!\d)",
        r"bulletin[-_](\d{1,2})(?![\d])(?![-_]?\d{4,})",
    ]
    for pat in patterns:
        m = re.search(pat, blob)
        if m:
            n = int(m.group(1))
            if 0 <= n <= 11:
                return str(n)
    # Topic-keyword identification (handles undated/number-less filenames).
    for num, kws in TB_TOPIC_KEYWORDS.items():
        if any(kw in blob for kw in kws):
            return num
    return None


def descriptive_tb_name(url: str, text: str) -> str:
    """Build a descriptive filename for a technical bulletin PDF."""
    base = urlparse(url).path.rsplit("/", 1)[-1]
    tbnum = detect_tb_number(url, text)
    if tbnum is not None and tbnum in TB_TOPIC:
        return f"FEMA-NFIP-TB-{tbnum}-{TB_TOPIC[tbnum]}.pdf"
    if tbnum is not None:
        return f"FEMA-NFIP-TB-{tbnum}.pdf"
    # Non-numbered TB docs (e.g. the "update to NFIP technical bulletins" memo).
    stem = re.sub(r"\.pdf$", "", base, flags=re.I)
    stem = re.sub(r"^fema[_-]", "", stem, flags=re.I)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()
    slug = re.sub(r"-+", "-", slug)[:80].strip("-")
    return f"FEMA-NFIP-{slug}.pdf" if slug else base


def main():
    results: list[tuple[str, str, str]] = []  # (status, dest, detail)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=UA, accept_downloads=True,
                                   viewport={"width": 1440, "height": 900})
        page = ctx.new_page()

        # Warm up: visit FEMA pages so Akamai sets clearance cookies.
        for w in WARMUP_URLS:
            try:
                page.goto(w, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(2000)
                print(f"[warmup] {w} -> ok")
            except Exception as e:
                print(f"[warmup] {w} -> {e}")

        # 1 + 2: static targets.
        for url, rel in STATIC_TARGETS:
            dest = CORPORA / rel
            ok, detail = fetch_pdf(ctx, url, dest, page)
            results.append(("OK" if ok else "FAIL", str(dest), detail))
            print(f"[{'OK' if ok else 'FAIL'}] {dest.name}: {detail}")
            time.sleep(1)

        # 3: technical bulletins.
        print(f"\n[tb] enumerating {TB_INDEX}")
        try:
            tb_links = enumerate_tb_pdfs(ctx, page)
        except Exception as e:
            tb_links = []
            print(f"[tb] enumeration error: {e}")
        print(f"[tb] found {len(tb_links)} PDF link(s)")
        for url, name in sorted(tb_links, key=lambda x: x[1]):
            dest = CORPORA / TB_DIR / name
            ok, detail = fetch_pdf(ctx, url, dest, page)
            results.append(("OK" if ok else "FAIL", str(dest), detail))
            print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}  <- {url}")
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
    if any(r[0] == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
