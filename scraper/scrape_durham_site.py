"""scrape_durham_site.py -- pull Durham City-County permit/zoning guidance pages
from durhamnc.gov + the searchable UDO into the building-codes corpus.

These are the city's own published "do I need a permit / how" pages (Inspections
+ Planning). They state the LOCAL rules (permit triggers, the $40k / load-bearing
threshold, fence/pool/accessory-structure requirements) that the model and state
codes don't restate -- so the RAG can answer Durham questions from Durham's own
authority instead of inferring from state scope rules.

Saves one cleaned .html per page under
    corpora/building-codes/durham/inspections/<slug>.html
with <meta name="book"> / <meta name="chapter"> provenance, so the HTML
extractor labels citations nicely. Re-run is idempotent (overwrites).

Usage:  python scraper/scrape_durham_site.py
Then:   python -m rag.index build --corpus building-codes --confirm
"""
import os
import re
import sys
import time
import urllib.request

from bs4 import BeautifulSoup  # type: ignore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "corpora", "building-codes", "durham", "inspections")

# (url, slug, short label) -- the permit/zoning-relevant content pages.
PAGES = [
    ("https://www.durhamnc.gov/294/Accessory-Structures", "accessory-structures", "Accessory Structures"),
    ("https://www.durhamnc.gov/303/Fences-Walls", "fences-walls", "Fences & Walls"),
    ("https://www.durhamnc.gov/327/Pools", "pools", "Pools, Spas & Hot Tubs"),
    ("https://www.durhamnc.gov/293/City-County-Building-Safety", "building-safety", "City-County Building & Safety"),
    ("https://www.durhamnc.gov/4147/Permits-Licenses", "permits-licenses", "Permits & Licenses"),
    ("https://www.durhamnc.gov/828/Fire-Protection-System-Permits", "fire-protection-permits", "Fire Protection System Permits"),
    ("https://www.durhamnc.gov/3992/Small-Project-Plan-Review", "small-project-plan-review", "Small Project Plan Review"),
    ("https://www.durhamnc.gov/4149/Guidelines-by-Project-Type", "guidelines-by-project-type", "Guidelines by Project Type"),
    ("https://www.durhamnc.gov/325/Plans-Review-Requirements", "plans-review-requirements", "Plans Review Requirements"),
    ("https://www.durhamnc.gov/5009/Approval-Processes-and-Inspections", "approval-processes-inspections", "Approval Processes and Inspections"),
    ("https://www.durhamnc.gov/304/Inspection-Requirements", "inspection-requirements", "Inspection Requirements"),
    ("https://www.durhamnc.gov/305/Mobile-Homes-Manufactured-Housing", "mobile-homes", "Mobile Homes / Manufactured Housing"),
    ("https://www.durhamnc.gov/4877/Right-of-Way-Permits", "right-of-way-permits", "Right-of-Way Permits"),
    ("https://www.durhamnc.gov/635/Permits-and-Fees", "permits-and-fees", "Permits and Fees"),
    ("https://www.durhamnc.gov/295/Applications-Forms", "applications-forms", "Applications & Forms"),
    ("https://udo.durhamnc.gov/udo/5_04_Accessory%20Uses%20and%20Structures.htm", "udo-5.4-accessory-uses", "UDO Sec. 5.4 Accessory Uses and Structures"),
    ("https://udo.durhamnc.gov/udo/9_09_Fences%20and%20Walls.htm", "udo-9.9-fences-walls", "UDO Sec. 9.9 Fences and Walls"),
]

_UA = "Mozilla/5.0 (docrag corpus builder; authorized personal/local use)"
_DROP = ("script", "style", "noscript", "nav", "header", "footer", "form",
         "iframe", "svg", "button")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "replace")


def main_content(soup: BeautifulSoup):
    """Best-effort main content region for a CivicPlus / UDO page."""
    for sel in ("main", "#displayBody", ".pageContent", "#ContentPlaceHolder1",
                "div[role=main]", ".contentArea"):
        node = soup.select_one(sel)
        if node and len(node.get_text(" ", strip=True)) > 200:
            return node
    return soup.body or soup


def clean(raw: str, label: str, url: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(_DROP):
        tag.decompose()
    title = (soup.title.get_text(strip=True) if soup.title else label)
    node = main_content(soup)
    body_html = node.decode_contents() if hasattr(node, "decode_contents") else str(node)
    book = "Durham City-County (durhamnc.gov): " + label
    return (
        "<html><head>"
        '<meta charset="utf-8">'
        '<meta name="book" content="%s">'
        '<meta name="chapter" content="%s">'
        "<title>%s</title></head><body>"
        "<p>Source: %s (City of Durham, NC official site).</p>\n%s"
        "</body></html>" % (book, label, title, url, body_html)
    )


def run():
    os.makedirs(OUT, exist_ok=True)
    ok = 0
    for url, slug, label in PAGES:
        try:
            raw = fetch(url)
            doc = clean(raw, label, url)
            text_len = len(BeautifulSoup(doc, "html.parser").get_text(" ", strip=True))
            if text_len < 150:
                sys.stderr.write("[durham] thin content (%d chars) %s\n" % (text_len, url))
            path = os.path.join(OUT, slug + ".html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(doc)
            print("[durham] saved %-34s %6d chars  %s" % (slug, text_len, url))
            ok += 1
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("[durham] FAILED %s: %s\n" % (url, e))
        time.sleep(1.0)  # be polite
    print("[durham] %d/%d pages saved to %s" % (ok, len(PAGES), OUT))


if __name__ == "__main__":
    run()
