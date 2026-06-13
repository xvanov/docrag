"""
ICC Digital Codes recon: log in with Playwright, open the North Carolina
category page, and capture every /api/ XHR the SPA fires (request + response).

Goal: learn the exact request shapes for category-books / book-info (TOC) /
chapter-xml, and enumerate the 2024 NC building-code documentIds -- WITHOUT
guessing. Everything captured is dumped to scraper/recon/ for inspection.

Run (headed so you can solve any captcha):
    .venv\\Scripts\\python.exe scraper\\recon.py
"""
import json
import os
import pathlib
import sys
import time

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "scraper" / "recon"
OUT.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")
USER = os.environ.get("ICC_USER")
PASS = os.environ.get("ICC_PASS")
if not USER or not PASS:
    sys.exit("ICC_USER / ICC_PASS not set in .env")

BASE = "https://codes.iccsafe.org"
NC_URL = f"{BASE}/codes/united-states/north-carolina"

captured = []  # list of {url, method, status, req_body, resp_json|resp_text}


def on_response(resp):
    url = resp.url
    if "/api/" not in url and "/content/chapter/" not in url:
        return
    req = resp.request
    rec = {
        "url": url,
        "method": req.method,
        "status": resp.status,
        "req_body": req.post_data,
        "resp_headers_ct": resp.headers.get("content-type", ""),
    }
    try:
        ct = rec["resp_headers_ct"]
        if "json" in ct:
            rec["resp"] = resp.json()
        else:
            t = resp.text()
            rec["resp_text_len"] = len(t)
            rec["resp_text_head"] = t[:2000]
    except Exception as e:
        rec["resp_err"] = str(e)
    captured.append(rec)
    print(f"  [xhr] {req.method} {resp.status} {url.replace(BASE,'')[:90]}")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=50)
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            viewport={"width": 1440, "height": 900},
        )
        page = ctx.new_page()
        page.on("response", on_response)

        print("[*] Opening login page ...")
        page.goto(f"{BASE}/login", wait_until="domcontentloaded")
        page.wait_for_timeout(3000)

        # Generic selectors -- the SPA renders standard email/password inputs.
        def first(selectors):
            for s in selectors:
                el = page.query_selector(s)
                if el and el.is_visible():
                    return el
            return None

        email = first([
            'input[type=email]', 'input[name=_username]', 'input[name=email]',
            'input#username', 'input#email', 'input[name=username]',
        ])
        pwd = first([
            'input[type=password]', 'input[name=_password]', 'input#password',
        ])
        if not email or not pwd:
            print("[!] Could not find login inputs automatically.")
            print("    Log in MANUALLY in the open browser window, then press Enter here.")
            input()
        else:
            print("[*] Filling credentials ...")
            email.fill(USER)
            pwd.fill(PASS)
            btn = first([
                'button[type=submit]', 'button:has-text("Log In")',
                'button:has-text("Login")', 'button:has-text("Sign In")',
                'input[type=submit]',
            ])
            if btn:
                btn.click()
            else:
                pwd.press("Enter")
            print("[*] Submitted. Waiting for login to settle ...")
            page.wait_for_timeout(6000)

        # Confirm login status via the SPA's own endpoint.
        try:
            status = page.evaluate(
                """async () => {
                    const r = await fetch('/api/user/login-status', {headers:{'X-Requested-With':'XMLHttpRequest'}});
                    return {status:r.status, body: await r.text()};
                }"""
            )
            print(f"[*] login-status -> {status['status']}: {status['body'][:200]}")
            (OUT / "login_status.json").write_text(json.dumps(status, indent=2))
        except Exception as e:
            print(f"[!] login-status check failed: {e}")

        # Save cookies for the bulk scraper.
        cookies = ctx.cookies()
        (OUT / "cookies.json").write_text(json.dumps(cookies, indent=2))
        print(f"[*] Saved {len(cookies)} cookies -> recon/cookies.json")

        # Now load the NC category page and capture its API traffic.
        print(f"[*] Opening NC category page: {NC_URL}")
        captured.clear()
        page.goto(NC_URL, wait_until="networkidle")
        page.wait_for_timeout(4000)
        # scroll to trigger any lazy loads
        for _ in range(6):
            page.mouse.wheel(0, 2000)
            page.wait_for_timeout(600)
        page.wait_for_timeout(2000)

        (OUT / "nc_category_xhr.json").write_text(json.dumps(captured, indent=2))
        print(f"[*] Captured {len(captured)} NC-page XHRs -> recon/nc_category_xhr.json")

        # Dump the rendered NC page HTML too (book tiles -> documentIds/links).
        (OUT / "nc_category.html").write_text(page.content(), encoding="utf-8")

        print("\n[*] Recon done. Inspect scraper/recon/. Browser stays open 20s.")
        page.wait_for_timeout(20000)
        browser.close()


if __name__ == "__main__":
    main()
