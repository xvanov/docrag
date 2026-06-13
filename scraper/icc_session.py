"""
Authenticated ICC Digital Codes session.

Logs in once with Playwright (handles the SPA form + any cookie banner),
verifies via /api/user/login-status, then hands the session cookies to a
plain `requests.Session` so the bulk scrape is fast HTTP (no DOM walking).

Cookies are cached to scraper/recon/cookies.json and reused while valid.
"""
import json
import os
import pathlib
import sys

import requests
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
RECON = ROOT / "scraper" / "recon"
RECON.mkdir(parents=True, exist_ok=True)
COOKIE_CACHE = RECON / "cookies.json"

BASE = "https://codes.iccsafe.org"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

load_dotenv(ROOT / ".env")


def _login_status(sess: requests.Session) -> tuple[int, str]:
    r = sess.get(f"{BASE}/api/user/login-status",
                 headers={"X-Requested-With": "XMLHttpRequest"}, timeout=30)
    return r.status_code, r.text[:200]


def _session_from_cookies(cookies: list[dict]) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "X-Requested-With": "XMLHttpRequest",
                      "Referer": BASE + "/"})
    for c in cookies:
        s.cookies.set(c["name"], c["value"], domain=c.get("domain"),
                      path=c.get("path", "/"))
    return s


def _try_cached() -> requests.Session | None:
    if not COOKIE_CACHE.exists():
        return None
    try:
        s = _session_from_cookies(json.loads(COOKIE_CACHE.read_text()))
        code, _ = _login_status(s)
        if code == 200:
            return s
    except Exception:
        pass
    return None


def login(headless: bool = False, force: bool = False) -> requests.Session:
    """Return an authenticated requests.Session. Reuses cached cookies if valid."""
    if not force:
        s = _try_cached()
        if s:
            print("[auth] reused cached cookies -> logged in")
            return s

    user, pw = os.environ.get("ICC_USER"), os.environ.get("ICC_PASS")
    if not user or not pw:
        sys.exit("ICC_USER / ICC_PASS missing in .env")

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(headless=headless, slow_mo=40)
        ctx = b.new_context(user_agent=UA, viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        print("[auth] opening login page ...")
        page.goto(f"{BASE}/login", wait_until="networkidle")
        page.wait_for_timeout(2500)

        # Dismiss the Termly cookie banner if present (it can overlay the form).
        for label in ("Accept", "Decline"):
            try:
                btn = page.query_selector(f"button:has-text('{label}')")
                if btn and btn.is_visible():
                    btn.click(); page.wait_for_timeout(500); break
            except Exception:
                pass

        page.fill("#emailAddress", user)
        page.fill("#password", pw)
        # The form's submit is a <button type=button>Sign In</button>; the last
        # visible "Sign In" button on the page is the form one.
        signins = [b for b in page.query_selector_all("button:has-text('Sign In')")
                   if b.is_visible()]
        if signins:
            signins[-1].click()
        else:
            page.press("#password", "Enter")
        print("[auth] submitted, waiting ...")
        try:
            page.wait_for_url(lambda u: "/login" not in u, timeout=15000)
        except Exception:
            page.wait_for_timeout(6000)
        page.wait_for_timeout(3000)

        cookies = ctx.cookies()
        b.close()

    COOKIE_CACHE.write_text(json.dumps(cookies, indent=2))
    s = _session_from_cookies(cookies)
    code, body = _login_status(s)
    if code != 200:
        sys.exit(f"[auth] login FAILED: login-status {code}: {body}")
    print(f"[auth] logged in OK (login-status {code})")
    return s


if __name__ == "__main__":
    sess = login(headless=False, force="--force" in sys.argv)
    code, body = _login_status(sess)
    print("login-status:", code, body)
