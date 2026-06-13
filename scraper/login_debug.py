"""Diagnose the ICC login form: dump URL + all inputs/buttons after load,
then attempt login and report login-status. Headed."""
import os, pathlib
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
USER, PASS = os.environ["ICC_USER"], os.environ["ICC_PASS"]
BASE = "https://codes.iccsafe.org"

def dump_inputs(page, tag):
    print(f"\n=== {tag}  url={page.url}")
    for f in page.frames:
        els = f.query_selector_all("input, button, a.btn, [role=button]")
        for e in els:
            try:
                if not e.is_visible():
                    continue
                info = e.evaluate(
                    "el=>({tag:el.tagName,type:el.type||'',name:el.name||'',id:el.id||'',ph:el.placeholder||'',txt:(el.innerText||el.value||'').slice(0,40)})"
                )
                print("   ", f.url[:50] if f != page.main_frame else "main", info)
            except Exception:
                pass

with sync_playwright() as p:
    b = p.chromium.launch(headless=False, slow_mo=80)
    ctx = b.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    page = ctx.new_page()
    page.goto(f"{BASE}/login", wait_until="networkidle")
    page.wait_for_timeout(3500)
    dump_inputs(page, "LOGIN PAGE LOADED")

    print("\n[*] Attempting login... fill email, click Continue/Login, then password.")
    # Step 1: email
    em = page.query_selector("input[type=email], input[name=email], input#email, input[name=_username], input[type=text]")
    if em:
        em.fill(USER); print("   filled email field:", em.evaluate("e=>e.name||e.id||e.type"))
    pw = page.query_selector("input[type=password]")
    if pw:
        pw.fill(PASS); print("   filled password field (single-step form)")
    # click submit
    btn = page.query_selector("button[type=submit], input[type=submit], button:has-text('Log In'), button:has-text('Login'), button:has-text('Continue'), button:has-text('Sign In')")
    if btn:
        print("   clicking:", btn.inner_text()[:30]); btn.click()
    page.wait_for_timeout(4000)
    dump_inputs(page, "AFTER FIRST SUBMIT")

    # maybe password appears now (2-step)
    pw2 = page.query_selector("input[type=password]")
    if pw2 and pw2.is_visible() and not pw:
        pw2.fill(PASS); print("   filled password (step 2)")
        b2 = page.query_selector("button[type=submit], button:has-text('Log In'), button:has-text('Continue'), button:has-text('Sign In')")
        if b2: b2.click()
        page.wait_for_timeout(5000)

    st = page.evaluate("""async()=>{const r=await fetch('/api/user/login-status',{headers:{'X-Requested-With':'XMLHttpRequest'}});return r.status+': '+(await r.text());}""")
    print("\n[*] login-status ->", st)
    print("[*] final url:", page.url)
    print("[*] Browser stays open 40s — finish login manually if needed.")
    page.wait_for_timeout(40000)
    b.close()
