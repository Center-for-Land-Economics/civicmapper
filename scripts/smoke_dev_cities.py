"""Headless smoke test: every city's dev page loads + renders data.

Detection strategy (no access to the viz's module-scoped map variable):
  1. Navigate to dev.civicmapper.org/app.html?city=<key> (the app is public —
     no login/token needed).
  2. Wait for the `#headline-stat-text` Land Value text to change from
     "Loading…" / blank to a real currency value, OR for the legend grid
     to populate — either signal means the dataset has rendered.
  3. Snapshot the failure cases (screenshot + console + network) for inspection.

Usage:
  python smoke_dev_cities.py                # run all 19, exit non-zero on failure
  python smoke_dev_cities.py --city nyc     # specific cities
  python smoke_dev_cities.py --screenshots  # always save screenshots
"""
import argparse, asyncio, json, os, re, sys, time
from pathlib import Path

ALL_CITIES = [
    'southbend','syracuse','spokane','rochester','bellingham','morgantown',
    'denver','fortcollins','cincinnati','cleveland','columbus','charlottesville',
    'ibx','stpaul','nyc','baltimore','albuquerque','pueblo','portland','houston',
    'austin','dallas','sanantonio','bcs','detroit','chicago',
    'tulsa','newportnews','richmond','olympia','seattle','vancouver','dmv','washington',
    'hartfordmetro',
    'tallinn','copenhagen',
]
DEV = "https://dev.civicmapper.org"
SHOTS_DIR = Path("/tmp/smoke_screenshots")

async def check_city(browser, city, *, timeout_s=30, always_screenshot=False):
    ctx = await browser.new_context(viewport={"width":1280,"height":800},
        user_agent="civicmapper-smoke/1.0 (+headless)")
    page = await ctx.new_page()
    failed = []
    console_errs = []
    page.on("response", lambda r: failed.append(f"HTTP {r.status} {r.url}") if r.status >= 400 else None)
    page.on("requestfailed", lambda req: failed.append(f"{req.method} {req.url} :: {req.failure}"))
    page.on("console", lambda m: console_errs.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: console_errs.append(f"PAGEERROR: {e}"))

    t0 = time.time()
    try:
        await page.goto(f"{DEV}/app.html?city={city}", wait_until="domcontentloaded", timeout=timeout_s*1000)
    except Exception as e:
        return {"city":city, "ok":False, "elapsed":time.time()-t0,
                "reason":f"nav: {e!s}", "failed":failed, "console":console_errs}

    # Strong signal: renderColorLegend (viz/src/main.ts) populates #legend with the
    # formatted low→high range labels ONLY after the dataset has loaded and stats are
    # computed — so "legend text contains a digit" == data rendered. (The legend was
    # redesigned; the old "Quantiles (p1–p99): ..." muted string no longer exists.)
    headline_loaded = False
    poll_until = t0 + timeout_s
    while time.time() < poll_until:
        try:
            done = await page.evaluate("""
                () => {
                    const legend = document.getElementById('legend');
                    const legendText = legend ? (legend.innerText || '') : '';
                    const hasRange = /\\d/.test(legendText);
                    const lo = document.getElementById('loadingOverlay');
                    const loadingHidden = !lo || !lo.classList.contains('show');
                    return {
                        hasRange,
                        legendText: hasRange ? legendText.replace(/\\s+/g, ' ').slice(0, 120) : null,
                        loadingHidden,
                    };
                }
            """)
        except Exception:
            done = {}
        if done.get("hasRange") and done.get("loadingHidden"):
            headline_loaded = True
            break
        await asyncio.sleep(0.5)

    elapsed = time.time() - t0

    # Filter noise
    def noise(s):
        if re.search(r"favicon|analytics|gtag|googletag|doubleclick|tile\.openstreetmap|basemaps\.cartocdn", s, re.I):
            return True
        if "ERR_ABORTED" in s and (".pmtiles" in s or ".mvt" in s or ".png" in s):
            return True
        return False
    real_failed = [r for r in failed if not noise(r)]
    real_errs = [e for e in console_errs if not re.search(r"AudioWorklet|fragment shader|webkit-text-size-adjust", e, re.I)]

    ok = headline_loaded and not real_failed and not real_errs
    result = {
        "city": city, "ok": ok, "elapsed": elapsed,
        "legend": done.get("legendText") if done else None,
        "loadingHidden": done.get("loadingHidden") if done else None,
        "failed": real_failed[:5], "console": real_errs[:5],
    }
    if not ok or always_screenshot:
        SHOTS_DIR.mkdir(exist_ok=True)
        path = SHOTS_DIR / f"{city}.png"
        try:
            await page.screenshot(path=str(path), full_page=False)
            result["screenshot"] = str(path)
        except Exception as e:
            result["screenshot_error"] = str(e)
    await ctx.close()
    return result

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", action="append")
    ap.add_argument("--screenshots", action="store_true", help="Save screenshots for every city (not just failures)")
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()
    targets = args.city or ALL_CITIES

    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True)
        results = []
        for city in targets:
            r = await check_city(b, city, timeout_s=args.timeout, always_screenshot=args.screenshots)
            flag = "✓" if r["ok"] else "✗"
            extra = ""
            if not r["ok"]:
                extra = f"  loadingHidden={r['loadingHidden']} fails={len(r['failed'])} errs={len(r['console'])}"
            q = (r.get("legend") or "").strip()
            print(f"{flag} {r['city']:<16} {r['elapsed']:>4.1f}s  legend={q[:50]!r}{extra}")
            results.append(r)
        await b.close()

    bad = [r for r in results if not r["ok"]]
    print()
    print("=" * 80)
    if bad:
        print(f"⚠️  {len(bad)} city(ies) failed:")
        for r in bad:
            print(f"\n--- {r['city']} ---")
            print(f"  loadingHidden: {r['loadingHidden']}")
            print(f"  legend: {r['legend']!r}")
            for f in r["failed"]:
                print(f"  fail: {f}")
            for e in r["console"]:
                print(f"  err:  {e}")
            if r.get("screenshot"):
                print(f"  screenshot: {r['screenshot']}")
    else:
        print(f"✅ All {len(results)} cities loaded successfully")
    sys.exit(0 if not bad else 1)

if __name__ == "__main__":
    asyncio.run(main())
