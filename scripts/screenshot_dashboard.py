"""Captura del dashboard Next.js con Playwright (entregable visual Sprint 4).

Uso:
    # con API (:8010) y frontend (:3000) corriendo:
    venv/Scripts/python scripts/screenshot_dashboard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT = Path("reports/dashboard_v1.png")


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto("http://localhost:3000/", wait_until="networkidle", timeout=45000)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT), full_page=True)
        browser.close()
    print(f"captura guardada: {OUT}")


if __name__ == "__main__":
    main()
