"""Captura de la vista /asset/[ticker] (grafico + niveles + AI Insights)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT = Path("reports/asset_v1.png")


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.goto("http://localhost:3000/asset/NVDA", wait_until="networkidle",
                  timeout=60000)
        page.wait_for_timeout(2500)   # velas + insights
        page.screenshot(path=str(OUT), full_page=True)
        browser.close()
    print(f"captura guardada: {OUT}")


if __name__ == "__main__":
    main()
