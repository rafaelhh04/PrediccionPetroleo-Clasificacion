"""Capture the README demo: dashboard GIF and screenshots of the dashboard and the API docs.

Needs a trained registry and a running stack, e.g.::

    make demo && uv run brent monitor drift      # models, report, explanations, drift report
    make -j2 serve dashboard                      # API on :8000, dashboard on :8501
    uv run --with playwright==1.56.0 python scripts/record_demo.py [--cdn-dir DIR]

Playwright drives a headless Chromium (``playwright install chromium`` once, unless a browser is
already available through ``PLAYWRIGHT_BROWSERS_PATH``). The frames are assembled with Pillow, a
matplotlib dependency, into ``docs/assets/demo.gif``. Nothing here runs in CI.
"""

import argparse
import io
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, Route, sync_playwright

TABS = ("Prices", "Statistical report", "Backtest", "Explanations", "Monitoring")
VIEWPORT = {"width": 1280, "height": 860}


def _serve_from(directory: Path) -> Callable[[Route], None]:
    """Answer CDN requests (the Swagger UI bundle) with the same-named file of ``directory``."""

    def handler(route: Route) -> None:
        route.fulfill(path=directory / route.request.url.rsplit("/", 1)[-1])

    return handler


def _settle(page: Page, seconds: float = 2.5) -> None:
    """Wait until Streamlit stops running and the charts have drawn."""
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=60_000)
    page.wait_for_timeout(seconds * 1000)


def _frame(page: Page) -> Image.Image:
    return Image.open(io.BytesIO(page.screenshot())).convert("RGB")


def record(dashboard_url: str, api_url: str, out_dir: Path, cdn_dir: Path | None = None) -> None:
    """Write ``demo.gif``, ``dashboard-report.png`` and ``api-docs.png`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        if cdn_dir is not None:
            page.route("https://cdn.jsdelivr.net/**", _serve_from(cdn_dir))

        page.goto(dashboard_url)
        page.wait_for_selector("h1")
        _settle(page, 4)
        frames.append(_frame(page))  # banner, prediction from the API
        for name in TABS:
            tab = page.get_by_role("tab", name=name)
            tab.click()
            _settle(page)
            tab.evaluate("el => el.scrollIntoView({block: 'start'})")
            page.mouse.wheel(0, -80)  # keep the tab bar below Streamlit's fixed header
            page.wait_for_timeout(800)
            frames.append(_frame(page))
            if name == "Statistical report":
                page.screenshot(path=out_dir / "dashboard-report.png")

        page.goto(f"{api_url}/docs")
        page.wait_for_selector(".opblock")
        page.wait_for_timeout(1500)
        page.screenshot(path=out_dir / "api-docs.png")
        browser.close()

    # A small palette and a reduced size keep the GIF light enough for a README.
    size = (960, round(960 * VIEWPORT["height"] / VIEWPORT["width"]))
    small = [f.resize(size, Image.Resampling.LANCZOS).quantize(colors=128) for f in frames]
    small[0].save(
        out_dir / "demo.gif",
        save_all=True,
        append_images=small[1:],
        duration=[3500] + [2800] * (len(small) - 1),
        loop=0,
        optimize=True,
    )


def main() -> None:
    """Parse the URLs and record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8501")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument(
        "--cdn-dir",
        type=Path,
        help="serve the Swagger UI assets from this folder (e.g. an unpacked swagger-ui-dist "
        "package: `npm pack swagger-ui-dist@5`) when the browser cannot reach the CDN",
    )
    args = parser.parse_args()
    record(args.dashboard_url, args.api_url, args.out_dir, args.cdn_dir)


if __name__ == "__main__":
    main()
