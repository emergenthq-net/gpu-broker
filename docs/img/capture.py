"""Regenerate the README screenshots from a running `gpu-broker demo`.

    gpu-broker demo --no-browser                      # prints the dashboard link
    pip install playwright && playwright install chromium
    python docs/img/capture.py 'http://127.0.0.1:8096/dash#token=...' docs/img

Each picture is taken only once the dashboard itself shows the state it illustrates, so it is
what a viewer saw at that moment. Leave the demo running for a few minutes first (WARM_S) so
the live charts have some history. The event log is cropped to its newest rows by hand
(Pillow) to `events.png`; this script saves it whole as `events-full.png`.
"""
from __future__ import annotations

import re
import sys
import time

from playwright.sync_api import Locator, Page, sync_playwright

VIEWPORT = {"width": 1440, "height": 900}
WARM_S = 330          # chart history before the first picture
WAIT_S = 600          # longest wait for one state (a demo cycle is about 3.5 min)
POLL_S = 0.25
PAD_PX = 16

CHATTING = r"""() => $('res').textContent === 'qwen3-8b' && $('run').textContent === 'qwen3-8b'
  && /in parallel|^running/.test($('runsub').textContent) && /resident \(/.test($('idx').textContent)
  && !document.querySelectorAll('#q tr td:not(.mute)').length"""
VIDEO = r"""() => $('run').textContent === 'wan2.2-5b' && $('res').textContent === 'none'
  && /^running/.test($('runsub').textContent) && parseFloat($('vram').textContent) > 15
  && parseInt($('util').textContent.replace('util ', '')) > 80 && parseInt($('lUtil').textContent) > 80
  && parseFloat($('lVram').textContent) > 15 && !/resident \(/.test($('idx').textContent)
  && document.querySelectorAll('#q tr').length >= 2 && !document.querySelector('#q td.mute')"""
RESTORED = r"""() => [...document.querySelectorAll('#ev tr')].slice(0, 3)
  .some(r => /residency.idle_restore/.test(r.textContent))"""


def wait(pg: Page, js: str, what: str) -> None:
    end = time.time() + WAIT_S
    while time.time() < end:
        if pg.evaluate(js):
            print("captured:", what, flush=True)
            return
        time.sleep(POLL_S)
    raise SystemExit(f"timed out waiting for: {what}")


def card(pg: Page, title: str) -> Locator:
    return pg.locator(".card.wide").filter(has=pg.locator("h2", has_text=re.compile(f"^{title}")))


def main(url: str, out: str) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport=VIEWPORT, color_scheme="light")
        pg.goto(url)
        pg.wait_for_timeout(WARM_S * 1000)
        wait(pg, CHATTING, "chat model answering")
        pg.screenshot(path=f"{out}/overview.png")
        card(pg, "Models").screenshot(path=f"{out}/models.png")
        wait(pg, VIDEO, "video rendering, chats queued")
        box = card(pg, "Queue").bounding_box() or {"y": 0, "height": VIEWPORT["height"]}
        pg.set_viewport_size({"width": VIEWPORT["width"], "height": int(box["y"] + box["height"] + PAD_PX)})
        pg.screenshot(path=f"{out}/video-queue.png")
        pg.set_viewport_size(VIEWPORT)
        wait(pg, RESTORED, "chat model restored after idle")
        card(pg, "Events").screenshot(path=f"{out}/events-full.png")
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
