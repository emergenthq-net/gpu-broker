"""Model downloads, on their own thread so a multi-hour fetch never blocks the GPU queue.

A request for a catalog model that is not on disk yet (or for an unknown repository) queues
a download here; the job itself is served by a substitute meanwhile. When the files land,
the catalog entry is marked downloaded — and becomes ready if it already has a template.
"""
from __future__ import annotations

import queue
import threading

from .catalog import Catalog
from .constants import ERR_DETAIL, DownloadState
from .drivers import DRIVER_ERRORS, Driver
from .resolve import Download
from .store import Store


class Downloader:
    def __init__(self, store: Store, driver: Driver, catalog: Catalog, poll_s: float) -> None:
        self.store, self.driver, self.catalog, self.poll_s = store, driver, catalog, poll_s
        self._q: queue.Queue[tuple[Download, str | None]] = queue.Queue()

    def request(self, dl: Download, catalog_key: str | None) -> bool:
        """Queue a download unless the same slug is queued, running or done; True if queued."""
        new = self.store.upsert_download(dl.slug, dl.kind, dl.ref, catalog_key)
        if new:
            self._q.put((dl, catalog_key))
        return new

    def loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                dl, key = self._q.get(timeout=self.poll_s)
            except queue.Empty:
                continue
            self.store.set_download(dl.slug, DownloadState.RUNNING)
            try:
                r = self.driver.download(dl.kind, dl.ref, dl.slug, dl.include)
                if r.returncode != 0:
                    raise RuntimeError((r.stderr or r.stdout)[-ERR_DETAIL:])
            except DRIVER_ERRORS as e:
                self.store.set_download(dl.slug, DownloadState.FAILED, str(e)[:ERR_DETAIL])
                continue
            self.store.set_download(dl.slug, DownloadState.DONE)
            if key:
                self.catalog.mark_downloaded(key)
