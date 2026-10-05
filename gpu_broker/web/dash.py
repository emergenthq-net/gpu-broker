"""Dashboard: the static page and its token-protected JSON (GPU, live metrics, stats, labels).

The page itself carries no data, so it is served without authentication; it asks for the
broker token once and keeps it in the viewer's own browser.
"""
from __future__ import annotations

import threading
import time
from http import HTTPStatus
from importlib.resources import files
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

from ..broker import Broker
from ..constants import ERR_SHORT
from ..drivers import DRIVER_ERRORS
from ..gpu.auto import PROBING
from ..metrics import job_metrics, summarize

STATIC = files(__package__) / "static"
PAGE = "dash.html"
SCRIPTS = frozenset({"dash", "live", "index", "imagejob", "connect", "upstreams"})   # the only files /dash/<name>.js serves
JS_MEDIA_TYPE = "text/javascript"
HOURS = 3600


def page_router() -> APIRouter:
    r = APIRouter()

    @r.get("/dash", response_class=HTMLResponse)
    def dash() -> str:
        return (STATIC / PAGE).read_text()

    @r.get("/dash/{name}.js")
    def script(name: str) -> Response:
        if name not in SCRIPTS:
            raise HTTPException(HTTPStatus.NOT_FOUND)
        return Response((STATIC / f"{name}.js").read_text(), media_type=JS_MEDIA_TYPE)

    return r


def data_router(broker: Broker) -> APIRouter:
    r = APIRouter()
    s = broker.settings
    ui = {"gpu_label": s.ui.gpu_label, "resident_label": s.ui.resident_label, "groups": s.ui.groups,
          "power_max_w": s.ui.power_max_w, "temp_max_c": s.ui.temp_max_c, "comfy_url": s.comfy.browser_url}
    cache: dict[str, Any] = {"t": -s.intervals.gpu_cache_s, "v": None}
    lock = threading.Lock()

    @r.get("/v1/gpu")
    def gpu() -> dict[str, Any]:
        """A reading at most `gpu_cache_s` old: it may be an SSH round trip, so the page must not
        cause one per poll per viewer. While the GPU reader is still being chosen (or the choice
        failed), say so at once: choosing may wait on a hung nvidia-smi, never under this lock.
        A failed choice is remembered, so reading it below raises at once."""
        if broker.driver.gpu_state().state == PROBING:
            return {"state": PROBING}
        with lock:
            if time.monotonic() - cache["t"] > s.intervals.gpu_cache_s:
                try:
                    used, total, util = broker.driver.gpu()
                    cache["v"] = {"used_mib": used, "total_mib": total, "util_pct": util,
                                  "probe": broker.driver.gpu_probe()}
                except DRIVER_ERRORS as e:
                    cache["v"] = {"error": str(e)[:ERR_SHORT]}
                cache["t"] = time.monotonic()
            result: dict[str, Any] = cache["v"]
            return result

    @r.get("/v1/ui")
    def ui_view() -> dict[str, Any]:
        """Site labels: GPU name, the default model's owner, VRAM groups, chart scales, ComfyUI URL."""
        return ui

    @r.get("/v1/metrics")
    def metrics(since: float = 0, window_s: float = s.limits.metrics_window_s) -> dict[str, Any]:
        """GPU samples newer than `since`, plus job latency/throughput over `window_s`."""
        rows = job_metrics(broker.store, time.time() - min(window_s, s.limits.metrics_window_s), s.limits.event_lookback_s)
        return {"gpu": broker.sampler.since(since), "gpu_error": broker.sampler.error,
                "jobs": rows[-s.limits.metrics_jobs:], "summary": summarize(rows)}

    @r.get("/v1/stats")
    def stats(hours: float = s.limits.stats_window_s / HOURS) -> dict[str, Any]:
        return broker.store.stats(time.time() - min(hours * HOURS, s.limits.stats_window_s))

    return r
