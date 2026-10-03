"""`gpu-broker demo`: wire the real broker and dashboard to the simulation and serve them.

`build` assembles everything without starting a thread or binding a port (tests use it);
`main` binds the port first, so nothing is printed, opened or started unless the demo really
owns it: a link to a port some other program holds (a real broker on 8095, say) would hand
that program the demo's token. Then it prints the link, opens it in a browser where there is
one, starts the broker and the traffic, and serves until Ctrl+C.
"""
from __future__ import annotations

import errno
import ipaddress
import os
import pathlib
import random
import secrets
import shutil
import signal
import socket
import sys
import tempfile
import webbrowser
from dataclasses import dataclass, field
from importlib.resources import as_file, files

from fastapi import FastAPI

from ..broker import Broker
from ..browser import open_browser
from ..catalog import Catalog
from ..constants import APP_NAME, Runner
from ..settings import Comfy, Driver, Inputs, Intervals, Server, Settings, Ui
from ..units import unit_ref
from . import content, web
from .backends import SimBackends
from .driver import SimDriver
from .gpu import COMFY_GROUP, SimGpu
from .traffic import TrafficGenerator
from .tuning import CATALOG_FILE, COMFY_PATH, DATA_PREFIX, DEMO_HOST, DEMO_PORT, INTERVALS, TOKEN_BYTES, SimCard, SimTimings, Traffic

DRIVER_KIND = "demo"   # shown in the broker.started event
DB, OUTPUTS, INPUTS = "broker.db", "outputs", "inputs"
LOG_LEVEL = "warning"  # keep the terminal to the banner; the dashboard shows what happens
BACKLOG = 128          # connections the bound port holds until the server starts accepting them
ANY_PORT = 0
EXIT_NO_PORT = 1
# Wildcard binds answer on every address; links and the ComfyUI stand-ins use loopback.
LOOPBACK = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}  # noqa: S104 — compared, not bound
PORT_TAKEN = "{app}: port {port} on {host} is in use (another gpu-broker?). Pick another --port, or leave it out."


@dataclass(frozen=True)
class Options:
    timings: SimTimings = field(default_factory=SimTimings)
    card: SimCard = field(default_factory=SimCard)
    traffic: Traffic = field(default_factory=Traffic)
    intervals: Intervals = INTERVALS
    seed: int | None = None   # fixed in tests, so the simulation repeats exactly


@dataclass(frozen=True)
class Demo:
    app: FastAPI
    broker: Broker
    traffic: TrafficGenerator
    url: str     # the broker: dashboard and API
    comfy: str   # the ComfyUI stand-ins (web.py), under a per-run secret path


def base_url(host: str, port: int) -> str:
    """http://host:port as a browser on this machine reaches the bind address."""
    host = LOOPBACK.get(host, host)
    try:
        bracket = isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address)
    except ValueError:
        bracket = False   # a host name
    return f"http://[{host}]:{port}" if bracket else f"http://{host}:{port}"


def settings(data: pathlib.Path, host: str, port: int, comfy: str, intervals: Intervals, card: SimCard) -> Settings:
    return Settings(catalog=str(data / CATALOG_FILE), db=str(data / DB), events_jsonl="", gpu_stream=True,
                    server=Server(host=host, port=port), driver=Driver(kind=DRIVER_KIND), intervals=intervals,
                    comfy=Comfy(url=comfy, public_url=comfy, unit=unit_ref(COMFY_GROUP), output_dir=str(data / OUTPUTS)),
                    inputs=Inputs(staging_dir=str(data / INPUTS)),
                    ui=Ui(gpu_label=content.GPU_LABEL, resident_label=content.RESIDENT_LABEL,
                          power_max_w=card.busy_w + card.jitter_w, groups=content.UI_GROUPS))


def memory(catalog: Catalog) -> dict[str, int]:
    """MiB each simulated unit or recipe holds while it runs, from the catalog's vram_mib."""
    out = {}
    for m in catalog.models.values():
        if "unit" in m:
            out[unit_ref(m["unit"]).name] = int(m.get("vram_mib", 0))
        if m.get("runner") == Runner.EXEC:
            out[m["exec"]["recipe"]] = int(m.get("vram_mib", 0))
    return out


def build(data: pathlib.Path, host: str, port: int, token: str, opt: Options | None = None) -> Demo:
    from ..web.app import create_app
    opt = opt or Options()
    with as_file(files(__package__) / CATALOG_FILE) as src:
        shutil.copy(src, data / CATALOG_FILE)   # the broker may add entries to its catalog: give it a copy
    url, key = base_url(host, port), secrets.token_urlsafe(TOKEN_BYTES)
    s = settings(data, host, port, url + COMFY_PATH + key, opt.intervals, opt.card)
    catalog, rng = Catalog(s.catalog), random.Random(opt.seed)  # noqa: S311 — simulated timings, not security
    gpu = SimGpu(opt.card, rng)
    resident = unit_ref(catalog.models[catalog.defaults["resident"]]["unit"]).name
    driver = SimDriver(gpu, memory(catalog), opt.timings, opt.card, s.comfy.output_dir, active=[resident])
    backends = SimBackends(catalog, driver, gpu, opt.timings, opt.card, s.comfy.output_dir, s.comfy.browser_url, rng)
    broker = Broker(s, env={}, driver=driver, backends=backends)
    app = create_app(broker, token, start=False)   # main() starts the broker before serving
    app.include_router(web.router(pathlib.Path(s.comfy.output_dir), COMFY_PATH, key))
    return Demo(app, broker, TrafficGenerator(broker, opt.traffic, rng), url, s.comfy.browser_url)


def bind(host: str, port: int | None) -> socket.socket:
    """A listening socket on host:port, or with no port on DEMO_PORT or else any free one.
    Raises OSError (EADDRINUSE) when an asked-for port is taken. No SO_REUSEADDR: with it, some
    systems let a second program bind a port another one is already serving."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    for p in (port,) if port is not None else (DEMO_PORT, ANY_PORT):
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.bind((host, p))
        except OSError as e:
            sock.close()
            if port is not None or e.errno != errno.EADDRINUSE:
                raise
            continue
        sock.listen(BACKLOG)
        return sock
    raise AssertionError("unreachable: binding port 0 picks a free port")


def serve(app: FastAPI, sock: socket.socket, graceful_s: int) -> None:
    import uvicorn
    uvicorn.Server(uvicorn.Config(app, log_level=LOG_LEVEL, timeout_graceful_shutdown=graceful_s)).run(sockets=[sock])


def _exit(signum: int, frame: object) -> None:
    raise SystemExit(0)


def main(host: str | None, port: int | None, quiet: bool, browser: bool = True) -> int:
    host = host or DEMO_HOST
    try:
        sock = bind(host, port)
    except OSError as e:
        taken = e.errno == errno.EADDRINUSE
        print(PORT_TAKEN.format(app=APP_NAME, host=host, port=port) if taken else f"{APP_NAME}: cannot listen on {host}: {e}",
              file=sys.stderr)
        return EXIT_NO_PORT
    port = sock.getsockname()[1]
    token = secrets.token_urlsafe(TOKEN_BYTES)
    with sock, tempfile.TemporaryDirectory(prefix=DATA_PREFIX, ignore_cleanup_errors=True) as d:
        demo = build(pathlib.Path(d), host, port, token)
        dash = f"{demo.url}/dash#token={token}"
        print(content.BANNER.format(url=demo.url, token=token,
                                    traffic=content.TRAFFIC_OFF if quiet else content.TRAFFIC_ON), flush=True)
        if browser:   # the port is listening: the page's first request waits until the server accepts it
            open_browser(dash, os.environ, sys.platform, webbrowser.open)
        # uvicorn re-raises SIGTERM once it has shut down; make that an exit that still runs the
        # cleanup below (the default would end the process and leave the data folder behind).
        signal.signal(signal.SIGTERM, _exit)
        demo.broker.start()
        if not quiet:
            demo.traffic.start()
        try:
            serve(demo.app, sock, demo.broker.settings.server.graceful_shutdown_s)
        finally:
            demo.traffic.close()
            demo.broker.stop()
    return 0
