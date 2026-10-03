"""Assembling the demo: the real broker, API and dashboard wired to the simulated card,
driver, model servers and traffic, in a data folder of its own. Nothing is started and no
port is bound here (run.py does that); tests build a demo this way and drive it directly.
"""
from __future__ import annotations

import ipaddress
import pathlib
import random
import secrets
import shutil
from dataclasses import dataclass, field
from importlib.resources import as_file, files

from fastapi import FastAPI

from ..broker import Broker
from ..catalog import Catalog
from ..constants import Runner
from ..settings import Comfy, Driver, Inputs, Intervals, Server, Settings, Ui
from ..units import unit_ref
from . import content, web
from .backends import SimBackends
from .driver import SimDriver
from .gpu import COMFY_GROUP, SimGpu
from .traffic import TrafficGenerator
from .tuning import CATALOG_FILE, COMFY_PATH, INTERVALS, TOKEN_BYTES, SimCard, SimTimings, Traffic

DRIVER_KIND = "demo"   # shown in the broker.started event
DB, OUTPUTS, INPUTS = "broker.db", "outputs", "inputs"
# Wildcard binds answer on every address; links and the ComfyUI stand-ins use loopback.
LOOPBACK = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}  # noqa: S104 — compared, not bound


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
