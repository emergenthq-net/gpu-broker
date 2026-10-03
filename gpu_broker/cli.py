"""`gpu-broker` console script: serve the API, or check a config and catalog without running anything."""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping

from . import drivers, execjob, settings
from .catalog import Catalog
from .constants import APP_NAME, ERR_SHORT, TOKEN_ENV, Runner
from .templates import TEMPLATES

EXIT_OK, EXIT_PROBLEMS, EXIT_NO_TOKEN = 0, 1, 2


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser(prog=APP_NAME, description="Share one GPU between LLM servers and ComfyUI.")
    ap.add_argument("-c", "--config", help=f"config file (default: ${settings.CONFIG_ENV} or {settings.DEFAULT_PATH})")
    sub = ap.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve", help="run the HTTP API (default)")
    serve.add_argument("--host", help="bind address (default: server.host from the config)")
    serve.add_argument("--port", type=int, help="port (default: server.port from the config)")
    sub.add_parser("check", help="load config + catalog, build the driver, read the GPU once, print a summary; "
                                 "starts and stops nothing")
    demo = sub.add_parser("demo", help="try it without a GPU: the dashboard and API on a simulated card")
    demo.add_argument("--host", help="bind address (default: this machine only, 127.0.0.1)")
    demo.add_argument("--port", type=int, help="port (default: a free one, 8096 if it is free)")
    demo.add_argument("--quiet", action="store_true", help="no simulated traffic; the card idles until you send something")
    demo.add_argument("--no-browser", dest="browser", action="store_false", help="print the dashboard link without opening it")
    a = ap.parse_args(argv)
    if a.cmd == "demo":   # needs no config file, token or host access; serve and check never load the demo
        from .demo import run
        return run.main(a.host, a.port, a.quiet, a.browser)
    cfg = settings.load(a.config, env)
    if a.cmd == "check":
        return check(cfg)
    token = env.get(TOKEN_ENV, "")
    if not token:
        print(f"{APP_NAME}: {TOKEN_ENV} is not set; every API call would be refused", file=sys.stderr)
        return EXIT_NO_TOKEN
    import uvicorn

    from .broker import Broker
    from .web.app import create_app
    app = create_app(Broker(cfg, env), token)
    uvicorn.run(app, host=getattr(a, "host", None) or cfg.server.host, port=getattr(a, "port", None) or cfg.server.port,
                timeout_graceful_shutdown=cfg.server.graceful_shutdown_s)
    return EXIT_OK


def check(cfg: settings.Settings) -> int:
    catalog = Catalog(cfg.catalog)
    driver = drivers.build(cfg, catalog.units())
    problems = [f"{k}: unknown template {m['template']!r}" for k, m in catalog.models.items()
                if m.get("template") and m["template"] not in TEMPLATES]
    for k, m in catalog.models.items():
        if m.get("runner") == Runner.EXEC:
            try:
                execjob.check_timeout(k, m, driver.recipe_info(m["exec"]["recipe"]), cfg.timeouts)
            except execjob.TimeoutTooShort as e:
                problems.append(str(e))
            except drivers.DRIVER_ERRORS as e:
                print(f"exec:    {k}: recipe not checked here ({str(e)[:ERR_SHORT]})")
    print(f"config:  {cfg.source or '(defaults)'}\ncatalog: {cfg.catalog} ({len(catalog.models)} models)")
    print(f"driver:  {type(driver).__name__}, units allowed: {sorted(driver.allowed or [])}")
    print(f"comfy:   {cfg.comfy.url} (start unit: {cfg.comfy.unit.key if cfg.comfy.unit else None})")
    print(f"listen:  {cfg.server.host}:{cfg.server.port}")
    print(gpu_summary(driver))
    for p in problems:
        print("problem:", p)
    return EXIT_PROBLEMS if problems else EXIT_OK


def gpu_summary(driver: drivers.Driver) -> str:
    """Which GPU probe was chosen and one reading from it. Not finding a GPU is not a config
    problem (check may run where the card is not); `serve` starts anyway and retries."""
    try:
        probe = driver.gpu_probe()
        sample = getattr(driver, "sample_line", None)   # local drivers; the proxmox one reads over SSH
        return f"gpu:     {probe}" + (f"\nsample:  {sample()}" if sample else "")
    except drivers.DRIVER_ERRORS as e:
        return f"gpu:     none usable here ({str(e)[:ERR_SHORT]}); serve would start and keep retrying"


if __name__ == "__main__":
    sys.exit(main())
