"""`gpu-broker` console script: serve the API, or check a config and catalog without running anything."""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping

from . import drivers, settings
from .catalog import Catalog
from .constants import APP_NAME, TOKEN_ENV
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
    sub.add_parser("check", help="load config + catalog, build the driver, print a summary; runs nothing")
    a = ap.parse_args(argv)
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
    print(f"config:  {cfg.source or '(defaults)'}\ncatalog: {cfg.catalog} ({len(catalog.models)} models)")
    print(f"driver:  {type(driver).__name__}, units allowed: {sorted(driver.allowed or [])}")
    print(f"comfy:   {cfg.comfy.url} (start unit: {cfg.comfy.unit.key if cfg.comfy.unit else None})")
    print(f"listen:  {cfg.server.host}:{cfg.server.port}")
    for p in problems:
        print("problem:", p)
    return EXIT_PROBLEMS if problems else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
