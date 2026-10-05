"""`gpu-broker` console script: serve the API, check a config and catalog without running anything,
write a starter config (init), set everything up from what is installed (setup), or run the demo."""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Mapping

from . import drivers, execjob, settings
from .catalog import Catalog
from .constants import APP_NAME, ERR_SHORT, TOKEN_ENV, Runner
from .modelmap import effective
from .policy import POLICIES
from .resolve import lookup
from .templates import TEMPLATES
from .tuning import Scheduling

EXIT_OK, EXIT_PROBLEMS, EXIT_NO_TOKEN = 0, 1, 2
INIT_NEXT = """
Next:
  1. Edit {catalog}: your model servers' units, endpoints and sizes.
  2. gpu-broker -c {config} check
  3. sudo BROKER_TOKEN=$(openssl rand -hex 24) gpu-broker -c {config} serve"""


CONNECT_COMMANDS = frozenset({"connect", "disconnect", "clients"})   # handled by gpu_broker.connect.cli
MCP_COMMAND = "mcp"   # MCP over stdio, relayed to a broker (gpu_broker.mcp_server.stdio)


def mcp_stdio(argv: list[str], env: Mapping[str, str]) -> int:
    from . import mcp_server
    if not mcp_server.available():
        print(f"{APP_NAME} mcp needs the MCP SDK: pip install '{APP_NAME}[mcp]'", file=sys.stderr)
        return EXIT_PROBLEMS
    from .mcp_server.stdio import main as stdio_main
    return stdio_main(argv, env)


def rules(base: Scheduling, a: argparse.Namespace) -> Scheduling:
    """The replay's scheduler settings: the config's, with the command line's overrides."""
    over = {k: getattr(a, k) for k in ("max_wait_s", "evict_wait_s") if getattr(a, k) is not None}
    return dataclasses.replace(base, **over)


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in CONNECT_COMMANDS:
        from .connect.cli import main as connect_main
        return connect_main(args, env)
    if args and args[0] == MCP_COMMAND:   # needs neither a config file nor the broker's token
        return mcp_stdio(args[1:], env)
    ap = argparse.ArgumentParser(prog=APP_NAME, description="Share one GPU between LLM servers and ComfyUI. "
                                 "Also: `connect`, `disconnect`, `clients` point this machine's AI apps at a broker; "
                                 "`mcp` serves its tools over stdio to Claude, Codex and other MCP apps.")
    ap.add_argument("-c", "--config", help=f"config file (default: ${settings.CONFIG_ENV} or {settings.DEFAULT_PATH})")
    sub = ap.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve", help="run the HTTP API (default)")
    serve.add_argument("--host", help="bind address (default: server.host from the config)")
    serve.add_argument("--port", type=int, help="port (default: server.port from the config)")
    sub.add_parser("check", help="load config + catalog, build the driver, read the GPU once, print a summary; "
                                 "starts and stops nothing")
    initp = sub.add_parser("init", help="write a starter config.yaml and catalog.yaml, and create the folders they name")
    initp.add_argument("--dir", default=None, help="where to write them (default: /etc/gpu-broker)")
    initp.add_argument("--force", action="store_true", help="replace existing files")
    setupp = sub.add_parser("setup", help="find the GPU and model servers, write the config, start the service "
                                           "and open the dashboard")
    setupp.add_argument("--yes", action="store_true", help="never ask and never run in the foreground (for scripts)")
    setupp.add_argument("--dry-run", action="store_true", help="print what it would do; change nothing")
    setupp.add_argument("--dir", default=None, help="where the config, catalog and broker.env go "
                                                    "(default: /etc/gpu-broker, or ~/.config/gpu-broker without root)")
    demo = sub.add_parser("demo", help="try it without a GPU: the dashboard and API on a simulated card")
    demo.add_argument("--host", help="bind address (default: this machine only, 127.0.0.1)")
    demo.add_argument("--port", type=int, help="port (default: a free one, 8096 if it is free)")
    demo.add_argument("--quiet", action="store_true", help="no simulated traffic; the card idles until you send something")
    demo.add_argument("--no-browser", dest="browser", action="store_false", help="print the dashboard link without opening it")
    replay = sub.add_parser("replay", help="replay an events.jsonl through queue policies in simulated time; "
                                           "contacts nothing")
    replay.add_argument("events", help="the broker's events_jsonl file")
    replay.add_argument("--catalog", help="catalog the log was written under (default: catalog from the config)")
    replay.add_argument("--policy", action="append", choices=sorted(POLICIES), help="policy to replay (repeatable; default: all)")
    replay.add_argument("--json", action="store_true", help="machine-readable output")
    replay.add_argument("--max-wait-s", type=float, help="scheduler.max_wait_s to replay with (default: the config's)")
    replay.add_argument("--evict-wait-s", type=float, help="scheduler.evict_wait_s to replay with (default: the config's)")
    a = ap.parse_args(args)
    if a.cmd == "demo":   # needs no config file, token or host access; serve and check never load the demo
        from .demo import run
        return run.main(a.host, a.port, a.quiet, a.browser)
    if a.cmd == "replay" and a.catalog:   # a catalog is all a replay needs
        from .replay.main import run as replay_run
        return replay_run(a.events, a.catalog, a.policy or sorted(POLICIES), a.json, rules(Scheduling(), a))
    if a.cmd == "init":
        return init(a.dir, a.force)
    if a.cmd == "setup":
        from . import setup
        return setup.main(setup.Options(a.yes, a.dry_run, a.dir), env)
    cfg = settings.load(a.config, env)
    if a.cmd == "replay":
        from .replay.main import run as replay_run
        return replay_run(a.events, cfg.catalog, a.policy or sorted(POLICIES), a.json, rules(cfg.scheduler, a))
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


def init(dest: str | None, force: bool) -> int:
    from . import starter
    dest = dest or starter.DEFAULT_DIR
    try:
        lines = starter.run(dest, force)
    except PermissionError as e:
        print(f"{APP_NAME}: cannot write {e.filename}: permission denied (run init with sudo)",
              file=sys.stderr)
        return EXIT_PROBLEMS
    print("\n".join(lines))
    print(INIT_NEXT.format(catalog=os.path.join(dest, starter.CATALOG), config=os.path.join(dest, starter.CONFIG)))
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
    mapped = ", ".join(f"{p} -> {t}" for p, t in effective(cfg.model_map).items()) or "off"
    print(f"names:   {'built-in default' if cfg.model_map is None else 'model_map'}: {mapped}")
    print(f"hosted:  fallback {'on' if cfg.fallback.enabled else 'off'}")
    up = cfg.upstreams
    print("cloud:   " + ("; ".join(f"{p} -> {' -> '.join(c)}" for p, c in up.routes.items()) if up.enabled else "off"))
    problems += [f"upstreams: {n!r} is neither a provider nor a catalog model" for n in sorted(up.local_names())
                 if lookup(catalog.data, n) is None and catalog.variant(n) is None]
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
