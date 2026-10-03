"""`gpu-broker setup`: one command from a fresh install to a running broker, no questions asked.

1. Find the GPU and the model servers already on this machine (detect.py: read-only).
2. Write a config and catalog for them (generate.py), through `init`'s writer, keeping any
   file that already exists. Nothing found: the starter catalog, to edit.
3. Put a new API token in broker.env (mode 0600), or keep the one there.
4. With root or passwordless sudo and systemd: install and start the gpu-broker service.
   Otherwise: print the exact `serve` command, and (interactive only) run it in the foreground.
5. Run `check`, wait for /health, open the dashboard (never over SSH), offer `connect`.

`--dry-run` detects and prints every change it would make, changing nothing. Running it again
is safe: existing files, the token and the unit are kept.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from importlib.metadata import PackageNotFoundError, version

from .. import settings, starter
from ..browser import open_browser
from ..catalog import Catalog
from ..constants import APP_NAME, TOKEN_ENV
from . import detect, generate, host
from .generate import Layout

SYSTEM_CONFIG, SYSTEM_DATA, SYSTEM_LOGS = starter.DEFAULT_DIR, "/var/lib/gpu-broker", "/var/log/gpu-broker"
HEALTH_WAIT_S = 60
HEALTH_POLL_S = 1
KEEP_HINT = "setup never replaces it"
EPHEMERAL = ("/uv/archive-v", "/.cache/uv/")   # `uvx` runs from a cache that may be cleaned
LOOPBACK = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}  # noqa: S104 — compared, not bound
EXIT_OK, EXIT_PROBLEMS = 0, 1


@dataclass(frozen=True)
class Options:
    yes: bool = False          # never ask; never block in the foreground
    dry_run: bool = False
    dir: str | None = None     # where config.yaml, catalog.yaml and broker.env go


def _healthy(base: str) -> bool:
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as r:  # noqa: S310 — our own loopback URL
            return bool(r.status == HTTPStatus.OK)
    except (OSError, ValueError):
        return False


def _foreground(argv: list[str], env: Mapping[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(argv, env=dict(env))  # noqa: S603 — our own argv


def own_command() -> list[str]:
    """How to run this same installation again: its `gpu-broker` script when it sits next to
    this interpreter, else `python -m gpu_broker`."""
    script = pathlib.Path(sys.executable).with_name(APP_NAME)
    return [str(script)] if script.is_file() else [sys.executable, "-m", "gpu_broker"]


@dataclass
class Machine:
    """Everything setup reads from or does to the machine, replaceable in tests."""
    probes: Callable[[], detect.Probes] = detect.real
    euid: int = field(default_factory=os.geteuid)
    can_sudo: Callable[[], bool] = host.can_sudo
    systemd: Callable[[], bool] = host.systemd_running
    run: host.Run = host.run_cmd
    healthy: Callable[[str], bool] = _healthy
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    system_dirs: tuple[str, str, str] = (SYSTEM_CONFIG, SYSTEM_DATA, SYSTEM_LOGS)
    unit_path: str = host.UNIT_PATH
    opener: Callable[[str], object] = webbrowser.open
    ask: Callable[[str], str] = input
    interactive: bool = field(default_factory=lambda: sys.stdin.isatty() and sys.stdout.isatty())
    platform: str = sys.platform
    prefix: str = sys.prefix
    exe: list[str] = field(default_factory=own_command)
    which: Callable[[str], str | None] = shutil.which
    start: Callable[[list[str], Mapping[str, str]], subprocess.Popen[bytes]] = _foreground


def layout(opts: Options, env: Mapping[str, str], system: tuple[str, str, str] | None) -> Layout:
    """System folders when installing as root or a service; else the user's XDG folders."""
    if system:
        config, data, logs = system
        return Layout(opts.dir or config, data, logs, True)
    home = pathlib.Path.home()
    xdg = {k: env.get(f"XDG_{k}_HOME") or str(home / d) for k, d in
           (("CONFIG", ".config"), ("DATA", ".local/share"), ("STATE", ".local/state"))}
    return Layout(opts.dir or f"{xdg['CONFIG']}/{APP_NAME}", f"{xdg['DATA']}/{APP_NAME}",
                  f"{xdg['STATE']}/{APP_NAME}", False)


def report(f: detect.Findings, say: Callable[[str], None]) -> None:
    say(f"gpu      {f.gpu.label} ({f.gpu.probe})" if f.gpu else f"gpu      none found ({f.gpu_error})")
    for s in f.servers:
        names = ", ".join(m.name for m in s.models[:generate.MAX_PER_SERVER])
        runs = f"unit {s.unit}" if s.unit else f"container {s.container}" if s.container else "no unit found"
        say(f"found    {s.kind.label} at {s.url}" + (f" ({names})" if names else "") + f", {runs}")
        if not (s.unit or s.container):
            say(f"note     no systemd unit or container runs {s.kind.label}: the catalog names "
                f"'{generate.runs_on(s)}'. Create it, so the broker can stop the server to free the GPU.")
    for name in f.idle():
        say(f"idle     {name} looks like a model server but did not answer: start it, then add it to "
            "the catalog (docs/catalog.md)")
    if not generate.llm_servers(f):
        say("found    no LLM server answering on its usual port (llama.cpp :8080, vLLM :8000, Ollama :11434): "
            "writing the starter catalog, to edit")


def validated(cfg_text: str, cat_text: str) -> None:
    """Load both files as `serve` would, before writing anything."""
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "c.yaml").write_text(cfg_text)
        (pathlib.Path(d) / "k.yaml").write_text(cat_text)
        settings.load(f"{d}/c.yaml", env={})
        Catalog(f"{d}/k.yaml")


def write_files(h: host.Host, lay: Layout, cfg_text: str, cat_text: str, say: Callable[[str], None]) -> bool:
    """config.yaml, catalog.yaml and their folders, through init's writer. True if anything was written."""
    if h.dry:
        h.mkdir(lay.config_dir)
        for path in (lay.config, lay.catalog):
            if h.exists(path):
                say(f"kept     {path} (exists; {KEEP_HINT})")
            else:
                h.write(path, "")
        cfg = settings.load(lay.config, env={}) if h.exists(lay.config) else _loaded(cfg_text)
        for d in starter.data_dirs(cfg):
            h.mkdir(d)
        return True
    lines = starter.run(lay.config_dir, mkdir=h.mkdir, config=cfg_text, catalog=cat_text,
                        write=h.write, keep_hint=KEEP_HINT)
    for line in lines:
        say(line)
    return any(line.startswith("wrote") for line in lines)


def _loaded(cfg_text: str) -> settings.Settings:
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "c.yaml").write_text(cfg_text)
        return settings.load(f"{d}/c.yaml", env={})


def executable(m: Machine, h: host.Host, say: Callable[[str], None]) -> list[str] | None:
    """What the service runs. A `uvx` run is installed for good first (`uv tool install`), since
    uv may clean its cache; None if that is not possible."""
    if not any(p in m.prefix for p in EPHEMERAL):
        return m.exe
    if m.which("uv") is None:
        say("note     running from uvx's cache, and uv is not on PATH: install gpu-broker for good "
            "(pip install gpu-broker) and run setup again to install the service")
        return None
    try:
        pinned = f"{APP_NAME}=={version(APP_NAME)}"
    except PackageNotFoundError:
        pinned = APP_NAME
    if h.cmd(["uv", "tool", "install", pinned], as_root=False).returncode != 0:
        say("problem  uv tool install failed; install gpu-broker for good and run setup again")
        return None
    bin_dir = m.run(["uv", "tool", "dir", "--bin"]).stdout.strip() or str(pathlib.Path.home() / ".local/bin")
    return [f"{bin_dir}/{APP_NAME}"]


def base_url(cfg: settings.Settings) -> str:
    h = LOOPBACK.get(cfg.server.host, cfg.server.host)
    return f"http://{f'[{h}]' if ':' in h else h}:{cfg.server.port}"


def wait_healthy(m: Machine, base: str) -> bool:
    deadline = m.clock() + HEALTH_WAIT_S
    while m.clock() < deadline:
        if m.healthy(base):
            return True
        m.sleep(HEALTH_POLL_S)
    return False


def offer_connect(m: Machine, opts: Options, base: str, token: str, env: Mapping[str, str],
                  say: Callable[[str], None]) -> None:
    """`gpu-broker connect` points this machine's AI apps at the broker, where it is installed."""
    if importlib.util.find_spec(f"{__package__.rpartition('.')[0]}.connect") is None:
        return
    if opts.yes or not m.interactive:
        say(f"next     gpu-broker connect --url {base}   (points this machine's AI apps at the broker)")
        return
    if m.ask("Point this machine's AI apps (Claude Code, Codex, editors) at it now? [y/N] ").strip().lower() in ("y", "yes"):
        cli = importlib.import_module(f"{__package__.rpartition('.')[0]}.connect.cli")
        cli.main(["connect", "--url", base], {**env, TOKEN_ENV: token})


def main(opts: Options, env: Mapping[str, str] | None = None, m: Machine | None = None,
         say: Callable[[str], None] = print) -> int:
    env, m = os.environ if env is None else env, m or Machine()
    say(f"{APP_NAME} setup" + (": dry run, nothing will be changed" if opts.dry_run else ""))
    found = detect.find(m.probes())
    report(found, say)

    systemd = m.systemd()
    sudo = m.euid != 0 and systemd and m.can_sudo()
    service = systemd and (m.euid == 0 or sudo)
    lay = layout(opts, env, m.system_dirs if service or m.euid == 0 else None)
    h = host.Host(opts.dry_run, sudo, m.run, say)
    say(f"layout   {lay.config_dir} (config), {lay.data_dir} (data)"
        + ("" if service else "; no root, passwordless sudo or systemd: not installing a service"))
    if not service and m.euid != 0 and any(s.unit and not s.user_unit for s in found.servers):
        say("note     serve will run as you, and starting or stopping system units needs root: "
            "`sudo gpu-broker setup` installs it as a service instead")

    cfg_text = generate.text("config", generate.config(found, lay), lay)
    cat = generate.catalog(found)
    cat_text = generate.text("catalog", cat, lay) if cat else starter.starter(starter.CATALOG)
    validated(cfg_text, cat_text)
    try:
        changed = write_files(h, lay, cfg_text, cat_text, say)
        token, new = host.token(h, lay.env_file)
    except PermissionError as e:
        say(f"problem  {e}: run setup with sudo, or pass --dir to a folder you can write")
        return EXIT_PROBLEMS
    if opts.dry_run and new:
        say(f"token    a new one would go in {lay.env_file}, mode 600")
    else:
        say(f"token    {host.masked(token)} ({'new, ' if new else 'kept, '}in {lay.env_file}, mode 600)")
    if opts.dry_run:
        exe = m.exe if service else None
        if service:
            h.write(m.unit_path, "")
            h.cmd(["systemctl", "enable", "--now", host.UNIT_NAME])
        say(f"would    run check, wait for /health and open the dashboard{'' if exe else ' (after you start serve)'}")
        return EXIT_OK

    from .. import cli
    cfg = settings.load(lay.config, env={})
    if cli.check(cfg) != EXIT_OK:
        say(f"problem  `check` found problems above: fix {lay.config} or {lay.catalog}, then run setup again")
        return EXIT_PROBLEMS
    base = base_url(cfg)
    exe = executable(m, h, say) if service else m.exe
    serve_argv = [*(exe or m.exe), "-c", lay.config, "serve"]
    proc = None
    if service and exe:
        unit = host.unit_text(exe, lay.config, lay.env_file)
        old = h.read(m.unit_path)
        if old is None:
            h.write(m.unit_path, unit)
            h.cmd(["systemctl", "daemon-reload"])
            changed = True
        elif old != unit:
            say(f"kept     {m.unit_path} (exists and differs from what setup would write; {KEEP_HINT})")
        h.cmd(["systemctl", "enable", host.UNIT_NAME])
        r = h.cmd(["systemctl", "restart" if changed or new else "start", host.UNIT_NAME])
        if r.returncode != 0:
            say(f"problem  could not start {host.UNIT_NAME}: {(r.stderr or r.stdout).strip()} "
                f"(see journalctl -u {host.UNIT_NAME})")
            return EXIT_PROBLEMS
        say(f"service  {host.UNIT_NAME} enabled and started ({m.unit_path})")
    else:
        say(f"serve    set -a; . {lay.env_file}; set +a; {' '.join(serve_argv)}")
        if opts.yes or not m.interactive:
            return EXIT_OK
        say("starting it now in the foreground; Ctrl-C stops it")
        proc = m.start(serve_argv, {**env, TOKEN_ENV: token})

    if not wait_healthy(m, base):
        say(f"problem  {base}/health did not answer within {HEALTH_WAIT_S} s"
            + (f" (see journalctl -u {host.UNIT_NAME})" if proc is None else ""))
        return EXIT_PROBLEMS
    open_browser(f"{base}/dash#token={token}", env, m.platform, m.opener)
    say(f"open     {base}/dash   (sign in with the token in {lay.env_file})")
    offer_connect(m, opts, base, token, env, say)
    return proc.wait() if proc else EXIT_OK
