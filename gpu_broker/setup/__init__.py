"""`gpu-broker setup`: one command from a fresh install to a running broker, no questions asked.

1. Find the GPU and the model servers already on this machine (detect.py: read-only).
2. Decide how the broker will run (`plan`):
   - model servers that are `systemctl --user` units: a user service, run as you, lingering;
   - otherwise, with root or passwordless sudo and systemd: a system service, never as root
     (service.py says who it runs as and refuses code others could change);
   - otherwise: print the exact `serve` command, and (interactive only) run it in the foreground.
3. Write a config and catalog for what was found (generate.py), through `init`'s writer,
   keeping any file that already exists. Nothing found: the starter catalog, to edit.
4. Put a new API token in broker.env (mode 0600), or keep the one there.
5. Run `check`, install and start the service, wait for /health, open the dashboard (never
   over SSH), offer `connect`.

`--dry-run` takes every one of these steps and branches, reading what is there, and only
skips the commands and writes (it prints them instead). Running setup again is safe:
existing files, the token, the unit and the sudoers rule are kept.
"""
from __future__ import annotations

import collections
import dataclasses
import importlib
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol

from .. import settings, starter
from ..browser import open_browser
from ..catalog import Catalog
from ..constants import APP_NAME, TOKEN_ENV
from . import detect, generate, host, service
from .generate import Layout
from .service import Account

SYSTEM_CONFIG, SYSTEM_DATA, SYSTEM_LOGS = starter.DEFAULT_DIR, "/var/lib/gpu-broker", "/var/log/gpu-broker"
HEALTH_WAIT_S = 60
HEALTH_POLL_S = 1
STOP_WAIT_S = 15                # serve's own shutdown takes server.graceful_shutdown_s (10 s)
TAIL_LINES = 20                 # of a foreground serve's stderr, shown when it exits early
PROBLEMS_SHOWN = 8              # install problems listed before "and N more"
KEEP_HINT = "setup never replaces it"
EPHEMERAL = ("/uv/archive-v", "/.cache/uv/")   # `uvx` runs from a cache that may be cleaned
LOOPBACK = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}  # noqa: S104 — compared, not bound
NOLOGIN = ("/usr/sbin/nologin", "/sbin/nologin")
EXIT_OK, EXIT_PROBLEMS = 0, 1
SYSTEM, USER, FOREGROUND = "system", "user", "foreground"


@dataclass(frozen=True)
class Options:
    yes: bool = False          # never ask; never block in the foreground
    dry_run: bool = False
    dir: str | None = None     # where config.yaml and broker.env go


class Child(Protocol):
    def poll(self) -> int | None: ...
    def wait(self) -> int: ...
    def stop(self) -> None: ...
    def tail(self) -> list[str]: ...


class Foreground:
    """`serve` run in the foreground: its stderr is passed through and the last lines kept, so an
    early exit can be reported; `stop` never leaves it running."""

    def __init__(self, argv: list[str], env: Mapping[str, str]) -> None:
        self.p = subprocess.Popen(argv, env=dict(env), stderr=subprocess.PIPE, text=True, errors="replace")  # noqa: S603 — our own argv
        self.lines: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()

    def _pump(self) -> None:
        for line in self.p.stderr or ():
            sys.stderr.write(line)
            self.lines.append(line.rstrip("\n"))

    def poll(self) -> int | None:
        return self.p.poll()

    def wait(self) -> int:
        return self.p.wait()

    def stop(self) -> None:
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.pump.join(timeout=1)

    def tail(self) -> list[str]:
        self.pump.join(timeout=1)
        return list(self.lines)


def _healthy(base: str) -> bool:
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as r:  # noqa: S310 — our own loopback URL
            return bool(r.status == HTTPStatus.OK)
    except (OSError, ValueError):
        return False


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
    sudoers_path: str = service.SUDOERS
    opener: Callable[[str], object] = webbrowser.open
    ask: Callable[[str], str] = input
    interactive: bool = field(default_factory=lambda: sys.stdin.isatty() and sys.stdout.isatty())
    platform: str = sys.platform
    prefix: str = sys.prefix
    exe: list[str] = field(default_factory=own_command)
    which: Callable[[str], str | None] = shutil.which
    start: Callable[[list[str], Mapping[str, str]], Child] = Foreground
    account: Callable[[str | int], Account | None] = service.account
    group_exists: Callable[[str], bool] = service.group_exists
    private_group: Callable[[int, int], bool] = service.private_group
    lstat: Callable[[str], os.stat_result] = os.lstat
    walk: Callable[[str], Iterable[str]] = service.walk


@dataclass(frozen=True)
class Plan:
    mode: str                  # SYSTEM, USER or FOREGROUND
    sudo: bool = False         # setup makes its system changes through `sudo -n`
    unit_sudo: bool = False    # the service starts and stops system units through sudo (driver.sudo)


def plan(m: Machine, f: detect.Findings) -> Plan | str:
    """How the broker will run, or why setup cannot install anything that would work."""
    units = [s for s in f.servers if s.unit]
    if any(s.user_unit for s in units) and not all(s.user_unit for s in units):
        return ("the model servers found are a mix of system and `systemctl --user` units, and one broker "
                "controls only one kind: move them to the same kind and run setup again")
    systemd = m.systemd()
    if generate.user_units(f) and systemd:
        if m.euid == 0:
            return ("the model servers found are `systemctl --user` units: run setup as the user they belong "
                    "to, without sudo, to install the broker as that user's service")
        return Plan(USER)
    sudo = m.euid != 0 and systemd and m.can_sudo()
    if not (systemd and (m.euid == 0 or sudo)):
        return Plan(FOREGROUND)
    unit_sudo = generate.driver_kind(f) == "systemd"
    if unit_sudo and not (m.which("sudo") and m.which("visudo")):
        return ("the service runs as an ordinary account and starts and stops the model-server units "
                "through sudo, which is not installed: install sudo and run setup again")
    return Plan(SYSTEM, sudo, unit_sudo)


def layout(opts: Options, env: Mapping[str, str], system: tuple[str, str, str] | None) -> Layout:
    """System folders when installing as root or a service; else the user's XDG folders."""
    if system:
        config, data, logs = system
        return Layout(opts.dir or config, data, logs, True)
    xdg = _xdg(env)
    return Layout(opts.dir or f"{xdg['CONFIG']}/{APP_NAME}", f"{xdg['DATA']}/{APP_NAME}",
                  f"{xdg['STATE']}/{APP_NAME}", False)


def _xdg(env: Mapping[str, str]) -> dict[str, str]:
    home = pathlib.Path.home()
    return {k: env.get(f"XDG_{k}_HOME") or str(home / d) for k, d in
            (("CONFIG", ".config"), ("DATA", ".local/share"), ("STATE", ".local/state"))}


def report(f: detect.Findings, say: Callable[[str], None]) -> None:
    say(f"gpu      {f.gpu.label} ({f.gpu.probe})" if f.gpu else f"gpu      none found ({f.gpu_error})")
    for s in f.servers:
        names = ", ".join(m.name for m in s.models[:generate.MAX_PER_SERVER])
        runs = f"unit {s.unit}" if s.unit else f"container {s.container}" if s.container else "no unit found"
        say(f"found    {s.kind.label} at {s.url}" + (f" ({names})" if names else "") + f", {runs}")
        if s.rivals:
            say(f"note     {', '.join((s.unit or '', *s.rivals))} could each run {s.kind.label}; nothing shows "
                f"which one serves {s.url}, so the catalog names {s.unit}. If that is wrong, change `unit:` "
                "in the catalog.")
        elif s.unit and s.unit_inactive:
            say(f"note     unit {s.unit} is not running, yet something answers at {s.url}: check that "
                f"{s.unit} is what serves it before relying on the broker to stop it.")
        if not (s.unit or s.container):
            say(f"note     no systemd unit or container runs {s.kind.label}: the catalog names "
                f"'{generate.runs_on(s)}'. Create it, so the broker can stop the server to free the GPU.")
    for name in f.idle():
        say(f"idle     {name} looks like a model server but did not answer: start it, then add it to "
            "the catalog (docs/catalog.md)")
    if not generate.llm_servers(f):
        say("found    no LLM server answering on its usual port (llama.cpp :8080, vLLM :8000, Ollama :11434): "
            "writing the starter catalog, to edit")


def _settings_from(text: str) -> settings.Settings:
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "c.yaml").write_text(text)
        return settings.load(f"{d}/c.yaml", env={})


def validated(cfg_text: str, cat_text: str) -> None:
    """Load both files as `serve` would, before writing anything."""
    _settings_from(cfg_text)
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "k.yaml").write_text(cat_text)
        Catalog(f"{d}/k.yaml")


def _load(path: str, text: str) -> settings.Settings:
    """The config at `path`, or the text setup would have written there (in a dry run)."""
    return settings.load(path, env={}) if os.path.exists(path) else _settings_from(text)


def write_files(h: host.Host, lay: Layout, cfg_text: str, cat_text: str, owner: str | None,
                say: Callable[[str], None]) -> bool:
    """config.yaml, catalog.yaml and their folders, through init's writer. The catalog and the data
    folders belong to `owner`, the service account. True if anything was (or would be) written."""
    lines = starter.run(lay.config_dir, mkdir=h.mkdir, config=cfg_text, catalog=cat_text, write=h.write,
                        keep_hint=KEEP_HINT, exists=h.exists, catalog_dir=lay.catalog_dir,
                        data_mkdir=lambda d: h.mkdir(d, owner),
                        write_catalog=lambda p, t: h.write(p, t, host.PUBLIC, owner), load=_load)
    for line in lines:
        if not (h.dry and line.startswith(("wrote", "folder"))):   # the host already said "would ..."
            say(line)
    return any(line.startswith("wrote") for line in lines)


def executable(m: Machine, h: host.Host, say: Callable[[str], None]) -> list[str] | None:
    """What the service runs. A `uvx` run is installed for good first (`uv tool install`), since
    uv may clean its cache; None if that is not possible."""
    if not any(p in m.prefix for p in EPHEMERAL):
        return m.exe
    if m.which("uv") is None:
        say("problem  running from uvx's cache, and uv is not on PATH: install gpu-broker for good "
            "(pip install gpu-broker) and run setup again to install the service")
        return None
    try:
        pinned = f"{APP_NAME}=={version(APP_NAME)}"
    except PackageNotFoundError:
        pinned = APP_NAME
    r = h.cmd(["uv", "tool", "install", pinned], as_root=False, timeout=host.INSTALL_TIMEOUT_S)
    if r.returncode != 0:
        say(f"problem  uv tool install failed ({(r.stderr or r.stdout).strip()[-300:]}); "
            "install gpu-broker for good and run setup again")
        return None
    bin_dir = h.query(["uv", "tool", "dir", "--bin"]).stdout.strip() or str(pathlib.Path.home() / ".local/bin")
    return [f"{bin_dir}/{APP_NAME}"]


def service_account(m: Machine, h: host.Host, p: Plan, lay: Layout, inst: service.Install,
                    say: Callable[[str], None]) -> Account | None:
    """Who the service runs as: you (user service); the installation's owner; or, when root owns
    it (or it is not installed yet and you are root), the dedicated account, created if needed."""
    if p.mode == USER:
        return m.account(m.euid)
    uid = service.owner_uid(inst, m.lstat)
    uid = m.euid if uid is None else uid
    if uid != 0:
        return m.account(uid)
    if acct := m.account(service.DEDICATED):
        return acct
    shell = next((s for s in NOLOGIN if os.path.exists(s)), "/bin/false")
    r = h.cmd(["useradd", "--system", "--user-group", "--home-dir", lay.data_dir, "--no-create-home",
               "--shell", shell, service.DEDICATED])
    if r.returncode != 0:
        say(f"problem  could not create the {service.DEDICATED} account: {(r.stderr or r.stdout).strip()}")
        return None
    return m.account(service.DEDICATED) or (Account(service.DEDICATED, -1, -1, lay.data_dir) if h.dry else None)


def safe_to_run(m: Machine, acct: Account, inst: service.Install, say: Callable[[str], None]) -> bool:
    """The service account can run the installation, and nobody but it and root can change it."""
    if acct.uid < 0 or not os.path.exists(inst.exe):
        say(f"would    check that only root or {acct.name} can change {inst.exe} and what it runs")
        return True
    if inst.site is None:
        say(f"problem  could not find where {inst.exe} is installed (its Python could not import {APP_NAME})")
        return False
    found = service.Checker(acct, m.lstat, m.private_group).problems(inst, m.walk)
    if not found:
        return True
    say(f"problem  the service would run as {acct.name}, but {len(found)} path(s) in what it runs "
        f"are not safe for that:")
    for line in found[:PROBLEMS_SHOWN]:
        say(f"         {line}")
    if len(found) > PROBLEMS_SHOWN:
        say(f"         and {len(found) - PROBLEMS_SHOWN} more")
    say("         install gpu-broker where only root or that account can change it and it can read it "
        "(e.g. a venv in /opt/gpu-broker), then run setup again")
    return False


def base_url(cfg: settings.Settings) -> str:
    h = LOOPBACK.get(cfg.server.host, cfg.server.host)
    return f"http://{f'[{h}]' if ':' in h else h}:{cfg.server.port}"


def wait_healthy(m: Machine, base: str, child: Child | None = None) -> str | None:
    """None once /health answers; else what went wrong. A foreground serve that exits stops the
    wait at once."""
    deadline = m.clock() + HEALTH_WAIT_S
    while m.clock() < deadline:
        if child is not None and (code := child.poll()) is not None:
            tail = "\n".join(f"         {line}" for line in child.tail())
            return f"serve exited with code {code} before {base}/health answered" + (f":\n{tail}" if tail else "")
        if m.healthy(base):
            return None
        m.sleep(HEALTH_POLL_S)
    return f"{base}/health did not answer within {HEALTH_WAIT_S} s"


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


def _write_new(h: host.Host, path: str, text: str, mode: int = host.PUBLIC) -> bool | None:
    """Write `path` unless something is there: True if written, False if kept (it differs), None
    if it is already exactly this."""
    old = h.read(path)
    if old is None and not h.exists(path):
        h.write(path, text, mode)
        return True
    return None if old == text else False


def install_sudoers(m: Machine, h: host.Host, acct: Account, units: list[str], say: Callable[[str], None]) -> bool:
    """The sudoers rule for the units, checked by visudo before it is put in place."""
    text, skipped = service.sudoers(acct.name, m.which("systemctl") or "/usr/bin/systemctl", units)
    for u in skipped:
        say(f"note     unit {u!r} has characters a sudoers rule would need escaped: add it to {m.sudoers_path} by hand")
    path = m.sudoers_path
    old = h.read(path)
    if old is not None or h.exists(path):
        if old != text:
            say(f"kept     {path} (exists and differs from what setup would write; {KEEP_HINT}): it must let "
                f"{acct.name} run `systemctl start|stop|is-active -- <unit>` for the catalog's units")
        return True
    staged = path + ".new"                    # sudo ignores files in sudoers.d whose names contain a dot
    h.write(staged, text, service.SUDOERS_MODE)
    r = h.cmd(["visudo", "-c", "-q", "-f", staged])
    if r.returncode != 0:
        h.cmd(["rm", "-f", "--", staged])
        say(f"problem  visudo rejected the sudoers rule: {(r.stderr or r.stdout).strip()}")
        return False
    return h.cmd(["mv", "-fT", "--", staged, path]).returncode == 0


def install_service(m: Machine, h: host.Host, p: Plan, lay: Layout, exe: list[str], acct: Account,
                    cfg: settings.Settings, env: Mapping[str, str], restart: bool,
                    say: Callable[[str], None]) -> bool:
    """Write the unit (and the sudoers rule), enable it and (re)start it. False on a problem."""
    as_root = p.mode == SYSTEM
    ctl = ["systemctl"] if as_root else ["systemctl", "--user"]
    if as_root:
        groups = service.supplementary(m.group_exists, cfg.driver.kind == "docker")
        unit, path = service.system_unit(exe, lay.config, lay.env_file, acct.name, groups), m.unit_path
    else:
        unit = service.user_unit(exe, lay.config, lay.env_file)
        path = f"{_xdg(env)['CONFIG']}/systemd/user/{host.UNIT_NAME}.service"
        h.mkdir(os.path.dirname(path))
    if p.unit_sudo:
        catalog = Catalog(cfg.catalog)
        units = [u.name for u in catalog.units()] + ([cfg.comfy.unit.name] if cfg.comfy.unit else [])
        if not install_sudoers(m, h, acct, units, say):
            return False
    wrote = _write_new(h, path, unit)
    if wrote:
        h.cmd([*ctl, "daemon-reload"], as_root=as_root)
    elif wrote is False:
        say(f"kept     {path} (exists and differs from what setup would write; {KEEP_HINT})")
    h.cmd([*ctl, "enable", host.UNIT_NAME], as_root=as_root)
    r = h.cmd([*ctl, "restart" if restart or wrote else "start", host.UNIT_NAME], as_root=as_root)
    journal = f"journalctl {'' if as_root else '--user '}-u {host.UNIT_NAME}"
    if r.returncode != 0:
        say(f"problem  could not start {host.UNIT_NAME}: {(r.stderr or r.stdout).strip()} (see {journal})")
        return False
    if not as_root:
        linger = h.query(["loginctl", "show-user", acct.name, "--property=Linger", "--value"]).stdout.strip()
        if linger != "yes" and h.cmd(["loginctl", "enable-linger", acct.name], as_root=False).returncode != 0:
            say(f"note     could not turn on lingering, so the broker stops when you log out: run "
                f"`sudo loginctl enable-linger {acct.name}`")
    if not h.dry:
        say(f"service  {host.UNIT_NAME} enabled and started as {acct.name} ({path})")
    return True


def main(opts: Options, env: Mapping[str, str] | None = None, m: Machine | None = None,
         say: Callable[[str], None] = print) -> int:
    env, m = os.environ if env is None else env, m or Machine()
    say(f"{APP_NAME} setup" + (": dry run, nothing will be changed" if opts.dry_run else ""))
    found = detect.find(m.probes())
    report(found, say)
    p = plan(m, found)
    if isinstance(p, str):
        say(f"problem  {p}")
        return EXIT_PROBLEMS
    lay = layout(opts, env, m.system_dirs if p.mode == SYSTEM or m.euid == 0 else None)
    h = host.Host(opts.dry_run, p.sudo, m.run, say)
    say(f"layout   {lay.config_dir} (config), {lay.data_dir} (data)"
        + ("" if p.mode != FOREGROUND else "; no root, passwordless sudo or systemd: not installing a service"))
    if p.mode == FOREGROUND and m.euid != 0 and any(s.unit and not s.user_unit for s in found.servers):
        say("note     serve will run as you, and starting or stopping system units needs root: "
            "`sudo gpu-broker setup` installs it as a service instead")

    exe, acct = m.exe, None
    if p.mode != FOREGROUND:
        found_exe = executable(m, h, say)
        if found_exe is None:
            return EXIT_PROBLEMS
        exe = found_exe
        inst = service.inspect(exe, h.read, lambda py: h.query([py, "-I", "-c", service.SITE_PROBE]).stdout.strip() or None)
        acct = service_account(m, h, p, lay, inst, say)
        if acct is None or not safe_to_run(m, acct, inst, say):
            return EXIT_PROBLEMS
    owner = acct.name if acct and p.mode == SYSTEM else None

    cfg_text = generate.text("config", generate.config(found, lay, p.unit_sudo), lay)
    cat = generate.catalog(found)
    cat_text = generate.text("catalog", cat, lay) if cat else starter.starter(starter.CATALOG)
    validated(cfg_text, cat_text)
    try:
        changed = write_files(h, lay, cfg_text, cat_text, owner, say)
        token, new = host.token(h, lay.env_file)
    except host.UnsafePath as e:
        say(f"problem  {e}: remove it and run setup again")
        return EXIT_PROBLEMS
    except PermissionError as e:
        say(f"problem  {e}: run setup with sudo, or pass --dir to a folder you can write")
        return EXIT_PROBLEMS
    if opts.dry_run and new:
        say(f"token    a new one would go in {lay.env_file}, mode 600")
    else:
        say(f"token    {host.masked(token)} ({'new, ' if new else 'kept, '}in {lay.env_file}, mode 600)")

    from .. import cli
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _load(lay.config, cfg_text)
        if not os.path.exists(cfg.catalog):          # a dry run: the catalog it would write
            pathlib.Path(tmp, starter.CATALOG).write_text(cat_text)
            cfg = dataclasses.replace(cfg, catalog=f"{tmp}/{starter.CATALOG}")
        if cli.check(cfg) != EXIT_OK:
            say(f"problem  `check` found problems above: fix {lay.config} or {lay.catalog}, then run setup again")
            return EXIT_PROBLEMS
        base = base_url(cfg)
        if acct is not None:
            try:
                if not install_service(m, h, p, lay, exe, acct, cfg, env, changed or new, say):
                    return EXIT_PROBLEMS
            except host.UnsafePath as e:
                say(f"problem  {e}: remove it and run setup again")
                return EXIT_PROBLEMS

    serve_argv = [*exe, "-c", lay.config, "serve"]
    child = None
    if acct is None:
        say(f"serve    set -a; . {lay.env_file}; set +a; {' '.join(serve_argv)}")
        if opts.yes or not m.interactive:
            return EXIT_OK
        if opts.dry_run:
            say("would    start it now in the foreground")
    if opts.dry_run:
        say("would    wait for /health and open the dashboard")
        return EXIT_OK
    if acct is None:
        say("starting it now in the foreground; Ctrl-C stops it")
        child = m.start(serve_argv, {**env, TOKEN_ENV: token})
    try:
        if problem := wait_healthy(m, base, child):
            journal = f"journalctl {'' if p.mode == SYSTEM else '--user '}-u {host.UNIT_NAME}"
            say(f"problem  {problem}" + (f" (see {journal})" if child is None else ""))
            code = child.poll() if child is not None else None
            return code if code else EXIT_PROBLEMS
        open_browser(f"{base}/dash#token={token}", env, m.platform, m.opener)
        say(f"open     {base}/dash   (sign in with the token in {lay.env_file})")
        offer_connect(m, opts, base, token, env, say)
        return child.wait() if child is not None else EXIT_OK
    finally:
        if child is not None:
            child.stop()
