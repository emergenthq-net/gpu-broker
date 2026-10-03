"""What `gpu-broker setup` finds on this machine: the GPU, model servers answering on their usual
ports, and the systemd units or Docker containers that run them. Read-only: nothing here starts,
stops or writes anything.

Every probe is passed in (`Probes`), so tests run on fakes and never touch the machine.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import os
import pathlib
import re
import socket
import subprocess
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any

from .. import drivers
from ..gpu import AMD, NVIDIA, auto
from ..gpu.amd import PROC_ROOT, SYS_ROOT
from ..gpu.nvidia import CSV, NVIDIA_SMI
from ..tuning import Gpu

HTTP_TIMEOUT_S = 1.5    # a model server on this machine answers well within this
CMD_TIMEOUT_S = 10      # systemctl, nvidia-smi
GPU_BUDGET_S = 15       # nvidia-smi may be slow to answer the first time after boot
LOCAL = "http://127.0.0.1"
DOCKER_SOCKET = "/var/run/docker.sock"
GIB = 1024


@dataclass(frozen=True)
class Kind:
    """A kind of model server: where it usually listens, what to ask it, and the words its
    systemd units, container names or images contain."""
    name: str
    label: str
    port: int
    path: str
    words: tuple[str, ...]


LLAMA = Kind("llama", "llama.cpp", 8080, "/v1/models", ("llama",))
VLLM = Kind("vllm", "vLLM", 8000, "/v1/models", ("vllm",))
OLLAMA = Kind("ollama", "Ollama", 11434, "/api/tags", ("ollama",))
COMFYUI = Kind("comfyui", "ComfyUI", 8188, "/system_stats", ("comfy",))
KINDS = (OLLAMA, VLLM, LLAMA, COMFYUI)    # most specific words first: "ollama" contains "llama"
OWNED_BY = {"vllm": VLLM, "llamacpp": LLAMA}   # what /v1/models says in `owned_by`


@dataclass(frozen=True)
class FoundGpu:
    vendor: str          # nvidia | amd
    probe: str           # the probe's own description
    total_mib: int
    label: str           # for the dashboard: the card's name where the vendor tool gives it


@dataclass(frozen=True)
class FoundModel:
    name: str
    size_mib: int | None = None   # weights on disk, when the server reports them


@dataclass(frozen=True)
class FoundServer:
    kind: Kind
    url: str
    models: list[FoundModel] = field(default_factory=list)
    unit: str | None = None        # the systemd unit or container that runs it, when one matches
    user_unit: bool = False        # the unit is a `systemctl --user` one
    container: str | None = None
    rivals: tuple[str, ...] = ()   # other units that fit as well as `unit`: setup says it picked one
    unit_inactive: bool = False    # `unit` is not running, though something answers on the port


@dataclass(frozen=True)
class Unit:
    name: str
    user: bool = False


@dataclass(frozen=True)
class Container:
    name: str
    image: str


@dataclass
class Findings:
    gpu: FoundGpu | None
    gpu_error: str
    servers: list[FoundServer]
    units: list[Unit]               # every unit that looks like a model server, answering or not
    containers: list[Container]
    checkpoints: list[str] = field(default_factory=list)   # ComfyUI's models/checkpoints

    def idle(self) -> list[str]:
        """Units and containers that look like model servers but did not answer: worth a hint."""
        used = {s.unit for s in self.servers} | {s.container for s in self.servers}
        return [u.name for u in self.units if u.name not in used] + \
               [c.name for c in self.containers if c.name not in used]


Get = Callable[[str], Any]          # URL -> parsed JSON, or None if nothing (good) answered
Run = Callable[[list[str]], str | None]   # argv -> stdout, or None if it failed or is missing


@dataclass(frozen=True)
class Probes:
    gpu: Callable[[], FoundGpu]     # raises RuntimeError when there is no usable GPU
    get: Get
    run: Run
    containers: Callable[[], list[Container] | None]   # None: no readable Docker socket
    serves: Callable[[Unit, int], bool | None] = lambda unit, port: None   # None: cannot tell


def kind_of(text: str) -> Kind | None:
    low = text.lower()
    return next((k for k in KINDS if any(w in low for w in k.words)), None)


def _mib(size: Any) -> int | None:
    return int(size) // (1024 * 1024) if isinstance(size, int) and size > 0 else None


def _openai_models(body: Any) -> tuple[Kind | None, list[FoundModel]]:
    """llama-server and vLLM both answer /v1/models; `owned_by` says which it is."""
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        return None, []
    rows = [m for m in body["data"] if isinstance(m, dict) and isinstance(m.get("id"), str)]
    kind = next((OWNED_BY[m["owned_by"]] for m in rows if m.get("owned_by") in OWNED_BY), None)
    return kind, [FoundModel(m["id"], _mib((m.get("meta") or {}).get("size"))) for m in rows]


def _server(get: Get, kind: Kind) -> FoundServer | None:
    url = f"{LOCAL}:{kind.port}"
    body = get(url + kind.path)
    if body is None:
        return None
    if kind is COMFYUI:
        return FoundServer(kind, url) if isinstance(body, dict) and "system" in body else None
    if kind is OLLAMA:
        rows = body.get("models") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            return None
        return FoundServer(kind, url, [FoundModel(m["name"], _mib(m.get("size"))) for m in rows
                                       if isinstance(m, dict) and isinstance(m.get("name"), str)])
    said, models = _openai_models(body)
    if not models:
        return None
    return FoundServer(said or kind, url, models)


def units(run: Run) -> list[Unit]:
    """Installed service units, system and user, whose names look like model servers."""
    found = []
    for user in (False, True):
        out = run(["systemctl", *(["--user"] if user else []), "list-unit-files", "--type=service",
                   "--no-legend", "--plain"])
        for line in (out or "").splitlines():
            name = line.split()[0] if line.split() else ""
            if name.endswith(".service") and "@" not in name and kind_of(name):
                found.append(Unit(name.removesuffix(".service"), user))
    return found


def active(run: Run, unit: Unit) -> bool:
    out = run(["systemctl", *(["--user"] if unit.user else []), "is-active", "--", unit.name])
    return (out or "").strip() == "active"


def _match(server: FoundServer, us: list[Unit], cs: list[Container], p: Probes) -> FoundServer:
    """Give a server the unit (preferred) or container that looks like it runs it. Among units of
    its kind, one shown to hold the server's port wins, then one that is running; a unit shown
    not to hold it is never picked. A tie is kept in `rivals`, for setup to report."""
    kind, port = server.kind, int(server.url.rsplit(":", 1)[1])
    ranked = []
    for u in (u for u in us if kind_of(u.name) is kind):
        serves = p.serves(u, port)
        if serves is not False:
            ranked.append(((serves is True, active(p.run, u)), u))
    if ranked:
        best = max(rank for rank, _ in ranked)
        top = sorted((u for rank, u in ranked if rank == best), key=lambda u: (u.user, u.name))
        return FoundServer(kind, server.url, server.models, top[0].name, top[0].user,
                           rivals=tuple(u.name for u in top[1:]), unit_inactive=not best[1])
    box = next((c for c in cs if kind_of(c.name) is kind or kind_of(c.image) is kind), None)
    return FoundServer(kind, server.url, server.models, container=box.name if box else None)


def find(p: Probes) -> Findings:
    try:
        gpu, gpu_error = p.gpu(), ""
    except RuntimeError as e:
        gpu, gpu_error = None, str(e)
    us, cs = units(p.run), p.containers()
    boxes = [c for c in cs or [] if kind_of(c.name) or kind_of(c.image)]
    seen: set[int] = set()
    servers = []
    for kind in KINDS:
        if kind.port in seen:
            continue
        if (s := _server(p.get, kind)) is not None:
            seen.add(kind.port)
            servers.append(_match(s, us, boxes, p))
    ckpts = p.get(f"{LOCAL}:{COMFYUI.port}/models/checkpoints") if any(s.kind is COMFYUI for s in servers) else None
    return Findings(gpu, gpu_error, servers, us, boxes,
                    [c for c in ckpts if isinstance(c, str)] if isinstance(ckpts, list) else [])


# The real probes.

def http_get(url: str) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as r:  # noqa: S310 — fixed local URLs
            return json.loads(r.read()) if r.status == HTTPStatus.OK else None
    except (OSError, ValueError):
        return None


def run_cmd(argv: list[str]) -> str | None:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=CMD_TIMEOUT_S, check=False)  # noqa: S603
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


class _UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path: str) -> None:
        super().__init__("localhost", timeout=HTTP_TIMEOUT_S)
        self.path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(HTTP_TIMEOUT_S)
        self.sock.connect(self.path)


def docker_containers(sock: str = DOCKER_SOCKET) -> list[Container] | None:
    if not os.access(sock, os.R_OK | os.W_OK):
        return None
    conn = _UnixHTTP(sock)
    try:
        conn.request("GET", "/containers/json?all=1")
        rows = json.loads(conn.getresponse().read())
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        conn.close()
    return [Container(r["Names"][0].lstrip("/"), str(r.get("Image", ""))) for r in rows
            if isinstance(r, dict) and r.get("Names")]


def local_gpu(run: Callable[..., subprocess.CompletedProcess[str]] = drivers.run) -> FoundGpu:
    """The card the systemd and docker drivers would read (gpu.vendor auto, card 0)."""
    probe = auto.local(Gpu(), run, GPU_BUDGET_S, NVIDIA_SMI, SYS_ROOT, PROC_ROOT)
    total = probe.read().total_mib
    vendor = NVIDIA if probe.name.startswith(NVIDIA) else AMD
    name = ""
    if vendor == NVIDIA:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            r = run([NVIDIA_SMI, "-i", "0", "--query-gpu=name", CSV], timeout=CMD_TIMEOUT_S)
            name = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
    return FoundGpu(vendor, probe.name, total, f"{name or vendor.upper()}, {round(total / GIB)} GB")


LISTEN = "0A"                        # TCP_LISTEN in /proc/net/tcp
SOCKET_LINK = re.compile(r"^socket:\[(\d+)\]$")


def _listening_inodes(port: int, proc: str = "/proc") -> set[str]:
    inodes = set()
    for table in ("tcp", "tcp6"):
        try:
            rows = pathlib.Path(proc, "net", table).read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            f = row.split()
            if len(f) > 9 and f[3] == LISTEN and int(f[1].rsplit(":", 1)[1], 16) == port:   # noqa: PLR2004 — /proc/net/tcp columns
                inodes.add(f[9])
    return inodes


def unit_serves(unit: Unit, port: int, run: Run = run_cmd, proc: str = "/proc",
                cgroup_root: str = "/sys/fs/cgroup") -> bool | None:
    """Whether a process of `unit` holds the listening socket on `port`: True or False when every
    process in the unit's cgroup could be looked at, else whether its command line names the
    port (True), or None when nothing says either way."""
    base = ["systemctl", *(["--user"] if unit.user else []), "show", "--value"]
    cg = (run([*base, "-p", "ControlGroup", "--", unit.name]) or "").strip()
    listening = _listening_inodes(port, proc)
    try:
        pids = pathlib.Path(cgroup_root + cg, "cgroup.procs").read_text().split() if cg else []
    except OSError:
        pids = []
    if pids and listening:
        try:
            held = {m.group(1) for pid in pids for fd in pathlib.Path(proc, pid, "fd").iterdir()
                    if (m := SOCKET_LINK.match(os.readlink(fd)))}
            return bool(held & listening)
        except OSError:
            pass                            # another user's process: fall back to the command line
    cmd = run([*base, "-p", "ExecStart", "--", unit.name]) or ""
    return True if re.search(rf"(?<![0-9]){port}(?![0-9])", cmd) else None


def real() -> Probes:
    return Probes(local_gpu, http_get, run_cmd, docker_containers, unit_serves)
