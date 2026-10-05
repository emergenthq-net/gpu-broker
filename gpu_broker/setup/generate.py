"""The config and catalog `gpu-broker setup` writes, generated from what detect.py found.

Estimates are deliberately safe: an LLM whose size the server does not report is assumed to
fill the card, so the broker always stops it before an image or video job rather than running
out of memory. The files say which values are estimates; edit them and run `gpu-broker check`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

from .. import starter
from .detect import COMFYUI, LLAMA, OLLAMA, VLLM, Findings, FoundServer, Kind

DEFAULT_TOTAL_MIB = 24564       # the starter catalog's card, used when no GPU was read
RESERVE_MIB = 600               # desktop / CUDA context headroom
KV_MIB = 1024                   # context and KV cache on top of the weights
WEIGHTS_FACTOR = 1.15           # runtime overhead on the file size
VLLM_SHARE = 0.9                # vLLM takes --gpu-memory-utilization (0.9) of the card up front
CHECKPOINT_MIB = 9000           # an SDXL-class checkpoint at 1024 px
MAX_PER_SERVER = 6              # Ollama can hold dozens of models; list the first few
MAX_CHECKPOINTS = 4
SDXL = re.compile(r"xl", re.I)  # checkpoints the stock single-checkpoint (sdxl) graph runs well
DEFAULT_UNITS = {LLAMA: "llama-server", VLLM: "vllm", OLLAMA: "ollama", COMFYUI: "comfyui"}
HEALTH = {OLLAMA: "/api/version"}   # servers without /health
COLORS = {LLAMA: "#3b5bdb", VLLM: "#0ca678", OLLAMA: "#7048e8", COMFYUI: "#e8590c"}
HEADER = """\
# gpu-broker {what}, written by `gpu-broker setup` from what it found on this machine.
# Edit freely: setup never overwrites this file. Check it with `gpu-broker -c {config} check`.
# Every key: https://github.com/emergenthq-net/gpu-broker/blob/main/docs/catalog.md (catalog)
# and gpu_broker/settingsschema.py (config).
"""


@dataclass(frozen=True)
class Layout:
    """Where everything lives: /etc + /var/lib when installed as a system service, else the
    user's own config and data folders. A system service's catalog is in its data folder: the
    broker rewrites the catalog, so the folder it is in must be the service account's, and the
    config folder (with broker.env, which systemd reads as root) must stay root's."""
    config_dir: str
    data_dir: str
    log_dir: str
    system: bool

    @property
    def config(self) -> str:
        return f"{self.config_dir}/{starter.CONFIG}"

    @property
    def catalog_dir(self) -> str:
        return self.data_dir if self.system else self.config_dir

    @property
    def catalog(self) -> str:
        return f"{self.catalog_dir}/{starter.CATALOG}"

    @property
    def env_file(self) -> str:
        return f"{self.config_dir}/broker.env"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-.") or "model"


def llm_servers(f: Findings) -> list[FoundServer]:
    return [s for s in f.servers if s.kind is not COMFYUI]


def runs_on(s: FoundServer) -> str:
    """What the driver starts and stops for this server."""
    return s.unit or s.container or DEFAULT_UNITS[s.kind]


def driver_kind(f: Findings) -> str:
    """docker when the servers found run in containers and none in systemd units."""
    servers = f.servers
    if any(s.container for s in servers) and not any(s.unit for s in servers):
        return "docker"
    return "systemd"


def user_units(f: Findings) -> bool:
    units = [s for s in f.servers if s.unit]
    return bool(units) and all(s.user_unit for s in units)


def vram_estimate(kind: Kind, size_mib: int | None, total: int) -> int:
    usable = total - RESERVE_MIB
    if size_mib:
        need = int(size_mib * WEIGHTS_FACTOR) + KV_MIB
    elif kind is VLLM:
        need = int(total * VLLM_SHARE)
    else:
        need = usable
    return min(usable, -(-need // 100) * 100)


def catalog(f: Findings) -> dict[str, object] | None:
    """The catalog for the servers found, or None when no LLM server answered (setup then
    writes the starter catalog)."""
    total = f.gpu.total_mib if f.gpu else DEFAULT_TOTAL_MIB
    models: dict[str, dict[str, object]] = {}
    for s in llm_servers(f):
        for m in s.models[:MAX_PER_SERVER]:
            key = slug(m.name)
            while key in models:
                key += "-2"
            entry: dict[str, object] = {
                "kind": "llm", "runner": "llm_unit", "unit": runs_on(s), "endpoint": s.url,
                "served_name": m.name, "vram_mib": vram_estimate(s.kind, m.size_mib, total),
                "caps": ["chat"], "quality": 50, "status": "ready",
                "notes": f"found by setup on {s.kind.label}" + ("" if m.size_mib else "; vram_mib is an estimate"),
            }
            if s.kind in HEALTH:
                entry["health_path"] = HEALTH[s.kind]
            models[key] = entry
    if not models:
        return None
    for ckpt in [c for c in f.checkpoints if SDXL.search(c)][:MAX_CHECKPOINTS]:
        key = slug(ckpt.rsplit(".", 1)[0])
        while key in models:
            key += "-2"
        models[key] = {"kind": "image", "runner": "comfy", "template": "sdxl", "params": {"ckpt": ckpt},
                       "vram_mib": CHECKPOINT_MIB, "caps": ["t2i"], "quality": 50, "status": "ready",
                       "notes": "found by setup in ComfyUI's checkpoints"}
    return {
        "defaults": {"resident": next(iter(models)), "idle_restore_s": 120, "session_idle_s": 900,
                     "session_yield_s": 120, "session_max_s": 14400, "vram_total_mib": total,
                     "vram_reserve_mib": RESERVE_MIB},
        "models": models,
    }


def config(f: Findings, lay: Layout, sudo: bool = False) -> dict[str, object]:
    """`sudo`: the broker runs as an account that starts and stops system units through sudo
    (the rule setup writes)."""
    comfy = next((s for s in f.servers if s.kind is COMFYUI), None)
    kind = driver_kind(f)
    drv: dict[str, object] = {"kind": kind, "models_root": f"{lay.data_dir}/models"}
    if kind == "systemd":
        drv.update(sudo=sudo, user=user_units(f))
    groups = {runs_on(s): {"label": s.kind.label, "color": COLORS[s.kind]} for s in f.servers}
    llms = llm_servers(f)
    return {
        "catalog": lay.catalog,
        "db": f"{lay.data_dir}/broker.db",
        "events_jsonl": f"{lay.log_dir}/events.jsonl",
        "gpu_stream": True,
        "server": {"host": "127.0.0.1", "port": 8095},
        "comfy": {"url": comfy.url if comfy else "http://127.0.0.1:8188",
                  "unit": (comfy.unit or comfy.container) if comfy else None},
        "driver": drv,
        "inputs": {"staging_dir": f"{lay.data_dir}/inputs"},
        "ui": {"gpu_label": f.gpu.label if f.gpu else "GPU",
               "resident_label": llms[0].kind.label if llms else "the default model",
               "groups": groups},
    }


def text(what: str, data: dict[str, object], lay: Layout) -> str:
    return HEADER.format(what=what, config=lay.config) + yaml.safe_dump(data, sort_keys=False, width=100)
