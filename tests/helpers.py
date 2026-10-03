"""Test doubles and helpers shared by the public and deployment test suites."""
from __future__ import annotations

import atexit
import dataclasses
import functools
import importlib.util
import json
import pathlib
import shutil
import subprocess
import tempfile
import threading
import time
import types
from collections.abc import Iterator
from typing import Any

from gpu_broker import settings as settings_mod
from gpu_broker.broker import Broker
from gpu_broker.constants import TERMINAL, Verb
from gpu_broker.drivers import RecipeInfo
from gpu_broker.gpu.auto import ProbeState
from gpu_broker.units import unit_ref

FIX = pathlib.Path(__file__).parent / "fixtures"
ROOT = pathlib.Path(__file__).parents[1]
TOKEN = "test-token"
COMFY_OUTPUT = "/srv/comfy/output"   # ComfyUI's output folder in test settings
FAST = settings_mod.Intervals(worker_poll_s=0.02, paused_s=0.01, health_poll_s=0.001, comfy_poll_s=0.001,
                              session_poll_s=0.001, gpu_sample_s=0.001, sampler_retry_s=0.01,
                              gpu_cache_s=0)
WAIT_S = 5


@functools.cache
def amdgpu_fixture() -> pathlib.Path:
    """tests/fixtures/amdgpu with its /proc/<pid>/fd symlinks, built once per run in a temp dir.

    The links are listed in fd-links.txt rather than committed: they point at /dev/dri and
    other paths that do not exist here, and an sdist drops such dangling symlinks."""
    src = FIX / "amdgpu"
    out = pathlib.Path(tempfile.mkdtemp(prefix="gpu-broker-amdgpu-"))
    atexit.register(shutil.rmtree, out, True)
    shutil.copytree(src, out, dirs_exist_ok=True)
    for line in (src / "fd-links.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            rel, target = line.split(" ", 1)
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).symlink_to(target)
    return out


def load_leak_scan() -> types.ModuleType:
    """scripts/leak_scan.py (a script, not part of the package) as a module."""
    spec = importlib.util.spec_from_file_location("leak_scan", ROOT / "scripts/leak_scan.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeDriver:
    """Units are names in a set; every call is recorded."""

    def __init__(self, active: set[str] | None = None) -> None:
        self.active = set(active or ())
        self.calls: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str, str]] = []
        self.allowed = None
        self.fail_start: set[str] = set()
        self.recipes: list[tuple[str, str, list[tuple[str, bytes]], float]] = []
        self.recipe_error: str | None = None
        self.recipe_outputs = ["{root}/broker/{jid}/scene.ply"]
        self.recipe_seconds = 600.0             # every recipe's own timeout_s
        self.vram = (8000, 24564)               # (used, total) MiB, as gpu() reports them
        self.streamed: list[str] = []           # inputs handed over as open files, by name
        self.cleans: list[tuple[str, str]] = []
        self.clean_error: str | None = None
        self.no_gpu, self.probe_name = False, "fake"   # no_gpu: GPU reads raise, as with no GPU

    def unit(self, spec: Any, verb: Verb) -> bool:
        name = unit_ref(spec).name
        self.calls.append((verb, name))
        if verb == Verb.IS_ACTIVE:
            return name in self.active
        if verb == Verb.START and name in self.fail_start:
            return False
        (self.active.add if verb == Verb.START else self.active.discard)(name)
        return True

    def gpu_probe(self) -> str:
        if self.no_gpu:
            raise RuntimeError("no GPU found")
        return self.probe_name

    def gpu_state(self) -> ProbeState:
        return ProbeState("failed", "no GPU found") if self.no_gpu else ProbeState("ready", self.probe_name)

    def gpu(self) -> tuple[int, int, int | None]:
        self.gpu_probe()   # raises without a GPU
        return *self.vram, 37

    def recipe_info(self, recipe) -> RecipeInfo:
        return RecipeInfo(self.recipe_seconds, 10, 15, 10)

    def clean_recipe(self, recipe, jid) -> None:
        self.cleans.append((recipe, jid))
        if self.clean_error:
            raise RuntimeError(self.clean_error)

    def gpu_stream(self):
        self.gpu_probe()
        yield "8000,24564,37,120.5,50,2500|llama-8b:7000"

    def close(self) -> None:
        self.closed = True

    def download(self, kind, ref, slug, include):
        self.downloads.append((kind, ref, slug))
        return subprocess.CompletedProcess([], 0, "", "")

    def link(self, rel, subdir):
        return subprocess.CompletedProcess([], 0, "", "")

    def run_recipe(self, recipe, jid, files, timeout_s):
        read = []
        for name, data in files:   # open files are only valid during the call: read them now
            if not isinstance(data, bytes):
                self.streamed.append(name)
            read.append((name, data if isinstance(data, bytes) else data.read()))
        self.recipes.append((recipe, jid, read, timeout_s))
        if self.recipe_error:
            raise RuntimeError(self.recipe_error)
        return [o.format(root=COMFY_OUTPUT, jid=jid) for o in self.recipe_outputs]


class FakeBackends:
    """An LLM is healthy when its unit is active; ComfyUI is alive unless told otherwise."""

    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver
        self.comfy_up, self.frees = True, 0
        self.queue: int | None = 0
        self.chat_gate: threading.Event | None = None   # set → chat calls block until released
        self.chats: list[str] = []
        self.streamed: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] = []   # llm_chat payloads
        self.uploads: list[tuple[str, bytes, str]] = []
        self.graphs: list[dict[str, Any]] = []

    def llm_healthy(self, model) -> bool:
        return unit_ref(model["unit"]).name in self.driver.active

    def llm_chat(self, model, payload) -> dict[str, Any]:
        self.chats.append(model["served_name"])
        self.sent.append(dict(payload))
        if self.chat_gate is not None:
            assert self.chat_gate.wait(WAIT_S)
        return {"choices": [{"message": {"content": model["served_name"]}}],
                "timings": {"predicted_per_second": 50.0, "predicted_n": 10}}

    def llm_stream(self, model, payload, summary) -> Iterator[str]:
        """The served name, one word per SSE chunk, then timings and [DONE]."""
        self.chats.append(model["served_name"])
        self.streamed.append(dict(payload))
        if self.chat_gate is not None:
            assert self.chat_gate.wait(WAIT_S)
        for word in model["served_name"].split("-"):
            yield "data: " + json.dumps({"object": "chat.completion.chunk", "choices": [{"delta": {"content": word}}]}) + "\n\n"
        summary["timings"] = {"predicted_per_second": 90.0}
        yield "data: " + json.dumps({"choices": [], "timings": summary["timings"]}) + "\n\n"
        yield "data: [DONE]\n\n"

    def comfy_alive(self) -> bool:
        return self.comfy_up

    def comfy_free(self) -> None:
        self.frees += 1

    def comfy_run(self, key, graph, jid) -> dict[str, Any]:
        self.graphs.append(graph)
        return {"model": key, "outputs": [{"file": f"broker/{jid}.png"}], "nodes": len(graph)}

    def comfy_queue_len(self) -> int | None:
        return self.queue

    def comfy_upload(self, name: str, data: bytes, kind: str) -> str:
        self.uploads.append((name, data, kind))
        return name


def make_settings(tmp_path: pathlib.Path, catalog: pathlib.Path = FIX / "catalog.yaml", **over: Any) -> settings_mod.Settings:
    cat = tmp_path / "catalog.yaml"
    shutil.copy(catalog, cat)
    s = settings_mod.Settings(catalog=str(cat), db=str(tmp_path / "b.db"), events_jsonl=str(tmp_path / "e.jsonl"),
                              gpu_stream=False, intervals=FAST,
                              comfy=settings_mod.Comfy(unit=unit_ref("comfyui"), output_dir=COMFY_OUTPUT),
                              inputs=settings_mod.Inputs(staging_dir=str(tmp_path / "inputs")))
    return dataclasses.replace(s, **over)


def wait_idle(b: Broker, timeout: float = WAIT_S) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        queued, running, inflight = b.scheduler.snapshot()
        if not (queued or running or inflight):
            return True
        time.sleep(0.01)
    return False


def done(b: Broker, jid: str) -> dict[str, Any]:
    """Wait for a job to finish; returns it as the API shows it (with `using`, no payload)."""
    j = b.wait(jid, WAIT_S)
    assert j is not None and j["state"] in TERMINAL, j
    view = b.view(jid)
    assert view is not None
    return {**view, "result": j.get("result")}


def leftover_staged(b: Broker, timeout: float = WAIT_S) -> list[str]:
    """Staged files still there once the GPU thread has dropped them: it does so just after a
    job's terminal state, so wait (bounded) for staging to empty; returns what is left."""
    end = time.monotonic() + timeout
    while True:
        d = b.staging.dir
        left = sorted(p.name for p in d.iterdir()) if d.exists() else []
        if not left or time.monotonic() >= end:
            return left
        time.sleep(0.01)

