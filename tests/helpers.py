"""Test doubles and helpers shared by the public and deployment test suites."""
from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from typing import Any

from gpu_broker import settings as settings_mod
from gpu_broker.broker import Broker
from gpu_broker.constants import TERMINAL, Verb
from gpu_broker.units import unit_ref

FIX = pathlib.Path(__file__).parent / "fixtures"
ROOT = pathlib.Path(__file__).parents[1]
TOKEN = "test-token"
FAST = settings_mod.Intervals(worker_poll_s=0.02, paused_s=0.01, health_poll_s=0.001, comfy_poll_s=0.001,
                              session_poll_s=0.001, job_wait_poll_s=0.01, gpu_sample_s=0.001, sampler_retry_s=0.01,
                              gpu_cache_s=0)
WAIT_S = 5


class FakeDriver:
    """Units are names in a set; every call is recorded."""

    def __init__(self, active: set[str] | None = None) -> None:
        self.active = set(active or ())
        self.calls: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str, str]] = []
        self.allowed = None
        self.fail_start: set[str] = set()

    def unit(self, spec: Any, verb: Verb) -> bool:
        name = unit_ref(spec).name
        self.calls.append((verb, name))
        if verb == Verb.IS_ACTIVE:
            return name in self.active
        if verb == Verb.START and name in self.fail_start:
            return False
        (self.active.add if verb == Verb.START else self.active.discard)(name)
        return True

    def gpu(self) -> tuple[int, int, int]:
        return 8000, 24564, 37

    def gpu_stream(self):
        yield "8000,24564,37,120.5,50,2500|llama-8b:7000"

    def close(self) -> None:
        self.closed = True

    def download(self, kind, ref, slug, include):
        self.downloads.append((kind, ref, slug))
        return subprocess.CompletedProcess([], 0, "", "")

    def link(self, rel, subdir):
        return subprocess.CompletedProcess([], 0, "", "")


class FakeBackends:
    """An LLM is healthy when its unit is active; ComfyUI is alive unless told otherwise."""

    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver
        self.comfy_up = True
        self.frees = 0
        self.queue: int | None = 0
        self.chat_gate: threading.Event | None = None   # set → chat calls block until released
        self.chats: list[str] = []
        self.streamed: list[dict[str, Any]] = []

    def llm_healthy(self, model) -> bool:
        return unit_ref(model["unit"]).name in self.driver.active

    def llm_chat(self, model, payload) -> dict[str, Any]:
        self.chats.append(model["served_name"])
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
        return {"model": key, "outputs": [{"file": f"broker/{jid}.png"}], "nodes": len(graph)}

    def comfy_queue_len(self) -> int | None:
        return self.queue


def make_settings(tmp_path: pathlib.Path, catalog: pathlib.Path = FIX / "catalog.yaml", **over: Any) -> settings_mod.Settings:
    cat = tmp_path / "catalog.yaml"
    shutil.copy(catalog, cat)
    s = settings_mod.Settings(catalog=str(cat), db=str(tmp_path / "b.db"), events_jsonl=str(tmp_path / "e.jsonl"),
                              gpu_stream=False, intervals=FAST,
                              comfy=settings_mod.Comfy(unit=unit_ref("comfyui")))
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
