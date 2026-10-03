"""The simulated host driver: model servers that "start" and "stop" in memory.

Starting an LLM server takes `llm_load_s` before its health check answers (as llama.cpp
loading weights does) and claims its catalog `vram_mib`; stopping takes `llm_stop_s` and
hands the memory back. The 3D recipe renders a placeholder under ComfyUI's output folder.
Downloads are refused: the demo fetches nothing.
"""
from __future__ import annotations

import pathlib
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from ..constants import Verb
from ..drivers import Input, RecipeInfo
from ..gpu.auto import READY, ProbeState
from ..scheduler import OUTPUT_PREFIX
from ..units import unit_ref
from . import placeholder
from .gpu import SimGpu
from .tuning import SimCard, SimTimings

SIM_PROBE = "simulated"   # the dashboard shows it beside utilisation, as it shows nvidia or amd
NO_DOWNLOADS = "the demo downloads nothing"
EXEC_KIND = "3d"
EXEC_PROMPT = "your photo, as a 3D scene"
FAILED = 1


class SimDriver:
    allowed: frozenset[str] | None = None

    def __init__(self, gpu: SimGpu, vram: Mapping[str, int], timings: SimTimings, card: SimCard,
                 output_dir: str, active: Sequence[str] = (), clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        """`vram`: MiB each unit holds once started; `active`: units running (and loaded) at start."""
        self.sim, self.vram, self.t, self.card = gpu, vram, timings, card
        self.output_dir, self.clock, self.sleep = output_dir, clock, sleep
        self._lock = threading.Lock()
        self._ready_at: dict[str, float] = {}   # running unit -> when its health check starts to answer
        self._closed = threading.Event()
        for name in active:
            self._ready_at[name] = clock()
            gpu.hold(name, vram.get(name, 0))

    def ready(self, name: str) -> bool:
        """An LLM server answers its health check: started, and done loading."""
        with self._lock:
            at = self._ready_at.get(name)
        return at is not None and self.clock() >= at

    def unit(self, spec: Any, verb: Verb) -> bool:
        name = unit_ref(spec).name
        if verb == Verb.IS_ACTIVE:
            with self._lock:
                return name in self._ready_at
        if verb == Verb.START:
            with self._lock:
                self._ready_at[name] = self.clock() + self.t.llm_load_s
            self.sim.hold(name, self.vram.get(name, 0))
            return True
        self.sleep(self.t.llm_stop_s)
        with self._lock:
            self._ready_at.pop(name, None)
        self.sim.drop(name)
        return True

    def gpu_probe(self) -> str:
        return SIM_PROBE

    def gpu_state(self) -> ProbeState:
        return ProbeState(READY, SIM_PROBE)   # nothing to find: the card is simulated

    def gpu(self) -> tuple[int, int, int | None]:
        return self.sim.reading()

    def gpu_stream(self) -> Iterator[str]:
        while not self._closed.is_set():
            yield self.sim.sample()
            self._closed.wait(self.t.sample_s)

    def close(self) -> None:
        self._closed.set()

    def download(self, kind: str, ref: str, slug: str, include: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([kind, ref], FAILED, "", NO_DOWNLOADS)

    def link(self, rel: str, subdir: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([rel, subdir], FAILED, "", NO_DOWNLOADS)

    def recipe_info(self, recipe: str) -> RecipeInfo:
        return RecipeInfo(self.t.recipe_timeout_s, 0, 0, 0)   # nothing to kill or reap: no process runs

    def clean_recipe(self, recipe: str, jid: str) -> None:
        return None

    def run_recipe(self, recipe: str, jid: str, files: Sequence[tuple[str, Input]], timeout_s: float) -> list[str]:
        self.sim.hold(recipe, self.vram.get(recipe, 0))
        try:
            with self.sim.working(self.card.render_load):
                self.sleep(self.t.run_s[EXEC_KIND])
        finally:
            self.sim.drop(recipe)
        out = placeholder.write(pathlib.Path(self.output_dir), OUTPUT_PREFIX + jid, jid, recipe, EXEC_KIND, EXEC_PROMPT)
        return [str(out)]
