"""The simulated card: who holds how much memory, how busy it is, and the samples the
dashboard draws. Shared by the simulated driver (model servers start and stop) and the
simulated backends (calls and renders keep it busy)."""
from __future__ import annotations

import contextlib
import random
import threading
from collections.abc import Iterator

from ..gpu import GpuSample, line
from .tuning import SimCard

COMFY_GROUP = "comfyui"   # ComfyUI's memory, as the dashboard's VRAM-by-owner legend names it


class SimGpu:
    def __init__(self, card: SimCard, rng: random.Random) -> None:
        self.card, self.rng = card, rng
        self._lock = threading.Lock()
        self._held: dict[str, int] = {COMFY_GROUP: card.comfy_idle_mib}   # owner -> MiB
        self._load = 0.0          # share of the card's compute in use (may exceed 1; capped when read)
        self._temp = card.idle_c

    def hold(self, owner: str, mib: int) -> None:
        """`owner` now holds `mib` (replacing what it held)."""
        with self._lock:
            self._held[owner] = mib

    def drop(self, owner: str) -> None:
        with self._lock:
            self._held.pop(owner, None)

    def held(self, owner: str) -> int:
        with self._lock:
            return self._held.get(owner, 0)

    @contextlib.contextmanager
    def working(self, load: float) -> Iterator[None]:
        """Keep `load` of the card busy for the duration of the block."""
        with self._lock:
            self._load += load
        try:
            yield
        finally:
            with self._lock:
                self._load -= load

    def _busy(self) -> float:
        return min(1.0, max(0.0, self._load))

    def reading(self) -> tuple[int, int, int]:
        """(used MiB, total MiB, utilisation %), as `nvidia-smi` would report them."""
        with self._lock:
            used = self.card.base_mib + sum(self._held.values())
            busy = self._busy()
        return used, self.card.total_mib, self._util(busy)

    def _util(self, busy: float) -> int:
        c = self.card
        if busy <= 0:
            return c.idle_util
        jitter = self.rng.randint(-c.jitter_util, c.jitter_util)
        return max(c.idle_util, min(c.busy_util, round(c.idle_util + (c.busy_util - c.idle_util) * busy) + jitter))

    def sample(self) -> str:
        """One sample line in the drivers' format (gpu_broker.gpu.line).
        Temperature eases toward its target, as a real card's does."""
        c = self.card
        with self._lock:
            busy, owners = self._busy(), dict(self._held)
            self._temp += (c.idle_c + (c.hot_c - c.idle_c) * busy - self._temp) * c.heat_step
            temp = self._temp
        used = c.base_mib + sum(owners.values())
        power = c.idle_w + (c.busy_w - c.idle_w) * busy + self.rng.uniform(-c.jitter_w, c.jitter_w) * busy
        mhz = c.busy_mhz if busy > 0 else c.idle_mhz
        sample = GpuSample(used, c.total_mib, self._util(busy), power, round(temp), mhz)
        return line(sample, [(o, mib) for o, mib in owners.items() if mib > 0])
