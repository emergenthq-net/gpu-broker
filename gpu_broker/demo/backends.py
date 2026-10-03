"""Simulated model servers: an OpenAI-style chat server and ComfyUI, answering in memory.

Chat calls take `chat_s` and keep part of the card busy; a render first loads its model if
ComfyUI does not hold it (`comfy_load_s`), then runs for the kind's `run_s` and writes a
placeholder image named the way ComfyUI names its outputs. Nothing is contacted.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from ..backends import view_url
from ..catalog import Catalog, Model
from ..scheduler import OUTPUT_PREFIX
from ..units import unit_ref
from . import placeholder
from .content import REPLIES
from .driver import SimDriver
from .gpu import COMFY_GROUP, SimGpu
from .tuning import SimCard, SimTimings

TOKENS_PER_WORD = 1.3      # for the usage/timings a llama.cpp server reports
GEN_TPS = (78.0, 96.0)     # simulated generation speed, tokens/s
PROMPT_TPS = (1800.0, 2600.0)
PROMPT_TOKENS = (40, 400)
MS_PER_S = 1000
DECIMALS = 1
EMBED_DIM = 8          # a demo embedding: the first bytes of the text's SHA-256, scaled to 0..1
EMBED_DECIMALS = 3
PROMPT_NODE = "CLIPTextEncode"   # the first one in a graph holds the positive prompt
SSE = "data: {}\n\n"
DONE = "data: [DONE]\n\n"


def prompt_of(graph: Mapping[str, Any]) -> str:
    for n in graph.values():
        if n.get("class_type") == PROMPT_NODE:
            return str(n["inputs"].get("text", ""))
    return ""


class SimBackends:
    def __init__(self, catalog: Catalog, driver: SimDriver, gpu: SimGpu, timings: SimTimings, card: SimCard,
                 output_dir: str, browser_url: str, rng: random.Random,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.catalog, self.driver, self.sim, self.t, self.card = catalog, driver, gpu, timings, card
        self.output_dir, self.browser_url, self.rng, self.sleep = pathlib.Path(output_dir), browser_url, rng, sleep
        self._lock = threading.Lock()
        self.loaded: str | None = None   # the model ComfyUI holds in memory

    # ---- chat server ----------------------------------------------------
    def llm_healthy(self, model: Model) -> bool:
        return self.driver.ready(unit_ref(model["unit"]).name)

    def _reply(self) -> tuple[str, dict[str, Any]]:
        text = self.rng.choice(REPLIES)
        gen_tps, n = self.rng.uniform(*GEN_TPS), round(len(text.split()) * TOKENS_PER_WORD)
        prompt_tps, prompt_n = self.rng.uniform(*PROMPT_TPS), self.rng.randint(*PROMPT_TOKENS)
        return text, {"prompt_n": prompt_n, "prompt_ms": round(prompt_n / prompt_tps * MS_PER_S, DECIMALS),
                      "prompt_per_second": round(prompt_tps, DECIMALS), "predicted_n": n,
                      "predicted_per_second": round(gen_tps, DECIMALS)}

    def llm_chat(self, model: Model, payload: Mapping[str, Any]) -> dict[str, Any]:
        text, timings = self._reply()
        with self.sim.working(self.card.chat_load):
            self.sleep(self.rng.uniform(*self.t.chat_s))
        return {"object": "chat.completion", "model": model["served_name"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": timings["prompt_n"], "completion_tokens": timings["predicted_n"]},
                "timings": timings}

    def llm_stream(self, model: Model, payload: Mapping[str, Any], summary: dict[str, Any]) -> Iterator[str]:
        text, timings = self._reply()
        with self.sim.working(self.card.chat_load):
            for word in text.split():
                self.sleep(self.t.stream_word_s)
                yield SSE.format(json.dumps({"object": "chat.completion.chunk", "model": model["served_name"],
                                             "choices": [{"index": 0, "delta": {"content": word + " "}}]}))
        summary["timings"] = timings
        yield SSE.format(json.dumps({"choices": [], "timings": timings}))
        yield DONE

    def llm_embed(self, model: Model, payload: Mapping[str, Any]) -> dict[str, Any]:
        """A small vector per input, the same for the same text (the demo catalog has no embedder)."""
        texts = payload.get("input", "")
        texts = [texts] if isinstance(texts, str) else list(texts)
        data = [{"object": "embedding", "index": i,
                 "embedding": [round(b / 255, EMBED_DECIMALS) for b in hashlib.sha256(str(t).encode()).digest()[:EMBED_DIM]]}
                for i, t in enumerate(texts)]
        return {"object": "list", "model": model["served_name"], "data": data,
                "usage": {"prompt_tokens": 0, "total_tokens": 0}}

    # ---- ComfyUI ----------------------------------------------------------
    def comfy_alive(self) -> bool:
        return True

    def comfy_free(self) -> None:
        with self._lock:
            self.loaded = None
        self.sim.hold(COMFY_GROUP, self.card.comfy_idle_mib)

    def comfy_queue_len(self) -> int | None:
        return 0

    def comfy_upload(self, name: str, data: bytes, kind: str) -> str:
        return name

    def comfy_run(self, key: str, graph: dict[str, Any], jid: str) -> dict[str, Any]:
        m = self.catalog.models[key]
        t0 = time.monotonic()
        with self._lock:
            load = self.loaded != key
            self.loaded = key
        if load:
            with self.sim.working(self.card.model_load):
                self.sleep(self.t.comfy_load_s)
            self.sim.hold(COMFY_GROUP, self.card.comfy_idle_mib + int(m.get("vram_mib", 0)))
        with self.sim.working(self.card.render_load):
            self.sleep(self.t.run_s[m["kind"]])
        # ComfyUI saves under the graph's filename prefix, OUTPUT_PREFIX + job id: <subfolder>/<name>_00001_.png
        subfolder, _, name = (OUTPUT_PREFIX + jid).rpartition("/")
        path = placeholder.write(self.output_dir, subfolder, name, key, m["kind"], prompt_of(graph))
        return {"model": key, "outputs": [{"file": f"{subfolder}/{path.name}",
                                           "url": view_url(self.browser_url, path.name, subfolder)}],
                "wall_s": round(time.monotonic() - t0, DECIMALS)}
