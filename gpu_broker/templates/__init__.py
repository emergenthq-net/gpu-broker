"""ComfyUI API graphs, one builder per catalog `template`.

A builder turns a job request plus the catalog entry's `params` into a ComfyUI API graph.
Each template module keeps its tunables (file names, step counts, samplers, sizes) in a
`DEFAULTS` mapping at the top, so the graph code reads as structure only. Request keys
override the request-level defaults (prompt, size, seed, steps, ...); catalog `params`
choose model files.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from . import image, video

Graph = dict[str, dict[str, Any]]
Builder = Callable[[Mapping[str, Any], Mapping[str, Any], str], Graph]

TEMPLATES: dict[str, Builder] = {
    "qwen_image": image.qwen_image,
    "chroma": image.chroma,
    "flux2_klein": image.flux2_klein,
    "sdxl": image.sdxl,
    "wan14b": video.wan14b,
    "wan5b": video.wan5b,
    "hunyuan": video.hunyuan,
    "minimax": video.minimax,
    "ltx25": video.ltx25,
}
