"""ComfyUI API graphs, one builder per catalog `template`.

A builder turns a job request plus the catalog entry's `params` into a ComfyUI API graph.
Each template module keeps its tunables (file names, step counts, samplers, sizes) in a
`DEFAULTS` mapping at the top, so the graph code reads as structure only. Request keys
override the request-level defaults (prompt, size, seed, steps, ...); catalog `params`
choose model files. A catalog entry's `defaults` sits between the two: it overrides the
template's request-level defaults for that model, and the request still overrides it
(request > catalog `defaults` > template DEFAULTS); its keys must be ones the template reads. Input images arrive as request keys (`image`, `end_image`) holding the
name ComfyUI stored the upload under.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from . import edit, hunyuan, image, video, wan

DEFAULTS_KEY = "defaults"   # catalog entry field: per-model request-level defaults
PARAMS_KEY = "params"       # catalog entry field: model files

Graph = dict[str, dict[str, Any]]
Builder = Callable[[Mapping[str, Any], Mapping[str, Any], str], Graph]

TEMPLATES: dict[str, Builder] = {
    "qwen_image": image.qwen_image,
    "chroma": image.chroma,
    "flux2_klein": image.flux2_klein,
    "sdxl": image.sdxl,
    "flux2_klein_edit": edit.flux2_klein_edit,
    "qwen_edit": edit.qwen_edit,
    "wan14b": wan.wan14b,
    "wan5b": wan.wan5b,
    "hunyuan": hunyuan.hunyuan,
    "hunyuan_i2v": hunyuan.hunyuan_i2v,
    "minimax": video.minimax,
    "ltx25": video.ltx25,
}

# The request keys each template reads: the only keys a catalog entry's `defaults` may set.
TEMPLATE_KEYS: dict[str, tuple[str, ...]] = {
    "qwen_image": image.QWEN_KEYS,
    "chroma": image.CHROMA_KEYS,
    "flux2_klein": image.KLEIN_KEYS,
    "sdxl": image.SDXL_KEYS,
    "flux2_klein_edit": edit.KLEIN_EDIT_KEYS,
    "qwen_edit": edit.QWEN_EDIT_KEYS,
    "wan14b": wan.WAN14B_KEYS,
    "wan5b": wan.WAN5B_KEYS,
    "hunyuan": hunyuan.HUNYUAN_KEYS,
    "hunyuan_i2v": hunyuan.HUNYUAN_I2V_KEYS,
    "minimax": video.MINIMAX_KEYS,
    "ltx25": video.LTX25_KEYS,
}


def build(model: Mapping[str, Any], request: Mapping[str, Any], prefix: str) -> Graph:
    """The graph for a catalog entry: request > the entry's `defaults` > template DEFAULTS."""
    layered = {**model.get(DEFAULTS_KEY, {}), **request}
    return TEMPLATES[model["template"]](layered, model.get(PARAMS_KEY, {}), prefix)


def check_defaults(key: str, model: Mapping[str, Any]) -> None:
    """A catalog entry's `defaults` must be a mapping of keys its template reads."""
    if DEFAULTS_KEY not in model:
        return
    defaults, template = model[DEFAULTS_KEY], model.get("template")
    if template not in TEMPLATE_KEYS:
        raise ValueError(f"{key}: `{DEFAULTS_KEY}` needs a known ComfyUI template, not {template!r}")
    if not isinstance(defaults, Mapping):
        raise ValueError(f"{key}: `{DEFAULTS_KEY}` must be a mapping of request keys")
    if unknown := sorted(set(defaults) - set(TEMPLATE_KEYS[template])):
        raise ValueError(f"{key}: `{DEFAULTS_KEY}` keys {unknown} are not read by template {template!r} "
                         f"(it reads {list(TEMPLATE_KEYS[template])})")
