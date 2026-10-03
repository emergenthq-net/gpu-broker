"""Helpers shared by the graph builders."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

Node = dict[str, Any]
Link = list[Any]  # [node id, output index]

VIDEO_FORMAT = "auto"
VIDEO_CODEC = "auto"
FIRST = 0  # output index of a node's first output


def node(class_type: str, **inputs: Any) -> Node:
    return {"class_type": class_type, "inputs": inputs}


def link(node_id: str, output: int = FIRST) -> Link:
    return [node_id, output]


def options(request: Mapping[str, Any], defaults: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """The request-overridable values a template uses: request value, else template default."""
    return {k: request.get(k, defaults[k]) for k in keys}


def save_video(src: Link, prefix: str) -> Node:
    return node("SaveVideo", video=src, filename_prefix=prefix, format=VIDEO_FORMAT,
                **{"format.codec": VIDEO_CODEC})


def save_image(src: Link, prefix: str) -> Node:
    return node("SaveImage", images=src, filename_prefix=prefix)


def load_image(name: str) -> Node:
    """An input image the broker uploaded to ComfyUI's input folder (see staging.py)."""
    return node("LoadImage", image=name)


def input_image(req: Mapping[str, Any], slot: str, template: str) -> str:
    """The uploaded file name for a required image slot; a missing one is a caller error."""
    name = req.get(slot)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{template}: needs an input image (`{slot}`)")
    return name
