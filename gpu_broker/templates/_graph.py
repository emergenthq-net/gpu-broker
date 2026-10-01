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
