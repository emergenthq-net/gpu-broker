"""Placeholder outputs: a PNG that names the job's model and repeats its prompt, written
where ComfyUI would have saved the real image or video. Standard library only."""
from __future__ import annotations

import pathlib
import struct
import textwrap
import zlib
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from . import font
from .content import PLACEHOLDER_NOTE, PLACEHOLDER_TITLE

WIDTH, HEIGHT = 640, 360
SCALE = 3                    # font pixels per glyph pixel
ADVANCE = font.WIDTH + 1     # glyph pixels per character, including the gap
LEADING = font.HEIGHT + 4    # glyph pixels per line
MARGIN = 24
TEXT_RGB = (255, 255, 255)
GRADIENT_DARKEN = 0.45       # the bottom row is this much darker than the top
DEFAULT_RGB = (70, 90, 160)
KIND_RGB: Mapping[str, tuple[int, int, int]] = MappingProxyType(
    {"image": (24, 128, 128), "video": (112, 72, 168), "3d": (196, 104, 40)})
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
RGB8 = (8, 2, 0, 0, 0)       # IHDR: bit depth 8, colour type 2 (RGB), default compression/filter/interlace
NO_FILTER = b"\x00"
SUFFIX = "_00001_.png"       # as ComfyUI names a first output


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def encode(rows: Sequence[bytes | bytearray], width: int) -> bytes:
    """An 8-bit RGB PNG from `rows` of width*3 bytes each."""
    raw = b"".join(NO_FILTER + bytes(r) for r in rows)
    return (PNG_SIGNATURE + _chunk(b"IHDR", struct.pack(">II5B", width, len(rows), *RGB8))
            + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b""))


def lines(model: str, kind: str, prompt: str) -> list[str]:
    per_line = (WIDTH - 2 * MARGIN) // (ADVANCE * SCALE)
    wrapped = textwrap.wrap(prompt, per_line) or [""]
    return [PLACEHOLDER_TITLE, "", f"{kind}: {model}"[:per_line], "", *wrapped, "", PLACEHOLDER_NOTE[:per_line]]


def render(text: Sequence[str], rgb: tuple[int, int, int]) -> bytes:
    rows = []
    for y in range(HEIGHT):
        shade = 1 - GRADIENT_DARKEN * y / HEIGHT
        rows.append(bytearray(bytes(round(c * shade) for c in rgb) * WIDTH))
    for n, line in enumerate(text):
        top = MARGIN + n * LEADING * SCALE
        for i, ch in enumerate(line):
            left = MARGIN + i * ADVANCE * SCALE
            for gy in range(font.HEIGHT):
                for gx in range(font.WIDTH):
                    if font.lit(ch, gx, gy):
                        _block(rows, left + gx * SCALE, top + gy * SCALE)
    return encode(rows, WIDTH)


def _block(rows: list[bytearray], x: int, y: int) -> None:
    for yy in range(y, min(y + SCALE, HEIGHT)):
        for xx in range(x, min(x + SCALE, WIDTH)):
            rows[yy][xx * 3:xx * 3 + 3] = bytes(TEXT_RGB)


def write(root: pathlib.Path, subfolder: str, name: str, model: str, kind: str, prompt: str) -> pathlib.Path:
    """Write a placeholder as ComfyUI names a first output, root/subfolder/<name>_00001_.png; returns its path."""
    path = root / subfolder / f"{name}{SUFFIX}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render(lines(model, kind, prompt), KIND_RGB.get(kind, DEFAULT_RGB)))
    return path
