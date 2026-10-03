"""Input file data: decode it, fetch it, and prove it is the image or video it claims to be.

Every file is checked when the job is submitted: decoded size against `inputs.max_bytes`
(images) or `inputs.video_max_bytes`, format by its magic bytes against `inputs.types` /
`inputs.video_types`, and any declared type (the data URL's or the server's Content-Type)
must agree. Which slots a model takes is inputs.py.

`<slot>_url` makes the broker fetch an address the caller chose, so it is refused unless
`inputs.allow_urls` is set. Only http(s) public addresses are fetched (netguard.py vets every
connection and redirect hop), the whole fetch runs within `inputs.fetch_s`, and every failure
— network, refusal, timeout, size or format — is the same error, so the answer never tells a
caller what the broker can reach.
"""
from __future__ import annotations

import base64
import binascii
import http.client
import re
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from . import netguard
from .constants import FRAMES, HTTP_SCHEMES, IMAGE_MIME, URL_SLOTS, URL_SUFFIX, VIDEO, VIDEO_MIME
from .deadline import Deadline
from .settings import Inputs

# Format -> (offset, bytes) signatures that must all match. Fixed by the file formats; the
# first match wins, so the QuickTime brand is listed before the generic ISO-BMFF `ftyp` box.
MAGIC: Mapping[str, tuple[tuple[int, bytes], ...]] = {
    "png": ((0, b"\x89PNG\r\n\x1a\n"),),
    "jpeg": ((0, b"\xff\xd8\xff"),),
    "webp": ((0, b"RIFF"), (8, b"WEBP")),
    "mov": ((4, b"ftypqt  "),),
    "mp4": ((4, b"ftyp"),),
    "webm": ((0, b"\x1a\x45\xdf\xa3"),),
}
EXTENSION = {"png": ".png", "jpeg": ".jpg", "webp": ".webp", "mp4": ".mp4", "mov": ".mov", "webm": ".webm"}
MIME = {mime: kind for kind, mime in (IMAGE_MIME | VIDEO_MIME).items()} | {"image/jpg": "jpeg"}   # declared -> format
GENERIC_MIME = frozenset({"", "application/octet-stream", "binary/octet-stream"})
DATA_URL = re.compile(r"^data:([\w/+.-]*)(?:;[\w=.-]+)*;base64,", re.IGNORECASE)
B64_RATIO = 3 / 4                       # decoded bytes per base64 character
B64_PAD_BYTES = 2                       # padding can make the estimate overshoot by this much
READ_CHUNK = 64 * 1024                  # bytes per read of a `<slot>_url` body (the deadline is checked between)
FETCH_FAILED = "`{name}` could not be fetched"   # every failure alike: no network map for callers
FETCH_ERRORS = (OSError, http.client.HTTPException, ValueError)


@dataclass(frozen=True)
class InputFile:
    slot: str
    data: bytes
    kind: str        # a MAGIC key
    source: str      # "inline" | "url"
    index: int | None = None   # position in `frames`

    @property
    def name(self) -> str:
        """The file's name for the job: <slot>[-NN].<ext>."""
        return f"{self.slot}{'' if self.index is None else f'-{self.index:02d}'}{EXTENSION[self.kind]}"

    def summary(self) -> dict[str, Any]:
        return {"bytes": len(self.data), "type": self.kind, "source": self.source}


def limits(slot: str, cfg: Inputs) -> tuple[int, tuple[str, ...]]:
    """(size cap, accepted formats) for a slot."""
    return (cfg.video_max_bytes, cfg.video_types) if slot == VIDEO else (cfg.max_bytes, cfg.types)


def sniff(data: bytes, allowed: tuple[str, ...], declared: str = "") -> str:
    """The format by magic bytes; must be allowed and agree with any declared type."""
    kind = next((k for k, sig in MAGIC.items() if all(data[o:o + len(b)] == b for o, b in sig)), None)
    if kind is None or kind not in allowed:
        raise ValueError(f"not an accepted file (accepted: {', '.join(allowed)})")
    declared = declared.split(";")[0].strip().lower()
    if declared not in GENERIC_MIME and MIME.get(declared) != kind:
        raise ValueError(f"declared type {declared!r} does not match the {kind} data")
    return kind


def decode(slot: str, value: str, cfg: Inputs, index: int | None = None) -> InputFile:
    cap, allowed = limits(slot, cfg)
    label = slot if index is None else f"{slot}[{index}]"
    declared, text = "", value
    if m := DATA_URL.match(value):
        declared, text = m.group(1), value[m.end():]
    if len(text) * B64_RATIO > cap + B64_PAD_BYTES:
        raise ValueError(f"`{label}` is larger than {cap} bytes")
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(f"`{label}` is not valid base64") from None
    if len(data) > cap:
        raise ValueError(f"`{label}` is larger than {cap} bytes")
    try:
        kind = sniff(data, allowed, declared)
    except ValueError as e:
        raise ValueError(f"`{label}`: {e}") from None
    return InputFile(slot, data, kind, "inline", index)


def _read(r: Any, limit: int, deadline: Deadline) -> bytes:
    """Up to `limit` bytes of the body; TimeoutError once the deadline passes, IncompleteRead if
    the body ends before its Content-Length (read1 just returns b"" there: the connection closed)."""
    buf = bytearray()
    while len(buf) < limit:
        deadline.left()
        chunk = r.read1(min(READ_CHUNK, limit - len(buf)))   # whatever has arrived, never waits for a full chunk
        if not chunk:
            if getattr(r, "length", None):   # bytes the server announced and never sent
                raise http.client.IncompleteRead(bytes(buf), r.length)
            break
        buf += chunk
    return bytes(buf)


def fetch(slot: str, url: str, cfg: Inputs, policy: netguard.Policy | None = None,
          clock: Callable[[], float] = time.monotonic) -> InputFile:
    """`policy` defaults to the allowlist in `cfg`; the broker adds its own ComfyUI endpoints."""
    name, (cap, allowed) = slot + URL_SUFFIX, limits(slot, cfg)
    if not cfg.allow_urls:
        raise ValueError(f"`{name}` is disabled on this broker (inputs.allow_urls); send `{slot}` as base64")
    if urlsplit(url).scheme not in HTTP_SCHEMES:
        raise ValueError(f"`{name}` must be an http(s) URL")
    deadline = None
    try:
        deadline = Deadline(cfg.fetch_s, clock)
        opener = netguard.opener(policy or netguard.Policy(cfg.url_allow_networks), deadline)
        with opener.open(urllib.request.Request(url), timeout=deadline.left()) as r:  # noqa: S310 — scheme checked above
            declared = r.headers.get("Content-Type", "")
            data = _read(r, cap + 1, deadline)
        deadline.left()   # a cut-off socket reads as a short body: it must not pass as complete
        if len(data) > cap:
            raise ValueError("too large")
        return InputFile(slot, data, sniff(data, allowed, declared), "url")
    except FETCH_ERRORS:   # what failed (and the URL's address) stays out of the answer
        raise ValueError(FETCH_FAILED.format(name=name)) from None
    finally:
        if deadline is not None:
            deadline.close()


def read(body: Mapping[str, Any], cfg: Inputs, policy: netguard.Policy | None = None) -> list[InputFile]:
    out = []
    for slot in URL_SLOTS:
        if body.get(slot):
            out.append(decode(slot, body[slot], cfg))
        elif body.get(slot + URL_SUFFIX):
            out.append(fetch(slot, body[slot + URL_SUFFIX], cfg, policy))
    frames = body.get(FRAMES) or []
    if len(frames) > cfg.max_frames:
        raise ValueError(f"`{FRAMES}` has {len(frames)} images; at most {cfg.max_frames} are accepted")
    out += [decode(FRAMES, f, cfg, i) for i, f in enumerate(frames)]
    return out


def summarize(files: list[InputFile]) -> dict[str, Any]:
    """What the job payload records instead of the data: per slot, a list for `frames`."""
    out: dict[str, Any] = {}
    for f in files:
        if f.index is None:
            out[f.slot] = f.summary()
        else:
            out.setdefault(f.slot, []).append(f.summary())
    return out


def strip(body: Mapping[str, Any]) -> dict[str, Any]:
    """The request without file data (stored as the job payload)."""
    drop = {FRAMES, *URL_SLOTS, *(s + URL_SUFFIX for s in URL_SLOTS)}
    return {k: v for k, v in body.items() if k not in drop}
