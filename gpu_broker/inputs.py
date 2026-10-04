"""Input files a job carries, and whether a model can take them.

Slots: `image` and `end_image` (single images: a start/end frame, or the source to edit),
`frames` (a list of images: several views of one scene) and `video`. A single slot arrives
inline — base64 or a `data:` URL — or as `<slot>_url`; `frames` is a list of inline images.
A catalog entry declares what it takes as `inputs: {slot: required|optional|one_of}`
(`one_of`: exactly one of those slots must be given). `frames` may instead be
`{need: <one of those>, min: N, max: M}`: how many views the model can use. Everything here is checked when the
job is submitted, so a job that cannot run is refused with a 400 instead of failing later.
Reading, checking and fetching the data itself is media.py; holding it until the job runs is
staging.py.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .constants import FRAMES, INPUT_SLOTS, NUM_FRAMES, URL_SLOTS, URL_SUFFIX, InputNeed

if TYPE_CHECKING:   # catalog validation imports this module
    from .catalog import Model


def _string(name: str, v: Any) -> None:
    if v is not None and not (isinstance(v, str) and v):
        raise ValueError(f"`{name}` must be a non-empty string")


def slots(body: Mapping[str, Any]) -> frozenset[str]:
    """The input slots a request fills; shape errors are ValueErrors (HTTP 400)."""
    given = set()
    for slot in URL_SLOTS:
        inline, url = body.get(slot), body.get(slot + URL_SUFFIX)
        _string(slot, inline)
        _string(slot + URL_SUFFIX, url)
        if inline and url:
            raise ValueError(f"give `{slot}` or `{slot}{URL_SUFFIX}`, not both")
        if inline or url:
            given.add(slot)
    frames = body.get(FRAMES)
    if isinstance(frames, int) and not isinstance(frames, bool):
        raise ValueError(f"`{FRAMES}` is the list of input views; give a video's length in frames as `{NUM_FRAMES}`")
    if frames is not None:
        if not (isinstance(frames, list) and frames and all(isinstance(f, str) and f for f in frames)):
            raise ValueError(f"`{FRAMES}` must be a non-empty list of base64 images")
        given.add(FRAMES)
    return frozenset(given)


RANGE_KEYS = frozenset({"need", "min", "max"})


def _needs(spec: Mapping[str, Any]) -> dict[str, str]:
    return {slot: v["need"] if isinstance(v, Mapping) else v for slot, v in spec.items()}


def needs(model: Model) -> Mapping[str, str]:
    return _needs(model.get("inputs") or {})


def frame_range(model: Model) -> tuple[int, int] | None:
    spec = (model.get("inputs") or {}).get(FRAMES)
    return (int(spec["min"]), int(spec["max"])) if isinstance(spec, Mapping) else None


def check_counts(key: str, model: Model, body: Mapping[str, Any]) -> None:
    """A model that declares how many frames it can use gets that many, or a 400."""
    frames, limits = body.get(FRAMES), frame_range(model)
    if isinstance(frames, list) and limits and not limits[0] <= len(frames) <= limits[1]:
        raise ValueError(f"'{key}' takes {limits[0]} to {limits[1]} frames, got {len(frames)}")


def _problem(want: Mapping[str, str], given: frozenset[str]) -> str | None:
    """Why `given` does not fit `want`, or None when it does."""
    if extra := sorted(given - set(want)):
        return f"takes no {' or '.join(extra)} input" + (f" (it takes: {', '.join(want)})" if want else "")
    if missing := sorted(s for s, n in want.items() if n == InputNeed.REQUIRED and s not in given):
        return f"needs an input: send {' and '.join(missing)}"
    one_of = sorted(s for s, n in want.items() if n == InputNeed.ONE_OF)
    if one_of and len(given & set(one_of)) != 1:
        return f"needs exactly one of {', '.join(one_of)}"
    return None


def accepts(model: Model, given: frozenset[str]) -> bool:
    """The model takes every given input and is given every input it requires."""
    return _problem(needs(model), given) is None


def check(key: str, model: Model, given: frozenset[str]) -> None:
    if (why := _problem(needs(model), given)) is not None:
        raise ValueError(f"'{key}' {why}")


def validate_spec(key: str, spec: Mapping[str, Any]) -> None:
    """A catalog `inputs` block: known slots and needs; `one_of` needs at least two slots."""
    for slot, value in spec.items():
        need = value
        if slot == FRAMES and isinstance(value, Mapping):
            lo, hi = value.get("min"), value.get("max")
            ok = (set(value) == RANGE_KEYS and isinstance(lo, int) and isinstance(hi, int)
                  and not isinstance(lo, bool) and not isinstance(hi, bool) and 1 <= lo <= hi)
            if not ok:
                raise ValueError(f"{key}: inputs.frames as a mapping is {{need, min, max}} with 1 <= min <= max, "
                                 f"got {dict(value)}")
            need = value["need"]
        if slot not in INPUT_SLOTS or not isinstance(need, str) or need not in {n.value for n in InputNeed}:
            raise ValueError(f"{key}: inputs must map {list(INPUT_SLOTS)} to required|optional|one_of, "
                             f"got {slot}: {need}")
    if sum(1 for n in _needs(spec).values() if n == InputNeed.ONE_OF) == 1:
        raise ValueError(f"{key}: one_of needs at least two slots")
