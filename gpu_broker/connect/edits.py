"""Edits a connector makes, and how each is undone later from whatever the file then holds.

- A marked block of lines (shell rc files): `# >>> gpu-broker connect >>>` ... `# <<< ... <<<`.
  Connecting again replaces the block in place; undoing removes exactly those lines.
- JSON keys set at a path: the undo record keeps what each key held before (or that it was
  absent), so undoing restores those values and leaves everything else as the user left it.
- A whole file we created: undone by deleting it, only while it is still ours.
A file that is not what we expect (JSON with comments, say) raises Unsupported: the connector
skips with that reason rather than guessing.
"""
from __future__ import annotations

import json
from typing import Any

from .core import Undo

BEGIN, END = ">>> gpu-broker connect >>>", "<<< gpu-broker connect <<<"
NOTE = "managed by `gpu-broker connect`; `gpu-broker disconnect` removes it"
ENCODING = "utf-8"
DEFAULT_INDENT = 2
BLOCK, JSON_KEYS, OWN_FILE = "block", "json", "file"


class Unsupported(Exception):
    """The file is in a form this connector will not edit safely."""


def _block_span(lines: list[str], comment: str) -> tuple[int, int] | None:
    begin = next((i for i, ln in enumerate(lines) if ln.strip() == f"{comment} {BEGIN}"), None)
    if begin is None:
        return None
    end = next((i for i in range(begin, len(lines)) if lines[i].strip() == f"{comment} {END}"), None)
    if end is None:
        raise Unsupported(f"a '{BEGIN}' marker has no matching end marker")
    return begin, end


def put_block(text: str, body: list[str], comment: str = "#") -> str:
    """`text` with our block (replaced in place, or appended)."""
    block = [f"{comment} {BEGIN}", f"{comment} {NOTE}", *body, f"{comment} {END}"]
    lines = text.splitlines()
    span = _block_span(lines, comment)
    if span:
        lines[span[0]:span[1] + 1] = block
        return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    sep = "" if not text or text.endswith("\n") else "\n"
    return text + sep + "\n".join(block) + "\n"


def strip_block(text: str, comment: str = "#") -> str:
    lines = text.splitlines()
    span = _block_span(lines, comment)
    if not span:
        return text
    del lines[span[0]:span[1] + 1]
    return "\n".join(lines) + ("\n" if lines and text.endswith("\n") else "")


def load_json(raw: bytes | None, what: str) -> dict[str, Any]:
    if raw is None or not raw.strip():
        return {}
    try:
        data = json.loads(raw.decode(ENCODING))
    except ValueError:
        raise Unsupported(f"{what} is not plain JSON (comments or trailing commas?); not editing it") from None
    if not isinstance(data, dict):
        raise Unsupported(f"{what} is not a JSON object; not editing it")
    return data


def dump_json(data: Any, like: bytes | None) -> bytes:
    """JSON in the indentation of the file it replaces (2 spaces for a new one), newline-ended."""
    indent = DEFAULT_INDENT
    if like:
        second = like.decode(ENCODING, "replace").splitlines()[1:2]
        if second and (n := len(second[0]) - len(second[0].lstrip(" "))):
            indent = n
    return (json.dumps(data, indent=indent, ensure_ascii=False) + "\n").encode(ENCODING)


REMOVE = object()   # a set_paths value: take the key out (undo puts back what it held)


def set_paths(data: dict[str, Any], values: dict[tuple[str, ...], Any]) -> Undo:
    """Set each path (creating parent objects), or remove it for REMOVE; returns the undo record."""
    sets = []
    for path, value in values.items():
        node, created_at = data, None
        for depth, k in enumerate(path[:-1]):
            if not isinstance(node.get(k), dict):
                if k in node:
                    raise Unsupported(f"'{'.'.join(path[:depth + 1])}' is not an object; not editing it")
                node[k] = {}
                created_at = depth if created_at is None else created_at
            node = node[k]
        had = path[-1] in node
        sets.append({"path": list(path), "had": had, "old": node.get(path[-1]), "created_at": created_at})
        if value is REMOVE:
            node.pop(path[-1], None)
        else:
            node[path[-1]] = value
    return {"kind": JSON_KEYS, "sets": sets}


def unset_paths(data: dict[str, Any], undo: Undo) -> dict[str, Any]:
    """Our keys back to what they held (or gone); an object we created goes only once it is
    empty, so keys the user has added to it since stay."""
    for s in reversed(undo["sets"]):
        path, cut = s["path"], s["created_at"]
        chain: list[Any] = [data]
        for k in path[:-1]:
            chain.append(chain[-1].get(k) if isinstance(chain[-1], dict) else None)
        node = chain[-1]
        if not isinstance(node, dict):
            continue
        if s["had"]:
            node[path[-1]] = s["old"]
        else:
            node.pop(path[-1], None)
        if cut is None:
            continue
        for depth in range(len(path) - 2, cut - 1, -1):   # objects we created, deepest first
            if isinstance(chain[depth], dict) and chain[depth + 1] == {}:
                chain[depth].pop(path[depth], None)
    return data


def merged(first: Undo, later: Undo) -> Undo:
    """The first connect's undo record, plus any path a later connect set that it lacks
    (the first record knows what those paths held before we ever touched the file)."""
    if first.get("kind") != JSON_KEYS or later.get("kind") != JSON_KEYS:
        return first
    known = {tuple(s["path"]) for s in first["sets"]}
    return {**first, "sets": first["sets"] + [s for s in later["sets"] if tuple(s["path"]) not in known]}


def undo(raw: bytes, spec: Undo) -> bytes | None:
    """`raw` with our change taken out; None = leave the file as it is (it is no longer ours)."""
    if spec["kind"] == BLOCK:
        return strip_block(raw.decode(ENCODING), spec["comment"]).encode(ENCODING)
    if spec["kind"] == JSON_KEYS:
        return dump_json(unset_paths(load_json(raw, "file"), spec), raw)
    return None
