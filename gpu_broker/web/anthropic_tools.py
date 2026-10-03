"""Anthropic Messages request: `tools`, `tool_choice` and the scalar fields, checked and
translated to their OpenAI equivalents (anthropic_req.py does the messages). A malformed
value is a ValueError, answered as a 400 invalid_request_error."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CUSTOM_TOOL = "custom"
CHOICES = {"auto": "auto", "any": "required", "none": "none"}
TOOL_CHOICE = "tool"
NUMBERS = ("temperature", "top_p")
INTEGERS = ("max_tokens", "top_k")


def text_field(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    return value


def _tool(t: Any, i: int) -> dict[str, Any]:
    where = f"tools[{i}]"
    if not isinstance(t, Mapping):
        raise ValueError(f"{where} must be an object")
    if t.get("type", CUSTOM_TOOL) not in (CUSTOM_TOOL, None):
        raise ValueError(f"hosted tool {t.get('type')!r} is not available on a local model")
    name = text_field(t.get("name"), f"{where}: `name`")
    if not name:
        raise ValueError(f"{where}: `name` must not be empty")
    schema = t.get("input_schema", {})
    if not isinstance(schema, Mapping):
        raise ValueError(f"{where}: `input_schema` must be an object")
    desc = t.get("description", "")
    return {"type": "function", "function": {"name": name, "description": text_field(desc, f"{where}: `description`"),
                                              "parameters": dict(schema)}}


def tools(body: Mapping[str, Any], out: dict[str, Any]) -> None:
    given = body.get("tools")
    if given is not None:
        if not isinstance(given, list):
            raise ValueError("`tools` must be a list")
        if given:
            out["tools"] = [_tool(t, i) for i, t in enumerate(given)]
    choice = body.get("tool_choice")
    if choice is None:
        return
    if not isinstance(choice, Mapping) or (kind := choice.get("type")) not in (*CHOICES, TOOL_CHOICE):
        raise ValueError(f"`tool_choice` must be an object with type one of {[*CHOICES, TOOL_CHOICE]}")
    out["tool_choice"] = ({"type": "function", "function": {"name": text_field(choice.get("name"), "tool_choice `name`")}}
                          if kind == TOOL_CHOICE else CHOICES[kind])
    if choice.get("disable_parallel_tool_use"):
        out["parallel_tool_calls"] = False


def scalars(body: Mapping[str, Any]) -> None:
    for k in NUMBERS + INTEGERS:
        v = body.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int if k in INTEGERS else (int, float))):
            raise ValueError(f"`{k}` must be {'an integer' if k in INTEGERS else 'a number'}")
    stops = body.get("stop_sequences")
    if stops is not None and not (isinstance(stops, list) and all(isinstance(x, str) for x in stops)):
        raise ValueError("`stop_sequences` must be a list of strings")
    if body.get("model") is not None:
        text_field(body["model"], "`model`")
