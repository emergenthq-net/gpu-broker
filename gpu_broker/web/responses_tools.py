"""OpenAI Responses request: `tools`, `tool_choice`, `text.format` and the scalar fields,
checked and translated to their chat equivalents (responses_req.py does the input).

Function tools pass through. A `custom` (freeform) tool becomes a function with one string
argument, `input`; the output side turns its calls back into custom_tool_call items. Hosted
tools (run by OpenAI: web search, file search, code interpreter, image generation, computer
use, MCP, namespace groups) cannot run on a local model and are left out, named by
`dropped_tools`. `local_shell`, and any other type, is refused. A malformed value is a
ValueError, answered as a 400."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .anthropic_tools import text_field

FUNCTION, CUSTOM, INPUT = "function", "custom", "input"
HOSTED = frozenset({"web_search", "web_search_preview", "file_search", "code_interpreter", "image_generation",
                    "computer_use", "computer_use_preview", "mcp", "namespace"})
CUSTOM_NOTE = "Free-form tool: put the whole input, as plain text, in the `input` argument."
CHOICES = frozenset({"auto", "none", "required"})
UNSUPPORTED = {"background": "background mode", "prompt": "a prompt template", "conversation": "a stored conversation"}
NUMBERS, INTEGERS = ("temperature", "top_p"), ("max_output_tokens", "top_logprobs")
FORMATS = frozenset({"json_schema", "json_object", "text"})


def label(value: Any, where: str) -> str | None:
    """A type/role/mode tag: absent, or a string (a list or object here is a 400, not a crash)."""
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    return value


def hosted(kind: str | None) -> bool:
    return kind in HOSTED or (kind or "").startswith("web_search_preview")


def dropped_tools(body: Mapping[str, Any]) -> list[str]:
    """Names of the tools left out because a local model cannot run them (hosted tools, namespaces)."""
    given = body.get("tools")
    if not isinstance(given, list):
        return []
    return [str(t.get("name") or t.get("type")) for t in given
            if isinstance(t, Mapping) and isinstance(t.get("type"), str) and hosted(t["type"])]


def _tool(t: Mapping[str, Any], kind: str, where: str) -> dict[str, Any]:
    name = text_field(t.get("name"), f"{where}: `name`")
    desc = text_field(t.get("description") or "", f"{where}: `description`")
    if kind == CUSTOM:
        params: Any = {"type": "object", "properties": {INPUT: {"type": "string"}}, "required": [INPUT]}
        desc = f"{desc}\n\n{CUSTOM_NOTE}" if desc else CUSTOM_NOTE
    else:
        params = t.get("parameters") or {}
    if not name or not isinstance(params, Mapping):
        raise ValueError(f"{where}: needs a non-empty `name` and an object `parameters`")
    fn = {"name": name, "description": desc, "parameters": dict(params)}
    return {"type": FUNCTION, FUNCTION: fn | ({"strict": t["strict"]} if isinstance(t.get("strict"), bool) else {})}


def tools(body: Mapping[str, Any], out: dict[str, Any]) -> frozenset[str]:
    """Translate the tools into `out`; the names of the custom (freeform) tools."""
    given = body.get("tools")
    if given is not None and not isinstance(given, list):
        raise ValueError("`tools` must be a list")
    fns, custom = [], set()
    for i, t in enumerate(given or []):
        if not isinstance(t, Mapping):
            raise ValueError(f"tools[{i}] must be an object")
        kind = label(t.get("type"), f"tools[{i}]: `type`")
        if kind in (FUNCTION, CUSTOM):
            fns.append(_tool(t, kind, f"tools[{i}]"))
            if kind == CUSTOM:
                custom.add(fns[-1][FUNCTION]["name"])
        elif not hosted(kind):
            raise ValueError(f"tools[{i}]: tool type {kind!r} is not supported by the broker "
                             "(it offers function and custom tools to the model)")
    fns = _choose(body.get("tool_choice"), fns, out)
    if fns:
        out["tools"] = fns
        if isinstance(body.get("parallel_tool_calls"), bool):
            out["parallel_tool_calls"] = body["parallel_tool_calls"]
    return frozenset(custom & {f[FUNCTION]["name"] for f in fns})


def _choose(choice: Any, fns: list[dict[str, Any]], out: dict[str, Any]) -> list[dict[str, Any]]:
    """Set `tool_choice`; the tools to offer (allowed_tools narrows them to its subset)."""
    if choice is None or not fns:
        return fns
    kind = label(choice.get("type"), "`tool_choice.type`") if isinstance(choice, Mapping) else None
    if isinstance(choice, str) and choice in CHOICES:
        out["tool_choice"] = choice
    elif kind in (FUNCTION, CUSTOM):
        out["tool_choice"] = {"type": FUNCTION, FUNCTION: {"name": text_field(choice.get("name"), "tool_choice `name`")}}
    elif kind == "allowed_tools" and label(choice.get("mode"), "`tool_choice.mode`") in CHOICES:
        allowed = choice.get("tools")
        if not isinstance(allowed, list) or not all(isinstance(a, Mapping) for a in allowed):
            raise ValueError("`tool_choice.tools` must be a list of tool objects")
        names = {a.get("name") for a in allowed if isinstance(a.get("name"), str)}
        fns = [f for f in fns if f[FUNCTION]["name"] in names]
        if not fns and choice["mode"] == "required":
            raise ValueError("`tool_choice` requires a tool, but none of its allowed tools can run here")
        if fns:
            out["tool_choice"] = choice["mode"]
    else:
        raise ValueError(f"`tool_choice` must be one of {sorted(CHOICES)}, a function choice or allowed_tools")
    return fns


def format_(body: Mapping[str, Any], out: dict[str, Any]) -> None:
    text = body.get("text")
    if text is None:
        return
    fmt = text.get("format") if isinstance(text, Mapping) else None
    if fmt is None:
        return
    if not isinstance(fmt, Mapping) or label(fmt.get("type"), "`text.format.type`") not in FORMATS:
        raise ValueError(f"`text.format.type` must be one of {sorted(FORMATS)}")
    if fmt["type"] == "json_object":
        out["response_format"] = {"type": "json_object"}
    elif fmt["type"] == "json_schema":
        schema = fmt.get("schema")
        if not isinstance(schema, Mapping):
            raise ValueError("`text.format.schema` must be an object")
        spec = {"name": fmt.get("name") or "response", "schema": dict(schema)}
        out["response_format"] = {"type": "json_schema", "json_schema": spec | ({"strict": fmt["strict"]} if "strict" in fmt else {})}


def scalars(body: Mapping[str, Any], out: dict[str, Any]) -> None:
    for key, what in UNSUPPORTED.items():
        if body.get(key):
            raise ValueError(f"`{key}`: {what} is not supported by the broker")
    for k in NUMBERS + INTEGERS:
        v = body.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int if k in INTEGERS else (int, float))):
            raise ValueError(f"`{k}` must be a number" if k in NUMBERS else f"`{k}` must be an integer")
    for k in NUMBERS:
        if body.get(k) is not None:
            out[k] = body[k]
    if body.get("max_output_tokens") is not None:
        out["max_tokens"] = body["max_output_tokens"]
