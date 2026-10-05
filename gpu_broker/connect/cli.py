"""`gpu-broker connect | disconnect | clients` — the command line for the connectors.

    gpu-broker connect [--url URL] [--key KEY | --key-stdin] [--only shell,continue] [--claude-code] [--dry-run]
    gpu-broker disconnect [--only ...] [--revoke] [--dry-run]
    gpu-broker clients                      # what is installed here, and what is connected

Without --key, connect reuses the key it issued last time for this broker, or asks the
broker for a new one named after this machine (needs the main token: $BROKER_TOKEN or
--token). `--key-stdin` reads the key from standard input, so it never shows in a process
list (connect.sh uses it). The default model is the broker's resident model. Every URL, key
and model, whether given or returned by the broker, is checked (core.checked) before use.

`disconnect --revoke` needs the main token and checks for it before changing anything; the
key record is dropped only once the key is revoked. With --only it is refused while other
clients still use the key. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import CLIENTS, engine, mcpinfo, openwebui, upstreams
from .core import Target, checked
from .mcpinfo import serves_mcp

URL_ENV, TOKEN_ENV = "GPU_BROKER_URL", "BROKER_TOKEN"
DEFAULT_URL = "http://127.0.0.1:8095"
PLACEHOLDER_KEY = "gbk_ISSUED-AT-CONNECT"   # (dry run) shaped to pass core.checked
PLACEHOLDER_MODEL = "BROKER-DEFAULT-MODEL"
TIMEOUT_S = 15
EXIT_OK, EXIT_ERROR = 0, 1
Api = Callable[[str, str, str, Any], Any]


def api(method: str, url: str, credential: str, body: Any = None) -> Any:
    """One JSON call to the broker."""
    if urlsplit(url).scheme not in ("http", "https"):
        raise ValueError(f"broker URL {url!r} is not http(s)")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Authorization": f"Bearer {credential}", "Content-Type": "application/json"},  # noqa: S310 — scheme checked
                                 method=method)
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:  # noqa: S310 — scheme checked above
        return json.load(r)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="gpu-broker", description="Point this machine's AI apps at a gpu-broker, or back.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("connect", "disconnect", "clients"):
        p = sub.add_parser(name)
        p.add_argument("--url", help=f"the broker (default: ${URL_ENV} or {DEFAULT_URL})")
        p.add_argument("--only", help=f"comma-separated clients: {', '.join(CLIENTS)}")
        p.add_argument("--dry-run", action="store_true", help="print the plan; change nothing")
        p.add_argument("--token", help=f"the broker's main token (default: ${TOKEN_ENV}); issues or revokes keys")
        p.add_argument("--openwebui-url")
        p.add_argument("--openwebui-token", help="an Open WebUI admin API key")
    c = sub.choices["connect"]
    keys = c.add_mutually_exclusive_group()
    keys.add_argument("--key", help="a client key already issued for this machine")
    keys.add_argument("--key-stdin", action="store_true", help="read the client key from standard input")
    c.add_argument("--name", help="name for a newly issued key (default: this machine's hostname)")
    c.add_argument("--model", help="model id clients ask for (default: the broker's resident model)")
    c.add_argument("--claude-code", action="store_true", help="also point Claude Code at the broker (opt-in)")
    sub.choices["disconnect"].add_argument("--revoke", action="store_true", help="also revoke the key connect issued")
    return ap


def _key(a: argparse.Namespace, url: str, token: str, home: Path, call: Api) -> tuple[str, dict[str, Any] | None, bool]:
    """(key, manifest record of a key we issued, whether it was issued just now). A new key is
    recorded before anything else can fail, so it is never left live with nothing recording it."""
    if a.key_stdin:
        return stdin().strip(), None, False
    if a.key:
        return a.key, None, False
    known = engine.load_manifest(home).get("key") or {}
    if known.get("broker") == url and known.get("key"):
        return known["key"], known, False
    if a.dry_run:
        return PLACEHOLDER_KEY, None, False
    if not token:
        raise SystemExit(f"connect needs --key, or the main token (${TOKEN_ENV} or --token) to issue one")
    issued = call("POST", url + "/v1/keys", token, {"name": a.name or socket.gethostname()})
    record = {"broker": url, "id": issued["id"], "name": issued["name"], "key": issued["key"]}
    engine.remember_key(home, record)
    return issued["key"], record, True


def _abandon(record: Mapping[str, Any], token: str, home: Path, call: Api, out: Callable[[str], None]) -> None:
    """Revoke a key this run issued and then could not use; if that fails, it stays recorded."""
    try:
        call("DELETE", f"{record['broker']}/v1/keys/{record['id']}", token, None)
    except (OSError, ValueError) as e:
        out(f"the key '{record['name']}' issued for this run could not be revoked ({e}); it is still recorded. "
            f"Remove it with `gpu-broker disconnect --revoke`, or revoke key {record['id']} on the dashboard")
        return
    engine.forget_key(home)
    out(f"revoked the key '{record['name']}' issued for this run")


def _model(a: argparse.Namespace, url: str, key: str, token: str, call: Api) -> str:
    """--model, else the resident model (/v1/status needs the main token), else the first listed."""
    if a.model:
        return str(a.model)
    try:
        status = call("GET", url + "/v1/status", token, None) if token else {}
        if status.get("resident_llm"):
            return str(status["resident_llm"])
        models = call("GET", url + "/v1/models", key, None)["data"]
        return str(models[0]["id"])
    except (OSError, ValueError, KeyError, IndexError):
        if a.dry_run:
            return PLACEHOLDER_MODEL
        raise SystemExit(f"could not ask {url} for its default model; give --model") from None


def target(a: argparse.Namespace, url: str, key: str, model: str, home: Path, env: Mapping[str, str], mcp: bool = False,
           own_keys: set[str] | None = None) -> Target:
    """`own_keys`: the APIs whose tools keep their own provider key (upstreams.passthrough_apis)."""
    opts = {"openwebui_url": a.openwebui_url or "", "openwebui_token": a.openwebui_token or "",
            "claude_code": "1" if getattr(a, "claude_code", False) else ""}
    return Target(url, key, model, home, env,
                  {k: v for k, v in opts.items() if v} | mcpinfo.options(mcp) | upstreams.options(own_keys or set()))


def stdin() -> str:
    return sys.stdin.readline()


def _disconnect(a: argparse.Namespace, only: set[str] | None, token: str, home: Path, call: Api, webui: openwebui.Http,
                out: Callable[[str], None]) -> int:
    key = engine.load_manifest(home).get("key")
    if a.revoke and not a.dry_run:
        if not token:
            out(f"--revoke needs the main token (${TOKEN_ENV} or --token); nothing was changed")
            return EXIT_ERROR
        if only is not None and (others := engine.connected(home) - only):
            out(f"--revoke with --only would cut off {', '.join(sorted(others))}, which use the same key; "
                "disconnect them too, or leave out --revoke; nothing was changed")
            return EXIT_ERROR
    restore = {openwebui.NAME: openwebui.restorer(a.openwebui_url, a.openwebui_token or "", webui)} if a.openwebui_url else {}
    for line in engine.disconnect(home, only, a.dry_run, restore):
        out(line)
    if a.revoke and key and not a.dry_run:
        try:
            call("DELETE", f"{key['broker']}/v1/keys/{key['id']}", token, None)
        except (OSError, ValueError) as e:
            out(f"could not revoke key '{key['name']}' ({e}); it is still recorded, run disconnect --revoke again")
            return EXIT_ERROR
        engine.forget_key(home)
        out(f"revoked key '{key['name']}'")
    return EXIT_OK


def main(argv: list[str], env: Mapping[str, str] | None = None, home: Path | None = None, call: Api = api,
         webui: openwebui.Http = openwebui.http, out: Callable[[str], None] = print) -> int:
    env = os.environ if env is None else env
    home = home or Path(env.get("HOME") or Path.home())
    a = parser().parse_args(argv)
    url = (a.url or env.get(URL_ENV) or DEFAULT_URL).rstrip("/")
    token = a.token or env.get(TOKEN_ENV, "")
    only = set(a.only.split(",")) if a.only else None
    unknown = (only or set()) - set(CLIENTS)
    if unknown:
        out(f"unknown clients: {', '.join(sorted(unknown))} (known: {', '.join(CLIENTS)})")
        return EXIT_ERROR
    chosen = [m for n, m in CLIENTS.items() if only is None or n in only]
    try:
        checked(url, "", "")
    except ValueError as e:
        out(f"{a.cmd} refused: {e}")
        return EXIT_ERROR
    if a.cmd == "clients":
        t, linked, modes = target(a, url, "", "", home, env), engine.connected(home), engine.key_modes(home)
        for m in chosen:
            found, detail = m.detect(t)
            mode = modes.get(m.NAME) or getattr(m, "KEY_MODE", "")   # what it sends now, else what it would
            out(f"{m.NAME:12} {'connected' if m.NAME in linked else 'found' if found else 'not found':10} {detail}"
                + (f"  [keys: {mode}]" if mode else ""))
        return EXIT_OK
    if a.cmd == "disconnect":
        return _disconnect(a, only, token, home, call, webui, out)
    key, record, fresh = _key(a, url, token, home, call)
    try:
        own = upstreams.passthrough_apis(url, key if key != PLACEHOLDER_KEY else token, call)
        t = target(a, url, key, _model(a, url, key, token, call), home, env, serves_mcp(url, call), own)
    except (ValueError, SystemExit) as e:   # an unsafe key or model, or no model to be had
        out(f"connect refused: {e}")
        if record and fresh:                # a reused key stays: other clients may use it
            _abandon(record, token, home, call, out)
        return EXIT_ERROR
    ours = openwebui.NAME in engine.connected(home)
    plans = [openwebui.plan(t, webui, ours) if m is openwebui else m.plan(t) for m in chosen]
    changed: list[Path] = []
    for line in engine.connect(home, plans, a.dry_run, record, changed):
        out(line)
    if not a.dry_run:   # exactly which files now differ (keys included), so nothing changes unseen
        out("files changed: " + (", ".join(map(str, changed)) or "none"))
    return EXIT_OK
