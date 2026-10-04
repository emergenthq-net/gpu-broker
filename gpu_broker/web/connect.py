"""Connect apps from the dashboard, or a remote machine with one command (see gpu_broker.connect).

Main token only:
    GET  /v1/connect/clients            apps found on the broker's own machine, and which are connected
    POST /v1/connect/{client}           connect one of them  {openwebui_url?, openwebui_token?, claude_code?}
    POST /v1/disconnect/{client}        undo it exactly
    POST /v1/connect/invite             {name}: a one-line installer for another machine
Public (the invite code is the credential, single use, short-lived):
    GET  /connect.sh?invite=CODE        the installer, with this broker's URL, a key and the model filled in
    GET  /connect/gpu-broker-connect.pyz   the connectors as a zipapp (no secrets in it)

The key is issued when the installer is fetched, not when the invite is made, so an invite
that expires unused leaves no key behind. The installer hands the key to the connectors on
standard input (never on a command line, where `ps` would show it), checks the zipapp's
SHA-256 (embedded in the script) before running it, and warns when the broker is reached over
plain http on a host that is neither this machine nor a private network.
"""
from __future__ import annotations

import hashlib
import io
import ipaddress
import secrets
import socket
import threading
import time
import zipfile
from http import HTTPStatus
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response

from .. import mcp_server
from ..broker import Broker
from ..connect import CLIENTS, engine, mcpinfo, openwebui
from ..connect.core import Target, checked
from ..keys import KeyStore

INVITE_TTL_S = 900
INVITE_BYTES = 24
PYZ = "gpu-broker-connect.pyz"
PYZ_MAIN = "import sys\nfrom gpu_broker.connect.cli import main\nsys.exit(main(sys.argv[1:]))\n"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)   # fixed, so the zipapp (and its checksum) is the same on every request
PRIVATE_SUFFIXES = (".local", ".lan", ".home.arpa", ".internal", ".localdomain")
DASHBOARD_KEY = "{host} (dashboard)"
REMOTE_KEY = "remote machine"
SCRIPT = """#!/bin/sh
# gpu-broker connect: points this machine's AI apps (shell, Continue, Cline, Roo, ...) at {url}
# Every file it changes is backed up; undo with: python3 ~/.gpu-broker/connect/{pyz} disconnect
set -eu
{warning}command -v python3 >/dev/null 2>&1 || {{ echo "gpu-broker connect needs python3 (3.9 or later)" >&2; exit 1; }}
umask 077
DIR="$HOME/.gpu-broker/connect"
mkdir -p "$DIR" && chmod 700 "$DIR"
curl -fsSL '{url}/connect/{pyz}' -o "$DIR/{pyz}.part"
GOT=$(python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$DIR/{pyz}.part")
if [ "$GOT" != '{sha}' ]; then
  rm -f "$DIR/{pyz}.part"
  echo "gpu-broker connect: the downloaded program does not match its checksum; stopping" >&2
  exit 1
fi
mv "$DIR/{pyz}.part" "$DIR/{pyz}"
printf '%s\\n' '{key}' | python3 "$DIR/{pyz}" connect --url '{url}' --key-stdin --model '{model}' "$@"
echo "Undo any time: python3 $DIR/{pyz} disconnect"
"""
HTTP_WARNING = ("echo \"warning: {url} is plain http to a host outside this machine and its private network;"
                " the key travels unencrypted\" >&2\n")


def pyz() -> bytes:
    """The connect package as a zipapp: standard library only, runnable with any python3 >= 3.9."""
    buf = io.BytesIO()
    pkg = files("gpu_broker.connect")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def add(name: str, text: str) -> None:
            z.writestr(zipfile.ZipInfo(name, ZIP_TIME), text, zipfile.ZIP_DEFLATED)
        add("__main__.py", PYZ_MAIN)
        add("gpu_broker/__init__.py", "")
        for f in sorted(pkg.iterdir(), key=lambda p: p.name):
            if f.is_file() and f.name.endswith(".py"):   # not __pycache__
                add(f"gpu_broker/connect/{f.name}", f.read_text())
    return buf.getvalue()


def private_host(url: str) -> bool:
    """This machine or a private network: loopback/private IPs, single-label and LAN names."""
    host = (urlsplit(url).hostname or "").lower()
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost" or "." not in host or host.endswith(PRIVATE_SUFFIXES)
    return ip.is_loopback or ip.is_private   # is_private covers link-local and the private ranges


def installer(url: str, key: str, model: str) -> str:
    checked(url, key, model)   # the values are pasted into a shell script
    warn = HTTP_WARNING.format(url=url) if urlsplit(url).scheme == "http" and not private_host(url) else ""
    return SCRIPT.format(url=url, pyz=PYZ, key=key, model=model, sha=hashlib.sha256(pyz()).hexdigest(), warning=warn)


def _safe(url: str, model: str) -> None:
    try:
        checked(url, "", model)
    except ValueError as e:
        raise HTTPException(HTTPStatus.BAD_REQUEST, f"cannot make an installer: {e}") from None


def routers(broker: Broker, keys: KeyStore, home: Path | None = None) -> tuple[APIRouter, APIRouter]:
    admin, public = APIRouter(), APIRouter()
    invites: dict[str, tuple[float, str, str]] = {}   # code -> (expiry, key name, broker URL)
    lock = threading.Lock()
    here = home or Path.home()

    def model() -> str:
        return broker.scheduler.pool.resident or broker.catalog.defaults["resident"]

    def base(request: Request) -> str:
        return str(request.base_url).rstrip("/")

    def local_target(request: Request, opts: dict[str, Any]) -> tuple[Target, dict[str, Any] | None]:
        url, known = base(request), engine.load_manifest(here).get("key") or {}
        _safe(url, model())   # before a key is issued for it
        record = None
        if known.get("broker") == url and known.get("key"):
            key = known["key"]
        else:
            issued = keys.issue(DASHBOARD_KEY.format(host=socket.gethostname()))
            key, record = issued["key"], {"broker": url, "id": issued["id"], "name": issued["name"], "key": issued["key"]}
        options = {k: str(v) for k, v in opts.items() if v and k in ("openwebui_url", "openwebui_token", "claude_code")}
        options |= mcpinfo.options(broker.settings.mcp.enabled and mcp_server.available())   # served at /mcp (web/app.py)
        return Target(url, key, model(), here, {"HOME": str(here)}, options), record

    def client_or_404(name: str) -> Any:
        if name not in CLIENTS:
            raise HTTPException(HTTPStatus.NOT_FOUND, f"unknown client {name!r}")
        return CLIENTS[name]

    @admin.post("/v1/connect/invite")   # before /v1/connect/{name}, which would match "invite"
    def invite(request: Request, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url, name = base(request), str((body or {}).get("name") or REMOTE_KEY).strip() or REMOTE_KEY
        _safe(url, model())
        code = secrets.token_urlsafe(INVITE_BYTES)
        with lock:
            now = time.monotonic()
            for c in [c for c, (exp, _, _) in invites.items() if exp < now]:
                del invites[c]
            invites[code] = (now + INVITE_TTL_S, name, url)
        return {"command": f"curl -fsSL '{url}/connect.sh?invite={code}' | sh", "name": name, "expires_in_s": INVITE_TTL_S}

    @admin.get("/v1/connect/clients")
    def clients(request: Request) -> list[dict[str, Any]]:
        t, linked = Target(base(request), "", "", here, {"HOME": str(here)}), engine.connected(here)
        rows = []
        for name, m in CLIENTS.items():
            found, detail = m.detect(t)
            rows.append({"name": name, "label": m.LABEL, "found": found, "detail": detail, "connected": name in linked})
        return rows

    @admin.post("/v1/connect/{name}")
    def connect(name: str, request: Request, body: dict[str, Any] | None = None) -> dict[str, Any]:
        m = client_or_404(name)
        t, record = local_target(request, body or {})
        plan = m.plan(t)
        if plan.skip and record:   # nothing will use the key just issued
            keys.revoke(record["id"])
            record = None
        return {"report": engine.connect(here, [plan], key=record)}

    @admin.post("/v1/disconnect/{name}")
    def disconnect(name: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        client_or_404(name)
        opts = body or {}
        restore = ({openwebui.NAME: openwebui.restorer(opts["openwebui_url"], opts.get("openwebui_token", ""))}
                   if opts.get("openwebui_url") else {})
        return {"report": engine.disconnect(here, {name}, restore_api=restore)}

    @public.get("/connect.sh", response_class=PlainTextResponse)
    def script(invite: str = "") -> str:
        with lock:
            exp, name, url = invites.pop(invite, (0.0, "", ""))
        if exp < time.monotonic():
            raise HTTPException(HTTPStatus.NOT_FOUND, "this install link has expired or was used; make a new one on the dashboard")
        _safe(url, model())                 # the resident model may have changed since the invite
        issued = keys.issue(name)           # minted now: an unused invite never leaves a key behind
        try:
            return installer(url, issued["key"], model())
        except ValueError:
            keys.revoke(issued["id"])
            raise

    @public.get(f"/connect/{PYZ}")
    def zipapp() -> Response:
        return Response(pyz(), media_type="application/zip")

    return admin, public
