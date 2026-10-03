"""`gpu-broker demo`: serve an assembled demo (assemble.py) until Ctrl+C.

`main` binds the port first, so nothing is printed, opened or started unless the demo really
owns it: a link to a port some other program holds (a real broker on 8095, say) would hand
that program the demo's token. Then it prints the link, opens it in a browser where there is
one, starts the broker and the traffic, and serves until Ctrl+C.
"""
from __future__ import annotations

import errno
import os
import pathlib
import secrets
import signal
import socket
import sys
import tempfile
import webbrowser

from fastapi import FastAPI

from ..browser import open_browser
from ..constants import APP_NAME
from . import content
from .assemble import Demo as Demo
from .assemble import Options as Options
from .assemble import base_url as base_url
from .assemble import build as build
from .assemble import memory as memory
from .tuning import DATA_PREFIX, DEMO_HOST, DEMO_PORT, TOKEN_BYTES

LOG_LEVEL = "warning"  # keep the terminal to the banner; the dashboard shows what happens
BACKLOG = 128          # connections the bound port holds until the server starts accepting them
ANY_PORT = 0
EXIT_NO_PORT = 1
PORT_TAKEN = "{app}: port {port} on {host} is in use (another gpu-broker?). Pick another --port, or leave it out."


def bind(host: str, port: int | None) -> socket.socket:
    """A listening socket on host:port, or with no port on DEMO_PORT or else any free one.
    Raises OSError (EADDRINUSE) when an asked-for port is taken. No SO_REUSEADDR: with it, some
    systems let a second program bind a port another one is already serving."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    for p in (port,) if port is not None else (DEMO_PORT, ANY_PORT):
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.bind((host, p))
        except OSError as e:
            sock.close()
            if port is not None or e.errno != errno.EADDRINUSE:
                raise
            continue
        sock.listen(BACKLOG)
        return sock
    raise AssertionError("unreachable: binding port 0 picks a free port")


def serve(app: FastAPI, sock: socket.socket, graceful_s: int) -> None:
    import uvicorn
    uvicorn.Server(uvicorn.Config(app, log_level=LOG_LEVEL, timeout_graceful_shutdown=graceful_s)).run(sockets=[sock])


def _exit(signum: int, frame: object) -> None:
    raise SystemExit(0)


def main(host: str | None, port: int | None, quiet: bool, browser: bool = True) -> int:
    host = host or DEMO_HOST
    try:
        sock = bind(host, port)
    except OSError as e:
        taken = e.errno == errno.EADDRINUSE
        print(PORT_TAKEN.format(app=APP_NAME, host=host, port=port) if taken else f"{APP_NAME}: cannot listen on {host}: {e}",
              file=sys.stderr)
        return EXIT_NO_PORT
    port = sock.getsockname()[1]
    token = secrets.token_urlsafe(TOKEN_BYTES)
    with sock, tempfile.TemporaryDirectory(prefix=DATA_PREFIX, ignore_cleanup_errors=True) as d:
        demo = build(pathlib.Path(d), host, port, token)
        dash = f"{demo.url}/dash#token={token}"
        print(content.BANNER.format(url=demo.url, token=token,
                                    traffic=content.TRAFFIC_OFF if quiet else content.TRAFFIC_ON), flush=True)
        if browser:   # the port is listening: the page's first request waits until the server accepts it
            open_browser(dash, os.environ, sys.platform, webbrowser.open)
        # uvicorn re-raises SIGTERM once it has shut down; make that an exit that still runs the
        # cleanup below (the default would end the process and leave the data folder behind).
        signal.signal(signal.SIGTERM, _exit)
        demo.broker.start()
        if not quiet:
            demo.traffic.start()
        try:
            serve(demo.app, sock, demo.broker.settings.server.graceful_shutdown_s)
        finally:
            demo.traffic.close()
            demo.broker.stop()
    return 0
