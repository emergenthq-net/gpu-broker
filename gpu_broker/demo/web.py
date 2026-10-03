"""The demo's stand-ins for ComfyUI's pages, served by the broker itself.

`/view` returns a placeholder output (ComfyUI's own /view, for the demo's output folder
only); `/` is where the dashboard's "Use in ComfyUI" button lands, and says what a real
install would show there. Both live under a per-run secret path, `/comfy/<key>/`: the
placeholders repeat their job's prompt, and the dashboard opens them as plain links, which
cannot carry the bearer token. The path reaches a browser only through answers that need the
token (`/v1/ui`, job results), so knowing it proves the same thing.
"""
from __future__ import annotations

import hmac
import pathlib
import re
from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from ..drivers import validate

FILE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
SUBFOLDER = re.compile(r"^[A-Za-z0-9_/-]{0,128}$")
OUTPUT_TYPE = "output"
PNG_MEDIA_TYPE = "image/png"
LANDING = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>gpu-broker demo</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:40rem;margin:4rem auto;padding:0 1rem}</style></head>
<body><h1>gpu-broker demo</h1>
<p>In a real install, ComfyUI (or the model's own web page) opens here once gpu-broker has freed
the GPU for you. The demo has no ComfyUI, so this page stands in for it.</p>
<p>Back to the <a href="/dash">dashboard</a>. Press <b>Done - give GPU back</b> there to hand
the GPU back to the chat model.</p></body></html>"""


def router(output_dir: pathlib.Path, prefix: str, key: str) -> APIRouter:
    """The stand-ins at `<prefix><key>/` and `<prefix><key>/view`; any other key is not found."""
    r = APIRouter(prefix=prefix.rstrip("/"))

    def check(given: str) -> None:
        if not hmac.compare_digest(given.encode(), key.encode()):
            raise HTTPException(HTTPStatus.NOT_FOUND)

    @r.get("/{given}/", response_class=HTMLResponse)
    def landing(given: str) -> str:
        check(given)
        return LANDING

    @r.get("/{given}/view")
    def view(given: str, filename: str, subfolder: str = "", kind: str = Query(OUTPUT_TYPE, alias="type")) -> FileResponse:
        check(given)
        if kind != OUTPUT_TYPE or not FILE_NAME.match(filename) or not SUBFOLDER.match(subfolder):
            raise HTTPException(HTTPStatus.NOT_FOUND)
        try:
            path = pathlib.Path(validate.inside(str(output_dir), subfolder, filename))
        except ValueError:
            raise HTTPException(HTTPStatus.NOT_FOUND) from None
        if not path.is_file():
            raise HTTPException(HTTPStatus.NOT_FOUND)
        return FileResponse(path, media_type=PNG_MEDIA_TYPE)

    return r
