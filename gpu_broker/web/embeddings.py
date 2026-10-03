"""POST /v1/embeddings — OpenAI-compatible, served by a catalog model with `caps: [embed]`.

The model asked for is used when it is an embedding model; otherwise (a hosted name such as
`text-embedding-3-small`, after `model_map`) the best ready embedding model stands in and
`x_broker.substitution` says so. With no embedding model in the catalog the answer is an
OpenAI-style 404 (`code: model_not_found`). Embeddings always run as queued jobs, so a model
switch, when one is needed, is decided by the GPU thread like any other.
"""
from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..broker import Broker, validate_request
from ..catalog import Catalog
from ..chat import is_interactive
from ..constants import EMBED_CAP, EMBED_KEY, INTERACTIVE_KEY, PRIORITY_HEADER, REQUESTER_HEADER, Kind
from ..modelmap import joined, map_name
from ..resolve import lookup, runnable
from . import completion
from .jobs import client

NO_EMBED_MODEL = "no embedding model is configured: add a catalog model with `caps: [embed]`"
NOT_READY = "no embedding model is ready to run"
NOT_FOUND_CODE = "model_not_found"


def embed_models(catalog: Catalog) -> list[str]:
    return [k for k, m in catalog.models.items() if m.get("kind") == Kind.LLM and EMBED_CAP in m.get("caps", [])]


def pick(broker: Broker, requested: str) -> tuple[str, str | None]:
    """(catalog key to run, substitution note); HTTP 404 when no embedding model can."""
    cat = broker.catalog
    candidates = embed_models(cat)
    if not candidates:
        raise HTTPException(HTTPStatus.NOT_FOUND, {"error": NO_EMBED_MODEL, "code": NOT_FOUND_CODE})
    key = lookup(cat.data, requested) if requested else None
    mapped = map_name(broker.settings.model_map, key is not None, requested, cat.defaults["resident"]) if requested else None
    if mapped:
        key = lookup(cat.data, mapped.target)
    if key in candidates:
        return str(key), mapped.note if mapped else None
    ready = [(cat.models[k].get("quality", 0), k) for k in candidates if runnable(cat.data, k)]
    if not ready:
        raise HTTPException(HTTPStatus.NOT_FOUND, {"error": NOT_READY, "code": NOT_FOUND_CODE})
    best = max(ready)[1]
    return best, joined(mapped.note if mapped else None, f"'{requested}' is not an embedding model; using '{best}'")


def router(broker: Broker) -> APIRouter:
    r = APIRouter()

    @r.post("/v1/embeddings")
    def embeddings(body: dict[str, Any], request: Request) -> dict[str, Any]:
        validate_request(body)
        requester = request.headers.get(REQUESTER_HEADER) or client(request)
        requested = body.get("model") or ""
        key, note = pick(broker, requested)
        interactive = is_interactive(broker.catalog, request.headers.get(PRIORITY_HEADER, ""), requester)
        job = {**body, "model": key, "kind": Kind.LLM, "caps": [EMBED_CAP], EMBED_KEY: True, INTERACTIVE_KEY: interactive}
        try:
            out = completion.queued(broker, job, requester, requested or key, note)
        except completion.ChatFailed as e:
            raise HTTPException(e.status, {"error": e.message, "x_broker": e.meta}) from None
        return out.result or {}

    return r
