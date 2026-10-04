"""Per-client keys (main token only): list, issue, revoke. See gpu_broker/keys.py.

    GET    /v1/keys          every key: id, name, first characters, created, last used, revoked
    POST   /v1/keys          {name}: a new key; the response is the only place it is shown
    DELETE /v1/keys/{id}     revoke it (the record is kept)
"""
from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException

from ..keys import KeyStore


def router(keys: KeyStore) -> APIRouter:
    r = APIRouter()

    @r.get("/v1/keys")
    def list_keys() -> list[dict[str, Any]]:
        return keys.list()

    @r.post("/v1/keys")
    def issue(body: dict[str, Any]) -> dict[str, Any]:
        return keys.issue(str(body.get("name") or ""))

    @r.delete("/v1/keys/{kid}")
    def revoke(kid: str) -> dict[str, bool]:
        if not keys.revoke(kid):
            raise HTTPException(HTTPStatus.NOT_FOUND, "no such live key")
        return {"revoked": True}

    return r
