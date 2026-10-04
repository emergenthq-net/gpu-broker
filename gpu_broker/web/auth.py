"""Who may call what.

Two credentials, each accepted as `Authorization: Bearer <value>` (OpenAI SDKs),
`x-api-key: <value>` (Anthropic SDKs) or `x-gpu-broker-key: <value>` (beside a cloud key the
client passes through to an upstream provider; see web/failover.py):
- the main token ($BROKER_TOKEN), compared in constant time on both headers every time;
- a per-client key issued by `gpu-broker connect` or the dashboard (keys.py), looked up by
  its SHA-256.
Each route has a scope:
- MODEL (chat, messages, embeddings, the model list): the main token or a client key. A
  client key's name becomes the job's requester, whatever the request claims.
- OPERATOR (jobs, status, events, stats, metrics, sessions, the dashboard's data): the main
  token only. These show every caller's prompts and results, so client keys get 403.
- ADMIN (keys, connect, quiesce): the main token, and only as `Authorization: Bearer`.
With no main token set, everything is refused, client keys included. Every credential header is
checked, the main token's included, and each that carried any broker credential is recorded on
the request (BROKER_CREDS), so none of them is ever passed upstream as a provider key. Values
are compared trimmed, the Bearer scheme in any case and spacing, and auth leaves a check on the
request (MAIN_CHECK) so web/failover.py can refuse to forward the main token in any form.
"""
from __future__ import annotations

import hmac
from collections.abc import Callable
from enum import StrEnum
from http import HTTPStatus

from fastapi import HTTPException, Request

from ..constants import API_KEY_HEADER, AUTH_SCHEME, BROKER_KEY_HEADER
from ..keys import KeyStore
from .jobs import CLIENT_ID_STATE, CLIENT_STATE

Auth = Callable[..., None]
BEARER_PREFIX = AUTH_SCHEME + " "
REFUSED = "bad token"
MAIN_ONLY = "this route needs the main broker token, not a client key"
AUTH_HEADER = "authorization"
BROKER_CREDS = "broker_creds"   # request.state: the header names that carried a broker credential
MAIN_CHECK = "is_main_token"    # request.state: value -> is it the main token (constant time)


class Scope(StrEnum):
    MODEL = "model"
    OPERATOR = "operator"
    ADMIN = "admin"


def bearer_value(raw: str) -> str | None:
    """The credential in an Authorization header (`Bearer x`, any case and spacing), else None."""
    scheme, _, rest = raw.strip().partition(" ")
    return rest.strip() if scheme.lower() == AUTH_SCHEME.lower() and rest.strip() else None


def presented(request: Request) -> dict[str, str]:
    """The credentials a request carries, by header name, trimmed."""
    found = {AUTH_HEADER: v} if (v := bearer_value(request.headers.get(AUTH_HEADER, ""))) else {}
    return found | {h: v for h in (API_KEY_HEADER, BROKER_KEY_HEADER) if (v := request.headers.get(h, "").strip())}


def same(value: str, token: str) -> bool:
    return bool(token) and hmac.compare_digest(value.encode(), token.encode())


def main_headers(request: Request, token: str, bearer_only: bool = False) -> set[str]:
    """The headers carrying the main token. All three are compared in constant time, every time."""
    values = {AUTH_HEADER: bearer_value(request.headers.get(AUTH_HEADER, "")) or "",
              API_KEY_HEADER: request.headers.get(API_KEY_HEADER, "").strip(),
              BROKER_KEY_HEADER: request.headers.get(BROKER_KEY_HEADER, "").strip()}
    by = {h: same(v, token) and (h == AUTH_HEADER or not bearer_only) for h, v in values.items()}
    return {h for h, ok in by.items() if ok}


def make_auth(token: str, keys: KeyStore | None = None, scope: Scope = Scope.MODEL) -> Auth:
    """A FastAPI dependency for routes of `scope`."""
    def require(request: Request) -> None:
        setattr(request.state, MAIN_CHECK, lambda value: same(value.strip(), token))
        main = main_headers(request, token, bearer_only=scope is Scope.ADMIN)
        found: tuple[str, str] | None = None   # (id, name) of the client key
        carried = set(main)
        if token and keys is not None:
            for header, value in presented(request).items():
                if header not in main and (hit := keys.identify(value)):
                    found, carried = found or hit, carried | {header}
        setattr(request.state, BROKER_CREDS, carried)
        if main:
            setattr(request.state, CLIENT_STATE, None)
            setattr(request.state, CLIENT_ID_STATE, None)
            return
        if found is None:
            raise HTTPException(HTTPStatus.UNAUTHORIZED, REFUSED)
        if scope is not Scope.MODEL:
            raise HTTPException(HTTPStatus.FORBIDDEN, MAIN_ONLY)
        setattr(request.state, CLIENT_ID_STATE, found[0])
        setattr(request.state, CLIENT_STATE, found[1])
    return require
