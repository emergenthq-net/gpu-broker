"""A provider credential and the header it travels in.

A client's key goes back out in the style it came in: an Anthropic OAuth or auth-token client
sends `Authorization: Bearer`, an API-key client `x-api-key`, and Anthropic answers each only in
its own header. An operator key uses the API's usual header.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..constants import AUTH_SCHEME
from .config import Api

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


@dataclass(frozen=True)
class Cred:
    value: str
    bearer: bool | None = None   # None: the API's usual header (Anthropic x-api-key, OpenAI Bearer)

    def __repr__(self) -> str:   # never the value
        return f"Cred(bearer={self.bearer})"


def auth_headers(api: Api, cred: Cred) -> dict[str, str]:
    bearer = cred.bearer if cred.bearer is not None else api is Api.OPENAI
    return {"authorization": f"{AUTH_SCHEME} {cred.value}"} if bearer else {"x-api-key": cred.value}
