"""Whether a tool should keep its own provider key: the broker has cloud failover on, with a
provider of the tool's API that passes the client's own key through (`pass_client_key`).

Then a connector that can send a custom header (Claude Code, Codex) keeps the tool's own key
and adds the broker key as `x-gpu-broker-key`, so a routed name is answered by the cloud with
the user's own account, and by the local model when the cloud fails. Otherwise nothing changes:
the broker key is the tool's only key, as before. Asked of the broker at connect time with the
client key (PASSTHROUGH_PATH, which a client key may read; /v1/upstreams is
operator-only); a broker that cannot say (older, unreachable, no credential) counts as off.
Standard library.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .core import OWN_KEY_OPT, PASSTHROUGH_PATH

SINGLE = "broker key only"
DUAL = "own provider key + x-gpu-broker-key"


def passthrough_apis(url: str, credential: str, call: Callable[[str, str, str, Any], Any]) -> set[str]:
    """The APIs ('anthropic', 'openai') for which the broker forwards a client's own key upstream."""
    if not credential:
        return set()
    try:
        view = call("GET", url + PASSTHROUGH_PATH, credential, None)
        if not view.get("enabled"):
            return set()
        return {str(a) for a in view.get("apis", [])} & set(OWN_KEY_OPT)
    except (OSError, ValueError, AttributeError):
        return set()


def options(apis: set[str]) -> dict[str, str]:
    return {OWN_KEY_OPT[a]: "1" for a in sorted(apis) if a in OWN_KEY_OPT}


# Clients that get their own `gpu-broker` entry beside the user's provider settings, which
# keep their keys: that entry has no own provider key to keep, so it carries the broker key.
SEPARATE = f"{SINGLE} (a separate gpu-broker entry; your own provider entries keep their keys)"
SHELL = f"{SINGLE} (SDKs read no custom headers from the environment)"
