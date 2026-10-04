"""Conversations kept for `previous_response_id`, in memory, for a limited time.

A stored response is the conversation up to and including it, as chat messages (without the
request's `instructions`, which OpenAI does not carry over to the next response). It is kept
for `RESPONSE_TTL_S`, at most `RESPONSE_STORE_MAX` of them, within a byte budget per entry
(a larger conversation is not kept) and in total (the oldest go first until it fits), and
only for the caller that created it (`jobs.identity`): another caller naming its id gets
"not found", exactly as for an id that never existed. Requests with `store: false` (Codex
sends that and resends the whole conversation each turn) are not kept. Nothing survives a
restart.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any

RESPONSE_TTL_S = 3600.0
RESPONSE_STORE_MAX = 256
ENTRY_BYTES, TOTAL_BYTES = 8 * 2**20, 128 * 2**20   # the defaults; settings.limits overrides them
Entry = tuple[str, float, int, list[dict[str, Any]]]   # owner, expires, bytes, conversation


def size(conversation: list[dict[str, Any]]) -> int:
    return len(json.dumps(conversation, separators=(",", ":")).encode())


class ResponseStore:
    def __init__(self, ttl_s: float = RESPONSE_TTL_S, cap: int = RESPONSE_STORE_MAX,
                 entry_bytes: int = ENTRY_BYTES, total_bytes: int = TOTAL_BYTES) -> None:
        self.ttl_s, self.cap, self.entry_bytes, self.total_bytes = ttl_s, cap, entry_bytes, total_bytes
        self._lock = threading.Lock()
        self._kept: OrderedDict[str, Entry] = OrderedDict()
        self.bytes = 0

    def _drop(self, rid: str) -> None:
        self.bytes -= self._kept.pop(rid)[2]

    def _expire(self, now: float) -> None:
        for rid in [r for r, e in self._kept.items() if e[1] <= now]:
            self._drop(rid)

    def put(self, rid: str, owner: str, conversation: list[dict[str, Any]]) -> bool:
        """Keep it (False if it is over the per-entry budget, and so not kept)."""
        n = size(conversation)
        if n > self.entry_bytes:
            return False
        with self._lock:
            now = time.monotonic()
            self._expire(now)
            if rid in self._kept:
                self._drop(rid)
            self._kept[rid] = (owner, now + self.ttl_s, n, conversation)
            self.bytes += n
            while len(self._kept) > self.cap or self.bytes > self.total_bytes:
                self._drop(next(iter(self._kept)))
        return True

    def get(self, rid: str, owner: str) -> list[dict[str, Any]] | None:
        with self._lock:
            self._expire(time.monotonic())
            kept = self._kept.get(rid)
            return list(kept[3]) if kept is not None and kept[0] == owner else None
