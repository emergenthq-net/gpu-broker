"""Refusing new work while the broker is quiesced (POST /v1/admin/quiesce).

A quiesce comes before a restart. Jobs queued before it are re-queued by the next start, but a
caller arriving now should simply come back after the restart. So a quiesced broker queues nothing: a new job or chat gets HTTP 503 with Retry-After, which the
OpenAI and Anthropic SDKs (and other well-behaved clients) retry after the restart. Resume
undoes it without a restart.
"""
from __future__ import annotations

QUIESCED = "the broker is quiesced for a restart; retry shortly"


class Quiesced(Exception):
    def __init__(self, retry_after_s: int) -> None:
        super().__init__(QUIESCED)
        self.retry_after_s = retry_after_s
