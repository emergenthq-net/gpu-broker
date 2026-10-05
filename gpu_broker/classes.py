"""A job's class, interactive or background, and who may claim which.

Without a claim the requester decides: callers in `defaults.background_requesters` are
background, everyone else interactive. A claim is the `x-priority` header, else the request's
own `interactive` flag. Under `fair` (strict) a claim may always lower the class but raises it
only for a requester allowed to (`scheduler.may_claim_interactive`; default: every requester not
in `background_requesters`), so a batch client cannot jump the line by saying so. Under `fifo`
the header wins as it did before `fair`.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .catalog import Catalog
from .constants import INTERACTIVE_KEY, Priority
from .policy import Fifo


@dataclass(frozen=True)
class Classes:
    strict: bool = True
    may_claim: frozenset[str] | None = None   # None: every requester that is not background

    @classmethod
    def of(cls, policy: str, may_claim: Iterable[str] | None) -> Classes:
        """`fifo` leaves classes as sent; any other policy is strict."""
        return cls(policy != Fifo.name, None if may_claim is None else frozenset(may_claim))

    def interactive(self, catalog: Catalog, priority: str, requester: str, claimed: bool | None = None) -> bool:
        background = requester in catalog.defaults.get("background_requesters", [])
        p = priority.strip().lower()
        want = p == Priority.INTERACTIVE if p in set(Priority) else claimed
        if want is None:
            return not background
        if not want or not self.strict:
            return want
        return requester in self.may_claim if self.may_claim is not None else not background

    def of_request(self, catalog: Catalog, body: Mapping[str, Any], priority: str, requester: str) -> bool:
        """The class of a request: its header, else the body's own `interactive` claim, else its requester."""
        claimed = body.get(INTERACTIVE_KEY)
        return self.interactive(catalog, priority, requester, claimed if isinstance(claimed, bool) else None)

    def classify(self, catalog: Catalog, body: Mapping[str, Any], priority: str, requester: str) -> dict[str, Any]:
        """The body with its class set (`of_request`)."""
        return {**body, INTERACTIVE_KEY: self.of_request(catalog, body, priority, requester)}

    def for_queue(self, body: Mapping[str, Any], interactive: bool) -> dict[str, Any]:
        """The body a route queues once it has worked out the class: under `fair` the caller's own
        claim only (Broker.submit derives the class again from it, the header and the requester; a
        class written in here would be re-read as a claim), under `fifo` the class itself."""
        return dict(body) if self.strict else {**body, INTERACTIVE_KEY: interactive}
