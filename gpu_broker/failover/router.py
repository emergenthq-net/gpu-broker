"""Send one request down its fallback chain: each provider in turn, then the local model.

`route` returns Served (a provider's answer, success or a client error, to relay as-is),
Local (the chain reached its local model: the caller runs the broker path with that name),
or Exhausted (every provider failed and the chain has no local model). A provider is skipped
without a network call when its breaker is open, its API is not the request's, or no
credential is at hand. The client's own provider key (`client`) is used only for a provider
with `pass_client_key`, in the header style it arrived in; otherwise the provider's configured
key. Only the request's own headers named in PASS reach the provider, so the broker credential
never does. A request that continues a provider-side conversation (`local=False`) never goes
to the local model, which cannot continue it.

An unstreamed request goes upstream as a stream and is reassembled into the unstreamed answer
(assemble.py), so the first-byte and idle timeouts apply to it too; a provider that refuses
the stream (400, 415, 422) is asked once more unstreamed, and that refusal never counts
against its breaker.

A 2xx stream is classified by its first real event (comments and pings skipped); any stream not
relayed to the caller is closed.

A stream that breaks after its first byte is not handed to another model (the client has
already shown part of an answer): it ends with an error event in the request's API shape.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from ..constants import BROKER_FIELDS, Event
from . import assemble, transport
from .breaker import Board
from .classify import Kind, Verdict, classify, failed, first_event, first_event_failed
from .config import Api, Provider, Upstreams
from .creds import DEFAULT_ANTHROPIC_VERSION, Cred, auth_headers
from .outcome import Exhausted, Local, Served, error_event
from .prober import Prober
from .tally import Tally

PASS = ("anthropic-version", "anthropic-beta", "openai-organization", "openai-project", "accept")
BROKER_ONLY = BROKER_FIELDS - {"model", "stream"}   # never sent to a provider
LEAD_MAX = 65536   # bytes read past pings and comments looking for a stream's first real event
STREAM_REFUSED = frozenset({400, 415, 422})   # an unstreamed request sent as a stream: ask again plainly
BROKE = "the upstream stream broke off after it had started; no other model was substituted"
Send = Callable[..., transport.Answer]
Emit = Callable[..., None]   # Store.event


class Router:
    def __init__(self, cfg: Upstreams, emit: Emit, env: Mapping[str, str] | None = None,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 send: Send = transport.send) -> None:
        self.cfg, self.emit, self.env, self.clock, self.wall, self.send = cfg, emit, env, clock, wall, send
        self.board = Board(cfg.breaker, list(cfg.providers), self._changed)
        self.tally = Tally(self._emit, clock)
        self.prober = Prober(self)

    def _emit(self, kind: str, **fields: Any) -> None:
        self.emit(kind, **fields)
        self.prober.wake.set()   # something new to wait for: a probe or a window's end

    def _changed(self, provider: str, state: str, reason: str) -> None:
        kind = {"open": Event.UPSTREAM_OPEN, "quota": Event.UPSTREAM_QUOTA}.get(state, Event.UPSTREAM_CLOSED)
        self._emit(kind, provider=provider, reason=reason)

    def credential(self, p: Provider, client: Cred | None) -> Cred | None:
        if client is not None and p.pass_client_key:
            return client
        return Cred(key) if (key := p.configured_key(self.env)) else None

    def route(self, api: Api, path: str, model: str, body: Mapping[str, Any], client: Cred | None,
              headers: Mapping[str, str], stream: bool, local: bool = True) -> Served | Local | Exhausted | None:
        """`client`: the client's own key for this API (None: none sent). None: no route matches `model`."""
        chain = self.cfg.chain(model)
        if chain is None:
            return None
        skipped: list[str] = []
        last: Served | None = None
        for name in chain:
            provider = self.cfg.providers.get(name)
            if provider is None:   # the local model: always last
                if not local:
                    break
                reason = "; ".join(skipped)
                self.tally.add(chain[0], f"local:{name}", model, reason)
                return Local(name, reason)
            if provider.api is not api:
                skipped.append(f"{name} speaks the {provider.api} API")
                continue
            cred = self.credential(provider, client)
            if cred is None:
                skipped.append(f"{name}: no credential (set its key_env{'' if client is None else ', or pass_client_key'})")
                continue
            if why := self.board.allow(name, cred.value, self.clock()):
                skipped.append(why)
                continue
            got = self._attempt(provider, path, body, cred, headers, stream)
            if isinstance(got, Served):
                got.skipped = skipped
                return got
            verdict, answer = got
            skipped.append(verdict.reason)
            last = answer or last
        return Exhausted("; ".join(skipped), last, local)

    def _attempt(self, p: Provider, path: str, body: Mapping[str, Any], cred: Cred, headers: Mapping[str, str],
                 stream: bool) -> Served | tuple[Verdict, Served | None]:
        sent = {k: v for k, v in body.items() if k not in BROKER_ONLY}
        passed = {k: h for k in PASS if (h := headers.get(k))}
        out = {"content-type": "application/json", **passed, **auth_headers(p.api, cred)}
        if p.api is Api.ANTHROPIC:
            out.setdefault("anthropic-version", DEFAULT_ANTHROPIC_VERSION)
        url, t = p.url + path, self.cfg.timeouts
        first = sent if stream else assemble.as_stream(path, sent)   # unstreamed goes up as a stream
        try:
            a = self.send(url, "POST", out, json.dumps(first).encode(), True, t)
            verdict = classify(p.name, a.status, a.headers, a.body, self.wall)
            if not stream and verdict.kind is Kind.CLIENT and a.status in STREAM_REFUSED:
                a = self.send(url, "POST", out, json.dumps(sent).encode(), False, t)
                verdict = classify(p.name, a.status, a.headers, a.body, self.wall)
            if verdict.kind is Kind.OK and a.stream and (stream or a.headers.get("content-type", "").startswith(assemble.SSE)):
                a.body = self._lead(a)
                verdict = first_event_failed(p.name, first_event(a.body) or b"") or verdict
            if not stream and verdict.kind is Kind.OK and a.stream:
                a = assemble.gather(path, a)
        except transport.Unreachable as e:
            down = failed(p.name, str(e))
            self.board.failure(p.name, self.clock(), down.reason)
            return down, None
        relay = a.stream and verdict.kind is Kind.OK
        if a.stream and not relay:
            a.close()
        served = Served(p.name, a.status, a.headers, b"" if a.stream else a.body,
                        self._relay(p.name, path, a) if relay else None)
        if verdict.kind is Kind.OK:
            self.board.success(p.name, cred.value)
        elif verdict.kind is Kind.CLIENT:
            self.board.reachable(p.name)
            return served
        elif verdict.kind is Kind.QUOTA:
            self.board.exhausted(p.name, cred.value, self.clock(), verdict.reason, verdict.retry_after_s)
            return verdict, served
        else:
            self.board.failure(p.name, self.clock(), verdict.reason)
            return verdict, None
        return served

    @staticmethod
    def _lead(a: transport.Answer) -> bytes:
        """The stream read up to and including its first real event (raises Unreachable if it ends first)."""
        buf = a.body
        while first_event(buf) is None:
            if len(buf) > LEAD_MAX:
                a.close()
                raise transport.Unreachable("no event in the stream's opening")
            more = next(a.rest, b"")
            if not more:
                raise transport.Unreachable("stream closed before its first event")
            buf += more
        return buf

    def _relay(self, provider: str, path: str, a: transport.Answer) -> Iterator[bytes]:
        yield a.body
        try:
            yield from a.rest
        except transport.Unreachable as e:
            self.board.failure(provider, self.clock(), failed(provider, str(e)).reason)
            yield error_event(path, BROKE)

    def probe_once(self) -> None:
        """Probe every provider whose breaker is due (operator keys only), and wait for them."""
        self.prober.probe_once(block=True)

    def start(self) -> None:
        if self.cfg.enabled:
            threading.Thread(target=self.prober.run, name="upstream-probe", daemon=True).start()

    def stop(self) -> None:
        self.prober.stop()

    def view(self) -> dict[str, Any]:
        """Breakers and routes, plus each provider's API and whether it passes the client's own key
        through (connect reads that to decide whether a tool keeps its own key); no credentials."""
        board = self.board.view(self.clock())
        return {"enabled": self.cfg.enabled, "routes": {k: list(v) for k, v in self.cfg.routes.items()},
                "providers": {n: board[n] | {"api": p.api.value, "pass_client_key": p.pass_client_key}
                              for n, p in self.cfg.providers.items()}}
