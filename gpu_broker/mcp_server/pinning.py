"""The stdio relay's plain-http rule: a credential goes over plain http only to loopback or
private addresses, and only to the addresses that were checked.

`vetted` resolves the broker's name once and returns those addresses when every one is loopback or
private (an IPv6 form that carries an IPv4 address is judged by that too). `PinnedBackend` then
connects only to them, for every connection the relay opens over its lifetime: a name that
answers a private address at the check and a public one later (DNS rebinding) never receives the
key. The Host header still carries the name; https needs no pin, TLS checks the name.
"""
from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterable
from typing import Any

import httpcore2
from httpcore2._backends.auto import AutoBackend

from ..netguard import embedded

LOOPBACK_NAMES = {"localhost"}
Resolver = Callable[..., list[Any]]


def literal(host: str) -> bool:
    try:
        return ipaddress.ip_address(host.split("%")[0]) is not None
    except ValueError:
        return False


def vetted(host: str, resolve: Resolver = socket.getaddrinfo) -> list[str] | None:
    """The addresses to connect to when `host` is loopback or private, else None. A literal is
    itself; `localhost` is the system's (no DNS); a name is every address it resolves to, all of
    which must be loopback or private (a name that does not resolve is not private)."""
    if host.lower() in LOOPBACK_NAMES:
        return []
    try:
        found = [host] if literal(host) else list(dict.fromkeys(ai[4][0] for ai in resolve(host, None)))
    except OSError:
        return None
    ips = [ipaddress.ip_address(a.split("%")[0]) for a in found]
    ok = bool(ips) and all(not ip.is_global and not (embedded(ip) or ip).is_global for ip in ips)
    return [] if ok and literal(host) else (found if ok else None)


def private(host: str, resolve: Resolver = socket.getaddrinfo) -> bool:
    return vetted(host, resolve) is not None


class PinnedBackend(AutoBackend):
    """Connects `host` only to its vetted addresses, trying each in turn; any other host is refused."""

    def __init__(self, host: str, addresses: Iterable[str]) -> None:
        super().__init__()
        self.host, self.addresses = host.lower(), list(addresses)

    async def connect_tcp(self, host: str, port: int, timeout: float | None = None, local_address: str | None = None,
                          socket_options: Iterable[Any] | None = None) -> httpcore2.AsyncNetworkStream:
        if host.lower() != self.host:
            raise httpcore2.ConnectError(f"{host} is not the checked broker host")
        last: Exception = httpcore2.ConnectError(f"{host}: no checked address")
        for address in self.addresses:
            try:
                return await super().connect_tcp(address, port, timeout, local_address, socket_options)
            except (httpcore2.ConnectError, httpcore2.ConnectTimeout) as e:
                last = e
        raise last
