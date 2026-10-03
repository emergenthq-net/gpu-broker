"""Fetching a caller-chosen URL without letting the caller reach the broker's own network.

Used for every `<slot>_url` (image, end_image, video). Every connection the opener makes — the
first request and each redirect hop — resolves the host itself, refuses unless every address is
public (`is_global`) or inside an operator allowlist (`inputs.url_allow_networks`), and then
connects to the vetted address, so a DNS answer cannot change between the check and the
connection. IPv6 forms that carry an IPv4 address (mapped, NAT64, IPv4-compatible) must pass for
the embedded address too. TLS still verifies the certificate against the host name.
Environment proxies are ignored: a proxy would make the check meaningless.

One exception, for chained jobs (one job's output URL fed to the next): a GET of ComfyUI's
`/view?type=output` may reach the broker's own ComfyUI — the addresses that the hosts of
comfy.url / comfy.public_url resolve to, on their ports. Nothing else there is reachable:
not other paths, not other jobs' inputs (`type=input`), not other ports.

Redirect bodies are never read: a 30x is closed and followed (or refused) from its headers.
Time limits (deadline.py) bound the whole fetch.
"""
from __future__ import annotations

import http.client
import ipaddress
import socket
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .constants import HTTP_SCHEMES
from .deadline import Deadline, Resolver, resolve

Network = ipaddress.IPv4Network | ipaddress.IPv6Network
Address = ipaddress.IPv4Address | ipaddress.IPv6Address
NAT64 = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48"))   # RFC 6052, 8215
V4_COMPAT = ipaddress.ip_network("::/96")   # deprecated ::a.b.c.d (:: and ::1 are not IPv4)
V4_BITS = 0xFFFFFFFF
DEFAULT_PORTS = {"http": 80, "https": 443}
COMFY_VIEW, COMFY_OUTPUT = "/view", "output"   # the only ComfyUI request a fetch may make


class Refused(OSError):
    """The URL resolves to an address the broker may not contact."""


@dataclass(frozen=True)
class Policy:
    networks: tuple[Network, ...] = ()                    # operator allowlist
    endpoints: frozenset[tuple[str, int]] = frozenset()   # the broker's ComfyUI: (host, port)


def endpoints(*urls: str) -> frozenset[tuple[str, int]]:
    """(host, port) of configured service URLs, e.g. the broker's own ComfyUI."""
    out = set()
    for url in urls:
        u = urlsplit(url)
        if u.hostname and u.scheme in DEFAULT_PORTS:
            out.add((u.hostname.lower(), u.port or DEFAULT_PORTS[u.scheme]))
    return frozenset(out)


def embedded(ip: Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 address carries, if any."""
    if not isinstance(ip, ipaddress.IPv6Address):
        return None
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if any(ip in n for n in NAT64) or (ip in V4_COMPAT and int(ip) > 1):
        return ipaddress.IPv4Address(int(ip) & V4_BITS)
    return None


def _ip(address: str) -> Address:
    return ipaddress.ip_address(address.split("%")[0])   # drop an IPv6 zone id


def allowed(address: str, extra: tuple[Network, ...]) -> bool:
    ip = _ip(address)
    inner = embedded(ip)
    checked = (inner,) if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped else (ip, inner)
    return all(a.is_global or any(a in n for n in extra) for a in checked if a is not None)


def comfy_output(req: urllib.request.Request) -> bool:
    """A GET of one ComfyUI output file: the only request the ComfyUI exception covers."""
    u = urlsplit(req.full_url)
    q = parse_qs(u.query, keep_blank_values=True)
    names = [v for k in ("filename", "subfolder") for v in q.get(k, [])]
    return (req.get_method() == "GET" and u.path == COMFY_VIEW and q.get("type") == [COMFY_OUTPUT]
            and not any(".." in v for v in names))


def connector(policy: Policy, deadline: Deadline, resolver: Resolver = socket.getaddrinfo,
              comfy_ok: bool = False) -> Callable[..., socket.socket]:
    """A replacement for HTTPConnection._create_connection that only reaches allowed addresses.
    `comfy_ok`: this request may also reach the broker's own ComfyUI (see comfy_output)."""
    def own(port: int) -> set[Address]:
        return {_ip(a) for host, p in policy.endpoints if p == port for a in resolve(host, p, deadline, resolver)}

    def create_connection(address: tuple[str, int], timeout: Any = None, source_address: Any = None) -> socket.socket:
        host, port = address
        vetted = resolve(host, port, deadline, resolver)
        ok = bool(vetted) and (all(allowed(a, policy.networks) for a in vetted)
                               or (comfy_ok and {_ip(a) for a in vetted} <= own(port)))
        if not ok:
            raise Refused(f"{host} is not a public address")
        error: OSError = Refused(f"{host}: no address")
        for a in vetted:
            try:
                sock = socket.create_connection((a, port), deadline.left(), source_address)
            except OSError as e:
                error = e
                continue
            try:
                deadline.watch(sock)
            except BaseException:
                sock.close()   # not handed to anyone yet: ours to close
                raise
            return sock
        raise error
    return create_connection


def _guarded(base: type[http.client.HTTPConnection], create: Callable[..., socket.socket]) -> Callable[..., Any]:
    def make(*args: Any, **kwargs: Any) -> http.client.HTTPConnection:
        conn = base(*args, **kwargs)
        conn._create_connection = create  # type: ignore[attr-defined]
        return conn
    return make


class _Guard:
    def __init__(self, policy: Policy, deadline: Deadline, resolver: Resolver) -> None:
        self.args = (policy, deadline, resolver)

    def create(self, req: urllib.request.Request) -> Callable[..., socket.socket]:
        return connector(*self.args, comfy_ok=comfy_output(req))   # decided per request, so per hop


class _Http(urllib.request.HTTPHandler):
    def __init__(self, guard: _Guard) -> None:
        super().__init__()
        self.guard = guard

    def http_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(_guarded(http.client.HTTPConnection, self.guard.create(req)), req)


class _Https(urllib.request.HTTPSHandler):
    def __init__(self, guard: _Guard) -> None:
        super().__init__()
        self.guard = guard

    def https_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(_guarded(http.client.HTTPSConnection, self.guard.create(req)), req,
                            context=self._context)  # type: ignore[attr-defined]


class _HttpOnlyRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any,
                         newurl: str) -> urllib.request.Request | None:
        if urlsplit(newurl).scheme not in HTTP_SCHEMES:
            raise Refused(f"redirect to a non-HTTP address {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)

    def http_error_302(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any) -> Any:
        fp.close()   # the base class reads the body to the end, which may never come
        return super().http_error_302(req, fp, code, msg, headers)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def opener(policy: Policy, deadline: Deadline, resolver: Resolver = socket.getaddrinfo) -> urllib.request.OpenerDirector:
    guard = _Guard(policy, deadline, resolver)
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _HttpOnlyRedirects, _Http(guard), _Https(guard))
