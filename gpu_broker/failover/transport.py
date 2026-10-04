"""The HTTP client for cloud providers, with the timeouts failover needs kept apart: connecting,
the first byte, and (for a stream) the gap between events once it has started.

`send` returns once the call has *started*: the status, headers, and for a 2xx stream its first
chunk, or the whole body otherwise (any other status: errors and redirects are short). A caller
that does not relay a started stream must `close` it, or the connection stays open. Everything that can still go wrong before then
raises Unreachable with a fixed description (never the exception text, which can carry a URL
or more), so the caller may still fail over; `rest` is what is left of a stream.
The standard library's http.client is used directly rather than urllib so the connect timeout
and the first-byte timeout can differ; HTTP(S)_PROXY is therefore not honoured.
"""
from __future__ import annotations

import http.client
import socket
import ssl
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .config import UpTimeouts

CHUNK = 65536


class Unreachable(Exception):
    """No usable answer: the provider could not be reached, or went quiet or away before one."""


@dataclass
class Answer:
    status: int
    headers: dict[str, str]
    body: bytes                                    # the body, or a stream's first chunk
    rest: Iterator[bytes] = field(default_factory=lambda: iter(()))
    stream: bool = False
    close: Callable[[], None] = lambda: None       # drop a stream that will not be relayed


def _connection(url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    parts = urlsplit(url)
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    if parts.scheme == "https":
        return http.client.HTTPSConnection(parts.netloc, timeout=timeout, context=ssl.create_default_context()), path
    return http.client.HTTPConnection(parts.netloc, timeout=timeout), path


def _connect(conn: http.client.HTTPConnection) -> None:
    try:
        conn.connect()
    except socket.gaierror:
        raise Unreachable("DNS lookup failed") from None
    except ssl.SSLError:
        raise Unreachable("TLS handshake failed") from None
    except TimeoutError:
        raise Unreachable("connect timed out") from None
    except OSError:
        raise Unreachable("connection failed") from None


def send(url: str, method: str, headers: Mapping[str, str], body: bytes | None, stream: bool,
         t: UpTimeouts) -> Answer:
    conn, path = _connection(url, t.connect_s)
    _connect(conn)
    assert conn.sock is not None   # noqa: S101 — connect() succeeded
    conn.sock.settimeout(t.first_byte_s if stream else t.response_s)
    try:
        conn.request(method, path, body, dict(headers))
        resp = conn.getresponse()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        if not stream or not http.client.OK <= resp.status < http.client.MULTIPLE_CHOICES:
            data = resp.read()
            conn.close()
            return Answer(resp.status, hdrs, data)
        first = resp.read1(CHUNK)
    except TimeoutError:
        conn.close()
        raise Unreachable("timed out before the first byte") from None
    except (OSError, http.client.HTTPException):
        conn.close()
        raise Unreachable("connection dropped before the answer") from None
    if not first:
        conn.close()
        raise Unreachable("stream closed before its first event")
    conn.sock.settimeout(t.idle_s)
    return Answer(resp.status, hdrs, first, _rest(conn, resp), stream=True, close=conn.close)


def _rest(conn: http.client.HTTPConnection, resp: http.client.HTTPResponse) -> Iterator[bytes]:
    """The stream after its first chunk; a drop or an idle gap past `idle_s` raises Unreachable."""
    try:
        while chunk := resp.read1(CHUNK):
            yield chunk
    except TimeoutError:
        raise Unreachable("stream went quiet") from None
    except (OSError, http.client.HTTPException):
        raise Unreachable("stream broke off") from None
    finally:
        conn.close()
