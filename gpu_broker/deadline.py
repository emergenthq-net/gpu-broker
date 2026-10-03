"""Time limits for one `<slot>_url` fetch: a deadline that cuts its sockets off, and name lookups
that stop being waited for when the time is up.

A `Deadline` bounds the whole fetch: name resolution, each connection attempt, TLS, headers and
body, across every redirect hop. When it passes, a timer shuts down every socket the fetch
opened, which ends any read in progress however slowly the server drips bytes.

getaddrinfo cannot be interrupted, so lookups run on a small, fixed set of daemon threads and
the caller stops waiting at its deadline. A lookup that hangs holds one worker until the
resolver gives up; it never blocks the process from exiting, and never spawns more threads.
"""
from __future__ import annotations

import contextlib
import ipaddress
import queue
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from .constants import APP_NAME

Resolver = Callable[..., list[Any]]   # socket.getaddrinfo
LOOKUP_WORKERS = 4                    # name lookups in flight at once, across all fetches


class Deadline:
    """Time left for one fetch, and a timer that cuts its sockets off when it runs out."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.end, self.clock, self.expired = clock() + seconds, clock, False
        self._socks: list[socket.socket] = []
        self._lock = threading.Lock()
        self._timer = threading.Timer(seconds, self._expire)
        self._timer.daemon = True
        self._timer.start()

    def left(self) -> float:
        left = self.end - self.clock()
        if left <= 0 or self.expired:
            raise TimeoutError("the URL fetch deadline passed")
        return left

    def watch(self, sock: socket.socket) -> None:
        """Cut `sock` off at the deadline. If this raises, the caller still owns `sock`."""
        with self._lock:   # a dup: the original is detached when TLS wraps it
            self._socks.append(sock.dup())
            if self.expired:
                self._cut()

    def _cut(self) -> None:
        for s in self._socks:
            with contextlib.suppress(OSError):   # already closed by its connection
                s.shutdown(socket.SHUT_RDWR)

    def _expire(self) -> None:
        with self._lock:
            self.expired = True
            self._cut()

    def close(self) -> None:
        self._timer.cancel()
        with self._lock:
            for s in self._socks:
                s.close()
            self._socks.clear()


class _Lookups:
    """A fixed number of daemon threads, started on first use, that run name lookups."""

    def __init__(self, workers: int) -> None:
        self._jobs: queue.SimpleQueue[tuple[Future[Any], Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = \
            queue.SimpleQueue()
        self._workers, self._started, self._lock = workers, 0, threading.Lock()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        f: Future[Any] = Future()
        self._jobs.put((f, fn, args, kwargs))
        with self._lock:
            if self._started < self._workers:
                self._started += 1
                threading.Thread(target=self._work, name=f"{APP_NAME}-lookup", daemon=True).start()
        return f

    def _work(self) -> None:
        while True:
            f, fn, args, kwargs = self._jobs.get()
            if not f.set_running_or_notify_cancel():   # its caller already gave up
                continue
            try:
                f.set_result(fn(*args, **kwargs))
            except BaseException as e:  # noqa: BLE001 — handed to the waiting caller
                f.set_exception(e)


LOOKUPS = _Lookups(LOOKUP_WORKERS)


def resolve(host: str, port: int, deadline: Deadline, resolver: Resolver = socket.getaddrinfo) -> list[str]:
    """The addresses `host` names; an IP literal is itself, with no lookup."""
    with contextlib.suppress(ValueError):
        ipaddress.ip_address(host)
        return [host]
    left = deadline.left()
    f = LOOKUPS.submit(resolver, host, port, type=socket.SOCK_STREAM)
    try:
        infos = f.result(timeout=left)
    except TimeoutError:
        f.cancel()   # still queued: never run it
        raise
    return [info[4][0] for info in infos]
