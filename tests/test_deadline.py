"""deadline.py: the timer that cuts a fetch's sockets off, and name lookups bounded in time and
in threads. No network: socketpairs and fake resolvers."""
import socket
import threading
import time

import pytest

from gpu_broker import deadline, netguard
from gpu_broker.deadline import Deadline

DEADLINE_S = 1.0


def test_the_cut_reaches_a_socket_that_tls_has_taken_over():
    """ssl.wrap_socket detaches the socket the connector returned; the timer must still end
    the connection (it holds a duplicate of the descriptor)."""
    a, b = socket.socketpair()
    d = Deadline(0.2)
    d.watch(a)
    taken = socket.socket(fileno=a.detach())   # what TLS does to it
    taken.settimeout(DEADLINE_S * 3)
    try:
        assert taken.recv(1) == b""            # shut down by the timer, not by the peer
    finally:
        d.close(), taken.close(), b.close()


def test_a_hanging_name_lookup_is_bounded_too(monkeypatch):
    def slow(host, port, **kw):
        time.sleep(DEADLINE_S * 3)
        return []
    d, t0 = Deadline(DEADLINE_S / 4), time.monotonic()
    with pytest.raises(TimeoutError):
        netguard.connector(netguard.Policy(), d, slow)(("slow.test", 80), 999)
    assert time.monotonic() - t0 < DEADLINE_S
    d.close()


def test_once_expired_no_time_is_left_whatever_the_clock_says():
    d = Deadline(60, clock=lambda: 0.0)
    d._expire()
    with pytest.raises(TimeoutError):
        d.left()
    d.close()


def test_ip_literals_are_not_looked_up():
    def never(*a, **k):
        pytest.fail("an IP literal needs no lookup")
    d = Deadline(5)
    for ip in ("10.1.2.158", "2606:4700::1", "fe80::1%eth0"):
        assert deadline.resolve(ip, 80, d, never) == [ip]
    d.close()


def test_hanging_lookups_hold_a_fixed_number_of_daemon_threads(monkeypatch):
    """Five lookups that never return start only two workers; the three still queued when
    their callers give up are never run."""
    pool, gate, calls = deadline._Lookups(2), threading.Event(), []
    monkeypatch.setattr(deadline, "LOOKUPS", pool)
    before = {t for t in threading.enumerate() if t.name.endswith("-lookup")}

    def hang(host, port, **kw):
        calls.append(host)
        gate.wait(10)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", port))]
    for i in range(5):
        d = Deadline(0.05)
        with pytest.raises(TimeoutError):
            deadline.resolve(f"h{i}.test", 80, d, hang)
        d.close()
    workers = {t for t in threading.enumerate() if t.name.endswith("-lookup")} - before
    assert len(workers) == 2 and all(t.daemon for t in workers)
    gate.set()
    time.sleep(0.2)
    assert sorted(calls) == ["h0.test", "h1.test"]
