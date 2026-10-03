"""`<slot>_url` fetches (image, end_image, video) reach only public addresses (or allowlisted
networks), checked after DNS on every connection. No network: the connector is driven directly
(loopback-server tests: test_netguard_http.py)."""
import ipaddress
import socket
import urllib.request

import pytest

from gpu_broker import netguard
from gpu_broker.deadline import Deadline
from gpu_broker.settings import Inputs

REAL_OPEN = urllib.request.OpenerDirector.open   # captured before conftest blocks the network
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.1.2.3", "172.16.0.9", "192.168.1.5", "169.254.169.254",
                                "0.0.0.0", "100.64.0.1", "::1", "fe80::1%eth0", "fc00::1", "::",
                                "::ffff:127.0.0.1", "::ffff:10.1.2.1",
                                "64:ff9b::a01:201", "64:ff9b::7f00:1", "64:ff9b::a9fe:a9fe",   # NAT64 (/96)
                                "64:ff9b:1::a01:201", "::a00:1", "::7f00:1", "::a9fe:a9fe"])   # NAT64 local, v4-compat
def test_non_public_addresses_are_refused(ip):
    assert not netguard.allowed(ip, ())


def test_public_addresses_and_allowlisted_networks_pass():
    assert netguard.allowed("93.184.215.14", ()) and netguard.allowed("2606:4700::1", ())
    lan = Inputs(url_allow_networks=("10.1.2.0/24",)).url_allow_networks
    assert netguard.allowed("10.1.2.158", lan) and not netguard.allowed("10.1.3.1", lan)
    assert netguard.allowed("::ffff:10.1.2.7", lan)


def test_embedded_ipv4_must_pass_too():
    assert netguard.allowed("64:ff9b::5db8:d70e", ())   # NAT64 of 93.184.215.14: both are public
    assert netguard.allowed("::5db8:d70e", ())          # IPv4-compatible form of the same
    assert str(netguard.embedded(ipaddress.ip_address("64:ff9b::a01:201"))) == "10.1.2.1"
    assert netguard.embedded(ipaddress.ip_address("::1")) is None   # loopback, not ::0.0.0.1
    assert netguard.embedded(ipaddress.ip_address("2606:4700::1")) is None
    local64 = Inputs(url_allow_networks=("64:ff9b:1::/48",)).url_allow_networks
    assert not netguard.allowed("64:ff9b:1::a01:201", local64)          # outer allowed, inner 10.1.2.1 not
    assert netguard.allowed("64:ff9b:1::5db8:d70e", local64)
    assert not netguard.allowed("64:ff9b:1::5db8:d70e", ())   # inner public, outer (local NAT64) not


def test_endpoints_come_from_service_urls():
    assert netguard.endpoints("http://10.1.2.158:8188", "https://Comfy.LAN/x", "http://h") == {
        ("10.1.2.158", 8188), ("comfy.lan", 443), ("h", 80)}
    assert netguard.endpoints("", "ftp://h/x") == frozenset()


def deadline(s=5):
    return Deadline(s)


def resolver(*ips):
    def resolve(host, port, **kw):
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]
    return resolve


@pytest.mark.parametrize("ips", [("10.1.2.1",), ("93.184.215.14", "127.0.0.1"), ()])
def test_connector_refuses_unless_every_answer_is_allowed_and_never_connects(ips, monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("must not connect"))
    with pytest.raises(netguard.Refused):
        netguard.connector(netguard.Policy(), deadline(), resolver(*ips))(("evil.test", 80), 5)


class Sock:
    def dup(self):
        return self

    def shutdown(self, how):
        pass

    def close(self):
        pass


def test_connector_connects_to_the_vetted_address_within_the_time_left(monkeypatch):
    seen, sock = [], Sock()
    monkeypatch.setattr(socket, "create_connection", lambda addr, t, *a: seen.append((addr, t)) or sock)
    d = deadline(5)
    assert netguard.connector(netguard.Policy(), d, resolver("93.184.215.14"))(("cdn.test", 443), 999) is sock
    ((addr, t),) = seen
    assert addr == ("93.184.215.14", 443) and 4 < t <= 5   # the time left, not the caller's 999 s
    d.close()


def test_the_brokers_own_comfy_is_reachable_only_for_output_views_on_its_addresses(monkeypatch):
    """comfy.url names comfy.lan (10.1.2.158); the exception follows its resolved address and port."""
    monkeypatch.setattr(socket, "create_connection", lambda *a: Sock())
    own, d = netguard.Policy(endpoints=netguard.endpoints("http://comfy.lan:8188")), deadline()
    lookup = {"comfy.lan": "10.1.2.158", "alias.lan": "10.1.2.158", "other.lan": "10.1.2.159"}
    dns = lambda host, port, **kw: resolver(lookup[host])(host, port)  # noqa: E731
    connect = lambda host, port, ok: netguard.connector(own, d, dns, comfy_ok=ok)((host, port), 5)  # noqa: E731
    for host in ("comfy.lan", "alias.lan", "10.1.2.158"):   # by any name, the same address
        connect(host, 8188, True)
    for host, port, ok in [("10.1.2.158", 8188, False),      # not an output view
                           ("10.1.2.158", 22, True), ("other.lan", 8188, True), ("10.1.2.159", 8188, True)]:
        with pytest.raises(netguard.Refused):
            connect(host, port, ok)
    d.close()


@pytest.mark.parametrize(("method", "url", "ok"), [
    ("GET", "http://c/view?filename=a.png&type=output", True),
    ("GET", "http://c/view?filename=a.png&subfolder=s&type=output", True),
    ("GET", "http://c/view?filename=broker-0123456789ab-image.png&type=input", False),   # another job's input
    ("GET", "http://c/view?filename=a.png&type=temp", False),
    ("GET", "http://c/view?filename=a.png", False),
    ("GET", "http://c/view?filename=a.png&type=output&type=input", False),
    ("GET", "http://c/view?filename=../input/a.png&type=output", False),
    ("GET", "http://c/view?filename=a.png&subfolder=..&type=output", False),
    ("GET", "http://c/upload/image?type=output", False),
    ("GET", "http://c/viewer?filename=a.png&type=output", False),
    ("GET", "http://c/view/x?filename=a.png&type=output", False),
    ("GET", "http://c/history?type=output", False),
    ("POST", "http://c/view?filename=a.png&type=output", False)])
def test_only_a_get_of_one_output_file_counts_as_comfy_output(method, url, ok):
    assert netguard.comfy_output(urllib.request.Request(url, method=method)) is ok  # noqa: S310 — never opened


def test_a_socket_the_deadline_cannot_watch_is_closed(monkeypatch):
    class Owned(Sock):
        closed = False

        def close(self):
            self.closed = True
    sock, d = Owned(), deadline()
    monkeypatch.setattr(socket, "create_connection", lambda *a: sock)
    monkeypatch.setattr(d, "watch", lambda s: (_ for _ in ()).throw(OSError(24, "Too many open files")))
    with pytest.raises(OSError, match="Too many"):
        netguard.connector(netguard.Policy(), d, resolver("93.184.215.14"))(("cdn.test", 80), 5)
    assert sock.closed
    d.close()


def test_allowed_networks_are_parsed_once_in_settings():
    with pytest.raises(ValueError, match="not a network"):
        Inputs(url_allow_networks=("10.1.2.0/33",))
    (net,) = Inputs(url_allow_networks=("10.1.2.5/24",)).url_allow_networks
    assert net == ipaddress.ip_network("10.1.2.0/24")
